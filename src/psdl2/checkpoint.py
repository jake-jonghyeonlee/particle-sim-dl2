"""Atomic, fingerprinted local checkpoints with optimizer and RNG resume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .schema import ContractError

CHECKPOINT_SCHEMA = 'psdl2_checkpoint_v2'


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def model_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, model, optimizer, config, *, epoch, view_rng=None, target_epochs=None, epoch_history=None):
    state = np.random.get_state()
    # CUDA RNG state is saved only for a CUDA run, so a CPU checkpoint resumes on a CPU-only machine.
    on_cuda = any(p.is_cuda for p in model.parameters())
    payload = {'schema_version': CHECKPOINT_SCHEMA, 'input_schema_version': 'original_roi_bundle_v1',
               'config': config, 'fingerprint': fingerprint(config), 'model': model.state_dict(),
               'model_fingerprint': model_fingerprint(model.state_dict()),
               'optimizer': optimizer.state_dict(), 'epoch': epoch, 'target_epochs': target_epochs,
               'epoch_history': [] if epoch_history is None else epoch_history,
               'rng': {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
                       'torch': torch.get_rng_state(),
                       'cuda': torch.cuda.get_rng_state_all() if on_cuda else [],
                       'views': None if view_rng is None else view_rng.bit_generator.state}}
    atomic_torch_save(payload, path)


def read_checkpoint(path, config):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('schema_version') != CHECKPOINT_SCHEMA or
            payload.get('input_schema_version') != 'original_roi_bundle_v1'):
        raise ContractError('incompatible_checkpoint_schema')
    if 'views' in payload['config']:
        from .views import ViewConfig
        ViewConfig.from_dict(payload['config']['views'])
    if payload.get('fingerprint') != fingerprint(config) or payload['fingerprint'] != fingerprint(payload['config']):
        raise ContractError('incompatible_config_fingerprint')
    if payload.get('model_fingerprint') != model_fingerprint(payload['model']):
        raise ContractError('incompatible_model_weight_fingerprint')
    if type(payload['epoch']) is not int or payload['epoch'] < 0:
        raise ContractError('invalid_checkpoint_epoch')
    return payload


def load_checkpoint(path, model, config, *, optimizer=None, restore_rng=False, view_rng=None):
    payload = read_checkpoint(path, config)
    model.load_state_dict(payload['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if restore_rng:
        if optimizer is None:
            raise ContractError('optimizer_required_for_resume')
        rng = payload['rng']; random.setstate(rng['python'])
        state = rng['numpy']; np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(rng['torch'])
        if rng['cuda']:
            if not torch.cuda.is_available():
                raise ContractError('resume_cuda_rng_device_unavailable')
            torch.cuda.set_rng_state_all(rng['cuda'])
        if rng['views'] is not None:
            if view_rng is None:
                raise ContractError('view_generator_required_for_resume')
            view_rng.bit_generator.state = rng['views']
    return payload['epoch']
