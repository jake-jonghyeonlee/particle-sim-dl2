from dataclasses import asdict
import random

import numpy as np
import pytest
import torch

from psdl2.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from psdl2.data import load_export
from psdl2.model import Encoder
from psdl2.schema import ContractError
from test_model import model_config
from test_views import view_config


def checkpoint_config():
    return {'model': asdict(model_config()), 'views': asdict(view_config())}


def test_checkpoint_embedding_optimizer_and_all_rng_roundtrip(source, tmp_path):
    _, write, _ = source; roi = load_export(write()).ready[0]
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    rng = np.random.default_rng(42)
    model = Encoder(model_config()).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    # Synthetic optimizer state, installed without an optimisation/training step.
    for parameter in model.parameters():
        optimizer.state[parameter] = {'step': torch.tensor(2.), 'exp_avg': torch.ones_like(parameter)*.1,
                                      'exp_avg_sq': torch.ones_like(parameter)*.01}
    bundle = view_config().bundle(roi)
    expected_embedding = model([bundle])['embedding'].detach().clone()
    path = tmp_path/'checkpoint.pt'; config = checkpoint_config()
    save_checkpoint(path, model, optimizer, config, epoch=2, view_rng=rng)
    expected_random = (random.random(), np.random.random(), torch.rand(2), rng.random(2))
    copied = Encoder(model_config()); copied_optimizer = torch.optim.Adam(copied.parameters(), lr=.99)
    epoch = load_checkpoint(path, copied, config, optimizer=copied_optimizer, restore_rng=True, view_rng=rng)
    assert epoch == 2 and copied_optimizer.param_groups[0]['lr'] == .001
    torch.testing.assert_close(copied([bundle])['embedding'], expected_embedding)
    assert random.random() == expected_random[0] and np.random.random() == expected_random[1]
    torch.testing.assert_close(torch.rand(2), expected_random[2]); np.testing.assert_array_equal(rng.random(2), expected_random[3])
    for state in copied_optimizer.state.values():
        assert state['step'].item() == 2
        torch.testing.assert_close(state['exp_avg'], torch.full_like(state['exp_avg'], .1))


def test_incompatible_fingerprint_and_schema_fail(tmp_path):
    model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    path = tmp_path/'state.pt'; config = checkpoint_config()
    save_checkpoint(path, model, optimizer, config, epoch=1)
    with pytest.raises(ContractError, match='fingerprint'):
        load_checkpoint(path, model, {**config, 'extra': 'changed'})
    payload = torch.load(path, weights_only=True); payload['schema_version'] = 'old'
    torch.save(payload, path)
    with pytest.raises(ContractError, match='schema'): read_checkpoint(path, config)


def test_atomic_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    path = tmp_path/'state.pt'; config = checkpoint_config()
    save_checkpoint(path, model, optimizer, config, epoch=1)
    before = path.read_bytes()
    def fail(payload, stream):
        stream.write(b'partial'); raise OSError('synthetic storage failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError): save_checkpoint(path, model, optimizer, config, epoch=2)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_exact_weight_fingerprint_changes_index_identity(tmp_path):
    from psdl2.index import load_encoder
    model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    config = checkpoint_config(); first = tmp_path/'first.pt'; second = tmp_path/'second.pt'
    save_checkpoint(first, model, optimizer, config, epoch=1)
    with torch.no_grad(): next(model.parameters()).add_(.1)
    save_checkpoint(second, model, optimizer, config, epoch=2)
    assert load_encoder(first)[2] != load_encoder(second)[2]
    payload = torch.load(second, weights_only=True)
    payload['model'][next(iter(payload['model']))].add_(.1); torch.save(payload, second)
    with pytest.raises(ContractError, match='weight_fingerprint'): read_checkpoint(second, config)
