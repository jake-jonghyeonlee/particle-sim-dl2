"""Explicit local training entry point; nothing trains during import."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from .checkpoint import fingerprint, load_checkpoint, save_checkpoint
from .data import load_export
from .losses import LossConfig, objective, read_labels, relation_matrix, validate_label_ids
from .model import Encoder, ModelConfig, resolve_device
from .progress import Progress, stderr_sink
from .schema import ContractError, number
from .split import record_for_roi, validate_split
from .views import ViewConfig, ViewRejection, generate_pair, preflight, CONNECTION_SEMANTICS, COMPARISON_BASIS, REJECTION_REASONS


def read_config(path):
    document = json.loads(Path(path).read_text())
    config = document.get('config', document)
    required = {'schema_version', 'model', 'views', 'loss', 'epochs', 'batch_size', 'learning_rate', 'seed'}
    if set(config) != required or config['schema_version'] != 'psdl2_run_v1':
        raise ContractError('explicit_run_config_v1_required')
    ModelConfig(**config['model']); ViewConfig.from_dict(config['views']); LossConfig(**config['loss'])
    for key in ('epochs', 'batch_size', 'seed'):
        if type(config[key]) is not int or config[key] < (2 if key == 'batch_size' else 0 if key == 'seed' else 1):
            raise ContractError('invalid_epochs_batch_or_seed')
    number(config['learning_rate'], positive=True)
    return config


def _counters():
    return dict(attempted_pairs=0, retry_attempts=0, rejected_attempts_by_reason={},
                accepted_pairs_generated=0, accepted_pairs_used=0, skipped_batches=0, optimizer_steps=0)


def paired_forward(model, rois, view_config, loss_config, rng, *, labels=None):
    if len({r.row['roi_id'] for r in rois}) != len(rois):
        raise ContractError('batch_requires_distinct_original_locations')
    first, second, ids, rejected = [], [], [], []
    counts = _counters()
    # Each location is visited once, retried only after a typed view rejection.
    for roi in rois:
        for attempt in range(view_config.max_pair_attempts):
            counts['attempted_pairs'] += 1
            counts['retry_attempts'] += int(attempt > 0)
            try:
                a, b, _ = generate_pair(roi, view_config, rng)
            except ViewRejection as exc:
                if str(exc) not in REJECTION_REASONS:
                    raise ContractError('unknown_view_rejection_reason: '+str(exc)) from exc
                rejected.append({'roi_id': roi.row['roi_id'], 'attempt': attempt+1, 'reason': str(exc)})
                continue
            first.append(a); second.append(b); ids.append(roi.row['roi_id'])
            break
    reasons = dict(Counter(row['reason'] for row in rejected))
    counts.update(rejected_attempts_by_reason=reasons, accepted_pairs_generated=len(first),
                  accepted_pair_count=len(first), rejected_pair_count=len(rejected), rejection_reasons=reasons)
    if len(first) < 2:
        counts['skipped_batches'] = 1
        return counts, rejected
    z1 = model(first)['projection']; z2 = model(second)['projection']
    relations = None if labels is None else relation_matrix(labels, ids)
    losses = objective(z1, z2, loss_config, relations=relations)
    counts['accepted_pairs_used'] = len(first)
    losses.update(counts)
    return losses, rejected


def train(rois, config, output, *, lineages, assignments, labels=None, resume=None, device='cpu',
          progress_sink=None, progress_interval=10.0):
    device = resolve_device(device)
    progress = Progress(progress_sink, progress_interval)
    view_config = ViewConfig.from_dict(config['views'])
    if labels is not None:
        validate_label_ids(labels, [r.row['roi_id'] for r in rois])
    records = [record_for_roi(r, lineages[r.row['product_id']], view_config) for r in rois]
    split = validate_split(records, assignments)
    training = [r for r in rois if r.row['roi_id'] in split['train']]
    if len(training) < 2:
        raise ContractError('training_split_requires_two_locations')
    for roi in training:
        preflight(roi, view_config)
    # Fingerprint the exact data, split and optional supervision as well as the run config.
    # The requested total epoch budget controls how long to continue; it does not
    # change the model/data/objective or invalidate an otherwise compatible resume.
    semantic_config = {key: value for key, value in config.items() if key != 'epochs'}
    accepted = {**semantic_config, 'training_records': records, 'split': assignments,
                'source_versions': [r.versions for r in rois],
                'source_assets': [r.row['source_assets'] for r in rois], 'labels': labels}
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    model = Encoder(ModelConfig(**config['model'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    epoch = 0
    if resume is not None:
        epoch = load_checkpoint(resume, model, accepted, optimizer=optimizer, restore_rng=True, view_rng=rng)
        if config['epochs'] <= epoch:
            raise ContractError('resume_target_epochs_must_exceed_saved_epoch')
    history, epoch_history = [], []
    if resume is not None:
        from .checkpoint import read_checkpoint
        epoch_history = read_checkpoint(resume, accepted).get('epoch_history', [])
    progress.emit('train', state='started', device=str(device),
                  cuda_device=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  locations=len(training), start_epoch=epoch, epochs=config['epochs'])
    for current in range(epoch, config['epochs']):
        epoch_started = time.monotonic()
        order = rng.permutation(len(training)).tolist()
        batches = [order[i:i+config['batch_size']] for i in range(0, len(order), config['batch_size'])]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        model.train()
        epoch_counts = _counters()
        epoch_reasons = Counter()
        epoch_batches = []
        for batch_number, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            losses, rejected = paired_forward(model, [training[i] for i in batch], view_config,
                                               LossConfig(**config['loss']), rng, labels=labels)
            if not losses['skipped_batches']:
                losses['total'].backward(); optimizer.step()
                losses['optimizer_steps'] = 1
            for key in epoch_counts:
                if key != 'rejected_attempts_by_reason':
                    epoch_counts[key] += losses[key]
            epoch_reasons.update(losses['rejected_attempts_by_reason'])
            entry = {'epoch': current+1, 'batch': batch_number, 'roi_ids': [training[i].row['roi_id'] for i in batch],
                     'rejected_views': rejected,
                     **{key: float(value.detach()) if torch.is_tensor(value) else value
                        for key, value in losses.items() if key != 'alignment_per_location'}}
            history.append(entry); epoch_batches.append(entry)
            if progress.due():
                progress.emit('train', state='running', epoch=current+1, epochs=config['epochs'],
                              batch=batch_number+1, batches=len(batches),
                              loss=None if losses['skipped_batches'] else round(entry['total'], 6))
        epoch_counts['rejected_attempts_by_reason'] = dict(epoch_reasons)
        epoch_entry = {'epoch': current+1, **epoch_counts, 'batches': epoch_batches}
        if not epoch_counts['optimizer_steps']:
            # Do not replace the last completed checkpoint with a failed epoch.
            error = ContractError('no_usable_batch_in_epoch: '+json.dumps(epoch_entry, sort_keys=True))
            error.statistics = epoch_entry
            raise error
        epoch_history.append(epoch_entry)
        save_checkpoint(output, model, optimizer, accepted, epoch=current+1, view_rng=rng,
                        target_epochs=config['epochs'], epoch_history=epoch_history)
        progress.emit('train', state='epoch_completed', epoch=current+1, epochs=config['epochs'],
                      optimizer_steps=epoch_counts['optimizer_steps'], skipped_batches=epoch_counts['skipped_batches'],
                      elapsed_s=round(time.monotonic()-epoch_started, 1),
                      cuda_max_allocated_mb=round(torch.cuda.max_memory_allocated(device)/2**20, 1) if device.type == 'cuda' else None)
    return {'status': 'completed', 'epochs': config['epochs'], 'device': str(device), 'config_fingerprint': fingerprint(accepted),
            'split': split, 'history': history, 'epoch_history': epoch_history,
            'topology_policy': view_config.topology_policy, 'max_pair_attempts': view_config.max_pair_attempts,
            'connection_semantics': CONNECTION_SEMANTICS, 'comparison_basis': COMPARISON_BASIS,
            'unversioned_producer_accepted': any(r.versions.get('unversioned_producer_accepted', False) for r in rois), 'performance_status': 'real_performance_pending'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train label-free DL2 on explicit original ROI views offline.')
    parser.add_argument('export', type=Path); parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--config', required=True, type=Path); parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--lineages', required=True, type=Path, help='JSON product_id -> lineage_id mapping')
    parser.add_argument('--split', required=True, type=Path, help='JSON roi_id -> train/validation/test mapping')
    parser.add_argument('--labels', type=Path, help='Optional roi_relations_v1; never required')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='cuda requires a CUDA build of torch; no CPU fallback')
    parser.add_argument('--no-progress', action='store_true', help='suppress JSON progress lines on stderr')
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    parser.add_argument('--max-asset-memory-mb', type=float, help='Reject the export if retained decoded masks would exceed this (transient decode buffers are extra)')
    args = parser.parse_args(argv)
    try:
        resolve_device(args.device)  # fail before the slow export load
        export = load_export(args.export, assets_root=args.assets_root,
                             allow_unversioned_producer=args.allow_unversioned_producer,
                             max_asset_memory_mb=args.max_asset_memory_mb)
        result = train(export.ready, read_config(args.config), args.out,
                       lineages=json.loads(args.lineages.read_text()), assignments=json.loads(args.split.read_text()),
                       labels=None if args.labels is None else read_labels(args.labels), resume=args.resume,
                       device=args.device, progress_sink=None if args.no_progress else stderr_sink)
        print(json.dumps(result, allow_nan=False)); return 0
    except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc), **({'statistics': exc.statistics} if hasattr(exc, 'statistics') else {})}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
