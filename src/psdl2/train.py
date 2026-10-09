"""Explicit local training entry point; nothing trains during import."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from .checkpoint import fingerprint, load_checkpoint, save_checkpoint
from .data import load_export
from .losses import LossConfig, objective, read_labels, relation_matrix, validate_label_ids
from .model import Encoder, ModelConfig
from .schema import ContractError, number
from .split import record_for_roi, validate_split
from .views import ViewConfig, generate_pair


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


def paired_forward(model, rois, view_config, loss_config, rng, *, labels=None):
    first, second, ids, rejected = [], [], [], []
    for roi in rois:
        try:
            a, b, _ = generate_pair(roi, view_config, rng)
            first.append(a); second.append(b); ids.append(roi.row['roi_id'])
        except ContractError as exc:
            rejected.append({'roi_id': roi.row['roi_id'], 'reason': str(exc)})
    reason_counts = dict(Counter(row['reason'] for row in rejected))
    if len(first) < 2:
        raise ContractError('effective_batch_at_least_two_required_after_view_rejection: '+json.dumps(
            {'accepted_pair_count': len(first), 'rejected_pair_count': len(rejected), 'reasons': reason_counts}, sort_keys=True))
    z1 = model(first)['projection']; z2 = model(second)['projection']
    relations = None if labels is None else relation_matrix(labels, ids)
    losses = objective(z1, z2, loss_config, relations=relations)
    losses.update(accepted_pair_count=len(first), rejected_pair_count=len(rejected), rejection_reasons=reason_counts)
    return losses, rejected


def train(rois, config, output, *, lineages, assignments, labels=None, resume=None):
    view_config = ViewConfig.from_dict(config['views'])
    if labels is not None:
        validate_label_ids(labels, [r.row['roi_id'] for r in rois])
    records = [record_for_roi(r, lineages[r.row['product_id']], view_config) for r in rois]
    split = validate_split(records, assignments)
    training = [r for r in rois if r.row['roi_id'] in split['train']]
    if len(training) < 2:
        raise ContractError('training_split_requires_two_locations')
    # Fingerprint the exact data, split and optional supervision as well as the run config.
    # The requested total epoch budget controls how long to continue; it does not
    # change the model/data/objective or invalidate an otherwise compatible resume.
    semantic_config = {key: value for key, value in config.items() if key != 'epochs'}
    accepted = {**semantic_config, 'training_records': records, 'split': assignments,
                'source_versions': [r.versions for r in rois],
                'source_assets': [r.row['source_assets'] for r in rois], 'labels': labels}
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    model = Encoder(ModelConfig(**config['model']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    epoch = 0
    if resume is not None:
        epoch = load_checkpoint(resume, model, accepted, optimizer=optimizer, restore_rng=True, view_rng=rng)
        if config['epochs'] <= epoch:
            raise ContractError('resume_target_epochs_must_exceed_saved_epoch')
    history, epoch_history = [], []
    for current in range(epoch, config['epochs']):
        order = rng.permutation(len(training)).tolist()
        batches = [order[i:i+config['batch_size']] for i in range(0, len(order), config['batch_size'])]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        model.train()
        epoch_accepted, epoch_rejected, epoch_reasons = 0, 0, Counter()
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            losses, rejected = paired_forward(model, [training[i] for i in batch], view_config,
                                               LossConfig(**config['loss']), rng, labels=labels)
            losses['total'].backward(); optimizer.step()
            epoch_accepted += losses['accepted_pair_count']; epoch_rejected += losses['rejected_pair_count']
            epoch_reasons.update(losses['rejection_reasons'])
            history.append({'epoch': current+1, 'rejected_views': rejected,
                            **{key: float(value.detach()) if torch.is_tensor(value) else value
                               for key, value in losses.items() if key != 'alignment_per_location'}})
        epoch_history.append({'epoch': current+1, 'accepted_pair_count': epoch_accepted,
                              'rejected_pair_count': epoch_rejected, 'rejection_reasons': dict(epoch_reasons)})
        save_checkpoint(output, model, optimizer, accepted, epoch=current+1, view_rng=rng, target_epochs=config['epochs'])
    return {'status': 'completed', 'epochs': config['epochs'], 'config_fingerprint': fingerprint(accepted),
            'split': split, 'history': history, 'epoch_history': epoch_history,
            'unversioned_producer_accepted': any(r.versions.get('unversioned_producer_accepted', False) for r in rois), 'performance_status': 'real_performance_pending'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train label-free DL2 on explicit original ROI views offline.')
    parser.add_argument('export', type=Path); parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--config', required=True, type=Path); parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--lineages', required=True, type=Path, help='JSON product_id -> lineage_id mapping')
    parser.add_argument('--split', required=True, type=Path, help='JSON roi_id -> train/validation/test mapping')
    parser.add_argument('--labels', type=Path, help='Optional roi_relations_v1; never required')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    args = parser.parse_args(argv)
    try:
        export = load_export(args.export, assets_root=args.assets_root,
                             allow_unversioned_producer=args.allow_unversioned_producer)
        result = train(export.ready, read_config(args.config), args.out,
                       lineages=json.loads(args.lineages.read_text()), assignments=json.loads(args.split.read_text()),
                       labels=None if args.labels is None else read_labels(args.labels), resume=args.resume)
        print(json.dumps(result, allow_nan=False)); return 0
    except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
