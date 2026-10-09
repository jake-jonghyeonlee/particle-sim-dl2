"""Local embedding index preserving case bindings, physical frames and coverage."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch

from .checkpoint import fingerprint, load_checkpoint
from .data import load_export
from .model import Encoder, ModelConfig
from .schema import ContractError
from .views import ViewConfig

INDEX_SCHEMA = 'psdl2_index_v1'


def case_signature(metadata):
    endpoints = metadata['endpoints']
    # Binding is ordered by the producer's case, not guessed from mask proximity.
    return {'failure_type': metadata['failure_type'], 'analysis_layer': metadata['analysis_layer'],
            'endpoints': [{k: ep[k] for k in ('electrode_id', 'role', 'actual_layer')} for ep in endpoints],
            'connection_observation': metadata['connections']['observation'],
            'cnt_binding': metadata['connections']['case_endpoints'],
            'via_count': len(metadata['connections']['via_assets'])}


def candidate_record(bundle, embedding):
    metadata = bundle.metadata
    return {'roi_id': metadata['roi_id'], 'product_id': metadata['source_identity']['product_id'],
            'source_identity': deepcopy(metadata['source_identity']),
            'frame': deepcopy(metadata['source_frame']), 'case_id': metadata['source_identity']['case_id'],
            'center_xy_um': deepcopy(metadata.get('original_center_xy_um', metadata['center_xy_um'])),
            'source_version': metadata['source_version'], 'source_versions': deepcopy(metadata['versions']),
            'source_assets': deepcopy(metadata['source_assets']),
            'case_signature': case_signature(metadata),
            'valid_coverage': {'near': deepcopy(bundle.near.metadata['valid_coverage']),
                               'wide': deepcopy(bundle.wide.metadata['valid_coverage'])},
            'crop_geometry': {'near': deepcopy(bundle.near.metadata), 'wide': deepcopy(bundle.wide.metadata)},
            'embedding': [float(v) for v in embedding]}


def load_encoder(checkpoint):
    from .checkpoint import read_checkpoint
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = raw['config']
    read_checkpoint(checkpoint, config)
    model = Encoder(ModelConfig(**config['model']))
    load_checkpoint(checkpoint, model, config)
    model.eval()
    return model, ViewConfig.from_dict(config['views']), fingerprint(
        {'config': raw['fingerprint'], 'weights': raw['model_fingerprint']})


def build_index(rois, model, view_config, checkpoint_fingerprint):
    records, rejected = [], []
    model.eval()
    with torch.no_grad():
        for roi in rois:
            try:
                bundle = view_config.bundle(roi)
                embedding = model([bundle])['embedding'][0].cpu().tolist()
                records.append(candidate_record(bundle, embedding))
            except ContractError as exc:
                rejected.append({'roi_id': roi.row['roi_id'], 'reason': str(exc)})
    result = {'schema_version': INDEX_SCHEMA, 'checkpoint_fingerprint': checkpoint_fingerprint,
              'view_fingerprint': fingerprint(view_config_dict(view_config)),
              'records': sorted(records, key=lambda r: r['roi_id']), 'rejected': rejected}
    from .search import validate_index
    validate_index(result)
    return result


def view_config_dict(config):
    from dataclasses import asdict
    return asdict(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build an offline original-ROI embedding index with local weights.')
    parser.add_argument('export', type=Path); parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path); parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        export = load_export(args.export, assets_root=args.assets_root)
        model, views, fp = load_encoder(args.checkpoint)
        result = build_index(export.ready, model, views, fp)
        result['blocked_export_records'] = [dict(roi_id=r['roi_id'], reasons=r['reasons']) for r in export.blocked]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, allow_nan=False, indent=2))
        print(json.dumps({'n_indexed': len(result['records']), 'n_rejected': len(result['rejected']),
                          'n_blocked': len(export.blocked), 'checkpoint_fingerprint': fp}))
        return 0 if result['records'] else 3
    except (ContractError, KeyError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
