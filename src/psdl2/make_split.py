"""Build a leakage-checked split.json (roi_id -> train/validation/test) from product lineages and lineage assignments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import load_dataset
from .patches import publish, serialize
from .schema import ContractError
from .split import record_for_roi, validate_split
from .train import read_config
from .views import ViewConfig


def build_split(rois, view_config, lineages, assign):
    """Map every ready roi_id through product -> lineage -> split; the result passes ``validate_split``."""
    if not isinstance(lineages, dict) or not isinstance(assign, dict):
        raise ContractError('lineages_and_assign_objects_required')
    if not rois:
        raise ContractError('no_ready_rois_to_split')
    records, assignments = [], {}
    for roi in rois:
        product = roi.row['product_id']
        if product not in lineages:
            raise ContractError('product_without_lineage: '+str(product))
        lineage = lineages[product]
        if not isinstance(lineage, str) or not lineage:
            raise ContractError('invalid_lineage_id: '+str(product))
        if lineage not in assign:
            raise ContractError('lineage_unassigned: '+lineage)
        if assign[lineage] not in ('train', 'validation', 'test'):
            raise ContractError('invalid_split_name: '+lineage)
        records.append(record_for_roi(roi, lineage, view_config))
        assignments[roi.row['roi_id']] = assign[lineage]
    validate_split(records, assignments)
    return assignments


def main(argv=None):
    parser = argparse.ArgumentParser(description='Write split.json from a dataset, product lineages and lineage assignments.')
    parser.add_argument('--dataset', required=True, type=Path, help='psdl2_dataset_v1 manifest')
    parser.add_argument('--config', required=True, type=Path, help='run config; its views are checked against patch sets')
    parser.add_argument('--lineages', required=True, type=Path, help='JSON product_id -> lineage_id mapping')
    parser.add_argument('--assign', required=True, type=Path, help='JSON lineage_id -> train/validation/test mapping')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    parser.add_argument('--max-asset-memory-mb', type=float, help='Reject the dataset if retained decoded masks would exceed this')
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and not args.overwrite:
            raise ContractError('output_exists_use_overwrite')
        views = ViewConfig.from_dict(read_config(args.config)['views'])
        dataset = load_dataset(args.dataset, expected_views=views, allow_unversioned_producer=args.allow_unversioned_producer,
                               max_asset_memory_mb=args.max_asset_memory_mb)
        assignments = build_split(dataset.ready, views, json.loads(args.lineages.read_text()), json.loads(args.assign.read_text()))
        publish(args.out, serialize(assignments), args.overwrite)
        counts = {name: sum(1 for v in assignments.values() if v == name) for name in ('train', 'validation', 'test')}
        print(json.dumps({'out': str(args.out), 'records': len(assignments), 'split_counts': counts}, allow_nan=False))
        return 0
    except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
