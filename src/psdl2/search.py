"""Case-compatible cosine search with a fixed primary cohort and separate expansion."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

from .schema import ContractError, IDENTITY_FIELDS, text


def _embedding(value):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2 or not np.isfinite(arr).all() or np.linalg.norm(arr) <= 0:
        raise ContractError('finite_nonzero_embedding_required')
    return arr


def cosine(first, second):
    a, b = _embedding(first), _embedding(second)
    if a.shape != b.shape:
        raise ContractError('embedding_dimension_mismatch')
    value = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return round(float(np.clip(value, -1, 1)), 6)


def validate_record(record):
    for key in ('roi_id', 'product_id', 'case_id', 'source_version'):
        text(record[key])
    if set(record['source_identity']) != set(IDENTITY_FIELDS):
        raise ContractError('complete_candidate_identity_required')
    for value in record['source_identity'].values():
        text(value)
    if record['product_id'] != record['source_identity']['product_id'] or record['case_id'] != record['source_identity']['case_id'] or record['frame']['frame_id'] != record['source_identity']['frame_id']:
        raise ContractError('candidate_identity_mismatch')
    from .schema import Frame, xy
    frame = Frame.from_dict(record['frame']); xy(record['center_xy_um'])
    for scale in ('near', 'wide'):
        fraction = record['valid_coverage'][scale]['valid_fraction']
        from .schema import number
        if not 0 < number(fraction) <= 1:
            raise ContractError('candidate_has_no_observed_coverage')
    from .views import TOPOLOGY_POLICY, CONNECTION_SEMANTICS, COMPARISON_BASIS
    if (record.get('topology_policy') != TOPOLOGY_POLICY or
            record.get('connection_semantics') != CONNECTION_SEMANTICS or
            record.get('comparison_basis') != COMPARISON_BASIS):
        raise ContractError('incompatible_candidate_view_policy')
    signature = record['case_signature']
    if set(signature) != {'failure_type', 'analysis_layer', 'endpoints', 'connection_observation', 'cnt_binding', 'via_count'}:
        raise ContractError('complete_case_signature_required')
    endpoint_key = lambda ep: (ep['electrode_id'], ep['role'], ep['actual_layer'])
    if sorted(map(endpoint_key, signature['endpoints'])) != sorted(map(endpoint_key, signature['cnt_binding'])):
        raise ContractError('CNT_endpoint_binding_mismatch')
    if len(signature['endpoints']) != 2:
        raise ContractError('two_candidate_endpoints_required')
    _embedding(record['embedding'])


def validate_index(index):
    from .index import INDEX_SCHEMA
    if index['schema_version'] != INDEX_SCHEMA:
        raise ContractError('unknown_index_schema')
    from .views import TOPOLOGY_POLICY, CONNECTION_SEMANTICS, COMPARISON_BASIS
    if (index.get('topology_policy') != TOPOLOGY_POLICY or
            index.get('connection_semantics') != CONNECTION_SEMANTICS or
            index.get('comparison_basis') != COMPARISON_BASIS):
        raise ContractError('incompatible_index_view_policy')
    for key in ('checkpoint_fingerprint', 'view_fingerprint'):
        value = index[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ContractError('index_fingerprint_required')
    seen, dims = set(), set()
    for record in index['records']:
        validate_record(record)
        if record['roi_id'] in seen:
            raise ContractError('duplicate_index_roi_id')
        seen.add(record['roi_id']); dims.add(len(record['embedding']))
    if len(dims) > 1:
        raise ContractError('embedding_dimension_mismatch')


def read_cohort(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw or len(set(raw)) != len(raw):
        raise ContractError('explicit_nonempty_unique_product_cohort_required')
    return {text(value) for value in raw}


def endpoint_key_for_identity(ep):
    return (ep['electrode_id'], ep['role'], ep['actual_layer'])


def search(query, index, cohort, *, topk, expand=False):
    validate_record(query); validate_index(index)
    if not isinstance(cohort, (set, list, tuple)) or not cohort:
        raise ContractError('explicit_DL1_product_cohort_required')
    cohort = {text(v) for v in cohort}
    if type(topk) is not int or topk < 1:
        raise ContractError('positive_topk_required')
    primary, supplementary, rejected = [], [], []
    from .index import compatibility_signature
    for candidate in index['records']:
        in_primary = candidate['product_id'] in cohort
        if not in_primary and not expand:
            continue
        if candidate['roi_id'] == query['roi_id'] and candidate['source_identity'] == query['source_identity']:
            rejected.append({'roi_id': candidate['roi_id'], 'reason': 'query_itself_excluded'}); continue
        if compatibility_signature(candidate['case_signature']) != compatibility_signature(query['case_signature']):
            rejected.append({'roi_id': candidate['roi_id'], 'reason': 'case_or_CNT_binding_incompatible'}); continue
        same_location = (candidate['source_identity'] == query['source_identity']
                         and candidate['frame'] == query['frame']
                         and candidate['center_xy_um'] == query['center_xy_um']
                         and candidate['source_version'] == query['source_version']
                         and candidate['source_assets'] == query['source_assets']
                         and candidate['case_signature']['analysis_layer'] == query['case_signature']['analysis_layer']
                         and sorted(map(endpoint_key_for_identity, candidate['case_signature']['endpoints']))
                             == sorted(map(endpoint_key_for_identity, query['case_signature']['endpoints'])))
        result = {key: deepcopy(value) for key, value in candidate.items() if key != 'embedding'}
        result.update(score=cosine(query['embedding'], candidate['embedding']),
                      search_scope='primary' if in_primary else 'supplementary',
                      same_physical_case=same_location)
        (primary if in_primary else supplementary).append(result)
    for rows in (primary, supplementary):
        rows.sort(key=lambda row: (-row['score'], row['roi_id']))
    if primary:
        status, reasons = 'available', []
    else:
        cohort_records = [r for r in index['records'] if r['product_id'] in cohort]
        status = 'unavailable'
        reasons = ['no_candidates_in_primary_cohort' if not cohort_records else 'no_case_compatible_primary_candidates']
    return {'query': {key: deepcopy(value) for key, value in query.items() if key != 'embedding'},
            'status': status, 'no_match_reasons': reasons, 'primary': primary[:topk],
            'supplementary': supplementary[:topk] if expand else [], 'rejected': rejected,
            'primary_cohort': sorted(cohort), 'expanded': bool(expand),
            'index_roi_ids': [r['roi_id'] for r in index['records']],
            'interpretation': 'similarity_only_not_registration_cause_or_risk'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Search one original ROI within an explicit DL1 cohort offline.')
    parser.add_argument('export', type=Path); parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--roi-id', required=True); parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--index', required=True, type=Path); parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--topk', required=True, type=int); parser.add_argument('--expand', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    args = parser.parse_args(argv)
    try:
        import torch
        from .checkpoint import fingerprint
        from .data import load_export
        from .index import candidate_record, load_encoder, view_config_dict
        imported = load_export(args.export, assets_root=args.assets_root,
                             allow_unversioned_producer=args.allow_unversioned_producer)
        rois = [r for r in imported.ready if r.row['roi_id'] == args.roi_id]
        if len(rois) != 1:
            raise ContractError('one_ready_query_roi_required')
        model, views, fp = load_encoder(args.checkpoint)
        index = json.loads(args.index.read_text())
        if index['checkpoint_fingerprint'] != fp or index['view_fingerprint'] != fingerprint(view_config_dict(views)):
            raise ContractError('incompatible_index_checkpoint_or_transform')
        bundle = views.bundle(rois[0])
        with torch.no_grad():
            query = candidate_record(bundle, model([bundle])['embedding'][0].tolist())
        result = search(query, index, read_cohort(args.cohort), topk=args.topk, expand=args.expand)
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, allow_nan=False, indent=2))
        print(json.dumps({'status': result['status'], 'primary': len(result['primary']), 'supplementary': len(result['supplementary']),
                          'no_match_reasons': result['no_match_reasons']}))
        return 0 if result['status'] == 'available' else 3
    except (ContractError, KeyError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
