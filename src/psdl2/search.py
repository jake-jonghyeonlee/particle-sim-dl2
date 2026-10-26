"""Case-compatible cosine search with a fixed primary cohort and separate expansion."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
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


def validate_mask_derivation(value, signature=None):
    from .data import DERIVATION_SCHEMA
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'rule_version', 'processed_endpoint', 'removed_layers'}
            or value['schema_version'] != DERIVATION_SCHEMA):
        raise ContractError('invalid_record_mask_derivation')
    text(value['rule_version'])
    processed = value['processed_endpoint']
    if processed is not None:
        if not isinstance(processed, dict) or set(processed) != {'role', 'actual_layer'}:
            raise ContractError('invalid_record_mask_derivation')
        text(processed['role']); text(processed['actual_layer'])
    if signature is not None:
        # processed_endpoint is null exactly for lateral cases, else one of the signature endpoints.
        if (processed is None) != (signature['failure_type'] == 'lateral_short'):
            raise ContractError('invalid_record_mask_derivation')
        if processed is not None and sum(1 for ep in signature['endpoints'] if (ep['role'], ep['actual_layer']) == (processed['role'], processed['actual_layer'])) != 1:
            raise ContractError('invalid_record_mask_derivation')
    layers = value['removed_layers']
    if not isinstance(layers, list) or layers != sorted(set(text(layer) for layer in layers)):
        raise ContractError('invalid_record_mask_derivation')


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
    base_keys = {'failure_type', 'analysis_layer', 'endpoints', 'connection_observation', 'cnt_binding', 'via_count'}
    # 'mask_derivation' is optional: absent means unknown (single-export records carry none).
    if not base_keys <= set(signature) <= base_keys | {'mask_derivation'}:
        raise ContractError('complete_case_signature_required')
    endpoint_key = lambda ep: (ep['electrode_id'], ep['role'], ep['actual_layer'])
    if sorted(map(endpoint_key, signature['endpoints'])) != sorted(map(endpoint_key, signature['cnt_binding'])):
        raise ContractError('CNT_endpoint_binding_mismatch')
    if len(signature['endpoints']) != 2:
        raise ContractError('two_candidate_endpoints_required')
    if 'mask_derivation' in signature:
        validate_mask_derivation(signature['mask_derivation'], signature)
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


BATCH_SCHEMA = 'psdl2_search_batch_v1'
SINGLE_EXPORT_SOURCES = 'single_export'


def search(query, index, cohort, *, topk, expand=False, ignore_mask_derivation=False,
           exclude_query_product=False, per_product_topk=None, index_validated=False):
    validate_record(query)
    if not index_validated:
        validate_index(index)
    if per_product_topk is not None and (type(per_product_topk) is not int or per_product_topk < 1):
        raise ContractError('positive_per_product_topk_required')
    if not isinstance(cohort, (set, list, tuple)) or not cohort:
        raise ContractError('explicit_DL1_product_cohort_required')
    cohort = {text(v) for v in cohort}
    if type(topk) is not int or topk < 1:
        raise ContractError('positive_topk_required')
    primary, supplementary, rejected = [], [], []
    derivation_rejected = set()
    from .index import compatibility_signature
    for candidate in index['records']:
        in_primary = candidate['product_id'] in cohort
        if not in_primary and not expand:
            continue
        if candidate['roi_id'] == query['roi_id'] and candidate['source_identity'] == query['source_identity']:
            rejected.append({'roi_id': candidate['roi_id'], 'reason': 'query_itself_excluded'}); continue
        if exclude_query_product and candidate['product_id'] == query['product_id']:
            rejected.append({'roi_id': candidate['roi_id'], 'reason': 'query_product_excluded'}); continue
        if compatibility_signature(candidate['case_signature']) != compatibility_signature(query['case_signature']):
            rejected.append({'roi_id': candidate['roi_id'], 'reason': 'case_or_CNT_binding_incompatible'}); continue
        if not ignore_mask_derivation:
            wanted, offered = query['case_signature'].get('mask_derivation'), candidate['case_signature'].get('mask_derivation')
            if wanted is not None or offered is not None:
                if wanted is None or offered is None:
                    rejected.append({'roi_id': candidate['roi_id'], 'reason': 'mask_derivation_unknown'}); derivation_rejected.add(candidate['roi_id']); continue
                if wanted != offered:
                    rejected.append({'roi_id': candidate['roi_id'], 'reason': 'mask_derivation_mismatch'}); derivation_rejected.add(candidate['roi_id']); continue
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
    if per_product_topk is not None:
        # Applied after the ordering and before --topk; scores are not adjusted.
        for name, rows in (('primary', primary), ('supplementary', supplementary)):
            seen_count, kept = {}, []
            for row in rows:
                seen_count[row['product_id']] = seen_count.get(row['product_id'], 0) + 1
                if seen_count[row['product_id']] <= per_product_topk:
                    kept.append(row)
            rows[:] = kept
    if primary:
        status, reasons = 'available', []
    else:
        cohort_records = [r for r in index['records'] if r['product_id'] in cohort
                          and not (exclude_query_product and r['product_id'] == query['product_id'])]
        status = 'unavailable'
        reasons = ['no_candidates_in_primary_cohort' if not cohort_records else 'no_case_compatible_primary_candidates']
        others = {r['roi_id'] for r in cohort_records if not (r['roi_id'] == query['roi_id'] and r['source_identity'] == query['source_identity'])}
        if others and others <= derivation_rejected:
            reasons = ['no_mask_derivation_compatible_candidates']
    result = {'query': {key: deepcopy(value) for key, value in query.items() if key != 'embedding'},
            'status': status, 'no_match_reasons': reasons, 'primary': primary[:topk],
            'supplementary': supplementary[:topk] if expand else [], 'rejected': rejected,
            'primary_cohort': sorted(cohort), 'expanded': bool(expand),
            'index_roi_ids': [r['roi_id'] for r in index['records']],
            'interpretation': 'similarity_only_not_registration_cause_or_risk'}
    if ignore_mask_derivation:
        result['mask_derivation_comparison'] = 'ignored'
    if exclude_query_product:
        result['excluded_products'] = [query['product_id']]
    if per_product_topk is not None:
        result['per_product_topk'] = per_product_topk
    return result


def derive_rankings(results, query_export_sha256s=None):
    """The only place rankings are built: primary order first, then supplementary order, ranks from 1.
    ``query_export_sha256s`` gives, per result, the sha256 of the export the query ROI belongs to (None when unknown)."""
    rankings = []
    hashes = query_export_sha256s if query_export_sha256s is not None else [None] * len(results)
    for result, export_sha256 in zip(results, hashes):
        rows = list(result['primary']) + list(result['supplementary'])
        rankings.append({'query_roi_id': result['query']['roi_id'], 'query_product_id': result['query']['product_id'],
                         'query_export_sha256': export_sha256,
                         'candidates': [{'rank': rank, 'roi_id': row['roi_id'], 'product_id': row['product_id'],
                                         'frame_id': row['frame']['frame_id'], 'center_xy_um': row['center_xy_um'],
                                         'search_scope': row['search_scope']}
                                        for rank, row in enumerate(rows, 1)]})
    return rankings


def index_sources(index):
    """Per-source provenance of the index. A single-export index has no `sources`, so it reports the string
    'single_export'. A dataset index reports each source's export hash and patch-set hashes; `product_ids` lists the
    products of index records generated from those patch sets. A source without patch sets cannot be linked to its
    records: when it is the only such source it gets the products of the records without patch provenance, otherwise
    its `product_ids` is null (unknown)."""
    sources = index.get('sources')
    if sources is None:
        return SINGLE_EXPORT_SOURCES
    products, unlinked_products = {}, set()
    for record in index['records']:
        digest = record.get('provenance', {}).get('patch', {}).get('set_sha256')
        if digest is not None:
            products.setdefault(digest, set()).add(record['product_id'])
        else:
            unlinked_products.add(record['product_id'])
    n_unlinked = sum(1 for source in sources if not source.get('patch_sets'))
    result = []
    for source in sources:
        digests = [entry['sha256'] for entry in source.get('patch_sets', [])]
        if digests:
            product_ids = sorted(set().union(*(products.get(d, set()) for d in digests)))
        else:
            product_ids = sorted(unlinked_products) if n_unlinked == 1 else None
        result.append({'export_sha256': source['export_sha256'], 'patch_set_sha256s': digests, 'product_ids': product_ids})
    return result


def query_export_sha256s(imported, rois, manifest_path=None, export_path=None):
    """The export sha256 of each query ROI. A single export hashes the file itself; a dataset maps every ROI to its
    source (patch rows through their patch-set hash, other rows through the roi ids of the source's export file)."""
    if manifest_path is None:
        digest = _sha256_file(export_path)
        return [digest] * len(rois)
    by_set, by_roi = {}, {}
    base = Path(manifest_path).parent
    for source in imported.sources:
        entries = source.get('patch_sets', [])
        for entry in entries:
            by_set[entry['sha256']] = source['export_sha256']
        if not entries:
            location = Path(source['export'])
            document = json.loads((location if location.is_absolute() else base / location).read_text(encoding='utf-8'))
            for row in document['rois']:
                by_roi[row['roi_id']] = source['export_sha256']
    hashes = []
    for roi in rois:
        digest = by_set.get(roi.patch['set_sha256']) if roi.patch is not None else by_roi.get(roi.row['roi_id'])
        if digest is None:
            raise ContractError('query_export_unknown')
        hashes.append(digest)
    return hashes


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp%d' % os.getpid())
    try:
        temp.write_text(text)
        os.replace(str(temp), str(path))
    finally:
        if temp.exists():
            temp.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Search one original ROI within an explicit DL1 cohort offline.')
    parser.add_argument('export', type=Path, nargs='?'); parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--query-dataset', type=Path, help='psdl2_dataset_v1 manifest wrapping the query export (and its mask_derivation)')
    parser.add_argument('--ignore-mask-derivation', action='store_true',
                        help='skip the mask_derivation compatibility check; the result records that it was skipped')
    parser.add_argument('--roi-id'); parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--all-ready', action='store_true',
                        help='search every ready query ROI (sorted by roi_id) and write one psdl2_search_batch_v1 envelope')
    parser.add_argument('--exclude-query-product', action='store_true',
                        help='drop candidates whose product_id equals the query product_id (primary and supplementary)')
    parser.add_argument('--per-product-topk', type=int,
                        help='keep at most M rows per product (after ordering, before --topk) in primary and supplementary')
    parser.add_argument('--index', required=True, type=Path); parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--topk', required=True, type=int); parser.add_argument('--expand', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    parser.add_argument('--max-asset-memory-mb', type=float, help='Reject the export if retained decoded masks would exceed this (transient decode buffers are extra)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    current_roi = None  # the query being searched in a batch; named in the error so a failed batch can be traced
    try:
        if (args.roi_id is None) == (not args.all_ready):
            raise ContractError('exactly_one_of_roi_id_or_all_ready_required')
        if args.per_product_topk is not None and args.per_product_topk < 1:
            raise ContractError('positive_per_product_topk_required')
        import torch
        from .checkpoint import fingerprint
        from .model import resolve_device
        resolve_device(args.device)  # fail before the slow export load
        from .data import load_dataset, load_export
        from .index import candidate_record, load_encoder, view_config_dict
        if (args.export is None) == (args.query_dataset is None):
            raise ContractError('exactly_one_of_export_or_query_dataset_required')
        if args.query_dataset is not None:
            if args.assets_root is not None:
                raise ContractError('assets_root_not_allowed_with_dataset')
            model, views, fp = load_encoder(args.checkpoint, args.device)
            imported = load_dataset(args.query_dataset, expected_views=views,
                                    allow_unversioned_producer=args.allow_unversioned_producer,
                                    max_asset_memory_mb=args.max_asset_memory_mb)
        else:
            imported = load_export(args.export, assets_root=args.assets_root,
                                   allow_unversioned_producer=args.allow_unversioned_producer,
                                   max_asset_memory_mb=args.max_asset_memory_mb)
        if args.all_ready:
            rois = sorted(imported.ready, key=lambda r: r.row['roi_id'])
            if not rois:
                raise ContractError('ready_query_roi_required')
        else:
            rois = [r for r in imported.ready if r.row['roi_id'] == args.roi_id]
            if len(rois) != 1:
                raise ContractError('one_ready_query_roi_required')
        if args.query_dataset is None:
            model, views, fp = load_encoder(args.checkpoint, args.device)
        index = json.loads(args.index.read_text())
        if index['checkpoint_fingerprint'] != fp or index['view_fingerprint'] != fingerprint(view_config_dict(views)):
            raise ContractError('incompatible_index_checkpoint_or_transform')
        cohort = read_cohort(args.cohort)
        options = dict(topk=args.topk, expand=args.expand, ignore_mask_derivation=args.ignore_mask_derivation,
                       exclude_query_product=args.exclude_query_product, per_product_topk=args.per_product_topk)
        if args.all_ready:
            validate_index(index)  # once, not per query
            options['index_validated'] = True
        results = []
        for roi in rois:
            current_roi = roi.row['roi_id'] if args.all_ready else None
            bundle = views.bundle(roi)
            with torch.no_grad():
                query = candidate_record(bundle, model([bundle])['embedding'][0].cpu().tolist(), roi.mask_derivation)
            results.append(search(query, index, cohort, **options))
        current_roi = None
        if not args.all_ready:
            result = results[0]
            _write_atomic(args.out, json.dumps(result, allow_nan=False, indent=2))
            print(json.dumps({'status': result['status'], 'primary': len(result['primary']), 'supplementary': len(result['supplementary']),
                              'no_match_reasons': result['no_match_reasons']}))
            return 0 if result['status'] == 'available' else 3
        envelope = {'schema_version': BATCH_SCHEMA, 'checkpoint_fingerprint': index['checkpoint_fingerprint'],
                    'view_fingerprint': index['view_fingerprint'], 'index_sources': index_sources(index),
                    'inputs': {'query_export_sha256': _sha256_file(args.export) if args.export is not None else None,
                               'query_dataset_sha256': _sha256_file(args.query_dataset) if args.query_dataset is not None else None,
                               'index_sha256': _sha256_file(args.index)},
                    'arguments': {'topk': args.topk, 'expand': bool(args.expand), 'per_product_topk': args.per_product_topk,
                                  'exclude_query_product': bool(args.exclude_query_product),
                                  'ignore_mask_derivation': bool(args.ignore_mask_derivation), 'cohort': sorted(cohort)},
                    'results': results,
                    'rankings': derive_rankings(results, query_export_sha256s(imported, rois, args.query_dataset, args.export))}
        _write_atomic(args.out, json.dumps(envelope, allow_nan=False, indent=2))
        n_available = sum(1 for r in results if r['status'] == 'available')
        print(json.dumps({'status': 'available' if n_available == len(results) else 'partial', 'n_queries': len(results),
                          'n_available': n_available, 'n_unavailable': len(results) - n_available}))
        return 0 if n_available == len(results) else 3
    except (ContractError, KeyError, ValueError, TypeError, OSError) as exc:
        error = {'error': str(exc)}
        if current_roi is not None:
            error['roi_id'] = current_roi
        print(json.dumps(error), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
