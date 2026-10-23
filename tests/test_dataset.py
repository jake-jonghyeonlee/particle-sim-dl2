"""psdl2_dataset_v1: several exports, one asset cache, mask_derivation and its search compatibility."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from psdl2.data import AssetMemoryLimit, load_dataset, load_export
from psdl2.index import candidate_record
from psdl2.schema import ContractError
from psdl2.search import search, validate_index, validate_record
from test_search import fixed_records, make_index
from test_views import view_config

ROOT = Path(__file__).parents[1]
ENDPOINTS = [{'electrode_id': 'A', 'role': 'gate', 'actual_layer': 'M1'},
             {'electrode_id': 'B', 'role': 'data', 'actual_layer': 'M3'}]


def make_export(root, name, *, product, structures=(0, 1), shift=0, role='data', failure_type='vertical_short',
                id_prefix=None):
    """One hand-written export folder: root/name/review_rois.json plus assets/*.npy (8x8 frames)."""
    folder = Path(root) / name
    template = json.loads((ROOT / 'tests/fixtures/review_roi_v1_export/review_rois.json').read_text())
    ready = next(r for r in template['rois'] if r['status'] == 'ready')
    payload = {**template, 'producer_commit': '0' * 40, 'assets_root': '.', 'rois': [], 'excluded_records': [],
               'status': 'available'}
    for structure in structures:
        rid = f'{id_prefix or product}_{structure}'
        row = deepcopy(ready)
        row.update(roi_id=rid, product_id=product, case_id='SYNTHETIC_CASE', frame_id=f'frame_{rid}',
                   source_version='synthetic_dataset_v1', center_xy_um=[102, 202], failure_type=failure_type)
        row['source_identity'].update(product_id=product, case_id='SYNTHETIC_CASE', frame_id=row['frame_id'])
        row['frame'].update(frame_id=row['frame_id'], shape_yx=[8, 8])
        row['endpoints'] = deepcopy(ENDPOINTS)
        row['endpoints'][1]['role'] = role
        row['analysis_layer'] = 'M3'
        for layout, endpoint in zip(row['source_assets']['layouts'], row['endpoints']):
            layout['endpoint'] = endpoint.copy()
        a = np.zeros((8, 8), np.uint8); b = np.zeros_like(a); cnt = np.zeros_like(a)
        if structure == 0:
            a[2, 2] = 1; b[2, 5 - shift] = 1; cnt[3, 4] = 1
        else:
            a[2:4, 2:4] = 1; b[4:6, 4:6 - shift] = 1; cnt[3:5, 4] = 1
        refs = [layout['asset'] for layout in row['source_assets']['layouts']] + [
            row['source_assets']['cnt'], row['source_assets']['valid']]
        for channel, (ref, values) in enumerate(zip(refs, (a, b, cnt, np.ones_like(a)))):
            path = folder / 'assets' / f's{structure}_{channel}.npy'; path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values)
            ref.update(path=str(path.relative_to(folder)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        payload['rois'].append(row)
    payload['source_manifest_hash'] = hashlib.sha256(json.dumps(payload['rois'], sort_keys=True).encode()).hexdigest()
    path = folder / 'review_rois.json'
    path.write_text(json.dumps(payload))
    return path


def derivation(*, rule='between_stack_order_v1', removed=('M2',), processed='B', failure_type='vertical_short',
               endpoints=None):
    endpoints = deepcopy(ENDPOINTS if endpoints is None else endpoints)
    lateral = failure_type == 'lateral_short'
    return {'schema_version': 'psdl2_mask_derivation_v1', 'rule_version': rule,
            'cases': [{'case': {'failure_type': failure_type, 'endpoints': endpoints},
                       'processed_endpoint': None if lateral else processed,
                       'removed_layers': [] if lateral else list(removed)}]}


def source_for(export, root, **extra):
    root = Path(root)
    return {'export': os.path.relpath(export, root),
            'export_sha256': hashlib.sha256(Path(export).read_bytes()).hexdigest(), **extra}


def write_dataset(root, sources, name='dataset.json'):
    path = Path(root) / name
    path.write_text(json.dumps({'schema_version': 'psdl2_dataset_v1', 'sources': sources}))
    return path


def failure(call):
    with pytest.raises(ContractError) as info:
        call()
    return str(info.value)


def test_two_sources_share_one_cache_and_resolve_from_the_manifest_folder(tmp_path, monkeypatch):
    a = make_export(tmp_path, 'prodA', product='A')
    b = make_export(tmp_path, 'prodB', product='B', shift=1)
    manifest = write_dataset(tmp_path, [source_for(a, tmp_path), source_for(b, tmp_path)])
    elsewhere = tmp_path / 'elsewhere'; elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    dataset = load_dataset(manifest)
    assert [r.row['roi_id'] for r in dataset.ready] == ['A_0', 'A_1', 'B_0', 'B_1'] and not dataset.blocked
    assert [s['export'] for s in dataset.sources] == ['prodA/review_rois.json', 'prodB/review_rois.json']
    report = dataset.asset_memory
    # 2 products x 2 structures x 4 files; references are 4 per ROI.
    assert report['decoded_assets'] == 16 and report['asset_references'] == 16 and report['decoded_bytes'] == 16 * 64
    assert all(r.mask_derivation is None for r in dataset.ready)
    # expected_views is accepted but not enforced in this task.
    assert [r.row['roi_id'] for r in load_dataset(manifest, expected_views=object()).ready] == [r.row['roi_id'] for r in dataset.ready]


def test_export_sha256_mismatch_and_unknown_fields(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A')
    bad = source_for(a, tmp_path); bad['export_sha256'] = '0' * 64
    assert failure(lambda: load_dataset(write_dataset(tmp_path, [bad]))) == 'export_sha256_mismatch'
    missing = source_for(a, tmp_path); del missing['export_sha256']
    assert failure(lambda: load_dataset(write_dataset(tmp_path, [missing]))) == 'unsupported_dataset_source_field'
    assert failure(lambda: load_dataset(write_dataset(tmp_path, []))) == 'dataset_sources_required'
    manifest = write_dataset(tmp_path, [source_for(a, tmp_path)])
    payload = json.loads(manifest.read_text()); payload['schema_version'] = 'psdl2_dataset_v9'
    manifest.write_text(json.dumps(payload))
    assert failure(lambda: load_dataset(manifest)) == 'unknown_dataset_schema'


def test_duplicate_roi_id_across_sources(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A')
    b = make_export(tmp_path, 'prodB', product='B', id_prefix='A')
    message = failure(lambda: load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path), source_for(b, tmp_path)])))
    assert message.startswith('duplicate_roi_id_across_sources')
    # The same export listed twice is the same collision, never a silent de-duplication.
    assert failure(lambda: load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path)] * 2))).startswith(
        'duplicate_roi_id_across_sources')


def test_aggregate_memory_cap_where_each_source_alone_fits(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,))
    b = make_export(tmp_path, 'prodB', product='B', structures=(0,), shift=1)
    limit_mb = 300 / (1024 * 1024)  # one source needs 4 x 64 = 256 bytes; two need 512
    assert load_export(a, max_asset_memory_mb=limit_mb).asset_memory['decoded_bytes'] == 256
    assert load_export(b, max_asset_memory_mb=limit_mb).asset_memory['decoded_bytes'] == 256
    manifest = write_dataset(tmp_path, [source_for(a, tmp_path), source_for(b, tmp_path)])
    with pytest.raises(AssetMemoryLimit, match='asset_memory_limit_exceeded'):
        load_dataset(manifest, max_asset_memory_mb=limit_mb)
    assert load_dataset(manifest, max_asset_memory_mb=512 / (1024 * 1024)).asset_memory['decoded_bytes'] == 512
    for bad in (0, -1, True):
        assert failure(lambda: load_dataset(manifest, max_asset_memory_mb=bad)) == 'max_asset_memory_mb_must_be_positive'


def test_same_relative_path_with_different_bytes_is_not_mixed(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,))
    b = make_export(tmp_path, 'prodB', product='B', structures=(0,), shift=1)
    ra = json.loads(a.read_text())['rois'][0]['source_assets']; rb = json.loads(b.read_text())['rois'][0]['source_assets']
    assert ra['layouts'][1]['asset']['path'] == rb['layouts'][1]['asset']['path']
    assert ra['layouts'][1]['asset']['sha256'] != rb['layouts'][1]['asset']['sha256']
    dataset = load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path), source_for(b, tmp_path)]))
    first, second = dataset.ready
    assert first.channels['layout:1'][2, 5] == 1 and second.channels['layout:1'][2, 4] == 1
    assert first.channels['layout:1'][2, 4] == 0 and second.channels['layout:1'][2, 5] == 0
    assert first.channels['layout:1'] is not second.channels['layout:1']
    assert dataset.asset_memory['decoded_assets'] == 8


def test_alias_of_the_same_real_file_is_decoded_once(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,))
    # A second export whose assets_root is a symlink to the first one's folder.
    (tmp_path / 'alias').symlink_to(tmp_path / 'prodA', target_is_directory=True)
    payload = json.loads(a.read_text())
    payload['rois'][0]['roi_id'] = 'A_alias'; payload['assets_root'] = '../alias'
    folder = tmp_path / 'prodA2'; folder.mkdir()
    (folder / 'review_rois.json').write_text(json.dumps(payload))
    manifest = write_dataset(tmp_path, [source_for(a, tmp_path), source_for(folder / 'review_rois.json', tmp_path)])
    dataset = load_dataset(manifest)
    assert [r.row['roi_id'] for r in dataset.ready] == ['A_0', 'A_alias']
    assert dataset.asset_memory['decoded_assets'] == 4 and dataset.asset_memory['asset_references'] == 8
    assert dataset.ready[0].channels['cnt'] is dataset.ready[1].channels['cnt']
    assert dataset.asset_memory['decoded_bytes'] == 256


def test_nonempty_patch_sets_are_rejected_not_ignored(tmp_path):
    # T2 enables patch sets (see test_dataset_patches.py); a listed set is still never silently ignored.
    a = make_export(tmp_path, 'prodA', product='A')
    manifest = write_dataset(tmp_path, [source_for(a, tmp_path, patch_sets=['ps.json'])])
    assert failure(lambda: load_dataset(manifest, expected_views=view_config())).startswith('patch_set_missing')
    assert failure(lambda: load_dataset(manifest)) == 'patch_set_views_unchecked'
    assert len(load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, patch_sets=[])])).ready) == 2


# ---- mask_derivation validation ------------------------------------------------------------------

def load_with(tmp_path, value, structures=(0, 1)):
    a = make_export(tmp_path, 'prodA', product='A', structures=structures)
    return load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=value)]))


def test_derivation_is_attached_by_role_and_layer_regardless_of_endpoint_order(tmp_path):
    swapped = derivation(endpoints=list(reversed(ENDPOINTS)))
    inline = load_with(tmp_path, derivation())
    reordered = load_with(tmp_path, swapped)
    expected = {'schema_version': 'psdl2_mask_derivation_v1', 'rule_version': 'between_stack_order_v1',
                'processed_endpoint': {'role': 'data', 'actual_layer': 'M3'}, 'removed_layers': ['M2']}
    assert all(r.mask_derivation == expected for r in inline.ready)
    assert [r.mask_derivation for r in reordered.ready] == [r.mask_derivation for r in inline.ready]
    # Same canonical content however the endpoints are listed, so the fingerprint agrees too.
    assert reordered.fingerprint == inline.fingerprint
    assert inline.fingerprint['sources'][0]['mask_derivation']['cases'][0]['removed_layers'] == ['M2']


def test_derivation_file_reference_equals_inline_and_is_relative_to_the_manifest(tmp_path, monkeypatch):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta' / 'mask_derivation.json').write_text(json.dumps(derivation()))
    from_file = load_with(tmp_path, 'meta/mask_derivation.json')
    monkeypatch.chdir(tmp_path / 'meta')
    assert load_dataset(tmp_path / 'dataset.json').fingerprint == from_file.fingerprint
    assert from_file.fingerprint == load_with(tmp_path, derivation()).fingerprint


def test_lateral_is_a_known_empty_value_distinct_from_absent(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A', failure_type='lateral_short')
    value = derivation(failure_type='lateral_short')
    known = load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=value)]))
    unknown = load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path)]))
    assert known.ready[0].mask_derivation['processed_endpoint'] is None
    assert known.ready[0].mask_derivation['removed_layers'] == []
    assert unknown.ready[0].mask_derivation is None


def test_derivation_case_missing_unreferenced_and_duplicate(tmp_path):
    other = [{'electrode_id': 'A', 'role': 'gate', 'actual_layer': 'M1'},
             {'electrode_id': 'C', 'role': 'data', 'actual_layer': 'M3'}]
    missing = derivation(endpoints=other, processed='C')
    assert failure(lambda: load_with(tmp_path, missing)).startswith('mask_derivation_case_missing')
    extra = derivation()
    extra['cases'].append(derivation(endpoints=other, processed='C')['cases'][0])
    message = failure(lambda: load_with(tmp_path, extra))
    assert message.startswith('mask_derivation_case_unreferenced') and "'C'" in message and 'vertical_short' in message
    twice = derivation()
    twice['cases'].append(derivation(endpoints=list(reversed(ENDPOINTS)), removed=('M2', 'M9'))['cases'][0])
    assert failure(lambda: load_with(tmp_path, twice)) == 'mask_derivation_duplicate_case'


@pytest.mark.parametrize('value,message', [
    (derivation(processed='Z'), 'mask_derivation_processed_endpoint_invalid'),
    (derivation(processed=None), 'mask_derivation_processed_endpoint_required'),
    (derivation(removed=('M2', 'M2')), 'mask_derivation_removed_layers_duplicate'),
    (derivation(removed=('M1',)), 'mask_derivation_removed_layer_is_endpoint_layer'),
    (derivation(rule=''), 'nonempty_identifier_required'),
    ({**derivation(), 'schema_version': 'psdl2_mask_derivation_v2'}, 'unknown_mask_derivation_schema'),
    ({**derivation(), 'extra': 1}, 'mask_derivation_envelope_invalid'),
])
def test_invalid_derivations_are_rejected(tmp_path, value, message):
    assert failure(lambda: load_with(tmp_path, value)) == message


def test_lateral_derivation_with_content_is_rejected(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A', failure_type='lateral_short')
    value = derivation(failure_type='lateral_short'); value['cases'][0]['removed_layers'] = ['M2']
    message = failure(lambda: load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=value)])))
    assert message == 'mask_derivation_lateral_must_be_empty'


# ---- search compatibility matrix -------------------------------------------------------------

def record_with(base, rid, product, value):
    record = deepcopy(base)
    record.update(roi_id=rid, product_id=product)
    record['source_identity']['product_id'] = product
    if value is None:
        record['case_signature'].pop('mask_derivation', None)
    else:
        record['case_signature']['mask_derivation'] = deepcopy(value)
    return record


def derived(*, rule='between_stack_order_v1', removed=('M2',), role='data', layer='M3'):
    return {'schema_version': 'psdl2_mask_derivation_v1', 'rule_version': rule,
            'processed_endpoint': {'role': role, 'actual_layer': layer}, 'removed_layers': list(removed)}


def run_matrix(source, query_value, candidates, **kwargs):
    base, _ = fixed_records(source)
    query = record_with(base, 'query', 'Q', query_value)
    records = [record_with(base, rid, 'A', value) for rid, value in candidates]
    result = search(query, make_index(records), {'A'}, topk=5, **kwargs)
    return query, result, {row['roi_id']: row['reason'] for row in result['rejected']}


def test_compatibility_matrix_five_rows(source):
    same = derived()
    candidates = [('same', same), ('other_layers', derived(removed=('M2', 'M4'))),
                  ('other_rule', derived(rule='between_stack_order_v2')), ('other_endpoint', derived(role='gate', layer='M1')),
                  ('unknown', None)]
    # Row 1: neither side has a derivation -> legacy behaviour, no new key anywhere.
    query, result, rejected = run_matrix(source, None, [('plain', None)])
    assert [r['roi_id'] for r in result['primary']] == ['plain'] and not rejected
    assert 'mask_derivation' not in result['query']['case_signature'] and 'mask_derivation_comparison' not in result
    # Rows 2-4: both sides have one.
    query, result, rejected = run_matrix(source, same, candidates)
    assert [r['roi_id'] for r in result['primary']] == ['same']
    assert rejected == {'other_layers': 'mask_derivation_mismatch', 'other_rule': 'mask_derivation_mismatch',
                        'other_endpoint': 'mask_derivation_mismatch', 'unknown': 'mask_derivation_unknown'}
    assert result['primary'][0]['case_signature']['mask_derivation'] == same
    # Row 3 (one side only), both directions.
    _, result, rejected = run_matrix(source, None, [('with', same), ('without', None)])
    assert [r['roi_id'] for r in result['primary']] == ['without'] and rejected == {'with': 'mask_derivation_unknown'}
    # Row 5: explicit override skips the comparison and records that it did.
    _, result, rejected = run_matrix(source, same, candidates, ignore_mask_derivation=True)
    assert sorted(r['roi_id'] for r in result['primary']) == sorted(rid for rid, _ in candidates)
    assert not rejected and result['mask_derivation_comparison'] == 'ignored'


def test_same_role_layer_pair_with_different_removed_layers_is_excluded(source):
    # Identical (role, actual_layer) endpoints and failure type: the case signature alone cannot tell them apart.
    query, result, rejected = run_matrix(source, derived(removed=('M2',)), [('two_layers', derived(removed=('M2', 'M5'))),
                                                                          ('no_layers', derived(removed=()))])
    assert not result['primary'] and rejected == {'two_layers': 'mask_derivation_mismatch', 'no_layers': 'mask_derivation_mismatch'}
    assert result['status'] == 'unavailable'


def test_processed_endpoint_is_compared_by_role_and_layer_not_electrode_id(tmp_path):
    renamed = [{'electrode_id': 'X1', 'role': 'gate', 'actual_layer': 'M1'},
               {'electrode_id': 'X2', 'role': 'data', 'actual_layer': 'M3'}]
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,))
    first = load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=derivation())]))
    # Another product naming its electrodes differently.
    payload = json.loads(a.read_text())
    row = payload['rois'][0]; row['roi_id'] = 'P_0'
    row['endpoints'] = deepcopy(renamed)
    for layout, endpoint in zip(row['source_assets']['layouts'], renamed):
        layout['endpoint'] = deepcopy(endpoint)
    (tmp_path / 'prodP').mkdir()
    payload['assets_root'] = '../prodA'
    (tmp_path / 'prodP' / 'review_rois.json').write_text(json.dumps(payload))
    second = load_dataset(write_dataset(tmp_path, [source_for(tmp_path / 'prodP' / 'review_rois.json', tmp_path,
                                                              mask_derivation=derivation(endpoints=renamed, processed='X2'))],
                                        name='d2.json'))
    assert first.ready[0].mask_derivation == second.ready[0].mask_derivation


def test_record_validation_accepts_and_checks_the_optional_derivation(source):
    base, _ = fixed_records(source)
    validate_record(record_with(base, 'ok', 'A', derived()))
    validate_record(record_with(base, 'ok', 'A', None))
    for bad in ({'schema_version': 'x'}, {**derived(), 'removed_layers': ['b', 'a']}, {**derived(), 'extra': 1}):
        with pytest.raises(ContractError, match='invalid_record_mask_derivation'):
            validate_record(record_with(base, 'bad', 'A', bad))
    record = record_with(base, 'ok', 'A', derived())
    index = make_index([record])
    validate_index(index)


def test_build_index_transports_the_derivation(tmp_path):
    from psdl2.index import case_signature
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,))
    dataset = load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=derivation())]))
    plain = load_export(a).ready[0]
    bundle = view_config().bundle(plain)
    legacy = candidate_record(bundle, [1, 0])
    assert 'mask_derivation' not in legacy['case_signature'] and case_signature(bundle.metadata) == legacy['case_signature']
    record = candidate_record(bundle, [1, 0], dataset.ready[0].mask_derivation)
    assert record['case_signature']['mask_derivation'] == dataset.ready[0].mask_derivation


# ---- follow-up fixes (N1-N5) -----------------------------------------------------------------

def test_dl2_blocked_row_still_references_its_derivation_case(tmp_path):
    # Producer said ready, DL2 blocks it (cnt hash wrong): its case stays referenced, not a whole-dataset error.
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,))
    payload = json.loads(a.read_text()); payload['rois'][0]['source_assets']['cnt']['sha256'] = '0' * 64
    a.write_text(json.dumps(payload))
    assert (len(load_export(a).ready), len(load_export(a).blocked)) == (0, 1)
    loaded = load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=derivation())]))
    assert not loaded.ready and len(loaded.blocked) == 1
    # A case that no row (ready or blocked) references is still an error and is named.
    other = [{'electrode_id': 'A', 'role': 'gate', 'actual_layer': 'M1'},
             {'electrode_id': 'C', 'role': 'data', 'actual_layer': 'M3'}]
    extra = derivation(); extra['cases'].append(derivation(endpoints=other, processed='C')['cases'][0])
    assert failure(lambda: load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=extra)],
                                                      name='d2.json'))).startswith('mask_derivation_case_unreferenced')


def test_ready_rows_stay_strictly_covered_when_another_row_is_blocked(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A', structures=(0, 1))
    payload = json.loads(a.read_text()); payload['rois'][0]['source_assets']['cnt']['sha256'] = '0' * 64
    a.write_text(json.dumps(payload))
    other = [{'electrode_id': 'A', 'role': 'gate', 'actual_layer': 'M1'},
             {'electrode_id': 'C', 'role': 'data', 'actual_layer': 'M3'}]
    wrong = derivation(endpoints=other, processed='C')
    assert failure(lambda: load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path, mask_derivation=wrong)]))
                   ).startswith('mask_derivation_case_missing')


def test_unsorted_removed_layers_are_normalised(tmp_path):
    loaded = load_with(tmp_path, derivation(removed=('M4', 'M2')))
    assert all(r.mask_derivation['removed_layers'] == ['M2', 'M4'] for r in loaded.ready)
    assert loaded.fingerprint == load_with(tmp_path, derivation(removed=('M2', 'M4'))).fingerprint
    assert loaded.fingerprint['sources'][0]['mask_derivation']['cases'][0]['removed_layers'] == ['M2', 'M4']


def test_roi_id_collision_involving_a_blocked_row_is_rejected(tmp_path):
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,), id_prefix='SAME')
    b = make_export(tmp_path, 'prodB', product='B', structures=(0,), id_prefix='SAME')
    payload = json.loads(b.read_text()); payload['rois'][0]['source_assets']['cnt']['sha256'] = '0' * 64
    b.write_text(json.dumps(payload))
    assert len(load_export(b).blocked) == 1
    message = failure(lambda: load_dataset(write_dataset(tmp_path, [source_for(a, tmp_path), source_for(b, tmp_path)])))
    assert message.startswith('duplicate_roi_id_across_sources')


def test_search_reports_derivation_as_the_cause_when_it_excludes_every_candidate(source):
    _, result, _ = run_matrix(source, derived(removed=('M2',)), [('x', derived(removed=('M5',))), ('y', None)])
    assert result['status'] == 'unavailable' and result['no_match_reasons'] == ['no_mask_derivation_compatible_candidates']
    # Legacy (no derivation anywhere) keeps the old reason.
    _, result, _ = run_matrix(source, None, [])
    assert result['no_match_reasons'] == ['no_candidates_in_primary_cohort']


def test_record_derivation_must_agree_with_the_signature(source):
    base, _ = fixed_records(source)
    ftype = base['case_signature']['failure_type']
    assert ftype != 'lateral_short'
    for bad in ({**derived(), 'processed_endpoint': None},
                derived(role='nope', layer='M9'),
                derived(role='gate', layer='M3')):
        with pytest.raises(ContractError, match='invalid_record_mask_derivation'):
            validate_record(record_with(base, 'bad', 'A', bad))
    lateral = record_with(base, 'lat', 'A', {**derived(), 'processed_endpoint': None, 'removed_layers': []})
    lateral['case_signature']['failure_type'] = 'lateral_short'
    validate_record(lateral)
    lateral['case_signature']['mask_derivation']['processed_endpoint'] = {'role': 'data', 'actual_layer': 'M3'}
    with pytest.raises(ContractError, match='invalid_record_mask_derivation'):
        validate_record(lateral)


def test_load_export_rejects_cache_together_with_memory_limit(tmp_path):
    from psdl2.data import AssetCache
    a = make_export(tmp_path, 'prodA', product='A', structures=(0,))
    assert failure(lambda: load_export(a, cache=AssetCache(None), max_asset_memory_mb=1)) == 'cache_and_max_asset_memory_mb_exclusive'
