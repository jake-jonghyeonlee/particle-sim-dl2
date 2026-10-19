"""Decoded masks are shared across ROIs; the memory cap rejects the whole export."""
from copy import deepcopy
import json

import pytest

from psdl2.data import AssetMemoryLimit, load_export, main as data_main


def _rows(payload, n):
    row = payload['rois'][0]
    payload['rois'] = []
    for i in range(n):
        copy = deepcopy(row); copy['roi_id'] = f'roi_{i}'
        payload['rois'].append(copy)


def test_rois_over_the_same_files_share_one_decoded_array(source):
    payload, write, _ = source
    _rows(payload, 5)
    export = load_export(write())
    assert len(export.ready) == 5 and not export.blocked
    first, second = export.ready[0].channels, export.ready[1].channels
    assert all(first[key] is second[key] for key in first)
    assert not first['layout:0'].flags.writeable
    report = export.asset_memory
    assert report['decoded_assets'] == 4 and report['asset_references'] == 20
    assert report['decoded_bytes'] == 4 * 64 and report['max_bytes'] is None
    assert export.summary()['asset_memory'] == report


def test_bad_hash_row_is_blocked_without_poisoning_the_shared_cache(source):
    payload, write, _ = source
    _rows(payload, 2)
    payload['rois'][1]['source_assets']['cnt']['sha256'] = '0' * 64
    export = load_export(write())
    assert [r.row['roi_id'] for r in export.ready] == ['roi_0']
    assert export.blocked[0]['roi_id'] == 'roi_1' and 'asset_hash_mismatch' in export.blocked[0]['reasons']


def test_memory_cap_rejects_the_export_instead_of_blocking_rows(source, capsys):
    payload, write, _ = source
    _rows(payload, 3)
    path = write()
    # Four distinct 8x8 uint8 masks need 256 bytes; allow only 200.
    limit_mb = 200 / (1024 * 1024)
    with pytest.raises(AssetMemoryLimit, match='asset_memory_limit_exceeded'):
        load_export(path, max_asset_memory_mb=limit_mb)
    assert load_export(path, max_asset_memory_mb=256 / (1024 * 1024)).asset_memory['decoded_bytes'] == 256
    assert data_main([str(path), '--max-asset-memory-mb', str(limit_mb)]) == 2
    assert 'asset_memory_limit_exceeded' in json.loads(capsys.readouterr().err)['error']
    for bad in (0, -1, True):
        with pytest.raises(Exception, match='max_asset_memory_mb_must_be_positive'):
            load_export(path, max_asset_memory_mb=bad)


def _reason(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the exact exception type and message are compared
        return type(exc).__name__, str(exc)
    return None


def test_cache_keeps_the_uncached_reasons_for_bad_references(source):
    from psdl2.data import AssetCache, read_asset
    payload, write, root = source
    export = load_export(write())
    frame, good = export.ready[0].frame, payload['rois'][0]['source_assets']['cnt']
    bad = [{'path': '../x'}, {**good, 'observed': False, 'path': 'missing.npy'},
           {**good, 'extra': 1, 'path': 'missing.npy'}, {**good, 'value_semantics': 'zzz', 'path': 'missing.npy'},
           {**good, 'path': 5}, {**good, 'sha256': ['x']}, {**good, 'value_semantics': ['binary']},
           {**good, 'sha256': '0' * 64}, {**good, 'sha256': 'NOT-A-HASH'}]
    for ref in bad:
        cache = AssetCache()
        read_asset(root, good, frame, cache)  # a warm cache must not change any reason
        expected = _reason(lambda: read_asset(root, ref, frame))
        assert expected is not None and _reason(lambda: read_asset(root, ref, frame, cache)) == expected, ref


def test_bad_hash_near_the_limit_is_a_blocked_row_not_a_limit(source):
    payload, write, _ = source
    _rows(payload, 2)
    payload['rois'][1]['source_assets']['cnt']['sha256'] = '0' * 64
    export = load_export(write(), max_asset_memory_mb=256 / (1024 * 1024))
    assert len(export.ready) == 1 and 'asset_hash_mismatch' in export.blocked[0]['reasons']


def test_non_finite_memory_limit_is_a_contract_error(source, capsys):
    _, write, _ = source
    path = write()
    for bad in ('inf', 'nan', '1e308'):
        assert data_main([str(path), '--max-asset-memory-mb', bad]) == 2
        assert 'max_asset_memory_mb_must_be_positive' in json.loads(capsys.readouterr().err)['error']
