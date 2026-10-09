import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from psdl2.crop import build_bundle
from psdl2.data import load_export, main
from psdl2.schema import ContractError, CropConfig

FIXTURE = Path(__file__).parent / 'fixtures/review_roi_v1_export'
COMMIT = 'f5e86006d72faf16c3bb747051325fbba4b0f9fd'


def test_actual_producer_export_preserves_ready_and_blocked():
    before = (FIXTURE / 'review_rois.json').read_bytes()
    imported = load_export(FIXTURE / 'review_rois.json', assets_root=FIXTURE, expected_producer_commit=COMMIT)
    assert len(imported.ready) == 1 and len(imported.blocked) == 3
    roi = imported.ready[0]
    assert roi.center_xy_um == (102.5, 201.5)
    assert roi.row['case_id'] == 'A' and roi.frame.frame_id == 'FRAME'
    assert roi.frame.origin_xy_um == (100, 200) and roi.frame.pixel_pitch_um == .5
    assert roi.source_identity == ('PRODUCT', 'DESIGN', 'SOURCE', 'STEP', 'EXPOSURE', 'A', 'FRAME', 'synthetic_v2')
    assert sorted(r['reasons'] for r in imported.blocked) == [
        ['case_context_required', 'source_shape_required'], ['map_coverage_incomplete'], ['map_coverage_incomplete']]
    for row in imported.blocked:
        with pytest.raises(ContractError):
            build_bundle(row, CropConfig((2, 2), (4, 4), 'preserve_exported_layers_v1'))
    assert len(imported.excluded_records) == 3
    bundle = build_bundle(roi, CropConfig((2, 2), (4, 4), 'preserve_exported_layers_v1'))
    assert bundle.metadata['center_xy_um'] == [102.5, 201.5]
    assert before == (FIXTURE / 'review_rois.json').read_bytes()


@pytest.mark.parametrize('change,reason', [
    ('hash', 'asset_hash_mismatch'), ('identity', 'composite_source_identity_mismatch'),
    ('endpoint', 'layout_endpoint_binding_mismatch'), ('missing', 'missing_observation'),
    ('hotspot', 'original_layout_CNT_VIA_valid_only'), ('ca', 'original_layout_CNT_VIA_valid_only'),
    ('fab', 'original_layout_CNT_VIA_valid_only'), ('semantics', 'unsupported_value_semantics'),
    ('parent', 'unsafe_asset_path'), ('absolute', 'unsafe_asset_path'),
])
def test_bad_ready_row_is_blocked(source, change, reason):
    doc, write, root = source
    row = doc['rois'][0]; assets = row['source_assets']; ref = assets['layouts'][0]['asset']
    if change == 'hash': ref['sha256'] = '0' * 64
    elif change == 'identity': row['source_identity']['product_id'] = 'OTHER'
    elif change == 'endpoint': assets['layouts'][0]['endpoint']['role'] = 'wrong'
    elif change == 'missing': ref['observed'] = False
    elif change in ('hotspot', 'ca', 'fab'): assets[change] = ref.copy()
    elif change == 'semantics': ref['value_semantics'] = 'auto'
    elif change == 'parent': ref['path'] = '../layout.npy'
    elif change == 'absolute': ref['path'] = str(root / ref['path'])
    result = load_export(write())
    assert not result.ready and result.blocked[0]['reasons'] == [reason]


@pytest.mark.parametrize('values,dtype', [([[0, 2]], np.uint8), ([[0, .5]], np.float32),
                                        ([[0, 255]], np.float32), ([[0, -1]], np.int32),
                                        ([[0, float('nan')]], np.float32)])
def test_no_thresholding_invalid_source(source, values, dtype):
    doc, write, root = source
    ref = doc['rois'][0]['source_assets']['layouts'][0]['asset']
    a = np.zeros((8, 8), dtype=dtype); a[:1, :2] = values
    path = root / ref['path']; np.save(path, a)
    ref['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_export(write())
    assert not result.ready and result.blocked[0]['reasons'] == ['exact_binary_required']


@pytest.mark.parametrize('container', ['npy', 'png'])
@pytest.mark.parametrize('semantics,expected_marked', [('binary', 63), ('binary_zero_is_marked', 1)])
def test_exact_declared_polarity(source, container, semantics, expected_marked):
    doc, write, root = source
    ref = doc['rois'][0]['source_assets']['layouts'][0]['asset']
    mask = np.full((8, 8), 255, np.uint8); mask[2, 3] = 0
    ref['path'] = f'assets/black.{container}'
    path = root / ref['path']
    if container == 'npy': np.save(path, mask)
    else: Image.fromarray(mask).save(path)
    ref['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    ref['value_semantics'] = semantics
    roi = load_export(write()).ready[0]
    assert int(roi.channels['layout:0'].sum()) == expected_marked
    assert roi.channels['layout:0'][2, 3] == (semantics == 'binary_zero_is_marked')


@pytest.mark.parametrize('field,value', [('schema_version', 'review_roi_v999'), ('producer_commit', None),
                                         ('source_manifest_hash', 'bad')])
def test_envelope_versions_required(source, field, value):
    doc, write, _ = source; doc[field] = value
    with pytest.raises(ContractError): load_export(write())


def test_producer_revision_pin(source):
    _, write, _ = source
    with pytest.raises(ContractError, match='producer_commit_mismatch'):
        load_export(write(), expected_producer_commit='a' * 40)


def test_valid_hole_at_center_is_blocked(source):
    doc, write, _ = source; doc['rois'][0]['center_xy_um'] = [100.25, 200.25]
    result = load_export(write())
    assert not result.ready and result.blocked[0]['reasons'] == ['position_outside_valid_domain']


def test_offline_import_and_cli_help_and_json(capsys):
    assert importlib.import_module('psdl2').__version__ == '0.1.0'
    with pytest.raises(SystemExit) as exit:
        main(['--help'])
    assert exit.value.code == 0 and 'offline' in capsys.readouterr().out
    assert main([str(FIXTURE / 'review_rois.json'), '--assets-root', str(FIXTURE)]) == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary['n_ready'] == 1 and summary['n_blocked'] == 3


@pytest.mark.parametrize('field', ['hotspot', 'ca_um2', 'fab'])
def test_risk_fields_on_roi_rejected(source, field):
    doc, write, _ = source; doc['rois'][0][field] = 'forbidden'
    result = load_export(write())
    assert not result.ready
    assert result.blocked[0]['reasons'] == ['unsupported_roi_field_original_input_only']


def test_excluded_record_cannot_be_promoted(source):
    doc, write, _ = source
    doc['excluded_records'] = [{'roi_id': doc['rois'][0]['roi_id'], 'reasons': ['source_shape_required']}]
    result = load_export(write())
    assert not result.ready
    assert result.blocked[0]['reasons'] == ['excluded_record_not_candidate']
    assert result.excluded_records[0]['reasons'] == ['source_shape_required']


def test_via_uses_producer_asset_reference_and_common_transform(source):
    doc, write, root = source
    ref = doc['rois'][0]['source_assets']['cnt'].copy()
    doc['rois'][0]['source_assets']['via'] = [ref]
    roi = load_export(write()).ready[0]
    bundle = build_bundle(roi, CropConfig((2, 2), (4, 4), 'preserve_exported_layers_v1', output_pitch_um=.25))
    expected = np.zeros((8, 8), np.uint8); expected[2:4, 4:6] = 1
    np.testing.assert_array_equal(bundle.near.channels['via:0'], expected)
    assert bundle.metadata['connections']['via_assets'] == [ref]


def test_symlink_asset_escape_is_blocked(source, tmp_path):
    doc, write, root = source
    ref = doc['rois'][0]['source_assets']['layouts'][0]['asset']
    path = root / ref['path']
    outside = root.parent / (root.name + '-outside.npy')
    outside.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(outside)
    result = load_export(write())
    assert not result.ready and result.blocked[0]['reasons'] == ['missing_or_escaping_asset_path']


def test_wrong_asset_shape_is_blocked(source):
    doc, write, root = source
    ref = doc['rois'][0]['source_assets']['layouts'][0]['asset']
    path = root / ref['path']; np.save(path, np.ones((1, 1), np.uint8))
    ref['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_export(write())
    assert not result.ready and result.blocked[0]['reasons'] == ['asset_frame_shape_mismatch']


def test_python_module_help_offline(monkeypatch, capsys):
    import runpy
    import sys
    monkeypatch.setattr(sys, 'argv', ['psdl2.data', '--help'])
    with pytest.raises(SystemExit) as exit:
        runpy.run_module('psdl2.data', run_name='__main__')
    assert exit.value.code == 0 and '--assets-root' in capsys.readouterr().out


def test_palette_indices_are_not_binary_pixel_values(source):
    doc, write, root = source
    ref = doc['rois'][0]['source_assets']['layouts'][0]['asset']
    image = Image.fromarray(np.zeros((8, 8), np.uint8)).convert('P')
    image.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
    ref['path'] = 'assets/palette.png'; path = root / ref['path']; image.save(path)
    ref['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_export(write())
    assert not result.ready and result.blocked[0]['reasons'] == ['single_channel_numeric_image_required']


def test_cli_explicit_relative_assets_root_resolves_from_caller(source, monkeypatch, capsys):
    _, write, root = source
    path = write()
    monkeypatch.chdir(root.parent)
    assert main([str(path), '--assets-root', root.name]) == 0
    assert json.loads(capsys.readouterr().out)['n_ready'] == 1
