import numpy as np
import pytest

from psdl2.crop import build_bundle
from psdl2.data import load_export
from psdl2.schema import ContractError, CropConfig

POLICY = 'preserve_exported_layers_v1'


def test_shared_transform_cross_layer_oracle_and_roundtrip(source):
    _, write, _ = source
    roi = load_export(write()).ready[0]
    bundle = build_bundle(roi, CropConfig((2, 2), (4, 4), POLICY))
    assert bundle.near.metadata['center_xy_um'] == bundle.wide.metadata['center_xy_um'] == [102, 202]
    assert bundle.near.grid.origin_xy_um == (101, 201)
    assert bundle.wide.grid.origin_xy_um == (100, 200)
    assert bundle.near.grid.um_to_pixel((101.25, 201.25)) == (.5, .5)
    assert bundle.near.grid.pixel_to_um((.5, .5)) == (101.25, 201.25)
    assert roi.frame.um_to_pixel(bundle.near.grid.pixel_to_um((.5, .5))) == (2.5, 2.5)
    # A and B are 3 columns apart: independently centring each mask fails.
    expected_a = np.zeros((4, 4), np.uint8); expected_a[0, 0] = 1
    expected_b = np.zeros((4, 4), np.uint8); expected_b[0, 3] = 1
    expected_cnt = np.zeros((4, 4), np.uint8); expected_cnt[1, 2] = 1
    np.testing.assert_array_equal(bundle.near.channels['layout:0'], expected_a)
    np.testing.assert_array_equal(bundle.near.channels['layout:1'], expected_b)
    np.testing.assert_array_equal(bundle.near.channels['cnt'], expected_cnt)
    assert bundle.metadata['endpoints'][0]['actual_layer'] == 'M1'
    assert bundle.metadata['endpoints'][1]['actual_layer'] == 'M3'
    assert bundle.metadata['intermediate_layer_policy'] == POLICY
    assert bundle.metadata['connections']['case_endpoints'] == roi.row['endpoints']
    assert bundle.metadata['versions']['producer_commit'] == 'f5e86006d72faf16c3bb747051325fbba4b0f9fd'


def test_padding_valid_observed_empty_and_missing_are_distinct(source):
    _, write, _ = source
    bundle = build_bundle(load_export(write()).ready[0], CropConfig((4, 4), (6, 6), POLICY))
    expected = np.zeros((12, 12), np.uint8)
    expected[2:10, 2:10] = 1; expected[2, 2] = 0
    np.testing.assert_array_equal(bundle.wide.channels['valid'], expected)
    assert bundle.wide.metadata['valid_coverage'] == {
        'valid_cells': 63, 'total_cells': 144, 'valid_fraction': 63 / 144,
        'outside_frame_cells': 80, 'observed_invalid_cells': 1}
    assert bundle.wide.channels['layout:0'][3, 3] == 0 and expected[3, 3] == 1  # observed empty
    assert bundle.wide.channels['layout:0'][0, 0] == 0 and expected[0, 0] == 0  # padding
    assert bundle.wide.metadata['quantitative_CA_permitted'] is False


def test_required_intermediate_missing_is_not_invented(source):
    _, write, _ = source
    roi = load_export(write()).ready[0]
    with pytest.raises(ContractError, match='observation_missing'):
        build_bundle(roi, CropConfig((2, 2), (4, 4), 'require_listed_layers_v1', required_layers=('M1', 'M2', 'M3')))
    bundle = build_bundle(roi, CropConfig((2, 2), (4, 4), 'require_listed_layers_v1', required_layers=('M1', 'M3')))
    assert bundle.metadata['crop_config']['required_layers'] == ('M1', 'M3')


@pytest.mark.parametrize('pitch,native_pixels', [(.5, 20), (1., 10)])
def test_physical_window_extent_and_fixed_grid(source, pitch, native_pixels):
    doc, write, _ = source
    doc['rois'][0]['frame']['pixel_pitch_um'] = pitch
    roi = load_export(write()).ready[0]
    native = build_bundle(roi, CropConfig((10, 10), (12, 12), POLICY))
    assert native.near.channels['valid'].shape == (native_pixels, native_pixels)
    assert native.near.metadata['bbox_xyxy_um'] == [97, 197, 107, 207]
    fixed = build_bundle(roi, CropConfig((10, 10), (12, 12), POLICY, output_pitch_um=.5))
    assert fixed.near.channels['valid'].shape == (20, 20)
    assert fixed.near.grid.pixel_to_um((0, 0)) == (97, 197)
    assert fixed.near.grid.pixel_to_um((20, 20)) == (107, 207)
    assert fixed.near.metadata['native_size_xy_px'] == [native_pixels, native_pixels]


def test_resampling_same_transform_for_all_channels(source):
    _, write, _ = source
    bundle = build_bundle(load_export(write()).ready[0], CropConfig((2, 2), (4, 4), POLICY, output_pitch_um=.25))
    a = np.zeros((8, 8), np.uint8); a[:2, :2] = 1
    b = np.zeros((8, 8), np.uint8); b[:2, 6:] = 1
    cnt = np.zeros((8, 8), np.uint8); cnt[2:4, 4:6] = 1
    np.testing.assert_array_equal(bundle.near.channels['layout:0'], a)
    np.testing.assert_array_equal(bundle.near.channels['layout:1'], b)
    np.testing.assert_array_equal(bundle.near.channels['cnt'], cnt)
    np.testing.assert_array_equal(bundle.near.channels['valid'], np.ones((8, 8), np.uint8))


def test_roi_index_is_not_physical_position(source):
    doc, write, _ = source
    doc['rois'][0]['center_xy_um'] = [4, 4]
    imported = load_export(write())
    assert not imported.ready
    assert imported.blocked[0]['reasons'] == ['position_outside_source_frame']
    with pytest.raises(ContractError, match='ready_original'):
        build_bundle(imported.blocked[0], CropConfig((2, 2), (4, 4), POLICY))


def test_fractional_center_preserved_and_nonintegral_grid_rejected(source):
    doc, write, _ = source
    doc['rois'][0]['center_xy_um'] = [102.125, 202.125]
    roi = load_export(write()).ready[0]
    patch = build_bundle(roi, CropConfig((2, 2), (4, 4), POLICY)).near
    assert patch.metadata['center_xy_um'] == [102.125, 202.125]
    assert patch.metadata['bbox_xyxy_um'] == [101.125, 201.125, 103.125, 203.125]
    assert patch.metadata['resampled'] is True
    with pytest.raises(ContractError, match='integer_multiple'):
        build_bundle(roi, CropConfig((2.1, 2), (4, 4), POLICY))
