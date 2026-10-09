from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from psdl2.crop import Patch, build_bundle
from psdl2.data import load_export
from psdl2.schema import ContractError, CropConfig
from psdl2.views import ViewConfig, generate_pair, validate_pair


def view_config(*, jitter=(.1, .1), near=(3, 3), wide=(4, 4), near_pitch=.5, wide_pitch=1):
    return ViewConfig(CropConfig(near, wide, 'preserve_exported_layers_v1'),
                      near_pitch, wide_pitch, 'nearest', 'area_coverage', jitter, (2, 2), 'common_core_connectivity_v2')


def test_label_free_jitter_has_one_transform_for_all_layers(source):
    _, write, _ = source
    roi = load_export(write()).ready[0]
    first, second, evidence = generate_pair(roi, view_config(), np.random.default_rng(7))
    assert evidence['status'] == 'accepted' and evidence['location_id'] == roi.row['roi_id']
    assert first.metadata['roi_id'] == second.metadata['roi_id']
    assert first.metadata['center_xy_um'] != second.metadata['center_xy_um']
    for bundle in (first, second):
        assert bundle.near.grid.shape_yx == (6, 6) and bundle.wide.grid.shape_yx == (4, 4)
        assert bundle.near.metadata['transform_version'] == 'physical_cell_centres_nearest_exact_v2'
        assert bundle.wide.metadata['transform_version'] == 'physical_cell_area_coverage_exact_v2'
        assert len({a.shape for a in bundle.near.channels.values()}) == 1
        assert len({a.shape for a in bundle.wide.channels.values()}) == 1


def test_area_coverage_thin_line_oracle_and_valid(source):
    _, write, _ = source
    roi = load_export(write()).ready[0]
    channels = dict(roi.channels)
    line = np.zeros((8, 8), np.uint8); line[:, 0] = 1
    channels['layout:0'] = line
    roi = replace(roi, channels=channels)
    bundle = build_bundle(roi, CropConfig((4, 4), (6, 6), 'preserve_exported_layers_v1'),
                          near_output_pitch_um=1, wide_output_pitch_um=1,
                          near_resampling_mode='area_coverage', wide_resampling_mode='area_coverage')
    expected = np.zeros((4, 4), np.float32); expected[:, 0] = .5
    np.testing.assert_array_equal(bundle.near.channels['layout:0'], expected)
    valid = np.ones((4, 4), np.float32); valid[0, 0] = .75
    np.testing.assert_array_equal(bundle.near.channels['valid'], valid)
    expected_wide = np.zeros((6, 6), np.float32); expected_wide[1:5, 1] = .5
    np.testing.assert_array_equal(bundle.wide.channels['layout:0'], expected_wide)
    assert bundle.wide.metadata['valid_coverage']['outside_frame_cells'] == 20
    assert bundle.near.metadata['quantitative_CA_permitted'] is False
    # The old nearest mode samples centres in column 1, entirely losing column 0.
    nearest = build_bundle(roi, CropConfig((4, 4), (6, 6), 'preserve_exported_layers_v1'),
                           near_output_pitch_um=1, wide_output_pitch_um=1)
    assert not nearest.near.channels['layout:0'].any()


def test_bridge_removed_rejected_even_same_center(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    channels = dict(roi.channels)
    bridge = np.zeros((8, 8), np.uint8); bridge[3, 1:7] = 1
    channels['layout:0'] = bridge; roi = replace(roi, channels=channels)
    config = view_config(jitter=(0, 0), near=(4, 4), wide=(6, 6), wide_pitch=.5)
    first, second, _ = generate_pair(roi, config, np.random.default_rng(3))
    modified = dict(second.near.channels); a = modified['layout:0'].copy(); a[3, 4] = 0
    modified['layout:0'] = a
    second = replace(second, near=replace(second.near, channels=modified))
    assert first.metadata['center_xy_um'] == second.metadata['center_xy_um']
    with pytest.raises(ContractError, match='core_marked_content|core_connection'):
        validate_pair(roi, first, second, (2, 2), 'common_core_connectivity_v2')


def test_frame_crossing_line_accepted_and_different_location_rejected(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    channels = dict(roi.channels); line = np.zeros((8, 8), np.uint8); line[3, :] = 1
    channels['layout:0'] = line
    clipped = replace(roi, channels=channels)
    assert generate_pair(clipped, view_config(jitter=(0, 0)), np.random.default_rng(1))[2]['status'] == 'accepted'
    first, second, _ = generate_pair(roi, view_config(), np.random.default_rng(1))
    metadata = deepcopy(second.metadata); metadata['roi_id'] = 'different_location'
    with pytest.raises(ContractError, match='share_original_location'):
        validate_pair(roi, first, replace(second, metadata=metadata), (2, 2), 'common_core_connectivity_v2')


def test_invalid_common_core_and_missing_parameters(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    channels = dict(roi.channels); valid = channels['valid'].copy(); valid[3, 3] = 0
    channels['valid'] = valid
    with pytest.raises(ContractError, match='common_core_not_observed'):
        generate_pair(replace(roi, channels=channels), view_config(), np.random.default_rng(0))
    from dataclasses import asdict
    for field in ('near_pitch_um', 'wide_pitch_um', 'jitter_xy_um', 'core_size_xy_um'):
        raw = asdict(view_config()); del raw[field]
        with pytest.raises(ContractError, match='explicit_complete'):
            ViewConfig.from_dict(raw)


def test_nearest_resampling_loss_of_native_bridge_rejected(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    channels = dict(roi.channels); line = np.zeros((8, 8), np.uint8); line[2, 2:6] = 1
    channels['layout:0'] = line; roi = replace(roi, channels=channels)
    cfg = replace(view_config(jitter=(0, 0), near=(4, 4), wide=(4, 4), near_pitch=1, wide_pitch=1),
                  near_resampling_mode='nearest')
    with pytest.raises(ContractError, match='core_marked_content|core_connection'):
        generate_pair(roi, cfg, np.random.default_rng(1))


def test_unconnected_component_outside_core_may_leave_view(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    channels = dict(roi.channels); mask = channels['layout:0'].copy(); mask[1, 1] = 1
    channels['layout:0'] = mask; roi = replace(roi, channels=channels)
    first, second, _ = generate_pair(roi, view_config(jitter=(0, 0), near=(4, 4), wide=(6, 6), wide_pitch=.5),
                                     np.random.default_rng(0))
    changed = dict(second.near.channels); mask = changed['layout:0'].copy(); mask[1, 1] = 0
    changed['layout:0'] = mask; second = replace(second, near=replace(second.near, channels=changed))
    assert validate_pair(roi, first, second, (2, 2), 'common_core_connectivity_v2')['status'] == 'accepted'


def test_fake_original_center_is_rejected(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    first, second, _ = generate_pair(roi, view_config(), np.random.default_rng(0))
    metadata = deepcopy(second.metadata); metadata['original_center_xy_um'] = [101, 201]
    with pytest.raises(ContractError, match='original_physical_center'):
        validate_pair(roi, first, replace(second, metadata=metadata), (2, 2), 'common_core_connectivity_v2')
