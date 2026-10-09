"""Shared physical-cell sampling for near/wide original multi-layer bundles.

No quantitative CA is measured here. Resampling uses nearest source cell at each
output cell centre, with exact configured physical extents and a common transform.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from fractions import Fraction

import numpy as np

from .data import ROI
from .schema import ContractError, CropConfig, Frame


@dataclass(frozen=True)
class Patch:
    channels: dict[str, np.ndarray]
    grid: Frame
    metadata: dict


@dataclass(frozen=True)
class Bundle:
    near: Patch
    wide: Patch
    metadata: dict


def rational(value):
    """Exact decimal value of the declared number, never its binary expansion."""
    return Fraction(str(value))


def _count(size, pitch):
    cells = rational(size) / rational(pitch)
    if cells.denominator != 1 or cells < 1:
        raise ContractError('window_size_must_be_integer_multiple_of_output_pitch')
    return int(cells)


def exact_indices(origin, pitch, count, source_origin, source_pitch):
    offset = (origin + pitch/2 - source_origin) / source_pitch
    step = pitch / source_pitch
    return np.array([int((offset+i*step)//1) for i in range(count)], dtype=np.int64)


def exact_weights(origin, pitch, count, source_origin, source_pitch, source_count):
    weights = np.zeros((count, source_count), dtype=np.float64)
    for i in range(count):
        left = origin+i*pitch; right = left+pitch
        first = max(0, int((left-source_origin)//source_pitch))
        stop_ratio = (right-source_origin)/source_pitch
        stop = min(source_count, -(-stop_ratio.numerator//stop_ratio.denominator))
        for j in range(first, stop):
            overlap = min(right, source_origin+(j+1)*source_pitch)-max(left, source_origin+j*source_pitch)
            if overlap > 0:
                weights[i, j] = float(overlap/pitch)
    return weights


def exact_bounds(patch):
    origin = tuple(Fraction(*v) for v in patch.metadata['transform']['origin_xy_um_rational'])
    pitch = rational(patch.grid.pixel_pitch_um)
    return (*origin, origin[0]+patch.grid.shape_yx[1]*pitch, origin[1]+patch.grid.shape_yx[0]*pitch)


def _patch(roi, size, pitch, resampling_mode='nearest'):
    width, height = (_count(s, pitch) for s in size)
    cx, cy = roi.center_xy_um
    exact_origin = tuple(rational(c)-rational(s)/2 for c, s in zip((cx, cy), size))
    origin = tuple(float(v) for v in exact_origin)
    grid = Frame(roi.frame.frame_id, origin, pitch, (height, width))
    source = roi.frame
    # Every channel uses these same two index vectors. No channel-specific bbox.
    sx, sy = (exact_indices(o, rational(pitch), count, rational(so), rational(source.pixel_pitch_um))
              for o, count, so in zip(exact_origin, (width, height), source.origin_xy_um))
    inside_x = (sx >= 0) & (sx < source.shape_yx[1])
    inside_y = (sy >= 0) & (sy < source.shape_yx[0])
    inside = inside_y[:, None] & inside_x[None, :]
    clipped_x = np.clip(sx, 0, source.shape_yx[1] - 1)
    clipped_y = np.clip(sy, 0, source.shape_yx[0] - 1)
    channels = {}
    for key, mask in roi.channels.items():
        channels[key] = np.where(inside, mask[np.ix_(clipped_y, clipped_x)], 0).astype(np.uint8)
        channels[key].flags.writeable = False
    if resampling_mode == 'area_coverage':
        if pitch < source.pixel_pitch_um:
            raise ContractError('area_coverage_requires_coarser_or_equal_pitch')
        wx, wy = (exact_weights(o, rational(pitch), count, rational(so), rational(source.pixel_pitch_um), n)
                  for o, count, so, n in zip(exact_origin, (width, height), source.origin_xy_um,
                                            (source.shape_yx[1], source.shape_yx[0])))
        for key, mask in roi.channels.items():
            channels[key] = (wy @ mask.astype(np.float64) @ wx.T).clip(0, 1).astype(np.float32)
            channels[key].flags.writeable = False
        inside_fraction = wy.sum(axis=1)[:, None] * wx.sum(axis=1)[None, :]
        valid_count = float(channels['valid'].sum(dtype=np.float64))
    elif resampling_mode == 'nearest':
        valid_count = int(channels['valid'].sum())
    else:
        raise ContractError('unknown_resampling_mode')
    meta = {
        'center_xy_um': [cx, cy], 'size_xy_um': list(size),
        'bbox_xyxy_um': [*origin, float(exact_origin[0]+rational(size[0])), float(exact_origin[1]+rational(size[1]))],
        'shape_yx': [height, width], 'output_pitch_um': pitch,
        'source_pitch_um': source.pixel_pitch_um,
        'native_size_xy_px': [s / source.pixel_pitch_um for s in size],
        'transform_version': ('physical_cell_area_coverage_exact_v2' if resampling_mode == 'area_coverage'
                              else 'physical_cell_centres_nearest_exact_v2'),
        'boundary_tie_rule': 'higher_source_index_half_open',
        'source_indices_xy': [sx.tolist(), sy.tolist()],
        'transform': {'origin_xy_um': list(origin),
                      'origin_xy_um_rational': [[v.numerator, v.denominator] for v in exact_origin], 'pixel_pitch_um': pitch,
                      'axis_order': 'yx', 'origin_convention': 'top_left_cell_edge',
                      'source_origin_xy_um': list(source.origin_xy_um)},
        'resampled': pitch != source.pixel_pitch_um or any(
            ((o-rational(so))/rational(source.pixel_pitch_um)).denominator != 1
            for o, so in zip(exact_origin, source.origin_xy_um)),
        'valid_coverage': {'valid_cells': valid_count, 'total_cells': height * width,
                           'valid_fraction': valid_count / (height * width),
                           'outside_frame_cells': int((~inside).sum()),
                           'observed_invalid_cells': int(inside.sum()) - valid_count},
        'quantitative_CA_permitted': False,
    }
    if resampling_mode == 'area_coverage':
        meta['valid_coverage']['outside_frame_cells'] = float((1 - inside_fraction).sum())
        meta['valid_coverage']['observed_invalid_cells'] = max(0.0, float(inside_fraction.sum()) - valid_count)
        meta['valid_coverage']['counts_meaning'] = 'fractional_output_cell_coverage'
    return Patch(channels, grid, meta)


def build_bundle(roi, config, *, near_output_pitch_um=None, wide_output_pitch_um=None,
                 near_resampling_mode='nearest', wide_resampling_mode='nearest'):
    """One admitted ROI produces both patches; blocked dictionaries are rejected.

    Intermediate policy preserves all exported original layers, or additionally
    requires an explicit layer list. It never removes pixels or invents layers.
    """
    if not isinstance(roi, ROI) or roi.row.get('status') != 'ready' or roi.row.get('reasons'):
        raise ContractError('ready_original_roi_required')
    if not isinstance(config, CropConfig):
        raise ContractError('explicit_crop_config_required')
    layers = {binding['actual_layer'] for binding in roi.channel_bindings.values()}
    if not set(config.required_layers) <= layers:
        raise ContractError('required_intermediate_layer_observation_missing')
    pitch = roi.frame.pixel_pitch_um if config.output_pitch_um is None else config.output_pitch_um
    metadata = {
        'bundle_schema_version': 'original_roi_bundle_v1', 'roi_id': roi.row['roi_id'],
        'source_identity': deepcopy(roi.row['source_identity']),
        'center_xy_um': list(roi.center_xy_um), 'source_frame': deepcopy(roi.row['frame']),
        'endpoints': deepcopy(roi.row['endpoints']), 'failure_type': roi.row['failure_type'],
        'analysis_layer': roi.row['analysis_layer'], 'source_version': roi.row['source_version'],
        'source_assets': deepcopy(roi.row['source_assets']), 'versions': deepcopy(roi.versions),
        'channel_bindings': deepcopy(roi.channel_bindings),
        'connections': {'case_endpoints': deepcopy(roi.row['endpoints']),
                        'cnt_asset': deepcopy(roi.row['source_assets']['cnt']),
                        'via_assets': deepcopy(roi.row['source_assets']['via']),
                        'observation': 'exported_masks_only_no_inferred_via_endpoints'},
        'crop_config': asdict(config), 'intermediate_layer_policy': config.intermediate_layer_policy,
        'retained_actual_layers': sorted(layers),
    }
    from .schema import number
    near_pitch = pitch if near_output_pitch_um is None else number(near_output_pitch_um, positive=True)
    wide_pitch = pitch if wide_output_pitch_um is None else number(wide_output_pitch_um, positive=True)
    metadata['view_sampling'] = {'near_pitch_um': near_pitch, 'wide_pitch_um': wide_pitch,
                                 'near_mode': near_resampling_mode, 'wide_mode': wide_resampling_mode}
    return Bundle(_patch(roi, config.near_size_xy_um, near_pitch, near_resampling_mode),
                  _patch(roi, config.wide_size_xy_um, wide_pitch, wide_resampling_mode), metadata)
