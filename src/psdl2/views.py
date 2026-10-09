"""Physical jitter acceptance by observed common-core connectivity."""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .crop import build_bundle, exact_bounds, rational
from .data import ROI
from .schema import ContractError, CropConfig, number, xy

TOPOLOGY_POLICY = 'common_core_connectivity_v2'


@dataclass(frozen=True)
class ViewConfig:
    crop: CropConfig
    near_pitch_um: float
    wide_pitch_um: float
    near_resampling_mode: str
    wide_resampling_mode: str
    jitter_xy_um: tuple[float, float]
    core_size_xy_um: tuple[float, float]
    topology_policy: str

    def __post_init__(self):
        if not isinstance(self.crop, CropConfig):
            raise ContractError('crop_recipe_required')
        if self.topology_policy != TOPOLOGY_POLICY:
            raise ContractError('explicit_supported_topology_policy_required')
        for name in ('near_pitch_um', 'wide_pitch_um'):
            object.__setattr__(self, name, number(getattr(self, name), positive=True))
        for name in ('near_resampling_mode', 'wide_resampling_mode'):
            if getattr(self, name) not in ('nearest', 'area_coverage'):
                raise ContractError('explicit_resampling_mode_required')
        jitter, core = xy(self.jitter_xy_um), xy(self.core_size_xy_um)
        if any(j < 0 for j in jitter) or any(s <= 0 for s in core):
            raise ContractError('invalid_jitter_or_core')
        if any(rational(c)+2*rational(j) > rational(n)
               for c, j, n in zip(core, jitter, self.crop.near_size_xy_um)):
            raise ContractError('common_core_must_fit_every_jittered_near_window')
        object.__setattr__(self, 'jitter_xy_um', jitter)
        object.__setattr__(self, 'core_size_xy_um', core)

    @classmethod
    def from_dict(cls, raw):
        required = {'crop', 'near_pitch_um', 'wide_pitch_um', 'near_resampling_mode',
                    'wide_resampling_mode', 'jitter_xy_um', 'core_size_xy_um', 'topology_policy'}
        if set(raw) != required:
            raise ContractError('explicit_complete_view_config_required')
        return cls(**{**raw, 'crop': CropConfig.from_dict(raw['crop'])})

    def bundle(self, roi):
        bundle = build_bundle(roi, self.crop, near_output_pitch_um=self.near_pitch_um,
                              wide_output_pitch_um=self.wide_pitch_um,
                              near_resampling_mode=self.near_resampling_mode,
                              wide_resampling_mode=self.wide_resampling_mode)
        bundle.metadata['topology_policy'] = self.topology_policy
        return bundle


def components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining); remaining.remove(first)
        found, todo = {first}, [first]
        while todo:
            y, x = todo.pop()
            for point in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if point in remaining:
                    remaining.remove(point); found.add(point); todo.append(point)
        result.append(found)
    return result


def _labels(mask):
    labels = np.zeros(mask.shape, dtype=np.int32)
    for label, component in enumerate(components(mask), 1):
        for point in component:
            labels[point] = label
    return labels


def _partition(labels):
    """Canonical connected equivalence classes; component numbering is irrelevant."""
    names = {0: 0}
    return tuple(names.setdefault(int(value), len(names)) for value in labels)


def _location(bundle):
    return (bundle.metadata['roi_id'], bundle.metadata['source_identity'],
            bundle.metadata['source_frame'], bundle.metadata['endpoints'], bundle.metadata['source_assets'])


def _native_window(roi, patch, key):
    """Native graph inside this window, using actual native output when available."""
    frame = roi.frame; ox, oy = map(rational, frame.origin_xy_um); p = rational(frame.pixel_pitch_um)
    left, top, right, bottom = exact_bounds(patch)
    if patch.grid.pixel_pitch_um == frame.pixel_pitch_um and patch.metadata['transform_version'] == 'physical_cell_centres_nearest_exact_v2':
        sx, sy = patch.metadata['source_indices_xy']
        return patch.channels[key] > 0, patch.channels['valid'], sx, sy
    sx = [x for x in range(frame.shape_yx[1]) if left <= ox+(x+ rational(.5))*p < right]
    sy = [y for y in range(frame.shape_yx[0]) if top <= oy+(y+ rational(.5))*p < bottom]
    return roi.channels[key][np.ix_(sy, sx)] > 0, roi.channels['valid'][np.ix_(sy, sx)], sx, sy


def validate_pair(roi, first, second, core_size_xy_um, topology_policy):
    if topology_policy != TOPOLOGY_POLICY:
        raise ContractError('explicit_supported_topology_policy_required')
    expected = (roi.row['roi_id'], roi.row['source_identity'], roi.row['frame'], roi.row['endpoints'], roi.row['source_assets'])
    if _location(first) != _location(second) or _location(first) != expected:
        raise ContractError('two_views_must_share_original_location_and_bindings')
    for bundle in (first, second):
        original = bundle.metadata.get('original_center_xy_um', bundle.metadata['center_xy_um'])
        if tuple(original) != roi.center_xy_um:
            raise ContractError('two_views_must_share_original_physical_center')
        if bundle.metadata.get('topology_policy') != topology_policy:
            raise ContractError('pair_topology_policy_mismatch')
    core = xy(core_size_xy_um)
    cx, cy = map(rational, roi.center_xy_um)
    x0, y0 = cx-rational(core[0])/2, cy-rational(core[1])/2
    x1, y1 = cx+rational(core[0])/2, cy+rational(core[1])/2
    frame = roi.frame; ox, oy = map(rational, frame.origin_xy_um); p = rational(frame.pixel_pitch_um)
    if min(core) <= 0 or x0 < ox or y0 < oy or x1 > ox+frame.shape_yx[1]*p or y1 > oy+frame.shape_yx[0]*p:
        raise ContractError('common_core_not_observed')
    xs = [x for x in range(frame.shape_yx[1]) if ox+x*p < x1 and ox+(x+1)*p > x0]
    ys = [y for y in range(frame.shape_yx[0]) if oy+y*p < y1 and oy+(y+1)*p > y0]
    anchors = [(y, x) for y in ys for x in xs]
    if not anchors or not np.all(roi.channels['valid'][np.ix_(ys, xs)] == 1):
        raise ContractError('common_core_not_observed')
    for scale in ('near', 'wide'):
        partitions = []
        for bundle in (first, second):
            patch = getattr(bundle, scale)
            left, top, right, bottom = exact_bounds(patch)
            if left > x0 or top > y0 or right < x1 or bottom < y1:
                raise ContractError('common_core_not_observed')
            per_channel = {}
            for key in roi.channels:
                if key == 'valid':
                    continue
                marked, valid, sx, sy = _native_window(roi, patch, key)
                cols, rows = {x: i for i, x in enumerate(sx)}, {y: i for i, y in enumerate(sy)}
                if any(x not in cols or y not in rows or valid[rows[y], cols[x]] != 1 for y, x in anchors):
                    raise ContractError('common_core_not_observed')
                points = [(rows[y], cols[x]) for y, x in anchors]
                actual = np.array([marked[v, u] for v, u in points])
                expected_content = np.array([roi.channels[key][y, x] != 0 for y, x in anchors])
                if not np.array_equal(actual, expected_content):
                    raise ContractError('core_marked_content_changed')
                labels = _labels(marked & (valid == 1))
                native_relation = _partition([labels[v, u] for v, u in points])
                per_channel[key] = native_relation
                if patch.grid.pixel_pitch_um != frame.pixel_pitch_um or 'area_coverage' in patch.metadata['transform_version']:
                    sampled = _labels((patch.channels[key] > 0) & (patch.channels['valid'] > 0))
                    sample_labels = []
                    pitch = rational(patch.grid.pixel_pitch_um)
                    for (y, x), is_marked in zip(anchors, expected_content):
                        u = int((ox+(x+rational(.5))*p-left)//pitch)
                        v = int((oy+(y+rational(.5))*p-top)//pitch)
                        if not (0 <= u < sampled.shape[1] and 0 <= v < sampled.shape[0]) or patch.channels['valid'][v, u] <= 0:
                            raise ContractError('common_core_not_observed')
                        value = int(sampled[v, u])
                        if is_marked and value == 0:
                            raise ContractError('resampled_core_marked_content_lost')
                        sample_labels.append(value)
                    # Only loss is checked: coarse cells may merge distinct native
                    # shapes. That limitation remains explicit in the policy.
                    groups = {}
                    for native, sampled_label in zip(native_relation, sample_labels):
                        if native:
                            groups.setdefault(native, set()).add(sampled_label)
                    if any(len(values) != 1 or 0 in values for values in groups.values()):
                        raise ContractError('resampled_core_connection_lost')
            partitions.append(per_channel)
        if partitions[0] != partitions[1]:
            raise ContractError('core_connection_lost_between_views')
    identical = {scale: all(np.array_equal(getattr(first, scale).channels[key], getattr(second, scale).channels[key])
                            for key in roi.channels) for scale in ('near', 'wide')}
    evidence = {'status': 'accepted', 'location_id': roi.row['roi_id'],
                'core_bbox_xyxy_um': list(map(float, (x0, y0, x1, y1))),
                'topology_policy': topology_policy, 'identical_views': identical}
    for bundle in (first, second):
        bundle.metadata['pair_acceptance'] = deepcopy(evidence)
    return evidence


def generate_pair(roi, config, rng):
    if not isinstance(roi, ROI) or not isinstance(config, ViewConfig):
        raise ContractError('ready_roi_and_view_config_required')
    views = []
    for _ in range(2):
        jitter = rng.uniform(-np.array(config.jitter_xy_um), np.array(config.jitter_xy_um))
        row = deepcopy(roi.row); row['center_xy_um'] = (np.array(roi.center_xy_um)+jitter).tolist()
        view = config.bundle(replace(roi, row=row))
        view.metadata['original_center_xy_um'] = list(roi.center_xy_um)
        view.metadata['jitter_xy_um'] = jitter.tolist(); views.append(view)
    evidence = validate_pair(roi, *views, config.core_size_xy_um, config.topology_policy)
    return views[0], views[1], evidence
