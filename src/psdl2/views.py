"""Exact common-core joint graph, a geometric contact proxy, not physical nets."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .crop import build_bundle, exact_bounds, rational, _count
from .data import ROI
from .schema import ContractError, CropConfig, number, xy

TOPOLOGY_POLICY = 'common_core_connectivity_v3'
CONNECTION_SEMANTICS = 'global_contact_proxy_unverified'
COMPARISON_BASIS = 'between_views_only'
REJECTION_REASONS = frozenset({
    'common_core_not_observed', 'core_marked_content_changed', 'no_core_layout_anchor',
    'core_joint_connectivity_changed', 'resampled_core_marked_content_lost',
    'resampled_core_joint_relation_changed',
})


class ViewRejection(ContractError):
    """Only geometric view rejection may consume a bounded retry."""


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
    connection_semantics: str
    comparison_basis: str
    max_pair_attempts: int

    def __post_init__(self):
        if not isinstance(self.crop, CropConfig):
            raise ContractError('crop_recipe_required')
        if self.topology_policy != TOPOLOGY_POLICY:
            raise ContractError('explicit_supported_topology_policy_required')
        if self.connection_semantics != CONNECTION_SEMANTICS or self.comparison_basis != COMPARISON_BASIS:
            raise ContractError('explicit_supported_graph_semantics_required')
        if type(self.max_pair_attempts) is not int or self.max_pair_attempts < 1:
            raise ContractError('explicit_positive_max_pair_attempts_required')
        for name in ('near_pitch_um', 'wide_pitch_um'):
            object.__setattr__(self, name, number(getattr(self, name), positive=True))
        for name in ('near_resampling_mode', 'wide_resampling_mode'):
            if getattr(self, name) not in ('nearest', 'area_coverage'):
                raise ContractError('explicit_resampling_mode_required')
        jitter, core = xy(self.jitter_xy_um), xy(self.core_size_xy_um)
        if any(j < 0 for j in jitter) or any(s <= 0 for s in core):
            raise ContractError('invalid_jitter_or_core')
        for size in (self.crop.near_size_xy_um, self.crop.wide_size_xy_um):
            if any(rational(c)+2*rational(j) > rational(n) for c, j, n in zip(core, jitter, size)):
                raise ContractError('common_core_must_fit_every_jittered_window')
        object.__setattr__(self, 'jitter_xy_um', jitter)
        object.__setattr__(self, 'core_size_xy_um', core)

    @classmethod
    def from_dict(cls, raw):
        from dataclasses import fields
        if set(raw) != {f.name for f in fields(cls)}:
            raise ContractError('explicit_complete_view_config_required')
        return cls(**{**raw, 'crop': CropConfig.from_dict(raw['crop'])})

    def bundle(self, roi):
        bundle = build_bundle(roi, self.crop, near_output_pitch_um=self.near_pitch_um,
                              wide_output_pitch_um=self.wide_pitch_um,
                              near_resampling_mode=self.near_resampling_mode,
                              wide_resampling_mode=self.wide_resampling_mode)
        bundle.metadata.update(topology_policy=self.topology_policy,
                               connection_semantics=self.connection_semantics,
                               comparison_basis=self.comparison_basis)
        return bundle


def _ceil(value):
    return -(-value.numerator // value.denominator)


def core_indices(roi, size):
    """Positive-area overlap with the fixed half-open core, exact per axis."""
    p = rational(roi.frame.pixel_pitch_um)
    bounds, axes = [], []
    for c, s, o, n in zip(roi.center_xy_um, size, roi.frame.origin_xy_um,
                          reversed(roi.frame.shape_yx)):
        left, right = rational(c)-rational(s)/2, rational(c)+rational(s)/2
        first, stop = int((left-rational(o))//p), _ceil((right-rational(o))/p)
        if left < rational(o) or right > rational(o)+n*p or first >= stop:
            raise ViewRejection('common_core_not_observed')
        axes.append(np.arange(first, stop, dtype=np.int64)); bounds.extend((left, right))
    xs, ys = axes
    if not np.all(roi.channels['valid'][np.ix_(ys, xs)] == 1):
        raise ViewRejection('common_core_not_observed')
    return xs, ys, (bounds[0], bounds[2], bounds[1], bounds[3])


def preflight(roi, config):
    """Combine a recipe with source alignment before any optimizer step."""
    try:
        xs, ys, _ = core_indices(roi, config.core_size_xy_um)
    except ViewRejection as exc:
        raise ContractError('view_preflight_'+str(exc)) from exc
    p = rational(roi.frame.pixel_pitch_um)
    for scale in ('near', 'wide'):
        pitch = rational(getattr(config, scale+'_pitch_um'))
        mode = getattr(config, scale+'_resampling_mode')
        if pitch > p and mode != 'area_coverage':
            raise ContractError('coarse_nearest_not_supported_for_views: use area_coverage')
        if pitch < p:
            raise ContractError('views_require_native_or_coarser_pitch')
        size = getattr(config.crop, scale+'_size_xy_um')
        for c, s, j, o, core_axis in zip(roi.center_xy_um, size, config.jitter_xy_um,
                                        roi.frame.origin_xy_um, (xs, ys)):
            count = _count(s, float(pitch))
            c, s, j, o = map(rational, (c, s, j, o))
            if mode == 'nearest':
                max_start = int(((c+j-s/2-o)/p+rational(.5))//1)
                min_stop = int(((c-j-s/2-o)/p+rational(.5))//1)+count
            else:
                max_start = _ceil((c+j-s/2-o)/p-rational(.5))
                min_stop = _ceil((c-j+s/2-o)/p-rational(.5))
            if max_start > core_axis[0] or min_stop <= core_axis[-1]:
                raise ContractError('view_preflight_core_not_represented_at_jitter_extrema')
    return {'status': 'passed', 'roi_id': roi.row['roi_id']}


def joint_labels(marked, layout_indices):
    """Run nodes + vectorized union-find; O(cells+edges), no cell objects/AxA.

    Horizontal runs are implicit edges. Vertical and layout/contact edges hook
    roots in arrays, then pointer jumping compresses paths to convergence.
    """
    marked = np.asarray(marked, dtype=bool)
    starts = marked.copy(); starts[:, :, 1:] &= ~marked[:, :, :-1]
    labels = np.cumsum(starts.ravel(), dtype=np.int32).reshape(marked.shape)
    labels[~marked] = 0
    edges_a, edges_b = [], []
    a, b = labels[:, :-1], labels[:, 1:]
    both = (a != 0) & (b != 0)
    edges_a.append(a[both]); edges_b.append(b[both])
    contacts = [i for i in range(len(marked)) if i not in layout_indices]
    for layout in layout_indices:
        for contact in contacts:
            a, b = labels[layout], labels[contact]
            both = (a != 0) & (b != 0)
            edges_a.append(a[both]); edges_b.append(b[both])
    a, b = np.concatenate(edges_a), np.concatenate(edges_b)
    parent = np.arange(int(labels.max())+1, dtype=np.int32)
    while a.size:
        ra, rb = parent[a], parent[b]
        changed = ra != rb
        if not np.any(changed):
            break
        np.minimum.at(parent, np.maximum(ra[changed], rb[changed]), np.minimum(ra[changed], rb[changed]))
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    return parent[labels]


def _partition(values):
    """Canonical labels by first anchor occurrence, independent of graph IDs."""
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    canonical = np.empty(len(first), np.int32)
    canonical[np.argsort(first)] = np.arange(len(first))
    return canonical[inverse]


def _location(bundle):
    m = bundle.metadata
    return (m['roi_id'], m['source_identity'], m['source_frame'], m['endpoints'], m['source_assets'], m['analysis_layer'])


def _native_window(roi, patch, keys):
    p = rational(roi.frame.pixel_pitch_um)
    if rational(patch.grid.pixel_pitch_um) < p:
        raise ContractError('views_require_native_or_coarser_pitch')
    if patch.metadata['transform_version'] == 'physical_cell_centres_nearest_exact_v2':
        if rational(patch.grid.pixel_pitch_um) != p:
            raise ContractError('coarse_nearest_not_supported_for_views: use area_coverage')
        sx, sy = (np.asarray(a, dtype=np.int64) for a in patch.metadata['source_indices_xy'])
        return np.stack([patch.channels[k] > 0 for k in keys]), patch.channels['valid'], sx, sy
    left, top, right, bottom = exact_bounds(patch)
    axes = []
    for low, high, origin, count in zip((left, top), (right, bottom), roi.frame.origin_xy_um,
                                       reversed(roi.frame.shape_yx)):
        first = max(0, _ceil((low-rational(origin))/p-rational(.5)))
        stop = min(count, _ceil((high-rational(origin))/p-rational(.5)))
        axes.append(np.arange(first, stop, dtype=np.int64))
    sx, sy = axes
    return (np.stack([roi.channels[k][np.ix_(sy, sx)] > 0 for k in keys]),
            roi.channels['valid'][np.ix_(sy, sx)], sx, sy)


def _map_axis(source_indices, source_origin, source_pitch, output_origin, output_pitch):
    # Exact once per axis, not per anchor. A grid can have arbitrarily placed origin.
    base = (rational(source_origin)+rational(.5)*source_pitch-output_origin)/output_pitch
    step = source_pitch/output_pitch
    return np.array([int((base+int(i)*step)//1) for i in source_indices], dtype=np.int64)


def validate_pair(roi, first, second, core_size_xy_um, topology_policy):
    if topology_policy != TOPOLOGY_POLICY:
        raise ContractError('explicit_supported_topology_policy_required')
    expected = (roi.row['roi_id'], roi.row['source_identity'], roi.row['frame'], roi.row['endpoints'],
                roi.row['source_assets'], roi.row['analysis_layer'])
    if _location(first) != _location(second) or _location(first) != expected:
        raise ContractError('two_views_must_share_original_location_and_bindings')
    for bundle in (first, second):
        original = bundle.metadata.get('original_center_xy_um', bundle.metadata['center_xy_um'])
        if tuple(original) != roi.center_xy_um:
            raise ContractError('two_views_must_share_original_physical_center')
        if (bundle.metadata.get('topology_policy') != topology_policy or
                bundle.metadata.get('connection_semantics') != CONNECTION_SEMANTICS or
                bundle.metadata.get('comparison_basis') != COMPARISON_BASIS):
            raise ContractError('pair_topology_policy_mismatch')
    xs, ys, bounds = core_indices(roi, xy(core_size_xy_um))
    # Only involved endpoint layouts take part; exported intermediate layers still
    # have their common-core content checked but do not invent graph endpoints.
    endpoints = roi.row['endpoints']
    keys = [k for k in roi.channels if k != 'valid']
    involved = [i for i, k in enumerate(keys) if k.startswith('layout:') and
                roi.channel_bindings[k] in endpoints]
    graph_indices = involved+[i for i, k in enumerate(keys) if k == 'cnt' or k.startswith('via:')]
    graph_layout = list(range(len(involved)))
    content = np.stack([roi.channels[k][np.ix_(ys, xs)] > 0 for k in keys])
    anchor_channel, anchor_y, anchor_x = np.nonzero(content[involved])
    if not anchor_channel.size:
        raise ViewRejection('no_core_layout_anchor')
    p = rational(roi.frame.pixel_pitch_um)
    for scale in ('near', 'wide'):
        partitions = []
        for bundle in (first, second):
            patch = getattr(bundle, scale)
            marked, valid, sx, sy = _native_window(roi, patch, keys)
            if (not sx.size or not sy.size or xs[0] < sx[0] or xs[-1] > sx[-1] or
                    ys[0] < sy[0] or ys[-1] > sy[-1]):
                raise ViewRejection('common_core_not_observed')
            u, v = xs-sx[0], ys-sy[0]
            if not np.all(valid[np.ix_(v, u)] == 1):
                raise ViewRejection('common_core_not_observed')
            if not np.array_equal(marked[:, v[:, None], u], content):
                raise ViewRejection('core_marked_content_changed')
            labels = joint_labels(marked[graph_indices] & (valid == 1), graph_layout)
            native = _partition(labels[anchor_channel, v[anchor_y], u[anchor_x]])
            partitions.append(native)
            if 'area_coverage' in patch.metadata['transform_version']:
                left, top, _, _ = exact_bounds(patch)
                out_pitch = rational(patch.grid.pixel_pitch_um)
                su = _map_axis(xs, roi.frame.origin_xy_um[0], p, left, out_pitch)
                sv = _map_axis(ys, roi.frame.origin_xy_um[1], p, top, out_pitch)
                if (su.min() < 0 or sv.min() < 0 or su.max() >= patch.grid.shape_yx[1] or
                        sv.max() >= patch.grid.shape_yx[0] or
                        not np.all(patch.channels['valid'][np.ix_(sv, su)] > 0)):
                    raise ViewRejection('common_core_not_observed')
                sampled = np.stack([patch.channels[k] > 0 for k in keys])
                if np.any(content & ~sampled[:, sv[:, None], su]):
                    raise ViewRejection('resampled_core_marked_content_lost')
                labels = joint_labels(sampled[graph_indices] & (patch.channels['valid'] > 0), graph_layout)
                relation = _partition(labels[anchor_channel, sv[anchor_y], su[anchor_x]])
                if not np.array_equal(native, relation):
                    raise ViewRejection('resampled_core_joint_relation_changed')
        if not np.array_equal(partitions[0], partitions[1]):
            raise ViewRejection('core_joint_connectivity_changed')
    identical = {scale: all(np.array_equal(getattr(first, scale).channels[k], getattr(second, scale).channels[k])
                            for k in roi.channels) for scale in ('near', 'wide')}
    evidence = {'status': 'accepted', 'location_id': roi.row['roi_id'],
                'core_bbox_xyxy_um': list(map(float, bounds)), 'topology_policy': topology_policy,
                'connection_semantics': CONNECTION_SEMANTICS, 'comparison_basis': COMPARISON_BASIS,
                'resampled_valid_semantics': 'positive_observed_support_fraction_not_full_observation',
                'identical_views': identical}
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
