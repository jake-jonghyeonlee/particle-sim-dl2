"""Physical jitter views and conservative native-grid topology acceptance."""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .crop import Bundle, build_bundle
from .data import ROI
from .schema import ContractError, CropConfig, number, xy


@dataclass(frozen=True)
class ViewConfig:
    crop: CropConfig
    near_pitch_um: float
    wide_pitch_um: float
    near_resampling_mode: str
    wide_resampling_mode: str
    jitter_xy_um: tuple[float, float]
    core_size_xy_um: tuple[float, float]

    def __post_init__(self):
        if not isinstance(self.crop, CropConfig):
            raise ContractError('crop_recipe_required')
        for name in ('near_pitch_um', 'wide_pitch_um'):
            object.__setattr__(self, name, number(getattr(self, name), positive=True))
        for name in ('near_resampling_mode', 'wide_resampling_mode'):
            if getattr(self, name) not in ('nearest', 'area_coverage'):
                raise ContractError('explicit_resampling_mode_required')
        jitter = xy(self.jitter_xy_um)
        core = xy(self.core_size_xy_um)
        if any(j < 0 for j in jitter) or any(s <= 0 for s in core):
            raise ContractError('invalid_jitter_or_core')
        if any(c + 2 * j > n for c, j, n in zip(core, jitter, self.crop.near_size_xy_um)):
            raise ContractError('common_core_must_fit_every_jittered_near_window')
        object.__setattr__(self, 'jitter_xy_um', jitter)
        object.__setattr__(self, 'core_size_xy_um', core)

    @classmethod
    def from_dict(cls, raw):
        required = {'crop', 'near_pitch_um', 'wide_pitch_um', 'near_resampling_mode',
                    'wide_resampling_mode', 'jitter_xy_um', 'core_size_xy_um'}
        if set(raw) != required:
            raise ContractError('explicit_complete_view_config_required')
        return cls(**{**raw, 'crop': CropConfig.from_dict(raw['crop'])})

    def bundle(self, roi):
        return build_bundle(roi, self.crop, near_output_pitch_um=self.near_pitch_um,
                            wide_output_pitch_um=self.wide_pitch_um,
                            near_resampling_mode=self.near_resampling_mode,
                            wide_resampling_mode=self.wide_resampling_mode)


def components(mask):
    """Native 4-neighbour components; no probability/source-mask thresholding."""
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


def _location(bundle):
    return (bundle.metadata['roi_id'], bundle.metadata['source_identity'],
            bundle.metadata['source_frame'], bundle.metadata['endpoints'],
            bundle.metadata['source_assets'])


def validate_pair(roi, first, second, core_size_xy_um):
    """Fail closed when a core-touching native component is clipped or lost.

    Native component preservation is deliberately conservative: entire components
    touching the common core must survive all four patches, not just a pixel count.
    Backprojection only tests presence; it is never quantitative CA measurement.
    """
    expected = (roi.row['roi_id'], roi.row['source_identity'], roi.row['frame'],
                roi.row['endpoints'], roi.row['source_assets'])
    if _location(first) != _location(second) or _location(first) != expected:
        raise ContractError('two_views_must_share_original_location_and_bindings')
    for bundle in (first, second):
        original = bundle.metadata.get('original_center_xy_um', bundle.metadata['center_xy_um'])
        if tuple(original) != roi.center_xy_um:
            raise ContractError('two_views_must_share_original_physical_center')
    core = xy(core_size_xy_um)
    cx, cy = roi.center_xy_um
    x0, y0, x1, y1 = cx-core[0]/2, cy-core[1]/2, cx+core[0]/2, cy+core[1]/2
    frame = roi.frame
    ox, oy = frame.origin_xy_um; pitch = frame.pixel_pitch_um
    if core[0] <= 0 or core[1] <= 0 or x0 < ox or y0 < oy or x1 > ox+frame.shape_yx[1]*pitch or y1 > oy+frame.shape_yx[0]*pitch:
        raise ContractError('common_core_not_observed')
    xs = ox + np.arange(frame.shape_yx[1])*pitch
    ys = oy + np.arange(frame.shape_yx[0])*pitch
    # Every native cell intersecting the core must be observed (not just its centre).
    core_mask = ((ys[:, None] < y1) & (ys[:, None]+pitch > y0)
                 & (xs[None, :] < x1) & (xs[None, :]+pitch > x0))
    if not np.all(roi.channels['valid'][core_mask] == 1):
        raise ContractError('common_core_not_observed')
    patches = (first.near, first.wide, second.near, second.wide)
    for patch in patches:
        px0, py0, px1, py1 = patch.metadata['bbox_xyxy_um']
        if px0 > x0 or py0 > y0 or px1 < x1 or py1 < y1:
            raise ContractError('common_core_not_observed')
        core_x = (np.arange(patch.grid.shape_yx[1])+.5)*patch.grid.pixel_pitch_um+px0
        core_y = (np.arange(patch.grid.shape_yx[0])+.5)*patch.grid.pixel_pitch_um+py0
        covered = ((core_y[:, None] >= y0) & (core_y[:, None] < y1)
                   & (core_x[None, :] >= x0) & (core_x[None, :] < x1))
        if not covered.any() or np.any(patch.channels['valid'][covered] < 1-1e-6):
            raise ContractError('common_core_not_observed')
    for key, source_mask in roi.channels.items():
        if key == 'valid':
            continue
        for component in components(source_mask):
            def represented(patch):
                px0, py0, px1, py1 = patch.metadata['bbox_xyxy_um']
                points = set()
                for y, x in component:
                    physical = (ox+(x+.5)*pitch, oy+(y+.5)*pitch)
                    if px0 <= physical[0] < px1 and py0 <= physical[1] < py1:
                        u, v = patch.grid.um_to_pixel(physical)
                        if patch.channels[key][int(v), int(u)] > 0:
                            points.add((y, x))
                return points
            # Same-scale jitter views may not lose any represented native part of
            # an involved component, including components outside the common core.
            for a, b in ((first.near, second.near), (first.wide, second.wide)):
                if represented(a) != represented(b):
                    raise ContractError('involved_bridge_or_component_lost_between_views')
            if not any(core_mask[y, x] for y, x in component):
                continue
            if any(roi.channels['valid'][y, x] != 1 for y, x in component):
                raise ContractError('core_component_not_fully_observed')
            for patch in patches:
                px0, py0, px1, py1 = patch.metadata['bbox_xyxy_um']
                for y, x in component:
                    # Full native cell inclusion stops boundary clipping from silently
                    # losing the component even when nearest resampling still shows it.
                    left, top = ox+x*pitch, oy+y*pitch
                    if left < px0-1e-9 or top < py0-1e-9 or left+pitch > px1+1e-9 or top+pitch > py1+1e-9:
                        raise ContractError('core_component_clipped')
                    u, v = patch.grid.um_to_pixel((left+pitch/2, top+pitch/2))
                    if patch.channels[key][int(v), int(u)] <= 0:
                        raise ContractError('core_bridge_or_component_lost')
    return {'status': 'accepted', 'location_id': roi.row['roi_id'],
            'core_bbox_xyxy_um': [x0, y0, x1, y1],
            'topology_policy': 'full_core_touching_native_components_v1'}


def generate_pair(roi, config, rng):
    if not isinstance(roi, ROI) or not isinstance(config, ViewConfig):
        raise ContractError('ready_roi_and_view_config_required')
    views = []
    for _ in range(2):
        jitter = rng.uniform(-np.array(config.jitter_xy_um), np.array(config.jitter_xy_um))
        row = deepcopy(roi.row)
        row['center_xy_um'] = (np.array(roi.center_xy_um)+jitter).tolist()
        view = config.bundle(replace(roi, row=row))
        view.metadata['original_center_xy_um'] = list(roi.center_xy_um)
        view.metadata['jitter_xy_um'] = jitter.tolist()
        views.append(view)
    evidence = validate_pair(roi, *views, config.core_size_xy_um)
    return views[0], views[1], evidence
