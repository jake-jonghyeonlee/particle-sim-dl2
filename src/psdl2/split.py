"""Lineage and physical-overlap groups; caller explicitly assigns each group."""
from .schema import ContractError, text


def record_for_roi(roi, lineage, view_config):
    text(lineage)
    size = view_config.crop.wide_size_xy_um
    jitter = view_config.jitter_xy_um
    cx, cy = roi.center_xy_um
    return {'roi_id': roi.row['roi_id'], 'augmentation_of': roi.row['roi_id'],
            'lineage_id': lineage, 'product_id': roi.row['product_id'],
            'design_version': roi.row['design_version'], 'frame': roi.row['frame'],
            'bbox_xyxy_um': [cx-size[0]/2-jitter[0], cy-size[1]/2-jitter[1],
                             cx+size[0]/2+jitter[0], cy+size[1]/2+jitter[1]]}


def _related(a, b):
    if a['lineage_id'] == b['lineage_id'] or a['augmentation_of'] == b['augmentation_of']:
        return True
    # Different product frames must never be mistaken for spatially registered data.
    same_frame = all(a[k] == b[k] for k in ('product_id', 'design_version', 'frame'))
    x0, y0, x1, y1 = a['bbox_xyxy_um']; u0, v0, u1, v1 = b['bbox_xyxy_um']
    return same_frame and max(x0, u0) < min(x1, u1) and max(y0, v0) < min(y1, v1)


def groups(records):
    ids = [text(r['roi_id']) for r in records]
    if len(set(ids)) != len(ids):
        raise ContractError('unique_split_record_ids_required')
    for r in records:
        text(r['lineage_id']); text(r['augmentation_of'])
        bbox = r['bbox_xyxy_um']
        from .schema import number
        if len(bbox) != 4 or any(number(v) != v for v in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ContractError('valid_split_bbox_required')
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for i, a in enumerate(records):
        for j in range(i):
            if _related(a, records[j]):
                parents[find(i)] = find(j)
    result = {}
    for i, record in enumerate(records):
        result.setdefault(find(i), []).append(record['roi_id'])
    return sorted([sorted(group) for group in result.values()])


def validate_split(records, assignments):
    if set(assignments) != {r['roi_id'] for r in records} or any(
            split not in ('train', 'validation', 'test') for split in assignments.values()):
        raise ContractError('complete_explicit_split_assignments_required')
    for group in groups(records):
        if len({assignments[rid] for rid in group}) != 1:
            raise ContractError('lineage_location_or_augmentation_split_leakage')
    return {name: sorted(rid for rid, split in assignments.items() if split == name)
            for name in ('train', 'validation', 'test')}
