"""Lineage and physical-overlap groups; caller explicitly assigns each group."""
from __future__ import annotations

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
    result, _ = _components(records)
    return result


def _freeze(value):
    """Hashable key with the same equality as ``==`` on JSON-like values (1 == 1.0, list != tuple)."""
    if isinstance(value, dict):
        return ('d', frozenset((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return ('l', tuple(_freeze(v) for v in value))
    if isinstance(value, tuple):
        return ('t', tuple(_freeze(v) for v in value))
    return value


def _components(records):
    """Connected components of ``_related`` without the all-pairs scan; returns (groups, stats).

    Lineage and augmentation links are global hash unions. Overlap is only tested inside a bucket whose
    (product_id, design_version, complete frame) are equal, by an x-sweep, so no pair that ``_related``
    could link is skipped.
    """
    n = len(records)
    parents = list(range(n))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def union(i, j):
        parents[find(i)] = find(j)
    for field in ('lineage_id', 'augmentation_of'):
        first = {}
        for i, r in enumerate(records):
            j = first.setdefault(r[field], i)
            if j != i:
                union(i, j)
    buckets = {}
    for i, r in enumerate(records):
        buckets.setdefault(_freeze((r['product_id'], r['design_version'], r['frame'])), []).append(i)
    comparisons = 0
    for members in buckets.values():
        members.sort(key=lambda i: records[i]['bbox_xyxy_um'][0])
        active = []
        for i in members:
            x0, y0, _, y1 = records[i]['bbox_xyxy_um']
            active = [j for j in active if records[j]['bbox_xyxy_um'][2] > x0]
            for j in active:
                if find(i) == find(j):
                    continue  # already one component; the pair cannot change the result
                comparisons += 1
                _, v0, _, v1 = records[j]['bbox_xyxy_um']
                if max(y0, v0) < min(y1, v1):
                    union(i, j)
            active.append(i)
    grouped = {}
    for i, record in enumerate(records):
        grouped.setdefault(find(i), []).append(record['roi_id'])
    stats = {'records': n, 'buckets': len(buckets), 'pair_comparisons': comparisons, 'all_pairs': n*(n-1)//2}
    return sorted([sorted(group) for group in grouped.values()]), stats


def validate_split(records, assignments):
    if set(assignments) != {r['roi_id'] for r in records} or any(
            split not in ('train', 'validation', 'test') for split in assignments.values()):
        raise ContractError('complete_explicit_split_assignments_required')
    for group in groups(records):
        if len({assignments[rid] for rid in group}) != 1:
            raise ContractError('lineage_location_or_augmentation_split_leakage')
    return {name: sorted(rid for rid, split in assignments.items() if split == name)
            for name in ('train', 'validation', 'test')}
