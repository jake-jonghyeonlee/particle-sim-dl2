"""split.groups: differential test against a verbatim copy of the original all-pairs implementation."""
import random
import time

import pytest

from psdl2.schema import ContractError, text
from psdl2.split import _components, groups


# ---- verbatim copy of the original implementation (oracle) -------------------------------------------

def _related(a, b):
    if a['lineage_id'] == b['lineage_id'] or a['augmentation_of'] == b['augmentation_of']:
        return True
    same_frame = all(a[k] == b[k] for k in ('product_id', 'design_version', 'frame'))
    x0, y0, x1, y1 = a['bbox_xyxy_um']; u0, v0, u1, v1 = b['bbox_xyxy_um']
    return same_frame and max(x0, u0) < min(x1, u1) and max(y0, v0) < min(y1, v1)


def old_groups(records):
    ids = [text(r['roi_id']) for r in records]
    if len(set(ids)) != len(ids):
        raise ContractError('unique_split_record_ids_required')
    for r in records:
        text(r['lineage_id']); text(r['augmentation_of'])
        bbox = r['bbox_xyxy_um']
        from psdl2.schema import number
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


def rec(rid, lineage, *, product='P', design='D', frame=None, augmentation=None, bbox=(0, 0, 2, 2)):
    return {'roi_id': rid, 'lineage_id': lineage, 'augmentation_of': augmentation or rid, 'product_id': product,
            'design_version': design, 'frame': frame or {'frame_id': 'F', 'origin_xy_um': [0, 0], 'pixel_pitch_um': .5},
            'bbox_xyxy_um': list(bbox)}


def same(records):
    expected = old_groups(records)
    assert groups(records) == expected
    return expected


# ---- adversarial inputs -------------------------------------------------------------------------------

def test_non_representative_bbox_chain_links_lineages():
    # Lineage A = {A1 far, A2 near B1}; B1 overlaps only A2, never A1 (the lineage's first box).
    rows = [rec('A1', 'A', bbox=(100, 100, 102, 102)), rec('A2', 'A', bbox=(0, 0, 2, 2)),
            rec('B1', 'B', bbox=(1, 1, 3, 3)), rec('C1', 'C', bbox=(50, 50, 52, 52))]
    assert same(rows) == [['A1', 'A2', 'B1'], ['C1']]


def test_cross_product_augmentation_link_and_same_lineage_other_product():
    rows = [rec('A', 'L1', product='P1', bbox=(0, 0, 2, 2)), rec('B', 'L2', product='P2', augmentation='A', bbox=(50, 50, 52, 52)),
            rec('C', 'L3', product='P3', bbox=(0, 0, 2, 2)), rec('D', 'L1', product='P4', bbox=(80, 80, 82, 82))]
    assert same(rows) == [['A', 'B', 'D'], ['C']]


def test_same_frame_id_with_different_frame_dicts_never_overlap():
    f1 = {'frame_id': 'F', 'origin_xy_um': [0, 0], 'pixel_pitch_um': .5}
    f2 = {'frame_id': 'F', 'origin_xy_um': [5, 0], 'pixel_pitch_um': .5}
    f3 = {'frame_id': 'F', 'origin_xy_um': [0, 0], 'pixel_pitch_um': .25}
    f1_reordered = {'pixel_pitch_um': .5, 'origin_xy_um': [0, 0], 'frame_id': 'F'}
    rows = [rec('a', 'a', frame=f1), rec('b', 'b', frame=f2), rec('c', 'c', frame=f3), rec('d', 'd', frame=f1_reordered),
            rec('e', 'e', design='D2', frame=f1), rec('f', 'f', product='Q', frame=f1)]
    assert same(rows) == [['a', 'd'], ['b'], ['c'], ['e'], ['f']]


def test_numeric_equality_semantics_of_frame_values():
    one = {'frame_id': 'F', 'origin_xy_um': [0, 0], 'pixel_pitch_um': 1}
    one_float = {'frame_id': 'F', 'origin_xy_um': [0.0, 0.0], 'pixel_pitch_um': 1.0}
    as_tuple = {'frame_id': 'F', 'origin_xy_um': (0, 0), 'pixel_pitch_um': 1}
    rows = [rec('a', 'a', frame=one), rec('b', 'b', frame=one_float), rec('c', 'c', frame=as_tuple)]
    assert same(rows) == [['a', 'b'], ['c']]


def test_touching_only_boxes_and_positive_area_overlap():
    rows = [rec('a', 'a', bbox=(0, 0, 2, 2)), rec('b', 'b', bbox=(2, 0, 4, 2)), rec('c', 'c', bbox=(0, 2, 2, 4)),
            rec('d', 'd', bbox=(2, 2, 4, 4))]
    assert same(rows) == [['a'], ['b'], ['c'], ['d']]
    # A sliver of positive area bridges all four corners.
    assert same(rows + [rec('e', 'e', bbox=(1.999, 1.999, 2.5, 2.5))]) == [['a', 'b', 'c', 'd', 'e']]


def test_multi_hop_transitive_chain_across_mechanisms():
    rows = [rec('r0', 'L0', bbox=(0, 0, 2, 2)), rec('r1', 'L1', bbox=(1.5, 0, 3.5, 2)),            # overlap
            rec('r2', 'L1', product='Z', bbox=(500, 500, 502, 502)),                             # lineage
            rec('r3', 'L9', product='Z', augmentation='r2', bbox=(900, 900, 902, 902)),           # augmentation
            rec('r4', 'L8', product='Z', bbox=(901, 900, 903, 902)),                              # overlap again
            rec('lone', 'LL', bbox=(7000, 7000, 7001, 7001))]
    assert same(rows) == [['lone'], ['r0', 'r1', 'r2', 'r3', 'r4']]


def test_wide_box_containing_many_and_equal_left_edges():
    rows = [rec('w', 'w', bbox=(0, 0, 100, 1))] + [rec(f'n{i}', f'n{i}', bbox=(i, 0, i+.5, 1)) for i in range(20)]
    rows += [rec('s0', 's0', bbox=(0, 5, 3, 6)), rec('s1', 's1', bbox=(0, 5, 3, 6)), rec('s2', 's2', bbox=(0, 5.99, 3, 8))]
    same(rows)


@pytest.mark.parametrize('mutate,message', [
    (lambda rs: rs.append(dict(rs[0])), 'unique_split_record_ids_required'),
    (lambda rs: rs[0].update(bbox_xyxy_um=[0, 0, 0, 2]), 'valid_split_bbox_required'),
    (lambda rs: rs[0].update(bbox_xyxy_um=[0, 0, 2]), 'valid_split_bbox_required'),
    (lambda rs: rs[0].update(lineage_id=''), None),
])
def test_error_behaviour_is_identical(mutate, message):
    rows = [rec('a', 'a'), rec('b', 'b', bbox=(5, 5, 6, 6))]
    mutate(rows)
    with pytest.raises(ContractError) as old:
        old_groups(rows)
    with pytest.raises(ContractError) as new:
        groups(rows)
    assert str(old.value) == str(new.value)
    if message:
        assert message in str(new.value)


def test_empty_and_single():
    assert same([]) == [] and same([rec('a', 'a')]) == [['a']]


# ---- random inputs -----------------------------------------------------------------------------------

FRAMES = [{'frame_id': 'F', 'origin_xy_um': [0, 0], 'pixel_pitch_um': .5},
          {'frame_id': 'F', 'origin_xy_um': [3, 0], 'pixel_pitch_um': .5},
          {'frame_id': 'F', 'origin_xy_um': [0, 0], 'pixel_pitch_um': .25},
          {'frame_id': 'G', 'origin_xy_um': [0, 0], 'pixel_pitch_um': .5},
          {'pixel_pitch_um': .5, 'origin_xy_um': [0.0, 0.0], 'frame_id': 'F'}]


def random_records(rng, n):
    span = rng.choice([4, 10, 40])
    rows = []
    for i in range(n):
        x0 = rng.randrange(span) * rng.choice([1, .5]); y0 = rng.randrange(span) * rng.choice([1, .5])
        w = rng.choice([.5, 1, 2, 3]); h = rng.choice([.5, 1, 2, 3])
        augmentation = f'r{rng.randrange(i)}' if i and rng.random() < .15 else None
        rows.append(rec(f'r{i}', f'L{rng.randrange(max(1, n//2))}' if rng.random() < .5 else f'solo{i}',
                        product=rng.choice(['P', 'Q', 'R']), design=rng.choice(['D', 'D', 'E']), frame=rng.choice(FRAMES),
                        augmentation=augmentation, bbox=(x0, y0, x0+w, y0+h)))
    rng.shuffle(rows)
    return rows


@pytest.mark.parametrize('seed', range(300))
def test_random_inputs_match_the_original(seed):
    rng = random.Random(seed)
    same(random_records(rng, rng.choice([1, 2, 5, 12, 30, 60])))


def test_random_inputs_exercise_multi_member_groups():
    rng = random.Random(7)
    sizes = [max(map(len, old_groups(r))) for r in (random_records(rng, 40) for _ in range(30))]
    assert max(sizes) > 5 and min(sizes) >= 1


# ---- performance ------------------------------------------------------------------------------------

def test_twenty_thousand_patches_in_seconds(capsys):
    rows = []
    for i in range(20000):
        product = i % 3
        k = i // 3
        x = (k % 80) * 3.0; y = (k // 80) * 3.0           # overlapping 4 um boxes on a 3 um grid
        rows.append(rec(f'p{i}', 'lin_a' if product < 2 else 'lin_b', product=f'P{product}', augmentation=f'p{i}',
                        frame={'frame_id': f'F{product}', 'origin_xy_um': [0, 0], 'pixel_pitch_um': .5},
                        bbox=(x, y, x+4, y+4)))
    start = time.perf_counter()
    result, stats = _components(rows)
    elapsed = time.perf_counter() - start
    print(f'groups(20000 patches, 2 lineages): {elapsed:.3f}s, pair_comparisons={stats["pair_comparisons"]}, '
          f'all_pairs={stats["all_pairs"]}, buckets={stats["buckets"]}')
    assert len(result) == 2 and sorted(map(len, result)) == [6666, 13334]
    assert elapsed < 10 and stats['pair_comparisons'] < stats['all_pairs'] / 100
    assert groups(rows) == result
    # Geometry only: every patch its own lineage, so only overlap links them. Not the algorithm's worst case: the dense 3 um grid
    # connects components quickly, so the union-find skip removes most comparisons. Boxes stacked in one vertical strip (x overlap,
    # no y overlap) would still compare all pairs; that input cannot come from make_split, where the lineage is a product property.
    for i, row in enumerate(rows):
        row['lineage_id'] = f'u{i}'
    start = time.perf_counter()
    result, stats = _components(rows)
    elapsed = time.perf_counter() - start
    print(f'groups(20000 patches, geometry only): {elapsed:.3f}s, pair_comparisons={stats["pair_comparisons"]}, '
          f'all_pairs={stats["all_pairs"]}, buckets={stats["buckets"]}')
    assert len(result) == 3 and elapsed < 10 and stats['pair_comparisons'] < stats['all_pairs'] / 20
