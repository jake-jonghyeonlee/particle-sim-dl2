from copy import deepcopy
import pytest

from psdl2.schema import ContractError
from psdl2.split import groups, validate_split


def record(rid, lineage, *, product='P', augmentation=None, bbox=(100, 200, 102, 202)):
    return {'roi_id': rid, 'lineage_id': lineage, 'augmentation_of': augmentation or rid,
            'product_id': product, 'design_version': 'D',
            'frame': {'frame_id': 'F', 'origin_xy_um': [100, 200]}, 'bbox_xyxy_um': bbox}


def test_transitive_lineage_overlap_augmentation_groups():
    rows = [record('A', 'family1', bbox=(100, 200, 102, 202)),
            record('B', 'family1', product='OTHER'),
            record('C', 'family2', bbox=(101, 201, 103, 203)),
            record('D', 'family3', product='OTHER2', augmentation='C'),
            record('E', 'family4', product='UNRELATED')]
    assert groups(rows) == [['A', 'B', 'C', 'D'], ['E']]
    valid = validate_split(rows, {'A': 'train', 'B': 'train', 'C': 'train', 'D': 'train', 'E': 'test'})
    assert valid == {'train': ['A', 'B', 'C', 'D'], 'validation': [], 'test': ['E']}


@pytest.mark.parametrize('kind', ['lineage', 'overlap', 'augmentation'])
def test_leakage_fails(kind):
    a = record('A', 'family1'); b = record('B', 'family2')
    if kind == 'lineage': b['lineage_id'] = a['lineage_id']; b['product_id'] = 'OTHER'
    elif kind == 'augmentation': b['augmentation_of'] = 'A'; b['product_id'] = 'OTHER'
    with pytest.raises(ContractError, match='split_leakage'):
        validate_split([a, b], {'A': 'train', 'B': 'validation'})


def test_boundary_touch_only_and_complete_assignment():
    rows = [record('A', 'a'), record('B', 'b', bbox=(102, 200, 104, 202))]
    assert groups(rows) == [['A'], ['B']]
    with pytest.raises(ContractError, match='complete_explicit'):
        validate_split(rows, {'A': 'train'})
    assert validate_split(rows, {'A': 'train', 'B': 'test'})['test'] == ['B']
