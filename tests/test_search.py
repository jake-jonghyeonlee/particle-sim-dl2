from copy import deepcopy
import math

import pytest

from psdl2.data import load_export
from psdl2.evaluate import evaluate
from psdl2.index import candidate_record
from psdl2.schema import ContractError
from psdl2.search import cosine, search, validate_index
from test_views import view_config


def fixed_records(source):
    _, write, _ = source
    roi = load_export(write()).ready[0]
    query = candidate_record(view_config().bundle(roi), [1, 0])
    query['roi_id'] = 'query'
    def record(rid, product, embedding):
        result = deepcopy(query)
        result.update(roi_id=rid, product_id=product, embedding=embedding)
        result['source_identity']['product_id'] = product
        return result
    return query, [record('A2', 'A', [1, 1]), record('A1', 'A', [1, 1]), record('B', 'B', [1, 0])]


def make_index(records):
    return {'schema_version': 'psdl2_index_v2', 'checkpoint_fingerprint': 'a'*64,
            'view_fingerprint': 'b'*64, 'records': records, 'rejected': [],
            'topology_policy': 'common_core_connectivity_v3',
            'connection_semantics': 'global_contact_proxy_unverified', 'comparison_basis': 'between_views_only'}


def test_primary_cohort_is_unchanged_by_higher_supplementary_and_stable_ties(source):
    query, records = fixed_records(source)
    result = search(query, make_index(records), {'A'}, topk=3)
    assert [(r['roi_id'], r['score']) for r in result['primary']] == [('A1', .707107), ('A2', .707107)]
    assert not result['supplementary']
    expanded = search(query, make_index(records), {'A'}, topk=3, expand=True)
    assert expanded['primary'] == result['primary']
    assert [(r['roi_id'], r['score'], r['search_scope']) for r in expanded['supplementary']] == [('B', 1., 'supplementary')]
    assert all('source_identity' in r and 'frame' in r and 'case_id' in r and 'valid_coverage' in r
               for r in expanded['primary'])
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([1, 0], [-1, 0]) == -1


@pytest.mark.parametrize('change', ['role', 'layer', 'cnt'])
def test_same_xy_wrong_case_binding_not_same_physical_case(source, change):
    query, records = fixed_records(source); candidate = records[0]
    candidate['center_xy_um'] = query['center_xy_um']
    signature = candidate['case_signature']
    if change == 'role':
        signature['endpoints'][0]['role'] = 'other'; signature['cnt_binding'][0]['role'] = 'other'
    elif change == 'layer':
        signature['endpoints'][0]['actual_layer'] = 'M2'; signature['cnt_binding'][0]['actual_layer'] = 'M2'
    else:
        signature['via_count'] = 1
    result = search(query, make_index([candidate]), {'A'}, topk=3)
    assert result['primary'] == [] and result['status'] == 'unavailable'
    assert result['rejected'][0]['reason'] == 'case_or_CNT_binding_incompatible'
    assert result['no_match_reasons'] == ['no_case_compatible_primary_candidates']


def test_no_candidates_is_unavailable_not_risk_zero(source):
    query, records = fixed_records(source)
    result = search(query, make_index([]), {'A'}, topk=1, expand=True)
    assert result['status'] == 'unavailable' and result['primary'] == []
    assert result['no_match_reasons'] == ['no_candidates_in_primary_cohort']
    assert 'risk' not in result
    with pytest.raises(ContractError, match='cohort'):
        search(query, make_index(records), set(), topk=2)
    with pytest.raises(ContractError, match='topk'):
        search(query, make_index(records), {'A'}, topk=0)


def test_same_physical_source_requires_complete_identity_frame_versions_and_assets(source):
    query, records = fixed_records(source)
    exact = deepcopy(query); exact['roi_id'] = 'alternate_selection'
    result = search(query, make_index([exact]), {query['product_id']}, topk=1)
    assert result['primary'][0]['same_physical_case'] is True
    exact['frame']['origin_xy_um'] = [101, 200]
    result = search(query, make_index([exact]), {query['product_id']}, topk=1)
    assert result['primary'][0]['same_physical_case'] is False


def test_index_schema_dimension_and_zero_vectors_fail(source):
    query, records = fixed_records(source)
    with pytest.raises(ContractError): cosine([0, 0], [1, 0])
    with pytest.raises(ContractError): cosine([1, 0], [1, 0, 0])
    with pytest.raises(ContractError): cosine([math.nan, 1], [1, 1])
    index = make_index(records); index['schema_version'] = 'old'
    with pytest.raises(ContractError, match='schema'): validate_index(index)


def test_recall_labels_optional_and_hand_counted_primary_only(source):
    query, records = fixed_records(source)
    result = search(query, make_index(records), {'A'}, topk=3, expand=True)
    unavailable = evaluate([result], k=1, assignments={'query': 'test'}, partition='test')
    assert unavailable['status'] == 'unavailable' and unavailable['recall_at_k'] is None
    labels = [{'a': 'query', 'b': 'A1', 'relation': 'similar'},
              {'a': 'query', 'b': 'B', 'relation': 'similar'},
              {'a': 'query', 'b': 'A2', 'relation': 'different'}]
    measured = evaluate([result], k=2, labels=labels, assignments={'query': 'test'}, partition='test')
    assert measured['recall_at_k'] == .5 and measured['n_evaluable_queries'] == 1
    assert measured['counterexample_retrievals'] == [{'roi_id': 'query', 'recall': .5, 'known_different_retrieved': ['A2']}]


def test_CNT_endpoint_binding_cannot_be_forged(source):
    query, records = fixed_records(source)
    records[0]['case_signature']['cnt_binding'][0]['electrode_id'] = 'OTHER_ENDPOINT'
    with pytest.raises(ContractError, match='CNT_endpoint_binding'):
        search(query, make_index([records[0]]), {'A'}, topk=1)
