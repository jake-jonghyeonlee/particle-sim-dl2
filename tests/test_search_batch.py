"""psdl2.search: --exclude-query-product, --per-product-topk, --all-ready (psdl2_search_batch_v1).

The world is the one the ETX smoke bundle builds (three compatible products P0..P2 and a negative product N whose
mask_derivation differs): the bundle runs once on the development interpreter, then the search CLI runs in-process.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import pytest

from psdl2 import search as search_module
from psdl2.index import main as index_main
from psdl2.schema import ContractError
from psdl2.search import derive_rankings, index_sources, main as search_main, search
from test_etx_runtime import build_smoke_bundle, execute, git_show
from test_search import fixed_records, make_index

# The single-query outputs must stay byte-identical to this commit's search.py. It never moves with the D10 constant.
SEARCH_GOLDEN_COMMIT = 'efc6dac346c6e0fefb373cb850946ff377adb627'
PRODUCTS = ('P0', 'P1', 'P2')
QUERIES = [f'{p}_case0' for p in PRODUCTS]

def grid_count(index, product):
    """Index records of one product. The products' images differ a little (adv:3), so their grid sizes may differ too."""
    return sum(1 for r in index['records'] if r['product_id'] == product)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def call(main, capsys, *argv):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('batch')
    base = tmp / 'base'
    base.mkdir()
    proc = execute(Path(sys.executable), build_smoke_bundle(tmp, base), tmp)
    assert proc.returncode == 0, proc.err
    root = base / 'psdl2_smoke'
    return types.SimpleNamespace(root=root, out=tmp / 'results', index=json.loads((root / 'out/index.json').read_text()))


def args(world, *extra, cohort=None, topk=4):
    root = world.root
    return ['--query-dataset', root / 'query.json', '--checkpoint', root / 'out/model.pt', '--index', root / 'out/index.json',
            '--cohort', cohort or root / 'cohort.json', '--topk', topk, *extra]


def single(world, capsys, roi, *extra, **kw):
    out = world.out / f'single_{roi}_{abs(hash(extra))}.json'
    code, _, err = call(search_main, capsys, *args(world, **kw), '--roi-id', roi, '--out', out, *extra)
    assert code in (0, 3), err
    return json.loads(out.read_text()), out


def batch(world, capsys, name, *extra, **kw):
    out = world.out / name
    code, stdout, err = call(search_main, capsys, *args(world, **kw), '--all-ready', '--out', out, *extra)
    return code, (json.loads(out.read_text()) if out.exists() else None), stdout, err, out


# ---------------------------------------------------------------- 1. byte-identical single queries (efc6dac golden)

def efc6dac_search_main():
    """The efc6dac search.py, executed as a module of the same package (its relative imports resolve to the current psdl2)."""
    module = types.ModuleType('psdl2._search_efc6dac')
    module.__package__ = 'psdl2'
    exec(compile(git_show('src/psdl2/search.py', SEARCH_GOLDEN_COMMIT).decode('utf-8'), 'search_efc6dac.py', 'exec'), module.__dict__)
    return module.main


@pytest.mark.parametrize('flags', [(), ('--expand',), ('--ignore-mask-derivation',)])
def test_single_query_without_new_flags_is_byte_identical_to_efc6dac(world, capsys, flags):
    old_main = efc6dac_search_main()
    for roi in QUERIES:
        files = {}
        for label, main in (('old', old_main), ('new', search_main)):
            out = world.out / f'golden_{label}_{roi}_{len(flags)}.json'
            code, stdout, err = call(main, capsys, *args(world), '--roi-id', roi, '--out', out, *flags)
            assert code == 0, err
            files[label] = (out.read_bytes(), stdout)
        assert files['old'] == files['new']
        assert b'excluded_products' not in files['new'][0] and b'per_product_topk' not in files['new'][0]


def test_row_keys_do_not_change_with_the_new_flags(world, capsys):
    old_main = efc6dac_search_main()
    out = world.out / 'rows_old.json'
    assert call(old_main, capsys, *args(world), '--roi-id', 'P0_case0', '--out', out, '--expand')[0] == 0
    old = json.loads(out.read_text())
    new, _ = single(world, capsys, 'P0_case0', '--expand', '--exclude-query-product', '--per-product-topk', '2')
    old_keys = {frozenset(row) for key in ('primary', 'supplementary') for row in old[key]}
    new_keys = {frozenset(row) for key in ('primary', 'supplementary') for row in new[key]}
    assert len(old_keys) == 1 and new_keys == old_keys
    assert set(new) == set(old) | {'excluded_products', 'per_product_topk'}


# ---------------------------------------------------------------- 2/3/4. flags on the multi-product world

@pytest.mark.parametrize('expand', [False, True])
def test_exclude_query_product_leaves_no_query_product_candidate(world, capsys, expand):
    flags = ('--exclude-query-product', *(('--expand',) if expand else ()))
    for roi, product in zip(QUERIES, PRODUCTS):
        result, _ = single(world, capsys, roi, *flags, topk=100)
        rows = result['primary'] + result['supplementary']
        assert rows and product not in {r['product_id'] for r in rows}
        assert result['excluded_products'] == [product] and 'per_product_topk' not in result
        excluded = [r for r in result['rejected'] if r['reason'] == 'query_product_excluded']
        assert len(excluded) == grid_count(world.index, product) > 0
        assert {r['roi_id'] for r in excluded} == {r['roi_id'] for r in world.index['records'] if r['product_id'] == product}
        # the negative product is rejected as incompatible, never ranked
        assert 'N' not in {r['product_id'] for r in rows}
        assert {r['reason'] for r in result['rejected'] if r['roi_id'] in {x['roi_id'] for x in world.index['records'] if x['product_id'] == 'N'}} \
            == {'mask_derivation_mismatch'}


def test_without_the_flag_the_query_product_is_ranked(world, capsys):
    result, _ = single(world, capsys, 'P0_case0', topk=100)
    assert 'P0' in {r['product_id'] for r in result['primary']} and 'excluded_products' not in result


def test_per_product_topk_then_topk_gives_distinct_products(world, capsys):
    result, _ = single(world, capsys, 'P0_case0', '--exclude-query-product', '--per-product-topk', '1', topk=2)
    assert len(result['primary']) == 2 and len({r['product_id'] for r in result['primary']}) == 2
    assert result['per_product_topk'] == 1
    wide, _ = single(world, capsys, 'P0_case0', '--exclude-query-product', topk=100)
    best = {}
    for row in wide['primary']:
        best[row['product_id']] = max(best.get(row['product_id'], -2), row['score'])
    # which products survive and which row stands for each does not rest on a tie: two products remain, each keeps one row
    # whose score is its product's maximum, and scores are untouched
    assert sorted(best) == ['P1', 'P2'] and {r['product_id'] for r in result['primary']} == {'P1', 'P2'}
    assert all(r['score'] == best[r['product_id']] for r in result['primary'])
    assert [r['score'] for r in result['primary']] == sorted((r['score'] for r in result['primary']), reverse=True)


def test_unavailable_when_the_only_cohort_product_is_the_query_product(world, capsys, tmp_path):
    cohort = tmp_path / 'cohort_p0.json'
    cohort.write_text(json.dumps(['P0']))
    out = world.out / 'unavailable.json'
    code, stdout, err = call(search_main, capsys, *args(world, cohort=cohort), '--roi-id', 'P0_case0', '--out', out, '--exclude-query-product')
    result = json.loads(out.read_text())
    assert code == 3 and result['status'] == 'unavailable' and result['primary'] == []
    assert result['no_match_reasons'] == ['no_candidates_in_primary_cohort']


# ---------------------------------------------------------------- 5. batch envelope

BATCH_FLAGS = ('--exclude-query-product', '--per-product-topk', '2', '--expand')


def test_batch_results_equal_single_outputs_and_rankings_are_rederived(world, capsys):
    code, envelope, stdout, err, out = batch(world, capsys, 'batch.json', *BATCH_FLAGS)
    assert code == 0, err
    assert list(envelope) == ['schema_version', 'checkpoint_fingerprint', 'view_fingerprint', 'index_sources', 'inputs', 'arguments',
                              'results', 'rankings']
    assert envelope['schema_version'] == 'psdl2_search_batch_v1'
    assert [r['query']['roi_id'] for r in envelope['results']] == QUERIES  # roi_id order
    for roi, result in zip(QUERIES, envelope['results']):
        one, _ = single(world, capsys, roi, *BATCH_FLAGS)
        assert canonical(result) == canonical(one)
    assert derive_rankings(envelope['results'], [r['query_export_sha256'] for r in envelope['rankings']]) == envelope['rankings']
    for ranking, result in zip(envelope['rankings'], envelope['results']):
        assert list(ranking) == ['query_roi_id', 'query_product_id', 'query_export_sha256', 'candidates']
        rows = result['primary'] + result['supplementary']
        assert [c['rank'] for c in ranking['candidates']] == list(range(1, len(rows) + 1))
        assert [c['roi_id'] for c in ranking['candidates']] == [r['roi_id'] for r in rows]
        assert ranking['query_roi_id'] == result['query']['roi_id'] and ranking['query_product_id'] == result['query']['product_id']
        assert all(set(c) == {'rank', 'roi_id', 'product_id', 'frame_id', 'center_xy_um', 'search_scope'} for c in ranking['candidates'])
        assert [c['frame_id'] for c in ranking['candidates']] == [r['frame']['frame_id'] for r in rows]
    # nothing was added to the results
    assert all(set(r) == {'query', 'status', 'no_match_reasons', 'primary', 'supplementary', 'rejected', 'primary_cohort', 'expanded',
                          'index_roi_ids', 'interpretation', 'excluded_products', 'per_product_topk'}
               for r in envelope['results'])
    assert json.loads(stdout) == {'status': 'available', 'n_queries': 3, 'n_available': 3, 'n_unavailable': 0}
    assert sum(1 for r in envelope['results'] if r['status'] == 'available') > 0


def test_batch_envelope_fields(world, capsys):
    code, envelope, _, err, _ = batch(world, capsys, 'fields.json', *BATCH_FLAGS, topk=3)
    assert code == 0, err
    index, root = world.index, world.root
    assert envelope['checkpoint_fingerprint'] == index['checkpoint_fingerprint'] and envelope['view_fingerprint'] == index['view_fingerprint']
    assert envelope['inputs'] == {'query_export_sha256': None,
                                  'query_dataset_sha256': hashlib.sha256((root / 'query.json').read_bytes()).hexdigest(),
                                  'index_sha256': hashlib.sha256((root / 'out/index.json').read_bytes()).hexdigest()}
    assert envelope['arguments'] == {'topk': 3, 'expand': True, 'per_product_topk': 2, 'exclude_query_product': True,
                                     'ignore_mask_derivation': False, 'cohort': ['N', 'P0', 'P1', 'P2']}
    sources = envelope['index_sources']
    assert [s['export_sha256'] for s in sources] == [s['export_sha256'] for s in index['sources']]
    assert [s['product_ids'] for s in sources] == [['P0'], ['P1'], ['P2'], ['N']]
    assert all(len(s['patch_set_sha256s']) == 1 and len(s['patch_set_sha256s'][0]) == 64 for s in sources)
    assert sources == index_sources(index)


def test_batch_exit_code_is_3_when_a_query_is_unavailable(world, capsys, tmp_path):
    cohort = tmp_path / 'cohort_p0.json'
    cohort.write_text(json.dumps(['P0']))
    code, envelope, stdout, err, _ = batch(world, capsys, 'partial.json', '--exclude-query-product', cohort=cohort)
    assert code == 3, err
    assert [r['status'] for r in envelope['results']] == ['unavailable', 'available', 'available']
    assert json.loads(stdout) == {'status': 'partial', 'n_queries': 3, 'n_available': 2, 'n_unavailable': 1}
    assert [len(r['candidates']) for r in envelope['rankings']][0] == 0


def test_batch_with_an_export_and_a_single_export_index(world, capsys, tmp_path):
    root = world.root
    p0, p1 = root / 'templates/P0/review_rois.json', root / 'templates/P1/review_rois.json'
    index = tmp_path / 'one_export_index.json'
    code, _, err = call(index_main, capsys, p0, '--checkpoint', root / 'out/model.pt', '--out', index)
    assert code == 0, err
    cohort = tmp_path / 'cohort.json'
    cohort.write_text(json.dumps(['P0']))
    out = tmp_path / 'batch_export.json'
    code, _, err = call(search_main, capsys, p1, '--all-ready', '--checkpoint', root / 'out/model.pt', '--index', index, '--cohort', cohort,
                        '--topk', 2, '--out', out)
    assert code == 0, err
    envelope = json.loads(out.read_text())
    assert envelope['index_sources'] == 'single_export'
    assert envelope['inputs'] == {'query_export_sha256': hashlib.sha256(p1.read_bytes()).hexdigest(), 'query_dataset_sha256': None,
                                  'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest()}
    assert [r['query']['roi_id'] for r in envelope['results']] == ['P1_case0'] and envelope['rankings'][0]['candidates'][0]['product_id'] == 'P0'
    assert [r['query_export_sha256'] for r in envelope['rankings']] == [hashlib.sha256(p1.read_bytes()).hexdigest()]


def test_each_ranking_names_the_export_of_its_own_query(world, capsys):
    """adv:1: a dataset has several query sources, so the hash belongs to the ranking, not to `inputs`."""
    code, envelope, _, err, _ = batch(world, capsys, 'hashes.json')
    assert code == 0, err
    expected = [hashlib.sha256((world.root / 'templates' / p / 'review_rois.json').read_bytes()).hexdigest() for p in PRODUCTS]
    assert [r['query_roi_id'] for r in envelope['rankings']] == QUERIES
    assert [r['query_export_sha256'] for r in envelope['rankings']] == expected and len(set(expected)) == 3
    assert envelope['inputs']['query_export_sha256'] is None  # unchanged: a dataset has no single export
    assert all('query_export_sha256' not in r['query'] for r in envelope['results'])  # results are untouched


def test_index_sources_product_ids_are_null_when_unknown():
    """adv:2: [] would claim 'this source produced no records'; null says 'cannot be told'."""
    def index(sources, records):
        return {'sources': sources, 'records': [{'product_id': p, **({'provenance': {'patch': {'set_sha256': d}}} if d else {})}
                                                for p, d in records]}
    patched = {'export_sha256': 'e1', 'patch_sets': [{'sha256': 'a' * 64}]}
    plain = {'export_sha256': 'e2'}
    # one source without patch sets: the records without patch provenance are its records
    one = index([patched, plain], [('X', 'a' * 64), ('Y', None), ('Z', None)])
    assert [s['product_ids'] for s in index_sources(one)] == [['X'], ['Y', 'Z']]
    assert index_sources(one)[1] == {'export_sha256': 'e2', 'patch_set_sha256s': [], 'product_ids': ['Y', 'Z']}
    # two sources without patch sets: they cannot be told apart
    two = index([patched, plain, {'export_sha256': 'e3'}], [('X', 'a' * 64), ('Y', None)])
    assert [s['product_ids'] for s in index_sources(two)] == [['X'], None, None]
    # a source whose patch set produced no record is known to have none
    empty = index([patched, plain], [('Y', None)])
    assert [s['product_ids'] for s in index_sources(empty)] == [[], ['Y']]
    assert index_sources({'records': []}) == 'single_export'


def test_negative_product_never_wins_even_with_the_mask_derivation_check_off(world, capsys):
    """adv:3: the negative product has a different image, so ignoring its mask_derivation does not make it a tie."""
    for roi in QUERIES:
        result, _ = single(world, capsys, roi, '--ignore-mask-derivation', topk=100)
        negative = [r for r in result['primary'] if r['product_id'] == 'N']
        assert result['mask_derivation_comparison'] == 'ignored'
        assert len(negative) == grid_count(world.index, 'N') > 0 and all(r['score'] < 1.0 for r in negative)
    # A query that is an exact copy of an indexed P0 record scores 1.0 against it; N (another image) stays below 1.0, so N
    # cannot take first place. (With identical images N would also score 1.0 and tie.)
    original = next(r for r in world.index['records'] if r['product_id'] == 'P0')
    query = deepcopy(original)
    query['roi_id'] = 'copy_of_' + original['roi_id']
    result = search(query, world.index, {'P0', 'P1', 'P2', 'N'}, topk=100, ignore_mask_derivation=True)
    scores = {r['roi_id']: r['score'] for r in result['primary']}
    assert scores[original['roi_id']] == 1.0 and result['primary'][0]['score'] == 1.0 and result['primary'][0]['product_id'] != 'N'
    assert max(r['score'] for r in result['primary'] if r['product_id'] == 'N') < 1.0


def test_a_contract_error_in_a_batch_names_the_query(world, capsys, monkeypatch):
    """adv:4: the batch stops at the first failing query (no skipping) and says which one."""
    real = search_module.search

    def failing(query, *a, **kw):
        if query['roi_id'] == 'P1_case0':
            raise ContractError('simulated_query_failure')
        return real(query, *a, **kw)

    monkeypatch.setattr(search_module, 'search', failing)
    out = world.out / 'named_error.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('OLD')
    code, stdout, err = call(search_main, capsys, *args(world), '--all-ready', '--out', out)
    assert code == 2 and stdout == '' and json.loads(err) == {'error': 'simulated_query_failure', 'roi_id': 'P1_case0'}
    assert out.read_text() == 'OLD'
    code, stdout, err = call(search_main, capsys, *args(world), '--roi-id', 'P1_case0', '--out', out)
    assert code == 2 and json.loads(err) == {'error': 'simulated_query_failure'}  # a single query keeps the old message


@pytest.mark.parametrize('extra,error', [
    ([], 'exactly_one_of_roi_id_or_all_ready_required'),
    (['--roi-id', 'P0_case0', '--all-ready'], 'exactly_one_of_roi_id_or_all_ready_required'),
    (['--all-ready', '--per-product-topk', '0'], 'positive_per_product_topk_required'),
])
def test_contract_errors_exit_2_and_keep_an_older_output(world, capsys, extra, error):
    out = world.out / 'kept.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('OLD')
    code, stdout, err = call(search_main, capsys, *args(world), '--out', out, *extra)
    assert code == 2 and json.loads(err) == {'error': error} and stdout == ''
    assert out.read_text() == 'OLD'


def test_batch_write_is_atomic(world, capsys, monkeypatch):
    out = world.out / 'atomic.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('OLD')

    def interrupted(*_):
        raise OSError('interrupted before the rename')

    monkeypatch.setattr(search_module.os, 'replace', interrupted)
    code, _, err = call(search_main, capsys, *args(world), '--all-ready', '--out', out)
    assert code == 2 and 'interrupted' in err
    assert out.read_text() == 'OLD' and [p.name for p in out.parent.glob('atomic.json*')] == ['atomic.json']
    monkeypatch.undo()
    assert call(search_main, capsys, *args(world), '--all-ready', '--out', out)[0] == 0
    assert json.loads(out.read_text())['schema_version'] == 'psdl2_search_batch_v1'
    assert [p.name for p in out.parent.glob('atomic.json*')] == ['atomic.json']


# ---------------------------------------------------------------- search(): the same rules on hand-made records

def test_search_function_exclusion_and_per_product_limits(source):
    query, records = fixed_records(source)
    product = query['product_id']

    def clone(rid, prod, embedding):
        record = deepcopy(records[0])
        record.update(roi_id=rid, product_id=prod, embedding=embedding)
        record['source_identity']['product_id'] = prod
        return record

    itself = deepcopy(query)
    own = clone('own', product, [1, 0])
    index = make_index([itself, own, clone('a1', 'A', [1, 1]), clone('a2', 'A', [1, 2]), clone('a3', 'A', [1, 3]), clone('b1', 'B', [1, .1])])
    result = search(query, index, {product, 'A', 'B'}, topk=10, exclude_query_product=True)
    reasons = {r['roi_id']: r['reason'] for r in result['rejected']}
    assert reasons == {'query': 'query_itself_excluded', 'own': 'query_product_excluded'}  # query_itself is decided first
    assert result['excluded_products'] == [product] and [r['roi_id'] for r in result['primary']] == ['b1', 'a1', 'a2', 'a3']
    limited = search(query, index, {'A', 'B'}, topk=3, per_product_topk=2)
    assert [r['roi_id'] for r in limited['primary']] == ['b1', 'a1', 'a2'] and limited['per_product_topk'] == 2
    assert [r['score'] for r in limited['primary']] == [r['score'] for r in search(query, index, {'A', 'B'}, topk=10)['primary'][:3]]
    narrow = search(query, index, {'A'}, topk=10, expand=True, per_product_topk=1)
    assert [r['roi_id'] for r in narrow['primary']] == ['a1']
    assert [r['roi_id'] for r in narrow['supplementary']] == ['own', 'b1']
    assert 'excluded_products' not in narrow
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ContractError, match='per_product_topk'):
            search(query, index, {'A'}, topk=1, per_product_topk=bad)
    alone = search(query, make_index([own]), {product}, topk=1, exclude_query_product=True)
    assert alone['status'] == 'unavailable' and alone['no_match_reasons'] == ['no_candidates_in_primary_cohort']


# ---------------------------------------------------------------- index size on the synthetic world (reported, not extrapolated)

def test_index_size_on_the_synthetic_world(world):
    path = world.root / 'out/index.json'
    records = len(world.index['records'])
    assert records == sum(grid_count(world.index, p) for p in (*PRODUCTS, 'N')) and records >= 4
    size = path.stat().st_size
    assert size > 0 and os.path.getsize(path) / records > 0
    print(f'index: {records} records, {size} bytes, {size / records:.0f} bytes per record')
