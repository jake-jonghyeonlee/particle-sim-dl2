"""psdl2.eval_position: batch envelope + several baseline files (psdl2_eval_position_multi_v1), and the unchanged old paths.

The unit is the query; the truth is the query's labelled (product, frame, position) list. Hand-made files pin the counting rules,
the real search CLI output (the ETX smoke world of test_search_batch) pins the link from psdl2_search_batch_v1 to the evaluation.
"""
import json
import types

import pytest

from psdl2 import eval_position as eval_module
from psdl2.eval_position import main as eval_main
from psdl2.search import main as search_main
from test_etx_runtime import git_show
from test_search_batch import QUERIES, args as search_args, call, single, world  # noqa: F401  (world is a fixture)

# The existing --search / --baseline paths must stay byte-identical to this commit's eval_position.py.
EVAL_GOLDEN_COMMIT = 'efc6dac346c6e0fefb373cb850946ff377adb627'
TOLERANCE = 0.5


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def truth_file(path, labels):
    """labels: [(query_roi_id or None, product, frame, (x, y))] -> roi_correspondence_v1."""
    pairs = [{'pair_id': 'T%d' % i,
              'a': {'product_id': 'Q', 'design_version': 'D', 'frame_id': 'FQ', 'center_xy_um': [0, 0], 'analysis_layer': 'M2', 'roi_id': query},
              'b': {'product_id': product, 'design_version': 'D', 'frame_id': frame, 'center_xy_um': list(center), 'analysis_layer': 'M2'},
              'source': {'kind': 'manual', 'labelled_by': 't', 'labelled_at': '2026-10-26'}, 'status': 'ready', 'reasons': []}
             for i, (query, product, frame, center) in enumerate(labels, 1)]
    return write_json(path, {'schema_version': 'roi_correspondence_v1', 'producer': {'kind': 'manual', 'version': '1'}, 'pairs': pairs})


def baseline_file(path, product, rows):
    """rows: {query: [(frame, (x, y)), ...]} -> psdl2_baseline_ncc_v1 of one candidate product."""
    results = [{'query': {'roi_id': query}, 'status': 'available' if cands else 'unavailable',
                'primary': [{'rank': rank, 'score': 1.0 / rank, 'center_xy_um': list(center), 'product_id': product, 'frame_id': frame}
                            for rank, (frame, center) in enumerate(cands, 1)]}
               for query, cands in rows.items()]
    return write_json(path, {'schema_version': 'psdl2_baseline_ncc_v1', 'candidate_product_id': product, 'results': results})


def batch_file(path, rankings):
    """rankings: {query: [(scope, product, frame, (x, y)), ...]} -> psdl2_search_batch_v1 (only what the evaluation reads)."""
    out = [{'query_roi_id': query, 'query_product_id': 'Q', 'query_export_sha256': None,
            'candidates': [{'rank': rank, 'roi_id': 'r%d' % rank, 'product_id': product, 'frame_id': frame,
                            'center_xy_um': list(center), 'search_scope': scope}
                           for rank, (scope, product, frame, center) in enumerate(cands, 1)]}
           for query, cands in rankings.items()]
    return write_json(path, {'schema_version': 'psdl2_search_batch_v1', 'rankings': out})


def run(capsys, *argv):
    try:
        code = eval_main([str(a) for a in argv])
    except SystemExit as exc:  # argparse errors
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_json(capsys, out, *argv):
    code, stdout, err = run(capsys, *argv, '--out', out)
    return code, (json.loads(out.read_text(encoding='utf-8')) if out.exists() else None), stdout, err


# ---------------------------------------------------------------- 1. the old paths are byte-identical to efc6dac

def efc6dac_eval_main():
    """The efc6dac eval_position.py, executed as a module of the same package (its relative imports resolve to the current psdl2)."""
    module = types.ModuleType('psdl2._eval_efc6dac')
    module.__package__ = 'psdl2'
    exec(compile(git_show('src/psdl2/eval_position.py', EVAL_GOLDEN_COMMIT).decode('utf-8'), 'eval_position_efc6dac.py', 'exec'), module.__dict__)
    return module.main


def search_result(query, rows):
    return {'query': {'roi_id': query}, 'status': 'available' if rows else 'unavailable',
            'primary': [{'roi_id': 'c%d' % i, 'product_id': product, 'frame': {'frame_id': frame}, 'center_xy_um': list(center), 'score': 0.9}
                        for i, (product, frame, center) in enumerate(rows, 1)]}


def test_old_inputs_are_byte_identical_to_efc6dac(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('qa', 'B', 'F', (4.0, 4.5)), ('qb', 'B', 'F', (6.0, 6.0)), ('qc', 'C', 'F', (1.0, 1.0))])
    first = write_json(tmp_path / 's1.json', search_result('qa', [('B', 'F', (4.2, 4.5)), ('B', 'F', (9, 9))]))
    second = write_json(tmp_path / 's2.json', search_result('qb', [('B', 'F', (0, 0)), ('B', 'F', (6.1, 6.0))]))
    third = write_json(tmp_path / 's3.json', search_result('qc', []))
    baseline = baseline_file(tmp_path / 'b.json', 'B', {'qa': [('F', (4.0, 4.5))], 'qb': [('F', (7.9, 6.0)), ('F', (6.0, 6.0))], 'zz': [('F', (1, 1))]})
    unlabelled = baseline_file(tmp_path / 'u.json', 'B', {'zz': [('F', (1, 1))]})
    old_main = efc6dac_eval_main()
    cases = [('--search', first, second, third), ('--search', first), ('--search', first, first), ('--baseline', baseline), ('--baseline', unlabelled)]
    for number, case in enumerate(cases):
        seen = {}
        for label, main in (('old', old_main), ('new', eval_main)):
            out = tmp_path / ('%s_%d.json' % (label, number))
            code = main([*map(str, case), '--correspondence', str(truth), '--hit-tolerance-um', '0.5', '--k', '2', '--out', str(out)])
            seen[label] = (code, capsys.readouterr().out, out.read_bytes())
        assert seen['old'] == seen['new'], case
    # the schema of the single mode did not change either
    assert json.loads(seen['new'][2])['schema_version'] == 'psdl2_position_eval_v1' == eval_module.SCHEMA
    # contract errors behave as before (exit code 2, one JSON line on stderr)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'schema_version': 'other'}))
    for label, main in (('old', old_main), ('new', eval_main)):
        code = main(['--baseline', str(bad), '--correspondence', str(truth), '--hit-tolerance-um', '0.5', '--k', '2', '--out', str(tmp_path / 'x.json')])
        err = capsys.readouterr().err
        assert code == 2 and json.loads(err) == {'error': 'unsupported_baseline_result'}, label


# ---------------------------------------------------------------- 2. acceptance numbers (several baseline files)

LABELS = [('qb1', 'B', 'FB', (10.0, 10.0)), ('qb2', 'B', 'FB', (20.0, 20.0)), ('qb3', 'B', 'FB', (30.0, 30.0)), ('qc', 'C', 'FC', (5.0, 5.0))]


def two_product_world(tmp_path):
    truth = truth_file(tmp_path / 'truth.json', LABELS)
    b_file = baseline_file(tmp_path / 'B.json', 'B', {
        'qb1': [('FB', (10.1, 10.0))],                          # hit at rank 1
        'qb2': [('FB', (50.0, 50.0)), ('FB', (20.0, 20.2))],    # hit at rank 2
        'qb3': [('FB', (99.0, 99.0))],                          # miss
        'qc': [('FB', (1.0, 1.0))]})                            # Q: a B candidate for a query whose truth is in C
    c_file = baseline_file(tmp_path / 'C.json', 'C', {'qc': [('FC', (80.0, 80.0))]})   # Q: a different candidate in C, a miss
    return truth, b_file, c_file


def test_acceptance_numbers_per_product_and_summary(tmp_path, capsys):
    truth, b_file, c_file = two_product_world(tmp_path)
    code, result, stdout, err = run_json(capsys, tmp_path / 'multi.json', '--baseline', b_file, c_file, '--correspondence', truth,
                                         '--hit-tolerance-um', TOLERANCE, '--k', 3)
    assert code == 0, err  # the same query in both files is not a conflict: the key is (query, candidate product)
    assert result['schema_version'] == 'psdl2_eval_position_multi_v1' and result['status'] == 'available' and result['k'] == 3
    assert result['arguments']['mode'] == 'multi_baseline' and result['arguments']['baselines'] == [str(b_file), str(c_file)]
    b, c = result['per_product']['B'], result['per_product']['C']
    assert (b['n_hit_at_k'], b['n_queries_with_truth'], b['n_queries_with_prediction'], b['n_top1_hit']) == (2, 3, 3, 1)
    assert (c['n_hit_at_k'], c['n_queries_with_truth'], c['n_queries_with_prediction'], c['n_top1_hit']) == (0, 1, 1, 0)
    assert b['hit_rate_at_3'] == pytest.approx(2 / 3) and c['hit_rate_at_3'] == 0.0
    summary = result['summary']
    assert (summary['n_hit_at_k'], summary['n_queries_with_truth'], summary['n_queries_with_prediction']) == (2, 4, 4)
    assert summary['hit_rate_at_3'] == 0.5 and summary['n_top1_hit'] == 1 and summary['top1_hit_rate'] == 0.25
    assert summary['baseline_missing_for_truth_product'] == []
    rows = {r['query_roi_id']: r for r in result['queries']}
    assert [r['query_roi_id'] for r in result['queries']] == ['qb1', 'qb2', 'qb3', 'qc']
    assert [rows[q]['hit_at_k'] for q in ('qb1', 'qb2', 'qb3', 'qc')] == [True, True, False, False]
    assert [rows[q]['first_hit_rank'] for q in ('qb1', 'qb2', 'qb3', 'qc')] == [1, 2, None, None]
    assert list(rows['qc']['by_product']) == ['C']  # its candidate in the B file is not judged against a C truth
    assert json.loads(stdout)['summary'] == summary
    # the file order does not matter
    code, swapped, _, _ = run_json(capsys, tmp_path / 'swapped.json', '--baseline', c_file, b_file, '--correspondence', truth,
                                   '--hit-tolerance-um', TOLERANCE, '--k', 3)
    assert code == 0 and swapped['summary'] == summary and swapped['per_product'] == result['per_product']


def test_a_missing_truth_product_file_is_a_miss_and_is_listed(tmp_path, capsys):
    truth, b_file, c_file = two_product_world(tmp_path)
    truth = truth_file(tmp_path / 'truth_d.json', [*LABELS, ('qd', 'D', 'FD', (1.0, 1.0))])
    code, result, _, err = run_json(capsys, tmp_path / 'missing.json', '--baseline', b_file, c_file, '--correspondence', truth,
                                    '--hit-tolerance-um', TOLERANCE, '--k', 3)
    assert code == 0, err
    summary = result['summary']
    assert (summary['n_hit_at_k'], summary['n_queries_with_truth'], summary['n_queries_with_prediction']) == (2, 5, 4)
    assert summary['baseline_missing_for_truth_product'] == ['D'] and summary['n_queries_without_prediction'] == 1
    row = next(r for r in result['queries'] if r['query_roi_id'] == 'qd')
    assert row['status'] == 'unavailable' and row['hit_at_k'] is False and row['reasons'] == ['D:baseline_missing_for_truth_product']
    assert result['per_product']['D'] == {'n_queries_with_truth': 1, 'n_queries_with_prediction': 0, 'n_top1_hit': 0, 'n_hit_at_k': 0,
                                           'top1_hit_rate': 0.0, 'hit_rate_at_3': 0.0}
    assert result['per_product']['B']['n_queries_with_truth'] == 3  # the other products are unaffected


def test_a_query_with_truth_in_two_products_is_a_hit_when_either_hits(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'FB', (10.0, 10.0)), ('q', 'C', 'FC', (5.0, 5.0))])
    b_file = baseline_file(tmp_path / 'B.json', 'B', {'q': [('FB', (90.0, 90.0))]})
    c_file = baseline_file(tmp_path / 'C.json', 'C', {'q': [('FC', (5.0, 5.2))]})
    code, result, _, err = run_json(capsys, tmp_path / 'both.json', '--baseline', b_file, c_file, '--correspondence', truth,
                                    '--hit-tolerance-um', TOLERANCE, '--k', 1)
    assert code == 0, err
    assert result['summary']['n_queries_with_truth'] == 1 and result['summary']['n_hit_at_k'] == 1  # one query, counted once
    assert result['per_product']['B']['n_hit_at_k'] == 0 and result['per_product']['C']['n_hit_at_k'] == 1
    assert result['queries'][0]['truth_products'] == ['B', 'C'] and result['queries'][0]['first_hit_rank'] == 1


def test_the_same_query_and_product_in_two_files_is_an_error(tmp_path, capsys):
    truth, b_file, c_file = two_product_world(tmp_path)
    again = baseline_file(tmp_path / 'B2.json', 'B', {'qb1': [('FB', (10.0, 10.0))]})
    out = tmp_path / 'never.json'
    code, stdout, err = run(capsys, '--baseline', b_file, again, '--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 3, '--out', out)
    assert code == 2 and stdout == '' and json.loads(err) == {'error': 'duplicate_candidate_product: qb1 / B'} and not out.exists()
    # two files of the same product with disjoint queries are fine
    disjoint = baseline_file(tmp_path / 'B3.json', 'B', {'qb9': [('FB', (1.0, 1.0))]})
    code, result, _, err = run_json(capsys, tmp_path / 'ok.json', '--baseline', b_file, disjoint, '--correspondence', truth,
                                    '--hit-tolerance-um', TOLERANCE, '--k', 3)
    assert code == 0, err
    assert result['summary']['n_prediction_queries_without_truth'] == 1


def test_a_repeated_query_inside_one_file_follows_the_single_rule(tmp_path, capsys):
    """adv:2: inside one file an identical repeat is ignored and counted, a different one is a conflict; never duplicate_candidate_product."""
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'F', (1.0, 1.0))])
    row = {'rank': 1, 'score': 1.0, 'center_xy_um': [1.0, 1.0], 'product_id': 'B', 'frame_id': 'F'}
    other = dict(row, center_xy_um=[9.0, 9.0])

    def file(name, *candidate_lists):
        results = [{'query': {'roi_id': 'q'}, 'status': 'available', 'primary': [c]} for c in candidate_lists]
        return write_json(tmp_path / name, {'schema_version': 'psdl2_baseline_ncc_v1', 'candidate_product_id': 'B', 'results': results})

    common = ['--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 1]
    same = file('same.json', row, row)
    elsewhere = baseline_file(tmp_path / 'C.json', 'C', {'zz': []})
    code, result, _, err = run_json(capsys, tmp_path / 'inside.json', '--baseline', same, elsewhere, *common)
    assert code == 0, err
    assert result['summary']['n_duplicate_predictions_ignored'] == 1 and result['summary']['n_hit_at_k'] == 1
    conflicting = file('conflict.json', row, other)
    out = tmp_path / 'never.json'
    code, stdout, err = run(capsys, '--baseline', conflicting, elsewhere, *common, '--out', out)
    assert code == 2 and stdout == '' and json.loads(err) == {'error': 'duplicate_query_prediction_conflict: q'} and not out.exists()
    # the same (query, product) in two different files stays duplicate_candidate_product, even when the rows are identical
    code, stdout, err = run(capsys, '--baseline', same, file('twin.json', row), *common, '--out', out)
    assert code == 2 and json.loads(err) == {'error': 'duplicate_candidate_product: q / B'} and not out.exists()


def test_a_repeated_query_in_a_batch_envelope_still_follows_the_same_rule(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'F', (1.0, 1.0))])
    ranking = {'query_roi_id': 'q', 'query_product_id': 'Q', 'query_export_sha256': None,
               'candidates': [{'rank': 1, 'roi_id': 'r1', 'product_id': 'B', 'frame_id': 'F', 'center_xy_um': [1.0, 1.0], 'search_scope': 'primary'}]}
    moved = json.loads(json.dumps(ranking))
    moved['candidates'][0]['center_xy_um'] = [9.0, 9.0]
    common = ['--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 1]
    same = write_json(tmp_path / 'same.json', {'schema_version': 'psdl2_search_batch_v1', 'rankings': [ranking, ranking]})
    code, result, _, err = run_json(capsys, tmp_path / 'a.json', '--search-batch', same, *common)
    assert code == 0, err
    assert result['summary']['n_duplicate_predictions_ignored'] == 1
    different = write_json(tmp_path / 'different.json', {'schema_version': 'psdl2_search_batch_v1', 'rankings': [ranking, moved]})
    code, stdout, err = run(capsys, '--search-batch', different, *common, '--out', tmp_path / 'never.json')
    assert code == 2 and json.loads(err) == {'error': 'duplicate_query_prediction_conflict: q'}


def test_a_file_whose_candidates_belong_to_another_product_is_rejected(tmp_path, capsys):
    """adv:8: the product of a file is its candidate_product_id; a candidate row of another product would be judged under the wrong label."""
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'F', (1.0, 1.0))])
    good = baseline_file(tmp_path / 'B.json', 'B', {'q': [('F', (1.0, 1.0))]})
    wrong = baseline_file(tmp_path / 'C.json', 'C', {'q': [('F', (1.0, 1.0))]})
    payload = json.loads(wrong.read_text())
    payload['candidate_product_id'] = 'D'   # the file says D, its candidate row says C
    write_json(wrong, payload)
    out = tmp_path / 'never.json'
    code, stdout, err = run(capsys, '--baseline', good, wrong, '--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 1, '--out', out)
    assert code == 2 and stdout == '' and json.loads(err) == {'error': 'candidate_product_mismatch: q / D'} and not out.exists()


# ---------------------------------------------------------------- regression fixtures

def test_empty_candidates_a_missing_prediction_and_a_dropped_query(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('empty', 'B', 'F', (1.0, 1.0)), ('absent', 'B', 'F', (2.0, 2.0)), ('good', 'B', 'F', (3.0, 3.0))])
    b_file = baseline_file(tmp_path / 'B.json', 'B', {'empty': [], 'good': [('F', (3.0, 3.0))]})
    c_file = baseline_file(tmp_path / 'C.json', 'C', {'good': [('F', (3.0, 3.0))]})
    code, result, _, err = run_json(capsys, tmp_path / 'r.json', '--baseline', b_file, c_file, '--correspondence', truth,
                                    '--hit-tolerance-um', TOLERANCE, '--k', 2)
    assert code == 0, err
    rows = {r['query_roi_id']: r for r in result['queries']}
    assert rows['empty']['reasons'] == ['B:no_candidates'] and rows['absent']['reasons'] == ['B:no_prediction']
    assert rows['empty']['hit_at_k'] is False and rows['absent']['hit_at_k'] is False and rows['good']['hit_at_k'] is True
    summary = result['summary']
    assert (summary['n_queries_with_truth'], summary['n_queries_with_prediction'], summary['n_hit_at_k']) == (3, 1, 1)
    assert summary['hit_rate_at_2'] == pytest.approx(1 / 3) and summary['hit_rate_at_2_among_predicted'] == 1.0


def test_exit_3_when_no_query_can_be_evaluated(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'F', (1.0, 1.0))])
    b_file = baseline_file(tmp_path / 'B.json', 'B', {'q': []})
    c_file = baseline_file(tmp_path / 'C.json', 'C', {'q': []})
    code, result, stdout, err = run_json(capsys, tmp_path / 'r.json', '--baseline', b_file, c_file, '--correspondence', truth,
                                         '--hit-tolerance-um', TOLERANCE, '--k', 2)
    assert code == 3 and result['status'] == 'unavailable' and result['reasons'] == ['no_evaluable_queries']
    assert result['summary']['hit_rate_at_2'] == 0.0 and result['summary']['hit_rate_at_2_among_predicted'] is None
    assert json.loads(stdout)['status'] == 'unavailable'
    nothing = truth_file(tmp_path / 'nothing.json', [(None, 'B', 'F', (1.0, 1.0))])
    code, result, _, _ = run_json(capsys, tmp_path / 'r2.json', '--baseline', b_file, c_file, '--correspondence', nothing,
                                  '--hit-tolerance-um', TOLERANCE, '--k', 2)
    assert code == 3 and result['reasons'] == ['no_queries_with_truth'] and result['summary']['hit_rate_at_2'] is None


def test_supplementary_candidates_count_only_when_asked(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'F', (10.0, 10.0))])
    envelope = batch_file(tmp_path / 'batch.json', {'q': [('primary', 'B', 'F', (90.0, 90.0)), ('supplementary', 'B', 'F', (10.0, 10.2))]})
    base = ['--search-batch', envelope, '--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 5]
    code, default, _, err = run_json(capsys, tmp_path / 'default.json', *base)
    assert code == 0, err
    assert default['summary']['n_hit_at_k'] == 0 and default['arguments']['include_supplementary'] is False
    assert default['queries'][0]['by_product']['B']['n_candidates'] == 1
    code, wide, _, err = run_json(capsys, tmp_path / 'wide.json', *base, '--include-supplementary')
    assert code == 0, err
    assert wide['summary']['n_hit_at_k'] == 1 and wide['arguments']['include_supplementary'] is True
    assert wide['queries'][0]['first_hit_rank'] == 2 and wide['arguments']['mode'] == 'search_batch'
    # --k cuts the whole ranking, not one product's share of it
    code, narrow, _, _ = run_json(capsys, tmp_path / 'narrow.json', *base[:-1], 1, '--include-supplementary')
    assert narrow['summary']['n_hit_at_k'] == 0


def test_batch_ranking_is_judged_whole_for_each_truth_product(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('q1', 'B', 'FB', (10.0, 10.0)), ('q2', 'C', 'FC', (5.0, 5.0)), ('q3', 'C', 'FC', (7.0, 7.0))])
    envelope = batch_file(tmp_path / 'batch.json', {
        'q1': [('primary', 'C', 'FC', (5.0, 5.0)), ('primary', 'B', 'FB', (10.0, 10.0))],   # C first, B second
        'q2': [('primary', 'C', 'FC', (5.1, 5.0))],
        'q3': []})
    code, result, _, err = run_json(capsys, tmp_path / 'r.json', '--search-batch', envelope, '--correspondence', truth,
                                    '--hit-tolerance-um', TOLERANCE, '--k', 2)
    assert code == 0, err
    assert (result['per_product']['B']['n_hit_at_k'], result['per_product']['B']['n_top1_hit']) == (1, 0)  # q1: its B truth is hit at rank 2
    assert (result['per_product']['C']['n_hit_at_k'], result['per_product']['C']['n_queries_with_prediction']) == (1, 1)
    assert result['summary']['n_hit_at_k'] == 2 and result['summary']['n_queries_with_prediction'] == 2 and result['summary']['n_top1_hit'] == 1
    assert {r['query_roi_id']: r['reasons'] for r in result['queries'] if r['status'] == 'unavailable'} == {'q3': ['C:no_candidates']}


def test_contract_errors_exit_2_and_leave_no_output(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'F', (1.0, 1.0))])
    envelope = batch_file(tmp_path / 'batch.json', {'q': [('primary', 'B', 'F', (1.0, 1.0))]})
    baseline = baseline_file(tmp_path / 'B.json', 'B', {'q': [('F', (1.0, 1.0))]})
    out = tmp_path / 'out.json'
    common = ['--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--out', out]
    wrong = write_json(tmp_path / 'wrong.json', {'schema_version': 'psdl2_search_batch_v0', 'rankings': []})
    scope = write_json(tmp_path / 'scope.json', {'schema_version': 'psdl2_search_batch_v1', 'rankings': [
        {'query_roi_id': 'q', 'candidates': [{'rank': 1, 'roi_id': 'r', 'product_id': 'B', 'frame_id': 'F', 'center_xy_um': [1, 1], 'search_scope': 'other'}]}]})
    nameless = write_json(tmp_path / 'nameless.json', {'schema_version': 'psdl2_baseline_ncc_v1', 'results': []})
    for argv, error in [(['--search-batch', wrong, '--k', 1], 'unsupported_search_batch'),
                        (['--search-batch', scope, '--k', 1], 'unsupported_search_scope'),
                        (['--search-batch', envelope, '--k', 0], 'positive_k_required'),
                        (['--baseline', baseline, nameless, '--k', 1], 'nonempty_identifier_required'),
                        (['--baseline', baseline, baseline, '--include-supplementary', '--k', 1], 'include_supplementary_requires_search_batch'),
                        (['--baseline', baseline, '--include-supplementary', '--k', 1], 'include_supplementary_requires_search_batch')]:
        code, stdout, err = run(capsys, *argv, *common)
        assert code == 2 and stdout == '' and json.loads(err) == {'error': error}, (argv, err)
        assert not out.exists()
    # --search-batch excludes --search and --baseline (argparse itself exits with 2)
    code, _, err = run(capsys, '--search-batch', envelope, '--baseline', baseline, *common)
    assert code == 2 and 'not allowed with' in err


def test_multi_output_is_written_atomically(tmp_path, capsys, monkeypatch):
    truth, b_file, c_file = two_product_world(tmp_path)
    out = tmp_path / 'sub' / 'out.json'
    out.parent.mkdir()
    out.write_text('OLD')

    def interrupted(*_):
        raise OSError('interrupted before the rename')

    monkeypatch.setattr(eval_module.os, 'replace', interrupted)
    code, _, err = run(capsys, '--baseline', b_file, c_file, '--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 3, '--out', out)
    assert code == 2 and 'interrupted' in err
    assert out.read_text() == 'OLD' and [p.name for p in out.parent.iterdir()] == ['out.json']
    monkeypatch.undo()
    code, _, _ = run(capsys, '--baseline', b_file, c_file, '--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 3, '--out', out)
    assert code == 0 and json.loads(out.read_text())['schema_version'] == 'psdl2_eval_position_multi_v1'
    assert [p.name for p in out.parent.iterdir()] == ['out.json']


# ---------------------------------------------------------------- 3. the real envelope of psdl2.search --all-ready

def locations(result):
    return [(row['product_id'], row['frame']['frame_id'], tuple(row['center_xy_um'])) for row in result['primary']]


def test_real_search_batch_envelope_is_evaluated_to_the_fixture_truth(world, capsys, tmp_path):  # noqa: F811
    flags = ('--exclude-query-product', '--per-product-topk', '2')
    out = world.out / 'for_eval.json'
    code, _, err = call(search_main, capsys, *search_args(world, topk=4), '--all-ready', '--out', out, *flags)
    assert code == 0, err
    envelope = json.loads(out.read_text())
    where = {r['query']['roi_id']: locations(r) for r in envelope['results']}
    assert all(len(rows) >= 3 for rows in where.values())
    far = (where['P2_case0'][0][0], where['P2_case0'][0][1], (where['P2_case0'][0][2][0] + 500.0, where['P2_case0'][0][2][1]))
    labels = [('P0_case0', *where['P0_case0'][0]),      # truth at the rank-1 candidate: hit at rank 1
              ('P1_case0', *where['P1_case0'][2]),      # truth at the rank-3 candidate: hit only from k=3
              ('P2_case0', *far)]                       # truth 500 um away from every candidate: miss
    assert len({where['P1_case0'][0][:2] + where['P1_case0'][0][2], where['P1_case0'][2][:2] + where['P1_case0'][2][2]}) == 2
    truth = truth_file(tmp_path / 'truth.json', labels)
    expected = {1: (1, 1), 2: (1, 1), 3: (2, 1), 4: (2, 1)}   # k -> (queries hit within k, top-1 hits)
    for k, (hits, top1) in expected.items():
        code, result, _, err = run_json(capsys, tmp_path / ('k%d.json' % k), '--search-batch', out, '--correspondence', truth,
                                        '--hit-tolerance-um', 0.01, '--k', k)
        assert code == 0, err
        summary = result['summary']
        assert (summary['n_queries_with_truth'], summary['n_queries_with_prediction'], summary['n_hit_at_k'], summary['n_top1_hit']) == (3, 3, hits, top1), k
        assert [r['first_hit_rank'] for r in result['queries']] == [1, 3 if k >= 3 else None, None]
    # supplementary candidates are absent without --expand, so the flag changes nothing here
    code, same, _, _ = run_json(capsys, tmp_path / 'same.json', '--search-batch', out, '--correspondence', truth, '--hit-tolerance-um', 0.01,
                                '--k', 4, '--include-supplementary')
    assert same['summary'] == result['summary'] and same['arguments']['include_supplementary'] is True


@pytest.mark.parametrize('k', [1, 2, 4])
def test_one_result_and_its_one_query_envelope_give_the_same_metrics(world, capsys, tmp_path, k):  # noqa: F811
    """The old single-result evaluation and the batch evaluation of the same search agree on every figure over the predicted queries."""
    manifest = json.loads((world.root / 'query.json').read_text())
    manifest['sources'] = [s for s in manifest['sources'] if 'templates/P1/' in s['export']]
    assert len(manifest['sources']) == 1
    (world.root / 'query_p1.json').write_text(json.dumps(manifest))   # a dataset of one source: P1_case0 is its only query
    common = ['--query-dataset', world.root / 'query_p1.json', '--checkpoint', world.root / 'out/model.pt', '--index', world.root / 'out/index.json',
              '--cohort', world.root / 'cohort.json', '--topk', 4, '--exclude-query-product']
    one, envelope = world.out / ('one_%d.json' % k), world.out / ('one_envelope_%d.json' % k)
    assert call(search_main, capsys, '--roi-id', 'P1_case0', *common, '--out', one)[0] == 0
    assert call(search_main, capsys, '--all-ready', *common, '--out', envelope)[0] == 0
    assert [r['query_roi_id'] for r in json.loads(envelope.read_text())['rankings']] == ['P1_case0']
    result = json.loads(one.read_text())
    truth = truth_file(tmp_path / 'truth.json', [('P1_case0', *locations(result)[1])])   # the rank-2 candidate
    old_code, old, _, err = run_json(capsys, tmp_path / 'old.json', '--search', one, '--correspondence', truth, '--hit-tolerance-um', 0.01, '--k', k)
    new_code, new, _, err2 = run_json(capsys, tmp_path / 'new.json', '--search-batch', envelope, '--correspondence', truth,
                                      '--hit-tolerance-um', 0.01, '--k', k)
    assert old_code == new_code == 0, (err, err2)
    old_summary, new_summary = old['summary'], new['summary']
    assert new_summary['n_queries_with_prediction'] == old_summary['n_evaluable'] == 1
    assert new_summary['top1_hit_rate_among_predicted'] == old_summary['top1_hit_rate'] == 0.0
    assert new_summary['hit_rate_at_%d_among_predicted' % k] == old_summary['hit_rate_at_%d' % k] == (1.0 if k >= 2 else 0.0)
    assert new_summary['mean_top1_distance_um'] == old_summary['mean_top1_distance_um']
    assert new_summary['median_top1_distance_um'] == old_summary['median_top1_distance_um']
    assert new['queries'][0]['first_hit_rank'] == old['queries'][0]['first_hit_rank']
    assert [c['distance_um'] for c in new['queries'][0]['by_product'][result['primary'][1]['product_id']]['candidates']] == \
        [c['distance_um'] for c in old['queries'][0]['candidates']]


def test_a_real_baseline_payload_joins_the_multi_mode(tmp_path, capsys):
    """A psdl2.baseline_ncc result (not a hand-made one) carries candidate_product_id and is accepted as one product's file."""
    from copy import deepcopy
    from psdl2.baseline_ncc import run_baseline
    from psdl2.data import load_export
    from psdl2.eval_position import read_correspondence
    from test_position_baseline import correspondence, masks_a, shifted, write_export
    e1, e2 = masks_a()
    query = write_export(tmp_path / 'A', 'A', [('qa', 'A1', (2.5, 2.5), (e1, e2))])
    index = write_export(tmp_path / 'B', 'B', [('cb', 'B1', (4.0, 4.5), (shifted(e1, 4, 3), shifted(e2, 4, 3)))])
    truth = correspondence(tmp_path / 'truth.json', [('qa', (2.5, 2.5), (4.0, 4.5), 'ready')])   # side b is product B, frame F
    payload = run_baseline(load_export(query), load_export(index), read_correspondence(truth), window_um=2.0, topk=2)
    assert payload['candidate_product_id'] == 'B'
    b_file = write_json(tmp_path / 'B.json', payload)
    other = deepcopy(payload)
    other['candidate_product_id'] = 'C'
    for row in other['results']:
        for candidate in row['primary']:
            candidate['product_id'] = 'C'
    c_file = write_json(tmp_path / 'C.json', other)
    code, result, _, err = run_json(capsys, tmp_path / 'r.json', '--baseline', b_file, c_file, '--correspondence', truth,
                                    '--hit-tolerance-um', 0.5, '--k', 2)
    assert code == 0, err
    assert result['summary']['n_hit_at_k'] == 1 and result['summary']['n_top1_hit'] == 1 and list(result['per_product']) == ['B']


# ---------------------------------------------------------------- adv:7 the two judges agree; frame mismatch in the new modes

def test_judge_and_evaluate_positions_agree_on_random_candidates():
    """`_judge` (multi mode) must decide hits exactly like `evaluate_positions` (single mode): seeded random cases, no I/O."""
    import random
    rng = random.Random(20261026)
    products, frames = ('B', 'C'), ('F1', 'F2')
    for case in range(40):
        labelled = [{'pair_id': 'T%d' % i, 'product_id': rng.choice(products), 'frame_id': rng.choice(frames),
                     'center_xy_um': [rng.uniform(0, 10), rng.uniform(0, 10)], 'roi_id': None} for i in range(rng.randint(1, 3))]
        candidates = [{'rank': rank, 'roi_id': 'c%d' % rank, 'product_id': rng.choice(products), 'frame_id': rng.choice(frames),
                       'center_xy_um': [rng.uniform(0, 10), rng.uniform(0, 10)], 'score': rng.random()} for rank in range(1, rng.randint(2, 7))]
        k, tolerance = rng.randint(1, 5), rng.choice((0.5, 2.0, 5.0))
        result = eval_module.evaluate_positions([{'query_roi_id': 'q', 'status': 'available', 'candidates': candidates, 'source': 's'}],
                                                {'q': labelled}, tolerance_um=tolerance, k=k)
        row = result['queries'][0]
        rows, first_hit = eval_module._judge(candidates, labelled, tolerance, k)
        assert [{key: v for key, v in r.items() if key != 'product_id'} for r in rows] == row['candidates'], case
        assert first_hit == row['first_hit_rank'], case


def test_a_candidate_in_the_right_place_but_another_frame_is_no_hit_in_the_new_modes(tmp_path, capsys):
    truth = truth_file(tmp_path / 'truth.json', [('q', 'B', 'F', (1.0, 1.0))])
    common = ['--correspondence', truth, '--hit-tolerance-um', TOLERANCE, '--k', 2]
    single = baseline_file(tmp_path / 'B.json', 'B', {'q': [('OTHER', (1.0, 1.0))]})
    other = baseline_file(tmp_path / 'C.json', 'C', {'zz': []})
    envelope = batch_file(tmp_path / 'batch.json', {'q': [('primary', 'B', 'OTHER', (1.0, 1.0))]})
    for mode in (['--baseline', single, other], ['--search-batch', envelope]):
        code, result, _, err = run_json(capsys, tmp_path / 'out.json', *mode, *common)
        assert code == 0, err
        entry = result['queries'][0]['by_product']['B']
        assert entry['top1_hit'] is False and entry['hit_at_k'] is False and entry['top1_distance_um'] is None
        assert entry['candidates'][0]['reason'] == 'product_or_frame_mismatch' and result['summary']['n_hit_at_k'] == 0
