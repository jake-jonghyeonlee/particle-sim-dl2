"""Batch mode of the ZNCC baseline: shared candidate FFTs, template dedup, ambiguity flags, review sheet, self-check."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import psdl2.baseline_ncc as bn
import test_position_baseline as tp
from psdl2.baseline_ncc import run_baseline, run_baseline_batch, self_check_failures
from psdl2.data import load_export
from psdl2.eval_position import read_correspondence
from psdl2.schema import ContractError

N = 40  # 20 um × 20 um at 0.5 um/px
SRC = str(Path(__file__).resolve().parents[1] / 'src')


def random_masks(seed, tile=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        mask = (rng.random((tile or N, tile or N)) < 0.35).astype(np.uint8)
        out.append(np.tile(mask, (N // tile, N // tile)) if tile else mask)
    return out


@pytest.fixture(autouse=True)
def big_frame(monkeypatch):
    monkeypatch.setattr(tp, 'SHAPE', (N, N))


def pair_of_exports(tmp_path, query_rows, *, shift=(6, 4), index_masks=None, query_masks=None):
    e1, e2 = query_masks or random_masks(1)
    rows = [(rid, f'A{i}', center, (e1, e2)) for i, (rid, center) in enumerate(query_rows)]
    query = tp.write_export(tmp_path / 'A', 'A', rows)
    f1, f2 = index_masks or (tp.shifted(e1, *shift), tp.shifted(e2, *shift))
    index = tp.write_export(tmp_path / 'B', 'B', [('cb', 'B1', (10.0, 10.0), (f1, f2))])
    return query, index


@pytest.fixture
def batch_products(tmp_path):
    rows = [('q1', (8.0, 8.0)), ('q2', (8.0, 8.0)), ('q3', (6.0, 12.0)), ('q4', (12.0, 7.0)), ('q5', (10.0, 14.0))]
    query, index = pair_of_exports(tmp_path, rows)
    return query, index, tmp_path


def strip_cache(result):
    return {k: v for k, v in result.items() if k != 'candidate_fft_cache'}


def test_batch_row_equals_old_label_run(batch_products):
    query_path, index_path, tmp_path = batch_products
    query, index = load_export(query_path), load_export(index_path)
    truth = tp.correspondence(tmp_path / 'c.json', [('q3', (6.0, 12.0), (8.0, 15.0), 'ready'), ('q1', (8.0, 8.0), (10.0, 11.0), 'ready')])
    old = run_baseline(query, index, read_correspondence(truth), window_um=4.0, topk=3)
    new = run_baseline_batch(query, index, window_um=4.0, topk=3)
    by_id = {r['query']['roi_id']: r for r in new['results']}
    assert len(new['results']) == 5 and new['n_queries'] == 5
    for row in old['results']:
        got = by_id[row['query']['roi_id']]
        assert got['primary'] == row['primary'] and got['endpoints_used'] == row['endpoints_used'] and got['query'] == row['query']
    top = by_id['q3']['primary'][0]
    assert top['center_xy_um'] == pytest.approx([8.0, 15.0]) and top['score'] == pytest.approx(1.0)
    # label mode keeps its keys: no batch extras
    assert 'mode' not in old and 'flags' not in old['results'][0] and 'candidate_fft_cache' not in old


def test_shared_fft_equals_per_query_recomputation(batch_products):
    query_path, index_path, _ = batch_products
    query, index = load_export(query_path), load_export(index_path)
    shared = run_baseline_batch(query, index, window_um=4.0, topk=4, share_fft=True, dedupe=False)
    plain = run_baseline_batch(query, index, window_um=4.0, topk=4, share_fft=False, dedupe=False)
    assert strip_cache(shared) == strip_cache(plain) and plain['candidate_fft_cache'] is None
    # 5 queries × 2 endpoints: terms are built once per (role, layer) and template shape
    assert shared['candidate_fft_cache'] == {'keys': 2, 'builds': 2, 'hits': 8}
    image = np.random.default_rng(3).random((30, 31)); template = (image[4:12, 5:15] > 0.5).astype(float)
    cache = bn.CandidateFFTCache()
    assert np.array_equal(bn.zncc_map(image, template), bn.zncc_map(image, template, cache, 'k'))
    assert np.array_equal(bn.zncc_map(image, template), bn.zncc_map(image, template, cache, 'k')) and cache.stats()['hits'] == 1
    assert np.allclose(bn.zncc_map(image, template), bn.zncc_map(image, template, bn.CandidateFFTCache(), 'k'))


def test_identical_templates_are_matched_once(batch_products, monkeypatch):
    query_path, index_path, _ = batch_products
    query, index = load_export(query_path), load_export(index_path)
    calls = []
    real = bn.zncc_map
    monkeypatch.setattr(bn, 'zncc_map', lambda *a, **k: calls.append(1) or real(*a, **k))
    deduped = run_baseline_batch(query, index, window_um=4.0, topk=2)
    assert len(calls) == 4 * 2 and deduped['n_template_matches'] == 4  # q1 and q2 share one match
    rows = {r['query']['roi_id']: r for r in deduped['results']}
    assert rows['q1']['shared_with'] == ['q2'] and rows['q2']['shared_with'] == ['q1'] and rows['q3']['shared_with'] == []
    assert rows['q1']['template_sha256'] == rows['q2']['template_sha256'] != rows['q3']['template_sha256']
    assert rows['q1']['primary'] == rows['q2']['primary'] and rows['q1']['query']['roi_id'] != rows['q2']['query']['roi_id']
    calls.clear()
    plain = run_baseline_batch(query, index, window_um=4.0, topk=2, dedupe=False)
    assert len(calls) == 5 * 2 and plain['n_template_matches'] == 5
    assert [r['primary'] for r in plain['results']] == [r['primary'] for r in deduped['results']]


def test_flags_gap_clip_and_structureless(tmp_path):
    e1, e2 = random_masks(1)
    e1[:, :6] = 0; e2[:, :6] = 0
    query, index = pair_of_exports(tmp_path, [('empty', (1.0, 10.0)), ('edge', (0.5, 14.0)), ('good', (10.0, 10.0))],
                                   query_masks=(e1, e2), index_masks=(tp.shifted(e1, 2, 2), tp.shifted(e2, 2, 2)))
    result = run_baseline_batch(load_export(query), load_export(index), window_um=4.0, topk=3)
    rows = {r['query']['roi_id']: r for r in result['results']}
    assert rows['empty']['flags']['low_structure'] is True and rows['empty']['status'] == 'unavailable'
    assert rows['empty']['flags']['score_gap'] is None
    assert rows['edge']['flags']['window_clipped'] is True and rows['good']['flags']['window_clipped'] is False
    gap = rows['good']['flags']['score_gap']
    assert gap == pytest.approx(rows['good']['primary'][0]['score'] - rows['good']['primary'][1]['score']) and gap > 0
    assert 'near_equal_peaks' not in rows['good']['flags'] and 'ambiguity_margin' not in result
    with pytest.raises(ContractError, match='positive_topk_required'):
        run_baseline_batch(load_export(query), load_export(index), window_um=4.0, topk=0)
    with pytest.raises(ContractError, match='nonnegative'):
        run_baseline_batch(load_export(query), load_export(index), window_um=4.0, topk=2, ambiguity_margin=-1.0)


def test_periodic_repeats_show_near_equal_peaks_without_selection(tmp_path):
    masks = random_masks(5, tile=N // 2)  # 2×2 repetition of one unit
    query, index = pair_of_exports(tmp_path, [('qp', (5.0, 5.0))], query_masks=masks, index_masks=masks)
    q, i = load_export(query), load_export(index)
    result = run_baseline_batch(q, i, window_um=4.0, topk=4, ambiguity_margin=0.01)
    row = result['results'][0]
    assert row['flags']['near_equal_peaks'] is True and row['flags']['score_gap'] < 0.01
    scores = [p['score'] for p in row['primary']]
    assert all(s == pytest.approx(1.0) for s in scores)
    assert sorted(tuple(p['center_xy_um']) for p in row['primary']) == [(5.0, 5.0), (5.0, 15.0), (15.0, 5.0), (15.0, 15.0)]
    assert result['ambiguity_margin'] == 0.01 and not any(key.startswith('selected') or key == 'match' for key in row)
    assert 'near_equal_peaks' not in run_baseline_batch(q, i, window_um=4.0, topk=4)['results'][0]['flags']


def test_self_check_passes_and_fails_on_shifted_centre(batch_products, tmp_path):
    query_path, _, _ = batch_products
    out = tmp_path / 'self.json'
    run = subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', '--query-export', query_path, '--self-check',
                          '--window-um', '4', '--topk', '2', '--out', out], capture_output=True, text=True, env={'PYTHONPATH': SRC})
    assert run.returncode == 0, run.stderr
    result = json.loads(out.read_text())
    assert result['self_check'] == {'passed': True} and self_check_failures(result, pixel_pitch_um=0.5) == []
    bad = deepcopy(result)
    bad['results'][2]['query']['center_xy_um'][0] += 1.0  # two pixel pitches off its own match
    failures = self_check_failures(bad, pixel_pitch_um=0.5)
    assert len(failures) == 1 and bad['results'][2]['query']['roi_id'] in failures[0]
    low = deepcopy(result); low['results'][0]['primary'][0]['score'] = 0.9
    assert 'score' in self_check_failures(low, pixel_pitch_um=0.5)[0]
    assert self_check_failures({'results': []}, pixel_pitch_um=0.5) == ['no available query to check']
    both = subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', '--query-export', query_path, '--self-check', '--index-export', query_path,
                           '--window-um', '4', '--topk', '2', '--out', out], capture_output=True, text=True, env={'PYTHONPATH': SRC})
    assert both.returncode == 2 and 'self_check' in both.stderr


def test_cli_batch_sheet_hashes_and_deterministic_rerun(tmp_path):
    e1, e2 = random_masks(1)
    noisy1, noisy2 = tp.shifted(e1, 6, 4), tp.shifted(e2, 6, 4)
    rng = np.random.default_rng(9)
    noisy1[:, 24:] = rng.integers(0, 2, noisy1[:, 24:].shape); noisy2[:, 24:] = rng.integers(0, 2, noisy2[:, 24:].shape)
    query, index = pair_of_exports(tmp_path, [('roi_hi', (8.0, 8.0)), ('roi_lo', (14.0, 10.0)), ('roi_<b>&x', (6.0, 14.0))],
                                   index_masks=(noisy1, noisy2))
    def run(out, sheet):
        proc = subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', '--query-export', query, '--index-export', index, '--all-ready', '--max-memory-mb', '500',
                               '--window-um', '4', '--topk', '3', '--ambiguity-margin', '0.05', '--out', out, '--sheet-dir', sheet],
                              capture_output=True, text=True, env={'PYTHONPATH': SRC})
        assert proc.returncode == 0, proc.stderr
    run(tmp_path / 'a.json', tmp_path / 'sheet_a'); run(tmp_path / 'b.json', tmp_path / 'sheet_b')
    assert (tmp_path / 'a.json').read_bytes() == (tmp_path / 'b.json').read_bytes()
    assert (tmp_path / 'sheet_a/sheet.html').read_bytes() == (tmp_path / 'sheet_b/sheet.html').read_bytes()
    result = json.loads((tmp_path / 'a.json').read_text())
    assert result['inputs'] == {'query_export_sha256': hashlib.sha256(Path(query).read_bytes()).hexdigest(),
                                'index_export_sha256': hashlib.sha256(Path(index).read_bytes()).hexdigest()}
    assert result['arguments'] == {'window_um': 4.0, 'topk': 3, 'ambiguity_margin': 0.05, 'max_memory_mb': 500.0, 'self_check': False}
    assert [r['query']['roi_id'] for r in result['results']] == sorted(r['query']['roi_id'] for r in result['results'])
    page = (tmp_path / 'sheet_a/sheet.html').read_text(encoding='utf-8')
    assert 'http://' not in page and 'https://' not in page and '<script' not in page and 'lang="ko"' in page
    assert '<b>' not in page and 'roi_&lt;b&gt;&amp;x' in page
    top = {r['query']['roi_id']: r['primary'][0]['score'] for r in result['results']}
    assert len(set(round(v, 6) for v in top.values())) > 1
    order = sorted(page.index(f'질의 {html_id}') for html_id in ('roi_hi', 'roi_lo', 'roi_&lt;b&gt;&amp;x'))
    expected = sorted(top, key=lambda k: (-top[k], k))
    positions = {k: page.index('질의 ' + k.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')) for k in top}
    assert sorted(positions, key=positions.get) == expected and order
    assert len(list((tmp_path / 'sheet_a').glob('q*_query.png'))) == 3 and len(list((tmp_path / 'sheet_a').glob('q*_m0*.png'))) == 9


# ---- memory bound (T2b) ----

def synthetic_exports(n_layers, n_queries, side=400, pitch=0.1):
    """Fake exports (only the attributes baseline_ncc reads): n_layers role/layer masks of side×side px; query j
    binds three layers so most template groups span several layers."""
    from types import SimpleNamespace
    rng = np.random.default_rng(7)
    frame = SimpleNamespace(frame_id='F', origin_xy_um=(0.0, 0.0), pixel_pitch_um=pitch, shape_yx=(side, side))
    masks = [(rng.random((side, side)) < 0.3).astype(np.uint8) for _ in range(n_layers)]

    def roi(roi_id, center, layers):
        return SimpleNamespace(row={'roi_id': roi_id, 'product_id': 'P'}, frame=frame, center_xy_um=center,
                               channels={f'e{k}': masks[k] for k in layers},
                               channel_bindings={f'e{k}': {'role': 'r', 'actual_layer': f'L{k}'} for k in layers})
    queries = [roi(f'q{j:02d}', (6.0 + 1.3 * j, 7.0 + 0.9 * j), sorted({j % n_layers, (j + 1) % n_layers, (j + 3) % n_layers}))
               for j in range(n_queries)]
    return SimpleNamespace(ready=queries), SimpleNamespace(ready=[roi('idx', (20.0, 20.0), range(n_layers))])


def peak_traced(**kwargs):
    import tracemalloc
    query, index = synthetic_exports(kwargs.pop('n_layers'), kwargs.pop('n_queries'))
    tracemalloc.start()
    try:
        result = bn._run_batch(query, index, window_um=4.0, topk=3, **kwargs)[0]
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_peak_memory_estimate_formula():
    C, R, V = bn._fft_terms_bytes((100, 120), (10, 8))
    assert (C, R, V) == (109 * (127 // 2 + 1) * 16, 109 * 127 * 8, 91 * 113 * 8)
    unit = (('r', 'L0'), (10, 8))
    one = bn.estimate_batch_memory({unit: ((100, 120), (10, 8))}, [[unit], [unit]], sheet_bytes=50)
    assert one == {'memory_estimate_bytes': 4 * C + 4 * R + 6 * V + 50, 'largest_layer_fft_bytes': 2 * C}  # single-layer groups: no accumulator
    other = (('r', 'L1'), (10, 8))
    two = bn.estimate_batch_memory({unit: ((100, 120), (10, 8)), other: ((100, 120), (10, 8))}, [[unit, other]] * 3)
    assert two['memory_estimate_bytes'] == 4 * C + 4 * R + 6 * V + 3 * V  # three groups hold a running sum across both layers
    assert bn.estimate_batch_memory({}, []) == {'memory_estimate_bytes': 0, 'largest_layer_fft_bytes': 0}


def test_layer_by_layer_equals_all_at_once_and_reports_estimate():
    query, index = synthetic_exports(6, 8, side=200)
    new = bn._run_batch(query, index, window_um=2.0, topk=4)[0]
    old = bn._run_batch(query, index, window_um=2.0, topk=4, all_at_once=True)[0]
    assert [r['query']['roi_id'] for r in new['results']] == sorted(r['query']['roi_id'] for r in new['results'])
    for a, b in zip(new['results'], old['results']):
        assert [p['template_top_left_px_yx'] for p in a['primary']] == [p['template_top_left_px_yx'] for p in b['primary']]
        assert [p['score'] for p in a['primary']] == pytest.approx([p['score'] for p in b['primary']], abs=1e-12)
        assert {k: v for k, v in a.items() if k != 'primary'} == {k: v for k, v in b.items() if k != 'primary'}
    assert new['n_template_matches'] == old['n_template_matches'] == 8 and new['status'] == 'available'
    assert new['memory_estimate_bytes'] > new['largest_layer_fft_bytes'] > 0
    assert new['candidate_fft_cache']['builds'] == 6 and new['candidate_fft_cache']['hits'] == 8 * 3 - 6
    assert json.dumps(new) == json.dumps(bn._run_batch(query, index, window_um=2.0, topk=4)[0])


def test_peak_memory_with_six_layers_is_not_six_times_one_layer():
    one, peak_one = peak_traced(n_layers=1, n_queries=6)
    result, peak_six = peak_traced(n_layers=6, n_queries=20)
    _, peak_six_old = peak_traced(n_layers=6, n_queries=20, all_at_once=True)
    assert peak_six < 4 * peak_one and peak_six < 0.85 * peak_six_old  # 6 layers: ~3x, the rest is the running sums
    assert peak_six <= result['memory_estimate_bytes'] and peak_one <= one['memory_estimate_bytes']


def test_cli_max_memory_mb_required_and_enforced(batch_products, tmp_path):
    query_path, index_path, _ = batch_products
    def run(*extra):
        out = tmp_path / 'm.json'
        out.unlink(missing_ok=True)
        proc = subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', '--query-export', query_path, '--index-export', index_path,
                               '--all-ready', '--window-um', '4', '--topk', '2', '--out', out, *extra], capture_output=True, text=True,
                              env={'PYTHONPATH': SRC})
        return proc, out
    proc, out = run()
    assert proc.returncode == 2 and 'max_memory_mb_required_with_all_ready' in proc.stderr and not out.exists()
    proc, out = run('--max-memory-mb', '0.001')
    assert proc.returncode == 2 and 'memory_estimate_exceeds_limit' in proc.stderr and 'even for one query' in proc.stderr and not out.exists()
    proc, out = run('--max-memory-mb', '100')
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert 0 < result['largest_layer_fft_bytes'] < result['memory_estimate_bytes'] < 100 * 2 ** 20


# ---- query chunking (T2c) ----

CHUNK_KEYS = ('memory_estimate_bytes', 'chunk_count', 'chunk_sizes', 'fft_recomputations', 'candidate_fft_cache')


def core(result):
    return {k: v for k, v in result.items() if k not in CHUNK_KEYS}


def cap_for_chunks(query, index, wanted, **kwargs):
    """Smallest-effort cap (MB) that yields at least ``wanted`` chunks yet still succeeds."""
    full = bn._run_batch(query, index, **kwargs)[0]['memory_estimate_bytes'] / 2 ** 20
    for fraction in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
        try:
            if bn._run_batch(query, index, max_memory_mb=full * fraction, **kwargs)[0]['chunk_count'] >= wanted:
                return full * fraction
        except ContractError:
            break
    raise AssertionError('no cap gives the wanted chunk count')


def test_chunked_equals_unchunked_and_reports_chunks():
    query, index = synthetic_exports(6, 12, side=200)
    kw = dict(window_um=2.0, topk=4)
    plain = bn._run_batch(query, index, **kw)[0]
    assert plain['chunk_count'] == 1 and plain['chunk_sizes'] == [12] and plain['fft_recomputations'] == 0
    cap = cap_for_chunks(query, index, 3, **kw)
    chunked = bn._run_batch(query, index, max_memory_mb=cap, **kw)[0]
    assert chunked['chunk_count'] >= 3 and sum(chunked['chunk_sizes']) == 12 and len(chunked['chunk_sizes']) == chunked['chunk_count']
    assert chunked['fft_recomputations'] > 0 and chunked['memory_estimate_bytes'] <= cap * 2 ** 20
    assert chunked['candidate_fft_cache']['builds'] == chunked['candidate_fft_cache']['keys'] + chunked['fft_recomputations']
    for a, b in zip(chunked['results'], plain['results']):
        assert [p['template_top_left_px_yx'] for p in a['primary']] == [p['template_top_left_px_yx'] for p in b['primary']]
        assert [p['score'] for p in a['primary']] == pytest.approx([p['score'] for p in b['primary']], abs=1e-12)
    assert [r['query'] for r in chunked['results']] == [r['query'] for r in plain['results']]
    # a second, different succeeding cap gives the same results; rerunning one cap is byte-identical
    other = bn._run_batch(query, index, max_memory_mb=cap * 2, **kw)[0]
    assert other['chunk_count'] != chunked['chunk_count'] and core(other) == core(chunked)
    assert json.dumps(bn._run_batch(query, index, max_memory_mb=cap, **kw)[0]) == json.dumps(chunked)


def test_chunked_cap_below_one_query_is_an_error():
    query, index = synthetic_exports(6, 4, side=200)
    with pytest.raises(ContractError, match=r'memory_estimate_exceeds_limit.*bytes.*even for one query'):
        bn._run_batch(query, index, window_um=2.0, topk=3, max_memory_mb=0.001)


def test_identical_templates_are_matched_once_across_chunks(monkeypatch):
    query, index = synthetic_exports(6, 6, side=200)
    for roi in query.ready[3:]:  # q03..q05 copy q00..q02: same bindings, same window centre
        src = query.ready[int(roi.row['roi_id'][1:]) - 3]
        roi.channel_bindings, roi.channels, roi.center_xy_um = src.channel_bindings, src.channels, src.center_xy_um
    kw = dict(window_um=2.0, topk=3)
    cap = cap_for_chunks(query, index, 3, **kw)
    calls = []
    real = bn.zncc_map
    monkeypatch.setattr(bn, 'zncc_map', lambda *a, **k: calls.append(1) or real(*a, **k))
    result = bn._run_batch(query, index, max_memory_mb=cap, **kw)[0]
    assert result['chunk_count'] >= 3 and result['n_template_matches'] == 3 and len(calls) == 3 * 3  # 3 layers each, once
    rows = {r['query']['roi_id']: r for r in result['results']}
    for j in range(3):
        a, b = rows[f'q{j:02d}'], rows[f'q{j + 3:02d}']
        assert a['primary'] == b['primary'] and a['shared_with'] == [f'q{j + 3:02d}']


def test_chunking_lowers_traced_peak_with_20_multilayer_queries():
    query, index = synthetic_exports(6, 20)
    kw = dict(window_um=4.0, topk=3)
    cap = cap_for_chunks(query, index, 4, **kw)
    plain, peak_plain = peak_traced(n_layers=6, n_queries=20)
    chunked, peak_chunked = peak_traced(n_layers=6, n_queries=20, max_memory_mb=cap)
    print(f'T2c peak: unchunked {peak_plain} B, chunks={chunked["chunk_count"]} {peak_chunked} B, '
          f'est {plain["memory_estimate_bytes"]} -> {chunked["memory_estimate_bytes"]}')
    assert chunked['chunk_count'] >= 4 and core(chunked) == core(plain)
    assert peak_chunked < peak_plain and peak_chunked <= chunked['memory_estimate_bytes']
