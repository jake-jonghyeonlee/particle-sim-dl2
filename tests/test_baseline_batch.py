"""Batch mode of the ZNCC baseline: shared candidate FFTs, template dedup, ambiguity flags, review sheet, self-check."""
from copy import deepcopy
import hashlib
import json
import os
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
        assert [{k: v for k, v in m.items() if k != 'invalid_pixel_count'} for m in got['primary']] == row['primary'] and got['endpoints_used'] == row['endpoints_used'] and got['query'] == row['query']
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
    run = subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', '--query-export', query_path, '--self-check', '--max-memory-mb', '500',
                          '--window-um', '4', '--topk', '2', '--out', out], capture_output=True, text=True, env={'PYTHONPATH': SRC})
    assert run.returncode == 0, run.stderr
    result = json.loads(out.read_text())
    export = load_export(query_path)
    assert result['self_check'] == {'passed': True} and self_check_failures(result, export) == []
    assert all(r['self_map']['score'] == pytest.approx(1.0) for r in result['results'])
    bad = deepcopy(result)
    bad['results'][2]['query']['center_xy_um'][0] += 1.0  # two pixel pitches off its own match
    failures = self_check_failures(bad, export)
    assert len(failures) == 1 and bad['results'][2]['query']['roi_id'] in failures[0]
    low = deepcopy(result); low['results'][0]['self_map']['score'] = 0.9
    assert 'not ~1' in self_check_failures(low, export)[0]
    assert self_check_failures({'results': []}, export) == ['no available query to check']
    both = subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', '--query-export', query_path, '--self-check', '--index-export', query_path, '--max-memory-mb', '500',
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
    valid = np.ones((side, side), np.uint8)

    def roi(roi_id, center, layers):
        return SimpleNamespace(row={'roi_id': roi_id, 'product_id': 'P'}, frame=frame, center_xy_um=center,
                               channels={**{f'e{k}': masks[k] for k in layers}, 'valid': valid},
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


# ---- T2-fix: review findings B1-B4, A1, A2 ----


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', *map(str, args)], capture_output=True, text=True, env={'PYTHONPATH': SRC})


def test_self_check_accepts_periodic_self_matches_through_cli(tmp_path):
    masks = random_masks(5, tile=N // 2)  # 2×2 repetition: all four centres score ~1, rank 1 is always the first repeat
    rows = [(f'q{i}', f'A{i}', center, masks) for i, center in enumerate([(5.0, 5.0), (15.0, 5.0), (5.0, 15.0), (15.0, 15.0)])]
    query = tp.write_export(tmp_path / 'A', 'A', rows)
    for topk in ('4', '1'):  # independent of top-k / suppression
        out = tmp_path / f'p{topk}.json'
        run = cli('--query-export', query, '--self-check', '--max-memory-mb', '500', '--window-um', '4', '--topk', topk, '--out', out)
        assert run.returncode == 0, run.stderr  # 08ed566: q1..q3 reported as incorrect, exit 2
        result = json.loads(out.read_text())
        assert result['self_check'] == {'passed': True}
        assert all(r['self_map']['score'] == pytest.approx(1.0) for r in result['results'])
    rank1 = {tuple(r['primary'][0]['center_xy_um']) for r in result['results']}
    assert len(rank1) == 1 and rank1 != {(15.0, 15.0)}  # the periodic condition really holds: every row ranks the same repeat first
    assert self_check_failures(result, load_export(query)) == []


def test_self_check_requires_memory_cap(batch_products, tmp_path):
    query_path, _, _ = batch_products
    run = cli('--query-export', query_path, '--self-check', '--window-um', '4', '--topk', '2', '--out', tmp_path / 'x.json')
    assert run.returncode == 2 and 'max_memory_mb_required' in run.stderr and not (tmp_path / 'x.json').exists()  # 08ed566: exit 0


def batch_cli(query, index, out, *extra):
    return cli('--query-export', query, '--index-export', index, '--all-ready', '--max-memory-mb', '500', '--window-um', '4', '--topk', '2',
               '--out', out, *extra)


def test_sheet_refuses_symlinks_and_existing_files(batch_products, tmp_path):
    query, index, _ = batch_products
    outside, outside_png = tmp_path / 'outside.txt', tmp_path / 'outside_png.txt'
    outside.write_text('keep html'); outside_png.write_text('keep png')
    sheet = tmp_path / 'sheet'
    sheet.mkdir()
    (sheet / 'sheet.html').symlink_to(outside)
    (sheet / 'q0001_query.png').symlink_to(outside_png)
    run = batch_cli(query, index, tmp_path / 'o.json', '--sheet-dir', sheet)
    assert run.returncode == 2 and 'sheet_dir_not_empty' in run.stderr and not (tmp_path / 'o.json').exists()
    assert outside.read_text() == 'keep html' and outside_png.read_text() == 'keep png'  # 08ed566 wrote through both links
    assert sorted(p.name for p in sheet.iterdir()) == ['q0001_query.png', 'sheet.html']
    # an ordinary pre-existing file is refused by name policy, not replaced
    plain = tmp_path / 'plain'; plain.mkdir(); (plain / 'sheet.html').write_text('mine')
    run = batch_cli(query, index, tmp_path / 'o.json', '--sheet-dir', plain)
    assert run.returncode == 2 and 'sheet_dir_not_empty' in run.stderr and (plain / 'sheet.html').read_text() == 'mine'
    assert [p.name for p in plain.iterdir()] == ['sheet.html']
    # symlinked destination directory and symlinked parent
    real = tmp_path / 'real'; real.mkdir()
    (tmp_path / 'linkdir').symlink_to(real)
    for target in (tmp_path / 'linkdir', tmp_path / 'linkdir' / 'sub'):
        run = batch_cli(query, index, tmp_path / 'o.json', '--sheet-dir', target)
        assert run.returncode == 2 and 'sheet_dir_symlink_refused' in run.stderr
    assert list(real.iterdir()) == [] and not (tmp_path / 'o.json').exists()
    # an empty or new directory is fine
    assert batch_cli(query, index, tmp_path / 'o.json', '--sheet-dir', tmp_path / 'fresh' / 'deep').returncode == 0


def test_sheet_write_leaves_no_partial_output_on_conflict(tmp_path):
    from psdl2 import review_sheet
    directory = tmp_path / 'd'; directory.mkdir()
    (directory / 'sheet.html').write_text('raced in')  # appears after the up-front check
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(FileExistsError):
            review_sheet._write_all(fd, {'q0001_query.png': b'png', 'sheet.html': b'html'})
    finally:
        os.close(fd)
    assert [p.name for p in directory.iterdir()] == ['sheet.html'] and (directory / 'sheet.html').read_text() == 'raced in'


def sheet_entry():
    return {'roi_id': 'q', 'center_xy_um': [8, 8], 'status': 'available', 'reasons': [], 'score_gap': None, 'flags': {}, 'shared_with': [],
            'query_invalid_pixels': 0, 'query_crop': np.array([[0, 1], [1, 0]]), 'matches': []}


@pytest.mark.parametrize('swap_parent', [False, True])
def test_sheet_directory_substitution_cannot_redirect_writes_or_cleanup(tmp_path, monkeypatch, swap_parent):
    """After the first PNG is published the sheet dir (or its parent) is renamed away and replaced by a symlink to an outside
    directory holding sentinels. 440d329 created the next temp file there and its rollback deleted the outside sentinel."""
    from psdl2 import review_sheet
    parent = tmp_path / 'parent'; parent.mkdir()
    directory = parent / 'race'
    outside = tmp_path / 'outside'
    (outside / 'race' if swap_parent else outside).mkdir(parents=True)
    victims = [(outside / 'race' if swap_parent else outside) / name for name in ('q0001_query.png', 'sheet.html')]
    victims[0].write_bytes(b'EXTERNAL PNG SENTINEL'); victims[1].write_bytes(b'EXTERNAL HTML SENTINEL')
    swapped_to = tmp_path / ('parent-original' if swap_parent else 'race-original')
    real_open, swapped = os.open, []

    def swap_then_open(path, *args, **kwargs):
        if os.fspath(path).endswith('.sheet.html.tmp') and not swapped:  # the second file of the sheet is about to be created
            moved = parent if swap_parent else directory
            moved.rename(swapped_to)
            moved.symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as hook:
        hook.setattr(review_sheet.os, 'open', swap_then_open)
        with pytest.raises(ContractError, match='sheet_dir_(replaced_during_write|symlink_refused)'):
            review_sheet.write_sheet(directory, [sheet_entry()], window_um=4, topk=1)
    assert swapped
    assert [v.read_bytes() for v in victims] == [b'EXTERNAL PNG SENTINEL', b'EXTERNAL HTML SENTINEL']  # nothing outside written or deleted
    assert sorted(p.name for p in victims[0].parent.iterdir()) == ['q0001_query.png', 'sheet.html']  # no stray temp files outside
    assert list((swapped_to / 'race' if swap_parent else swapped_to).iterdir()) == []  # the original directory holds no partial output
    assert (parent if swap_parent else directory).is_symlink()  # the substituted link was never followed or replaced


def punch_hole(root, product, case_id, box):
    path = root / f'{product}_{case_id}_valid.npy'
    valid = np.load(path); r0, r1, c0, c1 = box; valid[r0:r1, c0:c1] = 0
    np.save(path, valid)
    manifest = root / 'review_rois.json'
    payload = json.loads(manifest.read_text())
    for row in payload['rois']:
        if row['source_assets']['valid']['path'] == path.name:
            row['source_assets']['valid']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))


def test_valid_domain_holes_are_flagged_and_shown_without_changing_scores(tmp_path):
    query, index = pair_of_exports(tmp_path, [('qh', (8.0, 8.0))])
    clean = run_baseline_batch(load_export(query), load_export(index), window_um=4.0, topk=2)['results'][0]
    assert clean['flags']['query_valid_hole'] is False and clean['flags']['candidate_valid_hole'] is False
    assert clean['query_invalid_pixel_count'] == 0 and clean['primary'][0]['invalid_pixel_count'] == 0
    punch_hole(tmp_path / 'A', 'A', 'A0', (12, 16, 12, 16))  # inside the 12:20 template window; centre pixel (16, 16) stays valid
    punch_hole(tmp_path / 'B', 'B', 'B1', (20, 22, 18, 20))  # inside the best candidate window rows 18:26, cols 16:24
    out = tmp_path / 'h.json'
    run = batch_cli(query, index, out, '--sheet-dir', tmp_path / 'sheet')
    assert run.returncode == 0, run.stderr
    row = json.loads(out.read_text())['results'][0]
    assert row['flags']['query_valid_hole'] is True and row['query_invalid_pixel_count'] == 16  # 08ed566: no such flag
    assert row['flags']['candidate_valid_hole'] is True and row['primary'][0]['template_top_left_px_yx'] == [18, 16]
    assert row['primary'][0]['invalid_pixel_count'] == 4
    assert row['status'] == 'available' and [p['score'] for p in row['primary']] == [p['score'] for p in clean['primary']]  # never auto-rejected
    page = (tmp_path / 'sheet' / 'sheet.html').read_text(encoding='utf-8')
    assert '질의 창 무효 영역 16px' in page and '무효 영역 4px' in page


def custom_export(root, product, rows):
    """rows: [(roi_id, case_id, center, [(role, layer, mask), (role, layer, mask)])] with free role/layer strings."""
    path = tp.write_export(root, product, [(rid, case, center, (m[0][2], m[1][2])) for rid, case, center, m in rows])
    payload = json.loads(path.read_text())
    for row, (_, _, _, binds) in zip(payload['rois'], rows):
        for k, (role, layer, _) in enumerate(binds):
            row['endpoints'][k].update(role=role, actual_layer=layer)
            row['source_assets']['layouts'][k]['endpoint'] = dict(row['endpoints'][k])
    path.write_text(json.dumps(payload))
    return path


def test_template_hash_framing_is_unambiguous(tmp_path):
    mask = random_masks(1)[0]
    zeros = np.zeros_like(mask)
    spec = lambda role, layer: {'pitch': 0.5, 'spec': [((role, layer), mask[:8, :8])]}
    assert bn._template_sha256(spec('gate/x', 'M2')) != bn._template_sha256(spec('gate', 'x/M2'))  # 08ed566: equal
    query = custom_export(tmp_path / 'A', 'A', [('q0', 'A0', (8.0, 8.0), [('gate/x', 'M2', mask), ('data', 'M2', zeros)]),
                                                  ('q1', 'A1', (8.0, 8.0), [('gate', 'x/M2', mask), ('data', 'M2', zeros)])])
    index = custom_export(tmp_path / 'B', 'B', [('c0', 'B0', (10.0, 10.0), [('gate/x', 'M2', tp.shifted(mask, 6, 4)), ('data', 'M2', zeros)]),
                                                  ('c1', 'B1', (10.0, 10.0), [('gate', 'x/M2', tp.shifted(mask, 9, 8)), ('data', 'M2', zeros)])])
    q, i = load_export(query), load_export(index)
    dedup = run_baseline_batch(q, i, window_um=4.0, topk=2)
    plain = run_baseline_batch(q, i, window_um=4.0, topk=2, dedupe=False)
    rows = {r['query']['roi_id']: r for r in dedup['results']}
    assert rows['q0']['template_sha256'] != rows['q1']['template_sha256'] and rows['q0']['shared_with'] == []
    assert rows['q0']['primary'][0]['center_xy_um'] != rows['q1']['primary'][0]['center_xy_um']
    key = lambda r: (r['query'], r['status'], r['primary'], r['endpoints_used'])
    assert [key(r) for r in dedup['results']] == [key(r) for r in plain['results']]


def test_output_records_deterministic_implementation_revision(batch_products, tmp_path):
    query, index, _ = batch_products
    outs = []
    for name in ('a', 'b'):
        assert batch_cli(query, index, tmp_path / f'{name}.json').returncode == 0
        outs.append((tmp_path / f'{name}.json').read_bytes())
    assert outs[0] == outs[1]
    assert json.loads(outs[0])['implementation_revision'] == bn.IMPLEMENTATION == 'baseline_ncc_batch_v1.3'  # 08ed566: key absent


# ---- T2-fix2: re-review findings R1-R3 ----


def test_sheet_dir_walk_is_fd_based_and_follows_only_trusted_links(tmp_path, monkeypatch):
    from psdl2 import review_sheet
    # new nested directories are created through the descriptor walk; a second write into the (now non-empty) dir is refused
    target = tmp_path / 'a' / 'b'
    assert review_sheet.write_sheet(target, [sheet_entry()], window_um=4, topk=1) == target / 'sheet.html'
    assert sorted(p.name for p in target.iterdir()) == ['q0001_query.png', 'sheet.html']
    with pytest.raises(ContractError, match='sheet_dir_not_empty'):
        review_sheet.write_sheet(target, [sheet_entry()], window_um=4, topk=1)
    # a user-owned link is refused anywhere on the path; a link is followed only when root owns it and its parent
    (tmp_path / 'link').symlink_to(tmp_path / 'a')
    with pytest.raises(ContractError, match='sheet_dir_symlink_refused'):
        review_sheet.check_sheet_dir(tmp_path / 'link' / 'c')
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        link = type('S', (), {'st_uid': 0})()
        assert not review_sheet._trusted_link(parent, link)  # the parent belongs to the test user, not root
        with monkeypatch.context() as fake:
            fake.setattr(review_sheet.os, 'fstat', lambda fd: type('S', (), {'st_uid': 0, 'st_mode': 0o40755})())
            assert review_sheet._trusted_link(parent, link) and not review_sheet._trusted_link(parent, type('S', (), {'st_uid': 501})())
            fake.setattr(review_sheet.os, 'fstat', lambda fd: type('S', (), {'st_uid': 0, 'st_mode': 0o40777})())
            assert not review_sheet._trusted_link(parent, link)  # world-writable parent: anyone could swap the link
    finally:
        os.close(parent)


@pytest.mark.skipif(not os.path.islink('/tmp') or os.lstat('/tmp').st_uid != 0, reason='needs the root-owned /tmp system link (macOS)')
def test_sheet_under_the_root_owned_tmp_link_still_works(tmp_path):
    from psdl2 import review_sheet
    import tempfile
    with tempfile.TemporaryDirectory(dir='/tmp') as scratch:
        target = Path(scratch) / 'sheet'
        assert review_sheet.write_sheet(target, [sheet_entry()], window_um=4, topk=1) == target / 'sheet.html'
        assert (Path(os.path.realpath(scratch)) / 'sheet' / 'sheet.html').is_file()


def self_check_cli(query, out, topk='2'):
    return ['--query-export', str(query), '--self-check', '--max-memory-mb', '500', '--window-um', '4', '--topk', topk, '--out', str(out)]


def test_self_check_fails_when_the_matcher_uses_the_wrong_template(batch_products, tmp_path, monkeypatch, capsys):
    query_path, _, _ = batch_products
    real = bn.zncc_map
    monkeypatch.setattr(bn, 'zncc_map', lambda image, template, *a, **k: real(image, 1 - template, *a, **k))
    assert bn.main(self_check_cli(query_path, tmp_path / 'wrong.json')) == 2  # 440d329: exit 0 (a second, independent calculation passed)
    err = capsys.readouterr().err
    assert 'self_check_failed' in err and 'score map value' in err and not (tmp_path / 'wrong.json').exists()


def test_self_check_fails_when_the_accumulated_map_is_corrupted(batch_products, tmp_path, monkeypatch, capsys):
    query_path, _, _ = batch_products
    real = bn.zncc_map
    monkeypatch.setattr(bn, 'zncc_map', lambda *a, **k: 0.9 * real(*a, **k))  # every accumulated map is scaled: self position ~0.9
    assert bn.main(self_check_cli(query_path, tmp_path / 'scaled.json')) == 2
    assert 'not ~1' in capsys.readouterr().err


def test_self_check_fails_when_the_emitted_coordinate_is_wrong(batch_products, tmp_path, monkeypatch, capsys):
    query_path, _, _ = batch_products
    real = bn._result_row

    def shifted(*args, **kwargs):
        row = real(*args, **kwargs)
        row['primary'][0]['center_xy_um'][0] += 10.0 if row['primary'] else 0.0
        return row

    monkeypatch.setattr(bn, '_result_row', shifted)
    assert bn.main(self_check_cli(query_path, tmp_path / 'moved.json', topk='1')) == 2  # 440d329: exit 0
    assert 'maps to' in capsys.readouterr().err


def test_self_check_reads_the_map_at_the_expected_index(batch_products, tmp_path, monkeypatch):
    query_path, _, _ = batch_products
    real = bn._finish_match

    def misread(plan, total, topk, candidate_unions, probes=None):
        return real(plan, total, topk, candidate_unions, None if probes is None else {k: (r + 1, c) for k, (r, c) in probes.items()})

    monkeypatch.setattr(bn, '_finish_match', misread)
    export = load_export(query_path)
    result = run_baseline_batch(export, export, window_um=4.0, topk=2, self_check=True)
    assert any('score map was read at' in f or 'not ~1' in f for f in self_check_failures(result, export))


def test_valid_hole_counts_use_only_the_contributors_of_the_role_layer(tmp_path):
    e1, e2 = random_masks(3)
    query = tp.write_export(tmp_path / 'q', 'A', [('qa', 'A0', (8.0, 8.0), (e1, e2))])
    index = tp.write_export(tmp_path / 'i', 'B', [('clean', 'B0', (8.0, 8.0), (e1, e2)), ('unrelated', 'B1', (8.0, 8.0), (e1, e2))])
    payload = json.loads(index.read_text())
    row = payload['rois'][1]
    for k in range(2):  # the second ROI contributes to M3 only; the query and the first ROI use M2
        row['endpoints'][k]['actual_layer'] = 'M3'
        row['source_assets']['layouts'][k]['endpoint'] = dict(row['endpoints'][k])
    row['analysis_layer'] = 'M3'
    index.write_text(json.dumps(payload))
    punch_hole(index.parent, 'B', 'B1', (12, 16, 12, 16))  # 4x4 hole on the M3-only contributor
    loaded = load_export(index)
    assert len(loaded.ready) == 2
    invalid = bn.role_layer_valid(loaded)
    assert not invalid[('gate', 'M2')].any() and int(invalid[('gate', 'M3')].sum()) == 16  # 440d329: one frame-wide array, 16 everywhere
    assert invalid[('gate', 'M2')] is invalid[('data', 'M2')] and invalid[('gate', 'M3')] is invalid[('data', 'M3')]  # identical contributor sets share
    assert invalid[('gate', 'M2')] is not invalid[('gate', 'M3')]
    best = run_baseline_batch(load_export(query), loaded, window_um=4.0, topk=1)['results'][0]
    assert [e['actual_layer'] for e in best['endpoints_used']] == ['M2', 'M2']
    assert best['primary'][0]['invalid_pixel_count'] == 0 and best['flags']['candidate_valid_hole'] is False


def asymmetric_query(tmp_path):
    return tp.write_export(tmp_path / 'asym', 'A', [('q', 'A0', (8.0, 10.0), random_masks(3))], pitch=0.5)


def asym_cli(query, out):
    return ['--query-export', str(query), '--self-check', '--max-memory-mb', '500', '--window-um', '4', '--topk', '1', '--out', str(out)]


def test_self_check_fails_when_peak_row_and_column_are_transposed(tmp_path, monkeypatch, capsys):
    query = asymmetric_query(tmp_path)
    assert bn.main(asym_cli(query, tmp_path / 'ok.json')) == 0
    real = bn._peaks
    monkeypatch.setattr(bn, '_peaks', lambda *a, **k: [(c, r, value) for r, c, value in real(*a, **k)])
    assert bn.main(asym_cli(query, tmp_path / 'transposed.json')) == 2  # c297774: exit 0 (centre [10, 8] reported with score 1.0)
    assert 'accumulated map holds' in capsys.readouterr().err and not (tmp_path / 'transposed.json').exists()


def test_self_check_fails_when_scores_are_swapped_between_rank_entries(batch_products, tmp_path, monkeypatch, capsys):
    query_path, _, _ = batch_products
    real = bn._peaks

    def swapped(*args, **kwargs):
        # Swap only the scores of ranks 2 and 3: rank 1 keeps its own score, so only a per-entry
        # map_score check catches it.
        peaks = real(*args, **kwargs)
        if len(peaks) < 3:
            return peaks
        (r1, c1, s1), (r2, c2, s2) = peaks[1], peaks[2]
        return [peaks[0], (r1, c1, s2), (r2, c2, s1), *peaks[3:]]

    monkeypatch.setattr(bn, '_peaks', swapped)
    assert bn.main(self_check_cli(query_path, tmp_path / 'swap.json', topk='3')) == 2  # c297774: exit 0
    assert 'accumulated map holds' in capsys.readouterr().err


def large_origin_export(root, origin, centre):
    path = tp.write_export(root, 'A', [('q', 'A0', (1.63, 1.97), random_masks(3))], pitch=0.1)
    payload = json.loads(path.read_text())
    payload['rois'][0]['frame']['origin_xy_um'] = [origin, origin]
    payload['rois'][0]['center_xy_um'] = list(centre)
    path.write_text(json.dumps(payload))
    return path


def test_self_check_geometry_tolerance_follows_the_origin_magnitude_but_stays_sub_pixel(tmp_path, monkeypatch):
    query = large_origin_export(tmp_path / 'big', 1e7, [1e7 + 1.63, 1e7 + 1.97])
    args = ['--query-export', str(query), '--self-check', '--max-memory-mb', '100', '--window-um', '0.8', '--topk', '4']
    assert bn.main([*args, '--out', str(tmp_path / 'big.json')]) == 0  # c297774: exit 2 (10000000.53 vs 10000000.530000001)
    real = bn._result_row

    def one_pixel(*a, **k):
        row = real(*a, **k)
        row['primary'][0]['center_xy_um'][0] += 0.1
        return row

    monkeypatch.setattr(bn, '_result_row', one_pixel)
    assert bn.main([*args, '--out', str(tmp_path / 'shift.json')]) == 2


def test_validity_maps_are_in_the_memory_estimate_and_the_cap(tmp_path):
    def exports(n_sets):
        e1, e2 = random_masks(3)
        rows = [(f'r{i}', f'B{i}', (8.0, 8.0), (e1, e2)) for i in range(n_sets)]
        path = tp.write_export(tmp_path / f's{n_sets}', 'B', rows)
        payload = json.loads(path.read_text())
        for i, row in enumerate(payload['rois']):  # every ROI contributes to its own layer: n distinct contributor sets
            for k in range(2):
                row['endpoints'][k]['actual_layer'] = f'M{i}'
                row['source_assets']['layouts'][k]['endpoint'] = dict(row['endpoints'][k])
            row['analysis_layer'] = f'M{i}'
        path.write_text(json.dumps(payload))
        return load_export(path)

    few, many = exports(1), exports(4)
    base = run_baseline_batch(few, few, window_um=4.0, topk=1)
    grown = run_baseline_batch(many, many, window_um=4.0, topk=1)
    assert base['validity_map_count'] == 1 and grown['validity_map_count'] == 4
    assert grown['validity_map_bytes'] == 4 * N * N and base['validity_map_bytes'] == N * N
    assert grown['memory_estimate_bytes'] - base['memory_estimate_bytes'] >= grown['validity_map_bytes'] - base['validity_map_bytes']
    one_fft = base['memory_estimate_bytes'] - 3 * N * N  # one layer unit without its validity map and construction temporaries
    with pytest.raises(ContractError, match='memory_estimate_exceeds_limit'):  # c297774: no error (the maps were not counted)
        run_baseline_batch(many, many, window_um=4.0, topk=1, max_memory_mb=(one_fft + 6 * N * N - 1) / 2 ** 20)
    assert run_baseline_batch(many, many, window_um=4.0, topk=1, max_memory_mb=(one_fft + 6 * N * N) / 2 ** 20)['status'] == 'available'
