"""Position evaluation and the ZNCC template-matching baseline (PLAN_v3 T6)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from psdl2.baseline_ncc import role_layer_unions, run_baseline, zncc_map
from psdl2.data import load_export
from psdl2.eval_position import (
    evaluate_positions, predictions_from_baseline, predictions_from_search, read_correspondence,
)
from psdl2.schema import ContractError

FIXTURE = Path(__file__).parent / 'fixtures/review_roi_v1_export'
SHAPE = (16, 16)  # 8 um × 8 um at 0.5 um/px


def masks_a():
    # both inside the 2 um window around (2.5, 2.5) um = cells rows/cols 3..6
    e1 = np.zeros(SHAPE, np.uint8); e1[3:7, 4] = 1; e1[6, 4:8] = 1          # L shape
    e2 = np.zeros(SHAPE, np.uint8); e2[4, 5:8] = 1                           # short bar beside it
    return e1, e2


def shifted(mask, dy, dx):
    out = np.zeros_like(mask)
    out[dy:, dx:] = mask[:SHAPE[0] - dy, :SHAPE[1] - dx]
    return out


def write_export(root, product, rows):
    """rows: [(roi_id, case_id, center_xy_um, layouts(e1,e2))] → review_rois.json under root."""
    payload = json.loads((FIXTURE / 'review_rois.json').read_text())
    template = deepcopy(next(row for row in payload['rois'] if row['status'] == 'ready'))
    root.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for roi_id, case_id, center, layouts in rows:
        row = deepcopy(template)
        row.update(roi_id=roi_id, product_id=product, case_id=case_id, source_kind='manual', source_region_id=None,
                   region_bbox_um=None, center_xy_um=list(center), analysis_layer='M2', failure_type='lateral_short',
                   frame_id='F', status='ready', reasons=[])
        row['source_identity'].update(product_id=product, case_id=case_id, frame_id='F')
        row['frame'] = {'frame_id': 'F', 'length_unit': 'um', 'origin_xy_um': [0.0, 0.0], 'pixel_pitch_um': 0.5,
                        'shape_yx': list(SHAPE), 'axis_order': 'yx', 'origin_convention': 'top_left_cell_edge'}
        row['endpoints'] = [{'electrode_id': f'{product}_G', 'role': 'gate', 'actual_layer': 'M2'},
                            {'electrode_id': f'{product}_D', 'role': 'data', 'actual_layer': 'M2'}]
        assets = {}
        for name, mask in (('e1', layouts[0]), ('e2', layouts[1]), ('cnt', np.zeros(SHAPE, np.uint8)), ('valid', np.ones(SHAPE, np.uint8))):
            path = root / f'{product}_{case_id}_{name}.npy'
            np.save(path, mask)
            assets[name] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'observed': True, 'value_semantics': 'binary'}
        row['source_assets'] = {'layouts': [{'endpoint': row['endpoints'][0], 'asset': assets['e1']},
                                            {'endpoint': row['endpoints'][1], 'asset': assets['e2']}],
                                'cnt': assets['cnt'], 'valid': assets['valid'], 'via': []}
        out_rows.append(row)
    payload.update(rois=out_rows, excluded_records=[], assets_root='.')
    path = root / 'review_rois.json'
    path.write_text(json.dumps(payload))
    return path


def correspondence(path, pairs):
    payload = {'schema_version': 'roi_correspondence_v1', 'producer': {'kind': 'manual', 'version': '1'},
               'pairs': [{'pair_id': f'P{i}', 'a': {'product_id': 'A', 'design_version': 'DESIGN', 'frame_id': 'F',
                                                   'center_xy_um': list(a_center), 'analysis_layer': 'M2', 'roi_id': a_id},
                          'b': {'product_id': 'B', 'design_version': 'DESIGN', 'frame_id': 'F', 'center_xy_um': list(b_center),
                                'analysis_layer': 'M2'},
                          'source': {'kind': 'manual', 'labelled_by': 't', 'labelled_at': '2026-10-18'},
                          'status': status, 'reasons': [] if status == 'ready' else ['x']}
                         for i, (a_id, a_center, b_center, status) in enumerate(pairs, 1)]}
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def products(tmp_path):
    e1, e2 = masks_a()
    query = write_export(tmp_path / 'A', 'A', [('qa', 'A1', (2.5, 2.5), (e1, e2)), ('qb', 'A2', (7.5, 7.5), (e1, e2))])
    # B is A shifted by 4 rows (2.0 um in y) and 3 columns (1.5 um in x)
    index = write_export(tmp_path / 'B', 'B', [('cb', 'B1', (4.0, 4.5), (shifted(e1, 4, 3), shifted(e2, 4, 3)))])
    truth = correspondence(tmp_path / 'c.json', [('qa', (2.5, 2.5), (4.0, 4.5), 'ready')])
    return query, index, truth, tmp_path


def test_zncc_exact_copy_peaks_at_one_and_constant_template_is_none():
    image = np.zeros((10, 12)); image[3:6, 4:8] = [[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]]
    template = image[3:6, 4:8]
    score = zncc_map(image, template)
    assert score.shape == (8, 9)
    assert score[3, 4] == pytest.approx(1.0) and score.max() == pytest.approx(1.0)
    assert np.unravel_index(np.argmax(score), score.shape) == (3, 4)
    assert zncc_map(image, np.ones((2, 2))) is None and zncc_map(np.ones((2, 2)), np.ones((3, 3))) is None


def test_baseline_recovers_pure_translation_exactly(products):
    query_path, index_path, truth_path, _ = products
    query, index = load_export(query_path), load_export(index_path)
    assert set(role_layer_unions(query)) == {('gate', 'M2'), ('data', 'M2')}
    truths = read_correspondence(truth_path)
    result = run_baseline(query, index, truths, window_um=2.0, topk=3)
    assert result['status'] == 'available' and result['method'] == 'zncc_fft_role_layer_union'
    top = result['results'][0]['primary'][0]
    assert top['center_xy_um'] == pytest.approx([4.0, 4.5]) and top['score'] == pytest.approx(1.0)
    assert top['product_id'] == 'B' and top['frame_id'] == 'F'
    assert [e['template_shape_yx'] for e in result['results'][0]['endpoints_used']] == [[4, 4], [4, 4]]
    evaluation = evaluate_positions(predictions_from_baseline(result), truths, tolerance_um=0.5, k=3)
    assert evaluation['summary'] == {'n_queries': 1, 'n_evaluable': 1, 'n_unavailable': 0, 'top1_hit_rate': 1.0,
                                     'hit_rate_at_3': 1.0, 'mean_top1_distance_um': pytest.approx(0.0),
                                     'median_top1_distance_um': pytest.approx(0.0)}
    assert evaluation['queries'][0]['candidates'][0]['hit'] is True


def test_structureless_window_is_reported_not_guessed(products):
    query_path, index_path, _, tmp_path = products
    far = correspondence(tmp_path / 'far.json', [('qb', (7.5, 7.5), (4.0, 4.5), 'ready')])
    query, index = load_export(query_path), load_export(index_path)
    # query ROI 'qb' sits in an empty corner: its window holds no structure for either endpoint
    result = run_baseline(query, index, read_correspondence(far), window_um=2.0, topk=1)
    row = result['results'][0]
    assert row['status'] == 'unavailable' and row['primary'] == []
    assert all(r.startswith('template_without_structure') for r in row['reasons'])
    missing = read_correspondence(correspondence(tmp_path / 'missing.json', [('absent', (1, 1), (4.0, 4.5), 'ready')]))
    result = run_baseline(query, index, missing, window_um=2.0, topk=1)
    assert result['results'][0]['reasons'] == ['query_roi_not_ready_in_export'] and result['status'] == 'unavailable'
    with pytest.raises(ContractError, match='positive_topk_required'):
        run_baseline(query, index, missing, window_um=2.0, topk=0)


def test_search_predictions_hit_only_within_tolerance_same_product_and_frame(products):
    *_, truth_path, _ = products
    truths = read_correspondence(truth_path)
    def result(center, product='B', frame='F'):
        return {'query': {'roi_id': 'qa'}, 'status': 'available',
                'primary': [{'roi_id': 'cb', 'product_id': product, 'frame': {'frame_id': frame}, 'center_xy_um': center, 'score': 0.9}]}
    near = evaluate_positions(predictions_from_search([result([4.3, 4.5])]), truths, tolerance_um=0.5, k=1)
    assert near['queries'][0]['top1_hit'] is True and near['queries'][0]['top1_distance_um'] == pytest.approx(0.3)
    far = evaluate_positions(predictions_from_search([result([4.0, 6.5])]), truths, tolerance_um=0.5, k=1)
    assert far['queries'][0]['top1_hit'] is False and far['summary']['top1_hit_rate'] == 0.0
    other = evaluate_positions(predictions_from_search([result([4.0, 4.5], product='C')]), truths, tolerance_um=0.5, k=1)
    assert other['queries'][0]['candidates'][0]['reason'] == 'product_or_frame_mismatch' and other['queries'][0]['top1_hit'] is False
    unknown = evaluate_positions(predictions_from_search([{**result([4.0, 4.5]), 'query': {'roi_id': 'zz'}}]), truths, tolerance_um=0.5, k=1)
    assert unknown['status'] == 'unavailable' and unknown['queries'][0]['reasons'] == ['no_labelled_position_for_query']
    empty = evaluate_positions(predictions_from_search([{**result([4.0, 4.5]), 'primary': []}]), truths, tolerance_um=0.5, k=1)
    assert empty['queries'][0]['reasons'] == ['no_candidates']
    with pytest.raises(ContractError, match='positive_k_required'):
        evaluate_positions([], truths, tolerance_um=0.5, k=0)


def test_blocked_or_unlabelled_pairs_are_not_truths(tmp_path):
    path = correspondence(tmp_path / 'c.json', [('qa', (1, 1), (2, 2), 'blocked'), (None, (1, 1), (2, 2), 'ready')])
    assert read_correspondence(path) == {}
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'schema_version': 'other', 'pairs': []}))
    with pytest.raises(ContractError, match='unsupported_correspondence_schema'):
        read_correspondence(bad)


def test_cli_baseline_then_position_eval(products):
    query_path, index_path, truth_path, tmp_path = products
    env = {'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    run = subprocess.run([sys.executable, '-m', 'psdl2.baseline_ncc', '--query-export', query_path, '--index-export', index_path,
                          '--correspondence', truth_path, '--window-um', '2', '--topk', '2', '--out', tmp_path / 'baseline.json'],
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    run = subprocess.run([sys.executable, '-m', 'psdl2.eval_position', '--baseline', tmp_path / 'baseline.json',
                          '--correspondence', truth_path, '--hit-tolerance-um', '0.5', '--k', '2', '--out', tmp_path / 'eval.json'],
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert json.loads((tmp_path / 'eval.json').read_text())['summary']['top1_hit_rate'] == 1.0
    missing = subprocess.run([sys.executable, '-m', 'psdl2.eval_position', '--baseline', tmp_path / 'baseline.json',
                              '--correspondence', truth_path, '--k', '2', '--out', tmp_path / 'x.json'],
                             capture_output=True, text=True, env=env)
    assert missing.returncode == 2 and 'hit-tolerance-um' in missing.stderr
