"""Template-matching baseline for position recovery (no learning).

For each labelled query ROI the window of side ``--window-um`` around its centre is
cut from the role/layer layout union of its product and matched by zero-mean
normalised cross-correlation (ZNCC, FFT) against the same role/layer layout union of
the candidate product; each endpoint of the query case contributes one ZNCC map and
the maps are summed. Peaks become predicted centres. The output has the shape
``psdl2.eval_position`` reads, so the learned model and this baseline are judged on
identical terms. Exact source pixels, no resampling, no training. An empty or
structureless template is reported, never guessed. Nothing here is a similarity,
registration or risk claim.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

import numpy as np

from .data import load_export
from .eval_position import read_correspondence
from .schema import ContractError, number, text

SCHEMA = 'psdl2_baseline_ncc_v1'
METHOD = 'zncc_fft_role_layer_union'


def zncc_map(image, template):
    """Valid-mode ZNCC in [-1, 1]; None when the template has no variance or exceeds the image."""
    image = np.asarray(image, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    h, w = template.shape
    H, W = image.shape
    if h > H or w > W or h == 0 or w == 0:
        return None
    t = template - template.mean()
    t_norm = math.sqrt(float((t * t).sum()))
    if t_norm == 0:
        return None
    shape = (H + h - 1, W + w - 1)
    f_image = np.fft.rfft2(image, shape)
    corr = np.fft.irfft2(f_image * np.fft.rfft2(t[::-1, ::-1], shape), shape)[h - 1:H, w - 1:W]
    f_box = np.fft.rfft2(np.ones((h, w)), shape)
    s1 = np.fft.irfft2(f_image * f_box, shape)[h - 1:H, w - 1:W]
    s2 = np.fft.irfft2(np.fft.rfft2(image * image, shape) * f_box, shape)[h - 1:H, w - 1:W]
    var = np.maximum(s2 - s1 * s1 / (h * w), 0.0)
    denominator = t_norm * np.sqrt(var)
    out = np.zeros_like(corr)
    ok = denominator > 1e-9
    out[ok] = corr[ok] / denominator[ok]
    return np.clip(out, -1.0, 1.0)


def role_layer_unions(export):
    """{(role, actual_layer): (frame, uint8 union mask)} = OR of the endpoint layout masks carried by the
    ready ROI rows of one product export. This covers the whole product only when the export's rows
    reference every electrode of that role/layer; an export built from labelled positions alone yields
    only those electrodes, which the caller must keep in mind when reading hit rates."""
    unions, frames = {}, {}
    for roi in export.ready:
        for key, binding in roi.channel_bindings.items():
            pair = (binding['role'], binding['actual_layer'])
            frame = roi.frame
            if pair in frames and (frames[pair].frame_id, frames[pair].origin_xy_um, frames[pair].pixel_pitch_um, frames[pair].shape_yx) != (
                    frame.frame_id, frame.origin_xy_um, frame.pixel_pitch_um, frame.shape_yx):
                raise ContractError('multiple_candidate_frames_for_role_layer')
            frames[pair] = frame
            mask = roi.channels[key].astype(bool)
            unions[pair] = unions[pair] | mask if pair in unions else mask.copy()
    return {pair: (frames[pair], unions[pair].astype(np.uint8)) for pair in unions}


def _window_cells(frame, center, window_um):
    half = Fraction(str(window_um)) / 2
    cx, cy = Fraction(str(center[0])), Fraction(str(center[1]))
    ox, oy = Fraction(str(frame.origin_xy_um[0])), Fraction(str(frame.origin_xy_um[1]))
    pitch = Fraction(str(frame.pixel_pitch_um))
    c0 = max(0, math.floor((cx - half - ox) / pitch))
    c1 = min(frame.shape_yx[1], math.ceil((cx + half - ox) / pitch))
    r0 = max(0, math.floor((cy - half - oy) / pitch))
    r1 = min(frame.shape_yx[0], math.ceil((cy + half - oy) / pitch))
    if c1 <= c0 or r1 <= r0:
        raise ContractError('window_outside_query_frame')
    return r0, r1, c0, c1


def _peaks(score, count, suppress_hw):
    """Greedy non-maximum suppression; ties resolve by row-major position."""
    flat = np.argsort(-score, axis=None, kind='stable')
    picked = []
    sh, sw = suppress_hw
    for index in flat:
        r, c = divmod(int(index), score.shape[1])
        if any(abs(r - pr) < sh and abs(c - pc) < sw for pr, pc, _ in picked):
            continue
        picked.append((r, c, float(score[r, c])))
        if len(picked) >= count:
            break
    return picked


def match_query(query_roi, query_unions, candidate_unions, candidate_product, *, window_um, topk):
    frame = query_roi.frame
    r0, r1, c0, c1 = _window_cells(frame, query_roi.center_xy_um, window_um)
    total, used, reasons = None, [], []
    for key, binding in sorted(query_roi.channel_bindings.items()):
        pair = (binding['role'], binding['actual_layer'])
        if pair not in query_unions or pair not in candidate_unions:
            reasons.append(f'role_layer_absent:{pair[0]}/{pair[1]}')
            continue
        template = query_unions[pair][1][r0:r1, c0:c1]
        candidate_frame, image = candidate_unions[pair]
        if Fraction(str(candidate_frame.pixel_pitch_um)) != Fraction(str(frame.pixel_pitch_um)):
            reasons.append(f'pixel_pitch_mismatch:{pair[0]}/{pair[1]}')
            continue
        score = zncc_map(image, template)
        if score is None:
            reasons.append(f'template_without_structure:{pair[0]}/{pair[1]}')
            continue
        total = score if total is None else total + score
        used.append({'role': pair[0], 'actual_layer': pair[1], 'template_shape_yx': [r1 - r0, c1 - c0]})
    if total is None:
        return {'status': 'unavailable', 'reasons': reasons or ['no_matchable_endpoint'], 'primary': [], 'endpoints_used': used}
    candidate_frame = next(candidate_unions[(e['role'], e['actual_layer'])][0] for e in used)
    # offset of the physical query centre inside its own template, kept exactly for the prediction
    qx, qy = query_roi.center_xy_um
    dx = qx - (frame.origin_xy_um[0] + c0 * frame.pixel_pitch_um)
    dy = qy - (frame.origin_xy_um[1] + r0 * frame.pixel_pitch_um)
    peaks = _peaks(total, topk, (max(1, (r1 - r0) // 2), max(1, (c1 - c0) // 2)))
    primary = []
    for rank, (r, c, value) in enumerate(peaks, 1):
        x = candidate_frame.origin_xy_um[0] + c * candidate_frame.pixel_pitch_um + dx
        y = candidate_frame.origin_xy_um[1] + r * candidate_frame.pixel_pitch_um + dy
        primary.append({'rank': rank, 'score': value / len(used), 'center_xy_um': [x, y], 'product_id': candidate_product,
                        'frame_id': candidate_frame.frame_id, 'template_top_left_px_yx': [r, c]})
    return {'status': 'available', 'reasons': reasons, 'primary': primary, 'endpoints_used': used}


def run_baseline(query_export, index_export, truths, *, window_um, topk):
    number(window_um, positive=True)
    if type(topk) is not int or topk < 1:
        raise ContractError('positive_topk_required')
    query_unions = role_layer_unions(query_export)
    candidate_unions = role_layer_unions(index_export)
    candidate_products = {roi.row['product_id'] for roi in index_export.ready}
    if len(candidate_products) != 1:
        raise ContractError('single_candidate_product_required')
    candidate_product = next(iter(candidate_products))
    by_id = {roi.row['roi_id']: roi for roi in query_export.ready}
    results = []
    for query_id in sorted(truths):
        roi = by_id.get(query_id)
        base = {'query': {'roi_id': query_id, 'product_id': None if roi is None else roi.row['product_id'],
                          'frame_id': None if roi is None else roi.frame.frame_id,
                          'center_xy_um': None if roi is None else list(roi.center_xy_um)}}
        if roi is None:
            results.append({**base, 'status': 'unavailable', 'reasons': ['query_roi_not_ready_in_export'], 'primary': [], 'endpoints_used': []})
            continue
        try:
            results.append({**base, **match_query(roi, query_unions, candidate_unions, candidate_product, window_um=window_um, topk=topk)})
        except ContractError as exc:
            results.append({**base, 'status': 'unavailable', 'reasons': [str(exc)], 'primary': [], 'endpoints_used': []})
    return {'schema_version': SCHEMA, 'method': METHOD, 'window_um': number(window_um, positive=True), 'topk': topk,
            'candidate_product_id': candidate_product, 'n_queries': len(results),
            'status': 'available' if any(r['status'] == 'available' for r in results) else 'unavailable',
            'results': results, 'interpretation': 'non_learned_position_baseline_not_similarity_registration_or_risk'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='ZNCC template-matching baseline for labelled corresponding positions.')
    parser.add_argument('--query-export', required=True, type=Path)
    parser.add_argument('--query-assets-root', type=Path)
    parser.add_argument('--index-export', required=True, type=Path)
    parser.add_argument('--index-assets-root', type=Path)
    parser.add_argument('--correspondence', required=True, type=Path)
    parser.add_argument('--window-um', required=True, type=float)
    parser.add_argument('--topk', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    args = parser.parse_args(argv)
    try:
        query = load_export(args.query_export, assets_root=args.query_assets_root, allow_unversioned_producer=args.allow_unversioned_producer)
        index = load_export(args.index_export, assets_root=args.index_assets_root, allow_unversioned_producer=args.allow_unversioned_producer)
        result = run_baseline(query, index, read_correspondence(args.correspondence), window_um=args.window_um, topk=args.topk)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': result['status'], 'n_queries': result['n_queries']}, allow_nan=False))
        return 0 if result['status'] == 'available' else 3
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
