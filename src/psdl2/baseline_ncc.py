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
import hashlib
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
IMPLEMENTATION = 'baseline_ncc_batch_v1.2'  # bump on any change of matching, flags or batch output semantics; never a timestamp


class CandidateFFTCache:
    """Candidate-image terms shared by every query: rfft2(image), rfft2(image*image) folded with the box
    FFT into sqrt(window variance). They depend on the image and the template shape only, so they are
    built once per (role, actual_layer, template shape) and reused; the arithmetic is the one zncc_map
    always did, so results are numerically identical."""

    def __init__(self):
        self._terms = {}
        self._built = set()
        self.builds = 0
        self.hits = 0

    def get(self, key, image, template_shape):
        entry = (key, tuple(template_shape))
        if entry in self._terms:
            self.hits += 1
        else:
            self.builds += 1
            self._built.add(entry)
            self._terms[entry] = _candidate_terms(image, template_shape)
        return self._terms[entry]

    def release(self, key, template_shape):
        """Drop one layer's terms once every query that needs them has run (layer-by-layer batch)."""
        self._terms.pop((key, tuple(template_shape)), None)

    def stats(self):
        return {'keys': len(self._built), 'builds': self.builds, 'hits': self.hits}


def _candidate_terms(image, template_shape):
    image = np.asarray(image, dtype=np.float64)
    h, w = template_shape
    H, W = image.shape
    shape = (H + h - 1, W + w - 1)
    f_image = np.fft.rfft2(image, shape)
    f_box = np.fft.rfft2(np.ones((h, w)), shape)
    s1 = np.fft.irfft2(f_image * f_box, shape)[h - 1:H, w - 1:W]
    s2 = np.fft.irfft2(np.fft.rfft2(image * image, shape) * f_box, shape)[h - 1:H, w - 1:W]
    var = np.maximum(s2 - s1 * s1 / (h * w), 0.0)
    return shape, f_image, np.sqrt(var)


def _template_terms(image_shape, template):
    """(zero-mean template, its norm), or None when the template exceeds the image or has no variance."""
    template = np.asarray(template, dtype=np.float64)
    h, w = template.shape
    H, W = image_shape
    if h > H or w > W or h == 0 or w == 0:
        return None
    t = template - template.mean()
    t_norm = math.sqrt(float((t * t).sum()))
    if t_norm == 0:
        return None
    return t, t_norm


def zncc_map(image, template, cache=None, key=None):
    """Valid-mode ZNCC in [-1, 1]; None when the template has no variance or exceeds the image.
    With ``cache`` (a CandidateFFTCache) and ``key`` the candidate terms are shared across calls."""
    terms = _template_terms(image.shape, template)
    if terms is None:
        return None
    t, t_norm = terms
    h, w = t.shape
    H, W = image.shape
    shape, f_image, sqrt_var = (_candidate_terms(image, (h, w)) if cache is None else cache.get(key, image, (h, w)))
    corr = np.fft.irfft2(f_image * np.fft.rfft2(t[::-1, ::-1], shape), shape)[h - 1:H, w - 1:W]
    denominator = t_norm * sqrt_var
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
            if pair in unions:
                np.logical_or(unions[pair], roi.channels[key], out=unions[pair])  # in place: no per-ROI temporary
            else:
                unions[pair] = roi.channels[key].astype(bool)  # the one copy; never alias the ROI's own array
    return {pair: (frames[pair], unions[pair].view(np.uint8)) for pair in unions}


def role_layer_valid(export):
    """{(role, actual_layer): bool array, True = invalid pixel} for the candidate side: a pixel is invalid when any
    ready ROI that contributes to that role/layer (carries a binding for it) reports it outside its valid domain.
    ROIs that only contribute to other role/layers never enter. Pairs share one array only when their contributor sets
    are identical; the arrays are read-only in use."""
    contributors, frames = {}, {}
    for index, roi in enumerate(export.ready):
        frame = roi.frame
        frame_key = (frame.frame_id, frame.origin_xy_um, frame.pixel_pitch_um, frame.shape_yx)
        for binding in roi.channel_bindings.values():
            pair = (binding['role'], binding['actual_layer'])
            if frames.setdefault(pair, frame_key) != frame_key:
                raise ContractError('multiple_candidate_frames_for_role_layer')
            contributors.setdefault(pair, set()).add(index)
    by_set, invalid = {}, {}
    for pair, members in contributors.items():
        key = frozenset(members)
        if key not in by_set:
            bad = np.zeros(frames[pair][3], dtype=bool)
            for index in sorted(key):
                np.logical_or(bad, np.logical_not(export.ready[index].channels['valid'].astype(bool)), out=bad)
            by_set[key] = bad
        invalid[pair] = by_set[key]
    return invalid


def _window_bounds(frame, center, window_um):
    half = Fraction(str(window_um)) / 2
    cx, cy = Fraction(str(center[0])), Fraction(str(center[1]))
    ox, oy = Fraction(str(frame.origin_xy_um[0])), Fraction(str(frame.origin_xy_um[1]))
    pitch = Fraction(str(frame.pixel_pitch_um))
    return (math.floor((cy - half - oy) / pitch), math.ceil((cy + half - oy) / pitch),
            math.floor((cx - half - ox) / pitch), math.ceil((cx + half - ox) / pitch))


def _window_cells(frame, center, window_um):
    raw_r0, raw_r1, raw_c0, raw_c1 = _window_bounds(frame, center, window_um)
    c0, c1 = max(0, raw_c0), min(frame.shape_yx[1], raw_c1)
    r0, r1 = max(0, raw_r0), min(frame.shape_yx[0], raw_r1)
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


def _prepare(query_roi, query_unions, window_um):
    """Window cells and the per-endpoint template masks of one query (None where the query product lacks
    the role/layer), in the sorted binding order match_query always used."""
    frame = query_roi.frame
    r0, r1, c0, c1 = _window_cells(frame, query_roi.center_xy_um, window_um)
    clipped = _window_bounds(frame, query_roi.center_xy_um, window_um) != (r0, r1, c0, c1)
    spec = []
    for key, binding in sorted(query_roi.channel_bindings.items()):
        pair = (binding['role'], binding['actual_layer'])
        spec.append((pair, query_unions[pair][1][r0:r1, c0:c1] if pair in query_unions else None))
    invalid_px = int(np.count_nonzero(query_roi.channels['valid'][r0:r1, c0:c1] == 0))
    return {'cells': (r0, r1, c0, c1), 'clipped': clipped, 'spec': spec, 'pitch': Fraction(str(frame.pixel_pitch_um)),
            'invalid_px': invalid_px}


def _template_sha256(prep):
    """SHA256 over every endpoint role/layer, template shape and mask bytes of the window (plus pixel pitch):
    equal hashes mean equal matching inputs. Framing is unambiguous: each header is one JSON array (role and layer
    are free strings, so no separator may be assumed) that also carries the mask byte count."""
    digest = hashlib.sha256(json.dumps(['pitch', str(prep['pitch'])]).encode() + b'\n')
    for pair, template in prep['spec']:
        data = None if template is None else np.ascontiguousarray(template).tobytes()
        shape = None if template is None else list(template.shape)
        digest.update(json.dumps([pair[0], pair[1], shape, None if data is None else len(data)]).encode() + b'\n')
        if data is not None:
            digest.update(data)
    return digest.hexdigest()


def _plan_match(prep, candidate_unions):
    """Which endpoints of one query can be matched, decided before any FFT: (pair, template) items in the
    sorted binding order, the reasons for the rest, and the ``used`` rows match_prepared always reported."""
    r0, r1, c0, c1 = prep['cells']
    reasons, items, used = [], [], []
    for pair, template in prep['spec']:
        if template is None or pair not in candidate_unions:
            reasons.append(f'role_layer_absent:{pair[0]}/{pair[1]}')
            continue
        candidate_frame, image = candidate_unions[pair]
        if Fraction(str(candidate_frame.pixel_pitch_um)) != prep['pitch']:
            reasons.append(f'pixel_pitch_mismatch:{pair[0]}/{pair[1]}')
            continue
        if _template_terms(image.shape, template) is None:
            reasons.append(f'template_without_structure:{pair[0]}/{pair[1]}')
            continue
        items.append((pair, template))
        used.append({'role': pair[0], 'actual_layer': pair[1], 'template_shape_yx': [r1 - r0, c1 - c0]})
    return {'reasons': reasons, 'items': items, 'used': used, 'cells': prep['cells']}


def _finish_match(plan, total, topk, candidate_unions, probes=None):
    """Peaks of the summed ZNCC map (None = nothing matched) as the match dict _result_row reads. ``probes``
    {query_id: (r, c)} are read straight from the accumulated map while it is resident (S2 self-check): the mean-over-
    endpoints value at that map index, independent of top-k and suppression (None when the index is outside the map)."""
    r0, r1, c0, c1 = plan['cells']
    if total is None:
        return {'status': 'unavailable', 'reasons': plan['reasons'] or ['no_matchable_endpoint'], 'used': plan['used'], 'peaks': [],
                'candidate_frame': None}
    candidate_frame = next(candidate_unions[(e['role'], e['actual_layer'])][0] for e in plan['used'])
    peaks = _peaks(total, topk, (max(1, (r1 - r0) // 2), max(1, (c1 - c0) // 2)))
    matched = {'status': 'available', 'reasons': plan['reasons'], 'used': plan['used'], 'peaks': peaks, 'candidate_frame': candidate_frame}
    if probes is not None:
        matched['probe_values'] = {query_id: (float(total[r, c]) / len(plan['used']) if 0 <= r < total.shape[0] and 0 <= c < total.shape[1] else None)
                                   for query_id, (r, c) in probes.items()}
    return matched


def _match_prepared(prep, candidate_unions, topk, cache, probes=None):
    """ZNCC maps of the endpoint templates against the candidate unions, summed; peaks as (r, c, score)."""
    plan = _plan_match(prep, candidate_unions)
    total = None
    for pair, template in plan['items']:
        score = zncc_map(candidate_unions[pair][1], template, cache, pair)
        total = score if total is None else total + score
    return _finish_match(plan, total, topk, candidate_unions, probes)


def _result_row(query_roi, prep, matched, candidate_product):
    if matched['status'] == 'unavailable':
        return {'status': 'unavailable', 'reasons': matched['reasons'], 'primary': [], 'endpoints_used': matched['used']}
    candidate_frame = matched['candidate_frame']
    primary = []
    for rank, (r, c, value) in enumerate(matched['peaks'], 1):
        x, y = _result_row_center(query_roi, prep, candidate_frame, r, c)
        primary.append({'rank': rank, 'score': value / len(matched['used']), 'center_xy_um': [x, y], 'product_id': candidate_product,
                        'frame_id': candidate_frame.frame_id, 'template_top_left_px_yx': [r, c]})
    return {'status': 'available', 'reasons': matched['reasons'], 'primary': primary, 'endpoints_used': matched['used']}


def match_query(query_roi, query_unions, candidate_unions, candidate_product, *, window_um, topk, cache=None):
    prep = _prepare(query_roi, query_unions, window_um)
    return _result_row(query_roi, prep, _match_prepared(prep, candidate_unions, topk, cache), candidate_product)


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


def _fft_terms_bytes(image_shape, template_shape):
    """(C, R, V) for one candidate layer and template shape: bytes of one complex rfft2 on the padded grid,
    of one real padded array, and of one valid-correlation map (all float64/complex128)."""
    H, W = image_shape
    h, w = template_shape
    padded = (H + h - 1, W + w - 1)
    return padded[0] * (padded[1] // 2 + 1) * 16, padded[0] * padded[1] * 8, (H - h + 1) * (W - w + 1) * 8


def estimate_batch_memory(units, groups, *, sheet_bytes=0):
    """Peak resident bytes of the layer-by-layer batch, counted before any FFT is computed.

    ``units`` {unit key: (candidate image shape, template shape)}, unit key = (layer, template shape);
    ``groups`` one list of unit keys per template group. Layers run one after the other, so at most one layer's terms are
    alive: the FFT pair rfft2(image), rfft2(image*image) is 2*C (``largest_layer_fft_bytes``), and building
    or applying it holds about 4 complex + 4 real padded arrays and 6 valid-size maps at once
    (``layer_peak = 4*C + 4*R + 6*V``). On top of that only template groups that span several layers keep a
    running sum of one valid map V from their first to their last layer. The candidate/query mask unions
    are loaded before this point and not counted; ``sheet_bytes`` are the review-sheet crops."""
    order = sorted(units)
    index = {unit: i for i, unit in enumerate(order)}
    sizes = {unit: _fft_terms_bytes(*shapes) for unit, shapes in units.items()}
    spans = []
    for keys in groups:
        positions = sorted({index[k] for k in keys})
        if len(positions) > 1:
            spans.append((positions[0], positions[-1], sizes[order[positions[0]]][2]))
    peak, largest = 0, 0
    for i, unit in enumerate(order):
        C, R, V = sizes[unit]
        live = sum(v for first, last, v in spans if first <= i <= last)
        peak = max(peak, 4 * C + 4 * R + 6 * V + live)
        largest = max(largest, 2 * C)
    return {'memory_estimate_bytes': peak + sheet_bytes if order else sheet_bytes, 'largest_layer_fft_bytes': largest}


def _candidate_invalid_counts(row, candidate_invalid):
    """Per returned candidate window: pixels outside the valid domain of any used endpoint layer (in place, row['primary'])."""
    h, w = row['endpoints_used'][0]['template_shape_yx'] if row['endpoints_used'] else (0, 0)
    for match in row['primary']:
        r, c = match['template_top_left_px_yx']
        bad = np.zeros((h, w), dtype=bool)
        for e in row['endpoints_used']:
            np.logical_or(bad, candidate_invalid[(e['role'], e['actual_layer'])][r:r + h, c:c + w], out=bad)
        match['invalid_pixel_count'] = int(bad.sum())


def _result_row_center(query_roi, prep, candidate_frame, r, c):
    # offset of the physical query centre inside its own template, kept exactly for the prediction
    frame = query_roi.frame
    r0, _, c0, _ = prep['cells']
    dx = query_roi.center_xy_um[0] - (frame.origin_xy_um[0] + c0 * frame.pixel_pitch_um)
    dy = query_roi.center_xy_um[1] - (frame.origin_xy_um[1] + r0 * frame.pixel_pitch_um)
    return [candidate_frame.origin_xy_um[0] + c * candidate_frame.pixel_pitch_um + dx,
            candidate_frame.origin_xy_um[1] + r * candidate_frame.pixel_pitch_um + dy]


def _flags(prep, row, margin):
    scores = [p['score'] for p in row['primary']]
    flags = {'low_structure': any(r.startswith('template_without_structure') for r in row['reasons']),
             'window_clipped': prep['clipped'],
             'query_valid_hole': prep['invalid_px'] > 0,
             'candidate_valid_hole': any(p['invalid_pixel_count'] > 0 for p in row['primary']),
             'score_gap': scores[0] - scores[1] if len(scores) > 1 else None}
    if margin is not None:
        flags['near_equal_peaks'] = any(score >= scores[0] - margin for score in scores[1:])
    return flags


def _run_batch(query_export, index_export, *, window_um, topk, ambiguity_margin=None, share_fft=True, dedupe=True,
               max_memory_mb=None, sheet=False, all_at_once=False, self_check=False):
    """Every ready ROI of the query export is a query. Candidate FFTs are shared (R1) and queries with an
    identical template hash are matched once (R2); each query still gets its own row. Score never decides
    anything here: rows are ordered by roi_id and the peaks are the caller's topk. Layers are processed one at a time
    (see estimate_batch_memory); ``max_memory_mb`` splits the template groups into sequential chunks that fit it and makes a cap
    below one group's estimate a ContractError before any FFT."""
    number(window_um, positive=True)
    if type(topk) is not int or topk < 1:
        raise ContractError('positive_topk_required')
    if ambiguity_margin is not None and number(ambiguity_margin) < 0:
        raise ContractError('nonnegative_ambiguity_margin_required')
    query_unions = role_layer_unions(query_export)
    candidate_unions = role_layer_unions(index_export)
    candidate_invalid = role_layer_valid(index_export)
    candidate_products = {roi.row['product_id'] for roi in index_export.ready}
    if len(candidate_products) != 1:
        raise ContractError('single_candidate_product_required')
    candidate_product = next(iter(candidate_products))
    cache = CandidateFFTCache() if share_fft else None
    rois = sorted(query_export.ready, key=lambda roi: roi.row['roi_id'])
    preps, hashes, groups = {}, {}, {}
    for roi in rois:
        query_id = roi.row['roi_id']
        try:
            preps[query_id] = _prepare(roi, query_unions, window_um)
        except ContractError as exc:
            preps[query_id] = exc
            continue
        hashes[query_id] = _template_sha256(preps[query_id])
        groups.setdefault(hashes[query_id], []).append(query_id)
    group_order = []
    for roi in rois:
        query_id = roi.row['roi_id']
        if not isinstance(preps[query_id], ContractError):
            key = hashes[query_id] if dedupe else query_id
            if key not in group_order:
                group_order.append(key)
    plans = {key: _plan_match(preps[groups[key][0]] if dedupe else preps[key], candidate_unions) for key in group_order}
    group_units = {key: [(pair, template.shape) for pair, template in plans[key]['items']] for key in group_order}
    shapes = {u: (candidate_unions[u[0]][1].shape, u[1]) for units in group_units.values() for u in units}
    sheet_bytes = 0
    if sheet:  # one query crop + topk match crops (bool) per query
        for query_id, prep in preps.items():
            if not isinstance(prep, ContractError):
                r0, r1, c0, c1 = prep['cells']
                sheet_bytes += (1 + topk) * (r1 - r0) * (c1 - c0)

    def chunk_estimate(keys):
        used = {u for key in keys for u in group_units[key]}
        return estimate_batch_memory({u: shapes[u] for u in used}, [group_units[key] for key in keys], sheet_bytes=sheet_bytes)

    limit = None if max_memory_mb is None else max_memory_mb * 1024 * 1024
    # Query chunking: the template groups (queries with one template hash are one group, so identical templates are never
    # split across chunks and are matched once) are cut into contiguous chunks in group order, each the largest whose estimate
    # (running sums of its multi-layer groups + one FFT unit + sheet crops) fits the cap. Chunks run one after the other and
    # keep only peaks in ``matched``. Trade-off: every chunk recomputes the FFT terms of the units it touches, so more chunks
    # means more FFT recomputation (reported as fft_recomputations).
    chunks = [group_order]
    if limit is not None and not all_at_once:
        chunks, current = [], []
        for key in group_order:
            if current and chunk_estimate(current + [key])['memory_estimate_bytes'] > limit:
                chunks.append(current)
                current = []
            current.append(key)
        chunks.append(current)
    estimates = [chunk_estimate(keys) for keys in chunks]
    over = [e['memory_estimate_bytes'] for e in estimates if limit is not None and e['memory_estimate_bytes'] > limit]
    if over:  # a single group (one accumulator + one layer FFT unit) or the sheet alone does not fit: never truncate
        raise ContractError(f'memory_estimate_exceeds_limit: estimated peak {max(over)} bytes '
                            f'({max(over) / 2 ** 20:.1f} MB) > --max-memory-mb {max_memory_mb:g} even for one query; '
                            'reduce --window-um or the number of layers, or raise the limit')
    estimate = {'memory_estimate_bytes': max(e['memory_estimate_bytes'] for e in estimates),
                'largest_layer_fft_bytes': chunk_estimate(group_order)['largest_layer_fft_bytes']}
    probes = {key: {query_id: preps[query_id]['cells'][::2] for query_id in (groups[key] if dedupe else [key])} for key in group_order} if self_check else {}
    matched = {}
    if all_at_once:  # previous behaviour (every layer resident); kept as the equivalence reference for the tests
        for key in group_order:
            matched[key] = _match_prepared(preps[groups[key][0]] if dedupe else preps[key], candidate_unions, topk, cache, probes.get(key))
    else:
        for keys in chunks:
            units, totals, remaining = {}, {}, {key: len(plans[key]['items']) for key in keys}
            for key in keys:
                for pair, template in plans[key]['items']:
                    units.setdefault((pair, template.shape), []).append((key, template))
                if not remaining[key]:
                    matched[key] = _finish_match(plans[key], None, topk, candidate_unions, probes.get(key))
            for unit in sorted(units):
                pair, template_shape = unit
                image = candidate_unions[pair][1]
                for key, template in units[unit]:
                    score = zncc_map(image, template, cache, pair)
                    if key in totals:
                        totals[key] += score
                    else:
                        totals[key] = score
                    remaining[key] -= 1
                    if not remaining[key]:  # last layer of this group: peaks now, then free its accumulator
                        matched[key] = _finish_match(plans[key], totals.pop(key), topk, candidate_unions, probes.get(key))
                    del score
                if cache is not None:
                    cache.release(pair, template_shape)
    n_matches = len(group_order)
    chunk_units = [{u for key in keys for u in group_units[key]} for keys in chunks]
    results = []
    for roi in rois:
        query_id = roi.row['roi_id']
        base = {'query': {'roi_id': query_id, 'product_id': roi.row['product_id'], 'frame_id': roi.frame.frame_id,
                          'center_xy_um': list(roi.center_xy_um)}}
        prep = preps[query_id]
        if isinstance(prep, ContractError):
            results.append({**base, 'status': 'unavailable', 'reasons': [str(prep)], 'primary': [], 'endpoints_used': [],
                            'template_sha256': None, 'shared_with': [],
                            'flags': {'low_structure': False, 'window_clipped': False, 'query_valid_hole': False,
                                      'candidate_valid_hole': False, 'score_gap': None}, 'query_invalid_pixel_count': 0})
            continue
        group_key = hashes[query_id] if dedupe else query_id
        row = _result_row(roi, prep, matched[group_key], candidate_product)
        _candidate_invalid_counts(row, candidate_invalid)
        if self_check and row['status'] == 'available':  # the value the real accumulated map holds at the query's own window position
            row['self_map'] = {'window_top_left_px_yx': list(prep['cells'][::2]), 'score': matched[group_key]['probe_values'][query_id]}
        shared = [other for other in groups[hashes[query_id]] if other != query_id] if dedupe else []
        results.append({**base, **row, 'query_invalid_pixel_count': prep['invalid_px'], 'template_sha256': hashes[query_id], 'shared_with': shared,
                        'flags': _flags(prep, row, ambiguity_margin)})
    out = {'schema_version': SCHEMA, 'method': METHOD, 'implementation_revision': IMPLEMENTATION, 'mode': 'all_ready', 'window_um': number(window_um, positive=True), 'topk': topk,
           'candidate_product_id': candidate_product, 'n_queries': len(results),
           'status': 'available' if any(r['status'] == 'available' for r in results) else 'unavailable',
           'n_template_matches': n_matches, **estimate, 'chunk_count': len(chunks), 'chunk_sizes': [len(k) for k in chunks],
           'fft_recomputations': sum(len(u) for u in chunk_units) - len(set().union(*chunk_units)),
           'candidate_fft_cache': None if cache is None else cache.stats(),
           'results': results, 'interpretation': 'non_learned_position_baseline_not_similarity_registration_or_risk'}
    if ambiguity_margin is not None:
        out['ambiguity_margin'] = number(ambiguity_margin)
    return out, preps, query_unions, candidate_unions


def run_baseline_batch(query_export, index_export, **kwargs):
    return _run_batch(query_export, index_export, **kwargs)[0]


def self_check_failures(result, query_export):
    """S2: the export is matched against itself. For every available query (a) the accumulated score map the matcher
    really computed, read at the query's own window position (``self_map``: independent of rank, top-k and suppression, so
    periodic repeats cannot hide it), is ~1; (b) that position is the window position derived here from the export's frame and
    the query centre; (c) every emitted rank entry maps back, by the geometry derived here from its map index, to the
    centre it reports; (d) the rank-1 score is not below the map value at the own position. Periodic layouts may rank
    other, equally scoring repeats first; none is chosen here. Returns the failure descriptions; no available query is
    itself a failure."""
    rois = {roi.row['roi_id']: roi for roi in query_export.ready}
    failures, available = [], 0
    half = Fraction(str(result['window_um'])) / 2 if result['results'] else None
    for row in result['results']:
        if row['status'] != 'available':
            continue
        available += 1
        query_id = row['query']['roi_id']
        roi = rois.get(query_id)
        if roi is None:
            failures.append(f'{query_id}: not a ready ROI of the checked export')
            continue
        frame = roi.frame
        pitch = Fraction(str(frame.pixel_pitch_um))
        cx, cy = Fraction(str(roi.center_xy_um[0])), Fraction(str(roi.center_xy_um[1]))
        # own window origin (first pixel column/row of the window, clipped at the frame edge), from the physical centre
        own_c = max(0, math.floor((cx - half - Fraction(str(frame.origin_xy_um[0]))) / pitch))
        own_r = max(0, math.floor((cy - half - Fraction(str(frame.origin_xy_um[1]))) / pitch))
        if [float(cx), float(cy)] != row['query']['center_xy_um']:
            failures.append(f'{query_id}: reported query centre {row["query"]["center_xy_um"]} is not the export centre {[float(cx), float(cy)]}')
            continue
        own = row.get('self_map')
        if own is None or own.get('score') is None:
            failures.append(f'{query_id}: no score map value recorded at the own window position')
        elif own['window_top_left_px_yx'] != [own_r, own_c]:
            failures.append(f'{query_id}: score map was read at {own["window_top_left_px_yx"]}, not at the own window position {[own_r, own_c]}')
        elif not 0.999 <= own['score'] <= 1.0 + 1e-6:
            failures.append(f'{query_id}: score map value {own["score"]:.6f} at the own window position is not ~1')
        elif row['primary'] and row['primary'][0]['score'] < own['score'] - 1e-9:
            failures.append(f'{query_id}: rank-1 score {row["primary"][0]["score"]:.6f} is below the map value {own["score"]:.6f} at the own position')
        for entry in row['primary']:
            r, c = entry['template_top_left_px_yx']
            expected = [float(cx + (c - own_c) * pitch), float(cy + (r - own_r) * pitch)]
            if entry['frame_id'] != frame.frame_id or max(abs(a - b) for a, b in zip(entry['center_xy_um'], expected)) > 1e-9:
                failures.append(f'{query_id}: rank {entry["rank"]} map index {[r, c]} maps to {expected}, but {entry["center_xy_um"]} was reported')
    if not available:
        failures.append('no available query to check')
    return failures


def _sheet_entries(result, preps, query_unions, candidate_unions):
    entries = []
    for row in result['results']:
        query_id = row['query']['roi_id']
        entry = {'roi_id': query_id, 'center_xy_um': row['query']['center_xy_um'], 'status': row['status'], 'reasons': row['reasons'],
                 'score_gap': row['flags']['score_gap'], 'flags': row['flags'], 'shared_with': row['shared_with'],
                 'query_invalid_pixels': row['query_invalid_pixel_count'],
                 'query_crop': None, 'matches': []}
        if row['status'] == 'available':
            prep = preps[query_id]
            templates = [t for _, t in prep['spec'] if t is not None]
            entry['query_crop'] = np.logical_or.reduce([t > 0 for t in templates])
            h, w = row['endpoints_used'][0]['template_shape_yx']
            for match in row['primary']:
                r, c = match['template_top_left_px_yx']
                crops = [candidate_unions[(e['role'], e['actual_layer'])][1][r:r + h, c:c + w] > 0 for e in row['endpoints_used']]
                entry['matches'].append({'rank': match['rank'], 'score': match['score'], 'center_xy_um': match['center_xy_um'],
                                         'invalid_pixels': match['invalid_pixel_count'],
                                         'crop': np.logical_or.reduce(crops)})
        entries.append(entry)
    return entries


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description='ZNCC template-matching baseline for labelled corresponding positions, '
                                                 'or batch mode over every ready query ROI (--all-ready).')
    parser.add_argument('--query-export', required=True, type=Path)
    parser.add_argument('--query-assets-root', type=Path)
    parser.add_argument('--index-export', type=Path)
    parser.add_argument('--index-assets-root', type=Path)
    parser.add_argument('--correspondence', type=Path)
    parser.add_argument('--all-ready', action='store_true', help='query every ready ROI of the query export')
    parser.add_argument('--self-check', action='store_true', help='match the query export against itself; exit 2 unless every query finds itself')
    parser.add_argument('--ambiguity-margin', type=float, help='flag near_equal_peaks when rank 2..K score within this of rank 1 (no default)')
    parser.add_argument('--sheet-dir', type=Path, help='write PNG crops and sheet.html here (batch mode)')
    parser.add_argument('--max-memory-mb', type=float, help='batch mode: exit 2 if the estimated peak memory exceeds this; required with --all-ready and --self-check')
    parser.add_argument('--window-um', required=True, type=float)
    parser.add_argument('--topk', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    args = parser.parse_args(argv)
    try:
        batch = args.all_ready or args.self_check
        if batch == (args.correspondence is not None):
            raise ContractError('exactly_one_of_correspondence_or_all_ready_required')
        if args.index_export is None and not args.self_check:
            raise ContractError('index_export_required')
        if args.self_check and args.index_export is not None:
            raise ContractError('self_check_uses_the_query_export_as_index')
        if (args.sheet_dir or args.ambiguity_margin is not None) and not batch:
            raise ContractError('sheet_dir_and_ambiguity_margin_need_batch_mode')
        if args.max_memory_mb is not None and (not batch or not args.max_memory_mb > 0):
            raise ContractError('max_memory_mb_must_be_positive_and_needs_batch_mode')
        if batch and args.max_memory_mb is None:
            raise ContractError('max_memory_mb_required_with_all_ready' if args.all_ready else 'max_memory_mb_required_with_self_check')
        if args.sheet_dir is not None:
            from .review_sheet import check_sheet_dir
            check_sheet_dir(args.sheet_dir)  # refuse before any heavy work
        index_path = args.query_export if args.self_check else args.index_export
        query = load_export(args.query_export, assets_root=args.query_assets_root, allow_unversioned_producer=args.allow_unversioned_producer)
        if args.self_check:
            index = query
        else:
            index = load_export(index_path, assets_root=args.index_assets_root, allow_unversioned_producer=args.allow_unversioned_producer)
        if not batch:
            result = run_baseline(query, index, read_correspondence(args.correspondence), window_um=args.window_um, topk=args.topk)
        else:
            result, preps, query_unions, candidate_unions = _run_batch(query, index, window_um=args.window_um, topk=args.topk,
                                                                       ambiguity_margin=args.ambiguity_margin,
                                                                       max_memory_mb=args.max_memory_mb, sheet=args.sheet_dir is not None,
                                                                       self_check=args.self_check)
            result['inputs'] = {'query_export_sha256': _sha256(args.query_export), 'index_export_sha256': _sha256(index_path)}
            result['arguments'] = {'window_um': args.window_um, 'topk': args.topk, 'ambiguity_margin': args.ambiguity_margin,
                                   'max_memory_mb': args.max_memory_mb, 'self_check': args.self_check}
            if args.self_check:
                failures = self_check_failures(result, query)
                if failures:
                    raise ContractError('self_check_failed: ' + '; '.join(failures))
                result['self_check'] = {'passed': True}
            if args.sheet_dir is not None:
                from .review_sheet import write_sheet
                write_sheet(args.sheet_dir, _sheet_entries(result, preps, query_unions, candidate_unions),
                            window_um=args.window_um, topk=args.topk)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': result['status'], 'n_queries': result['n_queries']}, allow_nan=False))
        return 0 if result['status'] == 'available' else 3
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
