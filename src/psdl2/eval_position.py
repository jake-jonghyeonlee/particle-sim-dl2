"""Position-based evaluation of retrieval against labelled corresponding positions.

Truth comes from a ``roi_correspondence_v1`` file written by particle-sim-compare:
side ``a`` names the query ROI (``roi_id``) and side ``b`` the labelled position in
the candidate product. A query is a hit when a returned candidate centre lies within
``--hit-tolerance-um`` of a labelled centre in the same product and frame. The same
metric is applied to ``psdl2.search`` output and to ``psdl2.baseline_ncc`` output so
the learned model and the non-learned baseline are compared on identical terms.
This measures position recovery with tens of labels; it is not a statistical
performance claim, and no producer module is imported (JSON only).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from .schema import ContractError, number, text

CORRESPONDENCE_SCHEMA = 'roi_correspondence_v1'
SCHEMA = 'psdl2_position_eval_v1'


def read_correspondence(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.get('schema_version') != CORRESPONDENCE_SCHEMA or not isinstance(payload.get('pairs'), list):
        raise ContractError('unsupported_correspondence_schema')
    truths = {}
    for pair in payload['pairs']:
        if pair.get('status') != 'ready':
            continue
        a, b = pair['a'], pair['b']
        query = a.get('roi_id')
        if query is None:
            continue  # unlabelled query side: cannot be joined to a search result
        truth = {'pair_id': text(pair['pair_id']), 'product_id': text(b['product_id']), 'frame_id': text(b['frame_id']),
                 'center_xy_um': [number(v) for v in b['center_xy_um']], 'roi_id': b.get('roi_id')}
        truths.setdefault(text(query), []).append(truth)
    return truths


def predictions_from_search(results):
    predictions = []
    for result in results:
        if not isinstance(result, dict) or 'query' not in result or 'primary' not in result:
            raise ContractError('unsupported_search_result')
        candidates = []
        for rank, row in enumerate(result['primary'], 1):
            candidates.append({'rank': rank, 'roi_id': row.get('roi_id'), 'product_id': row['product_id'],
                               'frame_id': row['frame']['frame_id'], 'center_xy_um': [number(v) for v in row['center_xy_um']],
                               'score': row.get('score')})
        predictions.append({'query_roi_id': text(result['query']['roi_id']), 'status': result.get('status'),
                            'candidates': candidates, 'source': 'dl2_search'})
    return predictions


def predictions_from_baseline(payload):
    if not isinstance(payload, dict) or payload.get('schema_version') != 'psdl2_baseline_ncc_v1':
        raise ContractError('unsupported_baseline_result')
    predictions = []
    for result in payload['results']:
        candidates = [{'rank': row['rank'], 'roi_id': None, 'product_id': row['product_id'], 'frame_id': row['frame_id'],
                       'center_xy_um': [number(v) for v in row['center_xy_um']], 'score': row['score']}
                      for row in result.get('primary', [])]
        predictions.append({'query_roi_id': text(result['query']['roi_id']), 'status': result.get('status'),
                            'candidates': candidates, 'source': 'baseline_ncc'})
    return predictions


def evaluate_positions(predictions, truths, *, tolerance_um, k):
    tolerance = number(tolerance_um, positive=True)
    if type(k) is not int or k < 1:
        raise ContractError('positive_k_required')
    rows, top1_hits, topk_hits, top1_distances, seen, duplicates = [], 0, 0, [], {}, 0
    for prediction in predictions:
        query = prediction['query_roi_id']
        if query in seen:
            # an identical repeat is dropped and counted; a conflicting repeat is a contract error
            if prediction['candidates'] != seen[query]:
                raise ContractError('duplicate_query_prediction_conflict: ' + query)
            duplicates += 1
            continue
        seen[query] = prediction['candidates']
        labelled = truths.get(query, [])
        row = {'query_roi_id': query, 'source': prediction['source'], 'n_labelled_positions': len(labelled),
               'n_candidates': len(prediction['candidates']), 'status': 'available', 'reasons': [],
               'top1_distance_um': None, 'top1_hit': None, 'first_hit_rank': None, 'hit_at_k': None, 'candidates': []}
        if not labelled:
            row.update(status='unavailable', reasons=['no_labelled_position_for_query'])
        elif not prediction['candidates']:
            row.update(status='unavailable', reasons=['no_candidates'])
        else:
            first_hit = None
            for candidate in prediction['candidates'][:k]:
                distances = [math.dist(candidate['center_xy_um'], t['center_xy_um']) for t in labelled
                             if t['product_id'] == candidate['product_id'] and t['frame_id'] == candidate['frame_id']]
                distance = min(distances) if distances else None
                hit = distance is not None and distance <= tolerance
                row['candidates'].append({'rank': candidate['rank'], 'roi_id': candidate['roi_id'], 'score': candidate.get('score'),
                                          'center_xy_um': candidate['center_xy_um'], 'distance_um': distance,
                                          'hit': hit, 'reason': None if distances else 'product_or_frame_mismatch'})
                if hit and first_hit is None:
                    first_hit = candidate['rank']
            top1 = row['candidates'][0]
            row.update(top1_distance_um=top1['distance_um'], top1_hit=top1['hit'], first_hit_rank=first_hit,
                       hit_at_k=first_hit is not None)
            top1_hits += top1['hit']
            topk_hits += first_hit is not None
            if top1['distance_um'] is not None:
                top1_distances.append(top1['distance_um'])
        rows.append(row)
    evaluable = [r for r in rows if r['status'] == 'available']
    n = len(evaluable)
    ordered = sorted(top1_distances)
    median = None
    if ordered:
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    summary = {'n_queries': len(rows), 'n_evaluable': n, 'n_unavailable': len(rows) - n,
               'n_duplicate_predictions_ignored': duplicates,
               'top1_hit_rate': top1_hits / n if n else None, f'hit_rate_at_{k}': topk_hits / n if n else None,
               'mean_top1_distance_um': sum(top1_distances) / len(top1_distances) if top1_distances else None,
               'median_top1_distance_um': median}
    return {'schema_version': SCHEMA, 'hit_tolerance_um': tolerance, 'k': k,
            'status': 'available' if n else 'unavailable', 'reasons': [] if n else ['no_evaluable_queries'],
            'summary': summary, 'queries': rows,
            'interpretation': 'position_recovery_within_tolerance_not_similarity_quality_or_risk'}


# ---------------------------------------------------------------- multi-product mode (psdl2_eval_position_multi_v1)
#
# The unit is the query. Truth is the list of (product, frame, position) labels of that query. A query is judged per
# truth product P by the prediction that covers P: with a batch envelope that is the query's whole ranking, with several
# baseline files it is the (query, P) prediction of the file whose candidate product is P. The query is a hit when any
# truth product is a hit. NCC scores and cosine scores are never compared or re-ranked across products.

MULTI_SCHEMA = 'psdl2_eval_position_multi_v1'
BATCH_SCHEMA = 'psdl2_search_batch_v1'


def predictions_from_batch(payload, *, include_supplementary=False):
    """One prediction per query: the whole ``rankings[i].candidates`` list (primary only unless asked otherwise)."""
    if not isinstance(payload, dict) or payload.get('schema_version') != BATCH_SCHEMA or not isinstance(payload.get('rankings'), list):
        raise ContractError('unsupported_search_batch')
    predictions = []
    for ranking in payload['rankings']:
        if any(c['search_scope'] not in ('primary', 'supplementary') for c in ranking['candidates']):
            raise ContractError('unsupported_search_scope')
        kept = [c for c in ranking['candidates'] if include_supplementary or c['search_scope'] == 'primary']
        candidates = [{'rank': rank, 'roi_id': c.get('roi_id'), 'product_id': c['product_id'], 'frame_id': c['frame_id'],
                       'center_xy_um': [number(v) for v in c['center_xy_um']], 'score': None}
                      for rank, c in enumerate(kept, 1)]
        predictions.append({'query_roi_id': text(ranking['query_roi_id']), 'candidate_product_id': None,
                            'candidates': candidates, 'source': 'dl2_search_batch'})
    return predictions


def predictions_from_baselines(payloads):
    """One prediction per (query, candidate product); the product of a file is its ``candidate_product_id``.

    A repeat of (query, product) inside one file follows the single rule of ``evaluate_multi`` (identical = ignored and
    counted, different = conflict); the same pair in two different files is ``duplicate_candidate_product``."""
    predictions, owner = [], {}
    for number, payload in enumerate(payloads):
        if not isinstance(payload, dict) or payload.get('schema_version') != 'psdl2_baseline_ncc_v1':
            raise ContractError('unsupported_baseline_result')
        product = text(payload.get('candidate_product_id'))
        for prediction in predictions_from_baseline(payload):
            if any(c['product_id'] != product for c in prediction['candidates']):
                raise ContractError('candidate_product_mismatch: %s / %s' % (prediction['query_roi_id'], product))
            key = (prediction['query_roi_id'], product)
            if owner.setdefault(key, number) != number:
                raise ContractError('duplicate_candidate_product: %s / %s' % key)
            prediction['candidate_product_id'] = product
            predictions.append(prediction)
    return predictions


def _judge(candidates, labelled, tolerance, k):
    """The per-prediction hit judgement of ``evaluate_positions`` (same distance rule, same top-k cut)."""
    rows, first_hit = [], None
    for candidate in candidates[:k]:
        distances = [math.dist(candidate['center_xy_um'], t['center_xy_um']) for t in labelled
                     if t['product_id'] == candidate['product_id'] and t['frame_id'] == candidate['frame_id']]
        distance = min(distances) if distances else None
        hit = distance is not None and distance <= tolerance
        rows.append({'rank': candidate['rank'], 'roi_id': candidate['roi_id'], 'product_id': candidate['product_id'],
                     'score': candidate.get('score'), 'center_xy_um': candidate['center_xy_um'], 'distance_um': distance,
                     'hit': hit, 'reason': None if distances else 'product_or_frame_mismatch'})
        if hit and first_hit is None:
            first_hit = candidate['rank']
    return rows, first_hit


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def evaluate_multi(predictions, truths, *, tolerance_um, k, baseline_products=None):
    """``baseline_products``: the candidate products that have a baseline file (None for a batch envelope)."""
    tolerance = number(tolerance_um, positive=True)
    if type(k) is not int or k < 1:
        raise ContractError('positive_k_required')
    by_key, ignored, seen_queries = {}, 0, {}
    for prediction in predictions:
        key = (prediction['query_roi_id'], prediction['candidate_product_id'])
        if key in by_key:
            if prediction['candidates'] != by_key[key]['candidates']:
                raise ContractError('duplicate_query_prediction_conflict: ' + prediction['query_roi_id'])
            ignored += 1
            continue
        by_key[key] = prediction
        seen_queries[prediction['query_roi_id']] = True
    rows, missing_files = [], set()
    per_product = {}
    for query in sorted(truths):
        labelled_by_product = {}
        for truth in truths[query]:
            labelled_by_product.setdefault(truth['product_id'], []).append(truth)
        row = {'query_roi_id': query, 'truth_products': sorted(labelled_by_product), 'status': 'unavailable', 'reasons': [],
               'top1_distance_um': None, 'top1_hit': False, 'first_hit_rank': None, 'hit_at_k': False, 'by_product': {}}
        ranks, distances = [], []
        for product in sorted(labelled_by_product):
            if baseline_products is None:
                prediction = by_key.get((query, None))
            elif product not in baseline_products:
                prediction = None
                missing_files.add(product)
            else:
                prediction = by_key.get((query, product))
            entry = {'status': 'unavailable', 'reasons': [], 'n_candidates': 0, 'top1_distance_um': None, 'top1_hit': False,
                     'first_hit_rank': None, 'hit_at_k': False, 'candidates': []}
            if baseline_products is not None and product not in baseline_products:
                entry['reasons'] = ['baseline_missing_for_truth_product']
            elif prediction is None:
                entry['reasons'] = ['no_prediction']
            elif not prediction['candidates']:
                entry['reasons'] = ['no_candidates']
            else:
                judged, first_hit = _judge(prediction['candidates'], labelled_by_product[product], tolerance, k)
                entry.update(status='available', n_candidates=len(prediction['candidates']), candidates=judged,
                             top1_distance_um=judged[0]['distance_um'], top1_hit=judged[0]['hit'], first_hit_rank=first_hit,
                             hit_at_k=first_hit is not None)
            row['by_product'][product] = entry
            stats = per_product.setdefault(product, {'n_queries_with_truth': 0, 'n_queries_with_prediction': 0,
                                                      'n_top1_hit': 0, 'n_hit_at_k': 0})
            stats['n_queries_with_truth'] += 1
            if entry['status'] == 'available':
                stats['n_queries_with_prediction'] += 1
                stats['n_top1_hit'] += entry['top1_hit']
                stats['n_hit_at_k'] += entry['hit_at_k']
                if entry['first_hit_rank'] is not None:
                    ranks.append(entry['first_hit_rank'])
                if entry['top1_distance_um'] is not None:
                    distances.append(entry['top1_distance_um'])
            else:
                row['reasons'] += ['%s:%s' % (product, reason) for reason in entry['reasons']]
        available = [e for e in row['by_product'].values() if e['status'] == 'available']
        if available:
            row.update(status='available', reasons=[], top1_hit=any(e['top1_hit'] for e in available),
                       first_hit_rank=min(ranks) if ranks else None, hit_at_k=bool(ranks),
                       top1_distance_um=min(distances) if distances else None)
        rows.append(row)
    n_truth = len(rows)
    predicted = [r for r in rows if r['status'] == 'available']
    n_pred = len(predicted)
    top1_hits, topk_hits = sum(r['top1_hit'] for r in predicted), sum(r['hit_at_k'] for r in predicted)
    top1_distances = [r['top1_distance_um'] for r in predicted if r['top1_distance_um'] is not None]
    for stats in per_product.values():
        n = stats['n_queries_with_truth']
        stats.update(top1_hit_rate=stats['n_top1_hit'] / n, **{'hit_rate_at_%d' % k: stats['n_hit_at_k'] / n})
    summary = {'n_queries_with_truth': n_truth, 'n_queries_with_prediction': n_pred,
               'n_queries_without_prediction': n_truth - n_pred,
               'n_prediction_queries_without_truth': len([q for q in seen_queries if q not in truths]),
               'n_duplicate_predictions_ignored': ignored,
               'n_top1_hit': top1_hits, 'n_hit_at_k': topk_hits,
               'top1_hit_rate': top1_hits / n_truth if n_truth else None, 'hit_rate_at_%d' % k: topk_hits / n_truth if n_truth else None,
               'top1_hit_rate_among_predicted': top1_hits / n_pred if n_pred else None,
               'hit_rate_at_%d_among_predicted' % k: topk_hits / n_pred if n_pred else None,
               'mean_top1_distance_um': sum(top1_distances) / len(top1_distances) if top1_distances else None,
               'median_top1_distance_um': _median(top1_distances),
               'baseline_missing_for_truth_product': sorted(missing_files)}
    reasons = [] if n_pred else (['no_evaluable_queries'] if n_truth else ['no_queries_with_truth'])
    return {'schema_version': MULTI_SCHEMA, 'hit_tolerance_um': tolerance, 'k': k,
            'status': 'available' if n_pred else 'unavailable', 'reasons': reasons, 'summary': summary,
            'per_product': {p: per_product[p] for p in sorted(per_product)}, 'queries': rows,
            'interpretation': 'position_recovery_within_tolerance_not_similarity_quality_or_risk'}


def _write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp%d' % os.getpid())
    try:
        temp.write_text(content, encoding='utf-8')
        os.replace(str(temp), str(path))
    finally:
        if temp.exists():
            temp.unlink()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Position hit rate of DL2 search or the NCC baseline against labelled correspondences.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--search', nargs='+', type=Path, help='psdl2.search output JSON files')
    group.add_argument('--search-batch', type=Path, help='psdl2_search_batch_v1 envelope (psdl2.search --all-ready)')
    group.add_argument('--baseline', nargs='+', type=Path,
                       help='psdl2.baseline_ncc output JSON; one file = single mode, two or more = one file per candidate product')
    parser.add_argument('--include-supplementary', action='store_true',
                        help='--search-batch only: also count supplementary candidates (default: primary only)')
    parser.add_argument('--correspondence', required=True, type=Path)
    parser.add_argument('--hit-tolerance-um', required=True, type=float)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.include_supplementary and args.search_batch is None:
            raise ContractError('include_supplementary_requires_search_batch')
        truths = read_correspondence(args.correspondence)
        if args.search_batch is not None or len(args.baseline or ()) > 1:
            if args.search_batch is not None:
                predictions = predictions_from_batch(_load_json(args.search_batch), include_supplementary=args.include_supplementary)
                baseline_products = None
                arguments = {'mode': 'search_batch', 'search_batch': str(args.search_batch),
                             'include_supplementary': bool(args.include_supplementary)}
            else:
                payloads = [_load_json(p) for p in args.baseline]
                predictions = predictions_from_baselines(payloads)
                baseline_products = {p['candidate_product_id'] for p in payloads}
                arguments = {'mode': 'multi_baseline', 'baselines': [str(p) for p in args.baseline]}
            result = evaluate_multi(predictions, truths, tolerance_um=args.hit_tolerance_um, k=args.k, baseline_products=baseline_products)
            arguments.update(correspondence=str(args.correspondence), hit_tolerance_um=args.hit_tolerance_um, k=args.k)
            result = {'schema_version': result['schema_version'], 'arguments': arguments,
                      **{key: value for key, value in result.items() if key != 'schema_version'}}
            _write_atomic(args.out, json.dumps(result, allow_nan=False, indent=2) + '\n')
            print(json.dumps({'status': result['status'], 'summary': result['summary']}, allow_nan=False))
            return 0 if result['status'] == 'available' else 3
        if args.search:
            predictions = predictions_from_search([json.loads(p.read_text(encoding='utf-8')) for p in args.search])
        else:
            predictions = predictions_from_baseline(json.loads(args.baseline[0].read_text(encoding='utf-8')))
        result = evaluate_positions(predictions, truths, tolerance_um=args.hit_tolerance_um, k=args.k)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': result['status'], 'summary': result['summary']}, allow_nan=False))
        return 0 if result['status'] == 'available' else 3
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
