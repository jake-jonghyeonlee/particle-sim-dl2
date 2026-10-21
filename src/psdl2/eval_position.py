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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Position hit rate of DL2 search or the NCC baseline against labelled correspondences.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--search', nargs='+', type=Path, help='psdl2.search output JSON files')
    group.add_argument('--baseline', type=Path, help='psdl2.baseline_ncc output JSON')
    parser.add_argument('--correspondence', required=True, type=Path)
    parser.add_argument('--hit-tolerance-um', required=True, type=float)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        truths = read_correspondence(args.correspondence)
        if args.search:
            predictions = predictions_from_search([json.loads(p.read_text(encoding='utf-8')) for p in args.search])
        else:
            predictions = predictions_from_baseline(json.loads(args.baseline.read_text(encoding='utf-8')))
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
