"""Split-restricted optional-label evaluation with explicit index coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .losses import read_labels, relation_matrix
from .schema import ContractError


def evaluate(results, *, k, assignments, partition, labels=None, index_roi_ids=None):
    if type(k) is not int or k < 1:
        raise ContractError('positive_recall_k_required')
    if partition not in ('train', 'validation', 'test') or not isinstance(assignments, dict) or any(
            value not in ('train', 'validation', 'test') for value in assignments.values()):
        raise ContractError('explicit_valid_evaluation_split_required')
    selected, excluded = [], []
    for result in results:
        rid = result['query']['roi_id']
        if rid not in assignments:
            raise ContractError('evaluation_query_absent_from_split: '+rid)
        if assignments[rid] == partition:
            selected.append(result)
        else:
            excluded.append(rid)
    common = {'partition': partition, 'excluded_queries': excluded,
              'n_selected_queries': len(selected), 'scope': 'primary_only'}
    if labels is None:
        return {**common, 'status': 'unavailable', 'reason': 'correspondence_labels_not_supplied',
                'recall_at_k': None, 'counterexample_retrievals': None, 'n_evaluable_queries': 0,
                'positives_missing_from_index': None}
    ids = sorted({r[field] for r in labels for field in ('a', 'b')})
    relation_matrix(labels, ids)
    positives, negatives = {}, {}
    for relation in labels:
        target = positives if relation['relation'] == 'similar' else negatives if relation['relation'] == 'different' else None
        if target is not None:
            target.setdefault(relation['a'], set()).add(relation['b'])
            target.setdefault(relation['b'], set()).add(relation['a'])
    recalls, details, missing_records, seen = [], [], [], set()
    for result in selected:
        rid = result['query']['roi_id']
        if rid in seen:
            raise ContractError('duplicate_evaluation_query')
        seen.add(rid)
        available = index_roi_ids if index_roi_ids is not None else result.get('index_roi_ids')
        if available is None:
            raise ContractError('index_roi_ids_required_for_positive_coverage')
        available = set(available)
        retrieved = {r['roi_id'] for r in result['primary'][:k]}
        truth = positives.get(rid, set())
        missing = truth - available
        eligible = truth & available
        recall = len(eligible & retrieved)/len(eligible) if eligible else None
        if recall is not None:
            recalls.append(recall)
        missing_records.append({'roi_id': rid, 'missing_positive_roi_ids': sorted(missing)})
        details.append({'roi_id': rid, 'recall': recall,
                        'known_different_retrieved': sorted(retrieved & negatives.get(rid, set()))})
    return {**common, 'status': 'available' if recalls else 'unavailable',
            'reason': None if recalls else 'no_indexed_known_positive_query_labels',
            'recall_at_k': sum(recalls)/len(recalls) if recalls else None,
            'k': k, 'n_evaluable_queries': len(recalls), 'counterexample_retrievals': details,
            'positives_missing_from_index': missing_records,
            'recall_denominator': 'known_positives_present_in_index',
            'performance_status': 'evaluation_of_supplied_labels_and_partition_only'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate offline primary retrieval Recall@k in an explicit split partition.')
    parser.add_argument('results', type=Path, help='JSON list of search-result objects')
    parser.add_argument('--k', required=True, type=int); parser.add_argument('--labels', type=Path)
    parser.add_argument('--split', required=True, type=Path)
    parser.add_argument('--partition', required=True, choices=['train', 'validation', 'test'])
    parser.add_argument('--index', type=Path, help='Optional explicit index for older result files lacking index_roi_ids')
    args = parser.parse_args(argv)
    try:
        result = evaluate(json.loads(args.results.read_text()), k=args.k,
                          assignments=json.loads(args.split.read_text()), partition=args.partition,
                          labels=None if args.labels is None else read_labels(args.labels),
                          index_roi_ids=None if args.index is None else [r['roi_id'] for r in json.loads(args.index.read_text())['records']])
        print(json.dumps(result, allow_nan=False)); return 0 if result['status'] == 'available' else 3
    except (ContractError, KeyError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
