"""Optional label-based retrieval evaluation; absence of labels is unavailable."""
import argparse
import json
from pathlib import Path
import sys

from .losses import read_labels, relation_matrix
from .schema import ContractError


def evaluate(results, *, k, labels=None):
    if type(k) is not int or k < 1:
        raise ContractError('positive_recall_k_required')
    if labels is None:
        return {'status': 'unavailable', 'reason': 'correspondence_labels_not_supplied',
                'recall_at_k': None, 'counterexample_retrievals': None, 'n_evaluable_queries': 0}
    ids = sorted({r[field] for r in labels for field in ('a', 'b')})
    relation_matrix(labels, ids)
    positives, negatives = {}, {}
    for relation in labels:
        target = positives if relation['relation'] == 'similar' else negatives if relation['relation'] == 'different' else None
        if target is not None:
            target.setdefault(relation['a'], set()).add(relation['b'])
            target.setdefault(relation['b'], set()).add(relation['a'])
    recalls, details, seen = [], [], set()
    for result in results:
        rid = result['query']['roi_id']
        if rid in seen:
            raise ContractError('duplicate_evaluation_query')
        seen.add(rid)
        retrieved = {r['roi_id'] for r in result['primary'][:k]}
        truth = positives.get(rid, set())
        if truth:
            recall = len(truth & retrieved)/len(truth); recalls.append(recall)
        else:
            recall = None
        details.append({'roi_id': rid, 'recall': recall,
                        'known_different_retrieved': sorted(retrieved & negatives.get(rid, set()))})
    return {'status': 'available' if recalls else 'unavailable',
            'reason': None if recalls else 'no_known_positive_query_labels',
            'recall_at_k': sum(recalls)/len(recalls) if recalls else None,
            'k': k, 'n_evaluable_queries': len(recalls), 'counterexample_retrievals': details,
            'scope': 'primary_only', 'performance_status': 'evaluation_of_supplied_labels_only'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate offline primary retrieval Recall@k when labels are supplied.')
    parser.add_argument('results', type=Path, help='JSON list of search-result objects')
    parser.add_argument('--k', required=True, type=int); parser.add_argument('--labels', type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate(json.loads(args.results.read_text()), k=args.k,
                          labels=None if args.labels is None else read_labels(args.labels))
        print(json.dumps(result, allow_nan=False)); return 0 if result['status'] == 'available' else 3
    except (ContractError, KeyError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
