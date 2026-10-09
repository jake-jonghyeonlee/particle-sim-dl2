"""Label-free alignment/variance/covariance with an optional known-pair term."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .schema import ContractError, number


@dataclass(frozen=True)
class LossConfig:
    alignment_weight: float
    variance_weight: float
    covariance_weight: float
    variance_floor: float
    variance_epsilon: float
    supervised_weight: float
    supervised_temperature: float

    def __post_init__(self):
        for name in ('alignment_weight', 'variance_weight', 'covariance_weight',
                     'variance_floor', 'variance_epsilon', 'supervised_temperature'):
            number(getattr(self, name), positive=True)
        if number(self.supervised_weight) < 0:
            raise ContractError('nonnegative_supervised_weight_required')
        if self.variance_epsilon >= self.variance_floor**2:
            raise ContractError('variance_epsilon_must_not_disable_collapse_penalty')


def relation_matrix(records, roi_ids):
    if len(set(roi_ids)) != len(roi_ids):
        raise ContractError('unique_roi_ids_required_for_labels')
    positions = {rid: i for i, rid in enumerate(roi_ids)}
    labels = torch.full((len(roi_ids), len(roi_ids)), -1, dtype=torch.int64)
    resolved = {}
    for row in records:
        if set(row) != {'a', 'b', 'relation'} or row['relation'] not in ('similar', 'different', 'unknown'):
            raise ContractError('invalid_relation_record')
        if row['a'] == row['b']:
            raise ContractError('self_relation_not_allowed')
        pair = tuple(sorted([row['a'], row['b']]))
        if row['relation'] != 'unknown':
            if pair in resolved and resolved[pair] != row['relation']:
                raise ContractError('conflicting_relation_labels')
            resolved[pair] = row['relation']
    for (a, b), relation in resolved.items():
        if a in positions and b in positions:
            i, j = positions[a], positions[b]
            labels[i, j] = labels[j, i] = int(relation == 'similar')
    return labels


def validate_label_ids(records, export_ids):
    absent = sorted({row[key] for row in records for key in ('a', 'b')} - set(export_ids))
    if absent:
        raise ContractError('label_roi_ids_absent_from_export: '+','.join(absent))


def read_labels(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) != {'schema_version', 'relations'} or raw['schema_version'] != 'roi_relations_v1' or not isinstance(raw['relations'], list):
        raise ContractError('unknown_relation_schema')
    ids = sorted({row[k] for row in raw['relations'] for k in ('a', 'b')})
    relation_matrix(raw['relations'], ids)  # reject conflicts even outside this batch
    return raw['relations']


def supervised_term(logits, relations):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or logits.shape != relations.shape:
        raise ContractError('square_relation_logits_required')
    if not torch.isfinite(logits).all() or not torch.equal(relations, relations.T) or torch.any(
            (relations != -1) & (relations != 0) & (relations != 1)) or torch.any(relations.diag() != -1):
        raise ContractError('invalid_relation_matrix_or_logits')
    # Unknown pairs never enter BCE's numerator or denominator.
    known = torch.triu(relations != -1, diagonal=1)
    count = int(known.sum())
    value = F.binary_cross_entropy_with_logits(logits[known], relations[known].to(logits.dtype)) if count else logits.sum()*0
    return value, count


def objective(first, second, config, *, relations=None):
    if first.ndim != 2 or first.shape != second.shape or first.shape[1] < 2:
        raise ContractError('matching_projection_matrices_required')
    if first.shape[0] < 2:
        raise ContractError('effective_batch_at_least_two_required')
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ContractError('finite_projections_required')
    # The actual alignment pairing graph is the diagonal location correspondence.
    # Count cross-location edges from that graph; no negative-edge graph is built.
    alignment_pairs = torch.eye(len(first), dtype=torch.bool, device=first.device)
    cross_location_pairs = alignment_pairs & ~torch.eye(len(first), dtype=torch.bool, device=first.device)
    pair_indices = alignment_pairs.nonzero()
    per_location = (first[pair_indices[:, 0]]-second[pair_indices[:, 1]]).square().mean(dim=1)
    alignment = per_location.mean()
    variances, covariances = [], []
    for tensor in (first, second):
        centered = tensor-tensor.mean(dim=0)
        covariance = centered.T @ centered/(len(tensor)-1)
        std = torch.sqrt(covariance.diag()+config.variance_epsilon)
        variances.append(torch.relu(config.variance_floor-std).mean())
        off_diagonal = covariance-torch.diag(covariance.diag())
        covariances.append(off_diagonal.square().sum()/tensor.shape[1])
    variance = sum(variances)/2
    covariance = sum(covariances)/2
    supervised, known_count, different_count = first.sum()*0, 0, 0
    if relations is not None:
        combined = F.normalize((first+second)/2, dim=1)
        logits = combined @ combined.T / config.supervised_temperature
        supervised, known_count = supervised_term(logits, relations.to(first.device))
        different_count = int(torch.triu(relations == 0, diagonal=1).sum())
    total = (config.alignment_weight*alignment + config.variance_weight*variance
             + config.covariance_weight*covariance + config.supervised_weight*supervised)
    return {'total': total, 'alignment': alignment, 'variance': variance, 'covariance': covariance,
            'supervised': supervised, 'alignment_per_location': per_location,
            'positive_pair_count': int(alignment_pairs.sum()), 'negative_pair_count': int(cross_location_pairs.sum()),
            'supervised_different_pair_count': different_count,
            'alignment_pair_indices': pair_indices.tolist(),
            'supervised_known_pair_count': known_count}
