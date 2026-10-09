import math
import pytest
import torch

from psdl2.losses import LossConfig, objective, relation_matrix, supervised_term
from psdl2.schema import ContractError


def loss_config():
    return LossConfig(1, 1, 1, 2, .25, 1, 1)


def test_tiny_loss_terms_hand_computed_with_gradients():
    a = torch.tensor([[0., 0.], [2., 2.]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[1., 0.], [3., 2.]], dtype=torch.float64, requires_grad=True)
    result = objective(a, b, loss_config())
    # Alignment: (1^2+0^2)/2 = .5 per row. Sample var=2; std=sqrt(2+.25)=1.5.
    # Off-diagonal cov=2, so (2^2+2^2)/2 = 4.
    assert result['alignment'].item() == .5
    assert result['variance'].item() == .5
    assert result['covariance'].item() == 4
    assert result['supervised'].item() == 0
    assert result['total'].item() == 5
    assert result['negative_pair_count'] == 0 and result['positive_pair_count'] == 2
    result['total'].backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_collapse_and_effective_batch():
    a = torch.ones((2, 2), requires_grad=True)
    result = objective(a, a, loss_config())
    assert result['alignment'].item() == 0
    assert result['variance'].item() == 1.5
    result['total'].backward(); assert torch.isfinite(a.grad).all()
    with pytest.raises(ContractError, match='effective_batch'):
        objective(a[:1], a[:1], loss_config())


def test_identical_locations_have_no_cross_location_repulsion():
    a = torch.tensor([[1., 1.], [1., 1.]])
    b = a.clone()
    before = objective(a, b, loss_config())
    a[1] = torch.tensor([8., -2.]); b[1] = torch.tensor([-3., 4.])
    after = objective(a, b, loss_config())
    assert before['alignment_per_location'][0] == after['alignment_per_location'][0] == 0
    assert after['negative_pair_count'] == 0
    assert after['alignment_per_location'][1] == 78.5  # (11² + (-6)²)/2


def test_optional_supervised_hand_value_and_unknown_logit_invariance():
    labels = torch.tensor([[-1, 1, -1], [1, -1, 0], [-1, 0, -1]])
    logits = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    term, count = supervised_term(logits, labels)
    assert term.item() == pytest.approx(math.log(2)) and count == 2
    modified = logits.detach().clone(); modified[0, 2] = modified[2, 0] = 999
    assert supervised_term(modified, labels)[0].item() == term.item()
    term.backward(); assert logits.grad[0, 2] == logits.grad[2, 0] == 0
    a = torch.ones((2, 2), dtype=torch.float64)
    relations = torch.tensor([[-1, 1], [1, -1]])
    assert objective(a, a, loss_config(), relations=relations)['supervised'].item() == pytest.approx(math.log1p(math.exp(-1)))


def test_conflicting_labels_unknown_excluded_and_invalid_config():
    with pytest.raises(ContractError, match='conflicting'):
        relation_matrix([{'a': 'A', 'b': 'B', 'relation': 'similar'},
                         {'a': 'B', 'b': 'A', 'relation': 'different'}], ['A', 'B'])
    assert torch.equal(relation_matrix([{'a': 'A', 'b': 'B', 'relation': 'unknown'}], ['A', 'B']),
                       torch.full((2, 2), -1))
    with pytest.raises(ContractError): LossConfig(1, 0, 1, 2, .25, 0, 1)
