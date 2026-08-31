import torch

from dcg_sign4d.contact.balanced_sampler import (
    balanced_window_sample_weights,
    effective_number_weights,
)
from dcg_sign4d.contact.losses import balanced_event_loss


def test_rare_classes_receive_more_weight():
    weights = effective_number_weights(torch.tensor([1000, 10, 1, 0]))
    assert weights[2] > weights[1] > weights[0] > weights[3]


def test_uncertain_labels_have_exactly_zero_gradient():
    logits = torch.randn(1, 2, 1, 4, requires_grad=True)
    labels = torch.tensor([[[0], [2]]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    uncertain = torch.tensor([[[False], [True]]])
    loss = balanced_event_loss(logits, labels, valid, uncertain, torch.ones(4))
    loss.backward()
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[0, 0].abs().sum() > 0


def test_all_uncertain_batch_is_finite_zero():
    logits = torch.randn(1, 2, 1, 4, requires_grad=True)
    labels = torch.zeros(1, 2, 1, dtype=torch.long)
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss = balanced_event_loss(logits, labels, mask, mask, torch.ones(4))
    assert loss == 0
    loss.backward()
    assert logits.grad.abs().sum() == 0


def test_rare_edge_event_window_receives_higher_sampling_weight():
    labels = torch.zeros(3, 4, 2, dtype=torch.long)
    labels[2, 0, 1] = 2
    edge_valid = torch.ones(3, 2, dtype=torch.bool)
    frame_valid = torch.ones(3, 4, dtype=torch.bool)
    uncertain = torch.zeros_like(labels, dtype=torch.bool)
    weights = balanced_window_sample_weights(labels, edge_valid, frame_valid, uncertain, beta=0.9)
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]
