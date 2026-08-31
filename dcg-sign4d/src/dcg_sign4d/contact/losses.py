"""Masked class-balanced contact proposal objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor

from .balanced_sampler import effective_number_weights
from .ontology import VALID_FRAME_TRANSITIONS


def balanced_event_loss(
    logits: Tensor,
    labels: Tensor,
    valid: Tensor,
    uncertain: Tensor,
    class_counts: Tensor,
    beta: float = 0.999,
    sample_weight: Tensor | None = None,
) -> Tensor:
    active = valid & ~uncertain
    if not bool(active.any()):
        return logits.sum() * 0
    weights = effective_number_weights(class_counts.to(logits.device), beta)
    loss = functional.cross_entropy(
        logits[active], labels[active], weight=weights, reduction="none"
    )
    if sample_weight is None:
        return loss.mean()
    if sample_weight.shape != (logits.shape[0],):
        raise ValueError("sample_weight must be [B]")
    expanded = sample_weight[:, None, None].expand_as(labels)[active].to(loss)
    return (loss * expanded).sum() / expanded.sum().clamp_min(1e-8)


def invalid_transition_loss(
    logits: Tensor,
    frame_valid: Tensor,
    edge_valid: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    probabilities = logits.softmax(dim=-1)
    pair = probabilities[:, :-1, :, :, None] * probabilities[:, 1:, :, None, :]
    invalid = ~VALID_FRAME_TRANSITIONS.to(logits.device)
    valid_pair = frame_valid[:, :-1] & frame_valid[:, 1:]
    if edge_valid is None:
        edge_valid = torch.ones(
            logits.shape[0], logits.shape[2], dtype=torch.bool, device=logits.device
        )
    if edge_valid.shape != (logits.shape[0], logits.shape[2]) or edge_valid.dtype != torch.bool:
        raise ValueError("edge_valid must be bool [B,E]")
    if not bool(valid_pair.any()):
        return logits.sum() * 0
    penalty = pair[..., invalid].sum(dim=-1)
    active = valid_pair[:, :, None] & edge_valid[:, None, :]
    if not bool(active.any()):
        return logits.sum() * 0
    values = penalty[active]
    if sample_weight is None:
        return values.mean()
    if sample_weight.shape != (logits.shape[0],):
        raise ValueError("sample_weight must be [B]")
    expanded = sample_weight[:, None, None].expand_as(penalty)[active].to(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1e-8)


def masked_event_brier_loss(
    logits: Tensor,
    labels: Tensor,
    valid: Tensor,
    uncertain: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Differentiable multiclass calibration objective on certain valid events."""

    if labels.shape != logits.shape[:-1] or valid.shape != labels.shape:
        raise ValueError("event calibration tensors have incompatible shapes")
    active = valid & ~uncertain
    if not bool(active.any()):
        return logits.sum() * 0
    target = functional.one_hot(labels[active], num_classes=logits.shape[-1]).to(logits)
    values = (logits[active].softmax(-1) - target).square().sum(-1)
    if sample_weight is None:
        return values.mean()
    if sample_weight.shape != (logits.shape[0],):
        raise ValueError("sample_weight must be [B]")
    expanded = sample_weight[:, None, None].expand_as(labels)[active].to(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1e-8)


def masked_duration_loss(
    logits: Tensor,
    duration_frames: Tensor,
    valid: Tensor,
    uncertain: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Cross-entropy over 1..D frame durations with explicit supervision masks."""

    if logits.ndim != 4:
        raise ValueError("duration logits must be [B,T,E,D]")
    if duration_frames.shape != logits.shape[:-1]:
        raise ValueError("duration target shape mismatch")
    if valid.shape != duration_frames.shape or uncertain.shape != duration_frames.shape:
        raise ValueError("duration masks must be [B,T,E]")
    active = valid & ~uncertain
    if not bool(active.any()):
        return logits.sum() * 0
    maximum = logits.shape[-1]
    targets = duration_frames[active]
    if duration_frames.dtype != torch.long:
        raise ValueError("duration targets must be long")
    if bool(((targets < 1) | (targets > maximum)).any()):
        raise ValueError("duration targets must lie in [1,D]")
    loss = functional.cross_entropy(logits[active], targets - 1, reduction="none")
    if sample_weight is None:
        return loss.mean()
    if sample_weight.shape != (logits.shape[0],):
        raise ValueError("sample_weight must be [B]")
    expanded = sample_weight[:, None, None].expand_as(duration_frames)[active].to(loss)
    return (loss * expanded).sum() / expanded.sum().clamp_min(1e-8)
