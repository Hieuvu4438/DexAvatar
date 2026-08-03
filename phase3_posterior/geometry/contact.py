"""Probabilistic-contact targets and temporal persistence utilities."""

from __future__ import annotations

import torch


def contact_hysteresis(
    distance: torch.Tensor,
    valid: torch.Tensor,
    onset: float = 0.012,
    release: float = 0.020,
) -> torch.Tensor:
    """Create deterministic contact labels with enter/exit hysteresis."""
    if distance.shape != valid.shape:
        raise ValueError("distance and valid must have identical shapes")
    if onset <= 0 or release <= onset:
        raise ValueError("Require 0 < onset < release")
    result = torch.zeros_like(valid, dtype=torch.bool)
    active = torch.zeros_like(valid[0], dtype=torch.bool)
    for index in range(distance.shape[0]):
        active = torch.where(active, distance[index] < release, distance[index] < onset)
        active = active & valid[index]
        result[index] = active
    return result


def contact_persistence_target(contact: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(contact, dtype=torch.bool)
    if len(contact) > 1:
        result[1:] = contact[1:] & contact[:-1]
    return result


def contact_slip(
    relative_position: torch.Tensor, contact: torch.Tensor
) -> torch.Tensor:
    if relative_position.shape[:-1] != contact.shape:
        raise ValueError("relative_position/contact shape mismatch")
    velocity = torch.zeros_like(relative_position)
    if len(relative_position) > 1:
        velocity[1:] = relative_position[1:] - relative_position[:-1]
    return torch.linalg.vector_norm(velocity, dim=-1) * contact.float()


def gated_contact_probability(
    logits: torch.Tensor,
    valid: torch.Tensor,
    endpoint_reliability: torch.Tensor,
    threshold: float = 0.6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    if endpoint_reliability.shape != probability.shape:
        raise ValueError("endpoint reliability must match contact logits")
    enabled = valid & (endpoint_reliability > 0) & (probability >= threshold)
    return torch.where(enabled, probability, torch.zeros_like(probability))
