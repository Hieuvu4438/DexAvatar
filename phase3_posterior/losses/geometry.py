"""SO(3) and target-motion losses with partial-label masking."""

from __future__ import annotations

import torch

from phase2_refiner.geometry.rotations import geodesic_distance
from phase3_posterior.geometry.state_adapter import state_to_matrices


def masked_geodesic_loss(
    predicted_state: torch.Tensor,
    target_matrix: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted = state_to_matrices(predicted_state)
    distance = geodesic_distance(predicted.float(), target_matrix.float())
    weights = (
        torch.ones(distance.shape[0], device=distance.device)
        if sample_weight is None
        else sample_weight
    )
    mask = valid.to(distance.dtype) * weights[:, None, None]
    return (distance * mask).sum() / mask.sum().clamp_min(1.0)


def target_motion_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if predicted.shape[1] < 2:
        return predicted.sum() * 0.0
    pred_velocity = predicted[:, 1:] - predicted[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_valid = valid[:, 1:] & valid[:, :-1]
    error = (pred_velocity - target_velocity).square().mean(dim=-1)
    weights = (
        torch.ones(error.shape[0], device=error.device)
        if sample_weight is None
        else sample_weight
    )
    mask = velocity_valid.to(error.dtype) * weights[:, None, None]
    result = (error * mask).sum() / mask.sum().clamp_min(1.0)
    if predicted.shape[1] > 2:
        pred_accel = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        target_accel = target_velocity[:, 1:] - target_velocity[:, :-1]
        accel_valid = velocity_valid[:, 1:] & velocity_valid[:, :-1]
        accel_error = (pred_accel - target_accel).square().mean(dim=-1)
        accel_mask = accel_valid.to(accel_error.dtype) * weights[:, None, None]
        result = result + (accel_error * accel_mask).sum() / accel_mask.sum().clamp_min(
            1.0
        )
    return result
