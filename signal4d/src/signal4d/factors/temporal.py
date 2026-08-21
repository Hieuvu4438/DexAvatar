from __future__ import annotations

import torch

from ..geometry.so3 import log_map
from .base import FactorResult, pseudo_huber, summarize


def adaptive_weights(
    uncertainty: torch.Tensor,
    change_probability: torch.Tensor,
    base_weight: float = 1.0,
    alpha: float = 2.0,
    gamma: float = 2.0,
    minimum: float = 0.05,
    maximum: float = 5.0,
) -> torch.Tensor:
    if uncertainty.ndim != 2 or change_probability.shape != (uncertainty.shape[0],):
        raise ValueError("uncertainty [T,J] and change_probability [T] are required")
    normalized_uq = uncertainty / uncertainty.median().clamp_min(1e-6)
    weight = (
        base_weight * (1 - change_probability[:, None]).pow(gamma) * (1 + alpha * normalized_uq)
    )
    return weight.clamp(minimum, maximum)


def temporal_position_factor(
    joints: torch.Tensor,
    fps: float,
    weights: torch.Tensor,
    delta: float = 0.05,
) -> FactorResult:
    if joints.shape[0] < 3:
        zero = joints.sum() * 0
        return FactorResult(zero, 0, torch.zeros(joints.shape[0], device=joints.device))
    velocity = (joints[1:] - joints[:-1]) * fps
    acceleration = (velocity[1:] - velocity[:-1]) * fps
    local_weight = weights[2:]
    residual = torch.linalg.vector_norm(acceleration, dim=-1)
    loss_values = pseudo_huber(residual, delta) * local_weight
    per_frame = torch.zeros(joints.shape[0], device=joints.device, dtype=joints.dtype)
    per_frame[2:] = loss_values.mean(-1)
    return FactorResult(loss_values.mean(), loss_values.numel(), per_frame, summarize(residual))


def temporal_rotation_factor(
    rotations: torch.Tensor,
    fps: float,
    weights: torch.Tensor,
    delta: float = 0.5,
) -> FactorResult:
    if rotations.shape[0] < 3:
        zero = rotations.sum() * 0
        return FactorResult(zero, 0, torch.zeros(rotations.shape[0], device=rotations.device))
    angular_velocity = log_map(rotations[:-1].transpose(-1, -2) @ rotations[1:]) * fps
    acceleration = angular_velocity[1:] - angular_velocity[:-1]
    residual = torch.linalg.vector_norm(acceleration, dim=-1)
    loss_values = pseudo_huber(residual, delta) * weights[2:]
    per_frame = torch.zeros(rotations.shape[0], device=rotations.device, dtype=rotations.dtype)
    per_frame[2:] = loss_values.mean(-1)
    return FactorResult(loss_values.mean(), loss_values.numel(), per_frame, summarize(residual))
