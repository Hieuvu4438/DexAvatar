"""SO(3) operations following the right-composed residual convention in Methods."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    bound_rotation_vector,
    geodesic_distance,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)


def residual_target(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return Log(base^T target), so target = base Exp(delta)."""
    if base.shape != target.shape or base.shape[-2:] != (3, 3):
        raise ValueError("base and target must have equal (...,3,3) shapes")
    return matrix_to_axis_angle(base.transpose(-1, -2) @ target)


def compose_right(
    base: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor | None = None,
    max_angle: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compute base Exp(gate * residual); the gate is applied exactly once."""
    if max_angle is not None:
        residual = bound_rotation_vector(residual, max_angle)
    if gate is not None:
        residual = residual * gate[..., None]
    return base @ axis_angle_to_matrix(residual)


def gate_from_reliability(
    probability: torch.Tensor, tau_low: float, tau_high: float, dilation: int = 0
) -> torch.Tensor:
    if not 0 <= tau_low < tau_high <= 1:
        raise ValueError("Expected 0 <= tau_low < tau_high <= 1")
    if dilation < 0:
        raise ValueError("dilation must be non-negative")
    gate = ((tau_high - probability) / (tau_high - tau_low)).clamp(0.0, 1.0)
    if dilation == 0:
        return gate
    if gate.ndim < 2:
        raise ValueError("dilated gates require a time dimension at axis -2")
    shape = gate.shape
    flattened = gate.reshape(-1, shape[-2], shape[-1]).transpose(1, 2)
    flattened = F.max_pool1d(
        flattened, kernel_size=2 * dilation + 1, stride=1, padding=dilation
    )
    return flattened.transpose(1, 2).reshape(shape)


def joint_max_angles(
    device: torch.device,
    dtype: torch.dtype,
    body_degrees: float,
    hand_degrees: float,
) -> torch.Tensor:
    value = torch.full((51, 1), math.radians(hand_degrees), device=device, dtype=dtype)
    value[:21] = math.radians(body_degrees)
    return value


__all__ = [
    "axis_angle_to_matrix",
    "compose_right",
    "gate_from_reliability",
    "geodesic_distance",
    "joint_max_angles",
    "matrix_to_axis_angle",
    "matrix_to_rotation_6d",
    "residual_target",
]
