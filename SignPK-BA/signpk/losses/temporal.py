from __future__ import annotations

from torch import Tensor

from signpk.geometry.robustifiers import charbonnier, masked_mean
from signpk.geometry.rotations import so3_log


def target_velocity_loss(
    prediction: Tensor,
    target: Tensor,
    timestamps: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    dt = (timestamps[..., 1:] - timestamps[..., :-1]).clamp_min(1e-6)
    while dt.ndim < prediction.ndim:
        dt = dt.unsqueeze(-1)
    predicted_velocity = (prediction[..., 1:, :, :] - prediction[..., :-1, :, :]) / dt
    target_velocity = (target[..., 1:, :, :] - target[..., :-1, :, :]) / dt
    pair_valid = None if valid is None else valid[..., 1:] & valid[..., :-1]
    return masked_mean(charbonnier(predicted_velocity - target_velocity), pair_valid)


def angular_velocity_loss(
    prediction: Tensor,
    target: Tensor,
    timestamps: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    dt = (timestamps[..., 1:] - timestamps[..., :-1]).clamp_min(1e-6)
    predicted = so3_log(prediction[..., :-1, :, :].transpose(-1, -2) @ prediction[..., 1:, :, :])
    expected = so3_log(target[..., :-1, :, :].transpose(-1, -2) @ target[..., 1:, :, :])
    while dt.ndim < predicted.ndim:
        dt = dt.unsqueeze(-1)
    pair_valid = None if valid is None else valid[..., 1:] & valid[..., :-1]
    return masked_mean(charbonnier(predicted / dt - expected / dt), pair_valid)


def learned_motion_factor(
    rotations: Tensor,
    target_angular_velocity: Tensor,
    timestamps: Tensor,
    phase_gate: Tensor,
    hold_weight: float = 1.0,
    stroke_weight: float = 0.25,
    valid: Tensor | None = None,
) -> Tensor:
    dt = (timestamps[1:] - timestamps[:-1]).clamp_min(1e-6)
    observed = so3_log(rotations[:-1].transpose(-1, -2) @ rotations[1:]) / dt[:, None, None]
    residual = charbonnier(observed - target_angular_velocity[:-1]).mean(-1)
    alpha = stroke_weight * (1 - phase_gate[:-1]) + hold_weight * phase_gate[:-1]
    return masked_mean(residual * alpha, None if valid is None else valid[:-1] & valid[1:])

