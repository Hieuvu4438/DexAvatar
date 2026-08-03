"""Target-motion losses that do not reward zero-velocity oversmoothing."""

from __future__ import annotations

import torch

from phase2_refiner.geometry.rotations import geodesic_distance, matrix_to_axis_angle
from phase2_refiner.losses.geometry import masked_mean


def relative_rotation(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[:, 1:] @ matrix[:, :-1].transpose(-1, -2)


def rotation_motion_losses(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    time_delta_seconds: torch.Tensor | None = None,
    physical_time_motion: bool = False,
    motion_reference_seconds: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output.shape[1] <= 1:
        zero = output.new_zeros(())
        return zero, zero
    output_velocity = relative_rotation(output)
    target_velocity = relative_rotation(target)
    velocity_mask = mask[:, 1:] & mask[:, :-1]
    if physical_time_motion:
        if time_delta_seconds is None:
            raise ValueError(
                "physical_time_motion requires per-frame time_delta_seconds"
            )
        if motion_reference_seconds <= 0:
            raise ValueError("motion_reference_seconds must be positive")
        transition_dt = time_delta_seconds[:, 1:]
        velocity_mask = velocity_mask & (transition_dt[..., None] > 0)
        safe_dt = transition_dt.clamp_min(1e-6)[..., None, None]
        output_rate = matrix_to_axis_angle(output_velocity) / safe_dt
        target_rate = matrix_to_axis_angle(target_velocity) / safe_dt
        velocity_error = torch.linalg.vector_norm(
            output_rate - target_rate, dim=-1
        ) * motion_reference_seconds
        target_motion = (
            torch.linalg.vector_norm(target_rate, dim=-1).detach()
            * motion_reference_seconds
        )
        transition_weight = 1.0 + target_motion / target_motion.mean().clamp_min(1e-4)
        velocity = masked_mean(velocity_error * transition_weight, velocity_mask)
        if output.shape[1] <= 2:
            return velocity, output.new_zeros(())
        acceleration_dt = 0.5 * (transition_dt[:, 1:] + transition_dt[:, :-1])
        acceleration_mask = (
            mask[:, 2:]
            & mask[:, 1:-1]
            & mask[:, :-2]
            & (transition_dt[:, 1:, None] > 0)
            & (transition_dt[:, :-1, None] > 0)
        )
        output_acceleration = (output_rate[:, 1:] - output_rate[:, :-1]) / (
            acceleration_dt.clamp_min(1e-6)[..., None, None]
        )
        target_acceleration = (target_rate[:, 1:] - target_rate[:, :-1]) / (
            acceleration_dt.clamp_min(1e-6)[..., None, None]
        )
        acceleration_error = torch.linalg.vector_norm(
            output_acceleration - target_acceleration, dim=-1
        ) * (motion_reference_seconds**2)
        acceleration = masked_mean(acceleration_error, acceleration_mask)
        return velocity, acceleration

    velocity_error = geodesic_distance(output_velocity, target_velocity)
    target_motion = geodesic_distance(target[:, 1:], target[:, :-1]).detach()
    transition_weight = 1.0 + target_motion / target_motion.mean().clamp_min(1e-4)
    velocity = masked_mean(velocity_error * transition_weight, velocity_mask)
    if output.shape[1] <= 2:
        return velocity, output.new_zeros(())
    output_acceleration = relative_rotation(output_velocity)
    target_acceleration = relative_rotation(target_velocity)
    acceleration_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    acceleration = masked_mean(
        geodesic_distance(output_acceleration, target_acceleration), acceleration_mask
    )
    return velocity, acceleration
