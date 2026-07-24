"""Target-motion losses that do not reward zero-velocity oversmoothing."""

from __future__ import annotations

import torch

from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.losses.geometry import masked_mean


def relative_rotation(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[:, 1:] @ matrix[:, :-1].transpose(-1, -2)


def rotation_motion_losses(
    output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if output.shape[1] <= 1:
        zero = output.new_zeros(())
        return zero, zero
    output_velocity = relative_rotation(output)
    target_velocity = relative_rotation(target)
    velocity_mask = mask[:, 1:] & mask[:, :-1]
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
