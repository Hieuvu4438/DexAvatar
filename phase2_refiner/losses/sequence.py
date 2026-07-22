"""Rotation, motion-preservation, anchor, and uncertainty losses."""

from __future__ import annotations

import torch
from torch import nn

from phase2_refiner.geometry.rotations import geodesic_distance


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _relative_rotation(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[:, 1:] @ matrix[:, :-1].transpose(-1, -2)


class RefinerLoss(nn.Module):
    def __init__(
        self,
        rotation_weight: float = 1.0,
        velocity_weight: float = 0.25,
        acceleration_weight: float = 0.1,
        anchor_weight: float = 0.05,
        uncertainty_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.rotation_weight = rotation_weight
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.anchor_weight = anchor_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        initial_matrix: torch.Tensor,
        target_matrix: torch.Tensor,
        frame_valid: torch.Tensor,
        refine_mask: torch.Tensor,
        observation_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = prediction["matrix"]
        if refine_mask.ndim == 1:
            refine_mask = refine_mask[None].expand(output.shape[0], -1)
        mask = frame_valid[:, :, None] & refine_mask[:, None, :]
        rotation_error = geodesic_distance(output, target_matrix)
        rotation = _masked_mean(rotation_error, mask)

        if output.shape[1] > 1:
            output_velocity = _relative_rotation(output)
            target_velocity = _relative_rotation(target_matrix)
            velocity_mask = mask[:, 1:] & mask[:, :-1]
            velocity = _masked_mean(
                geodesic_distance(output_velocity, target_velocity), velocity_mask
            )
        else:
            velocity = rotation.new_zeros(())

        if output.shape[1] > 2:
            output_velocity = _relative_rotation(output)
            target_velocity = _relative_rotation(target_matrix)
            output_acceleration = _relative_rotation(output_velocity)
            target_acceleration = _relative_rotation(target_velocity)
            acceleration_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
            acceleration = _masked_mean(
                geodesic_distance(output_acceleration, target_acceleration),
                acceleration_mask,
            )
        else:
            acceleration = rotation.new_zeros(())

        anchor_error = geodesic_distance(output, initial_matrix)
        reliable_anchor = mask.to(output.dtype) * observation_confidence.clamp(0.0, 1.0)
        anchor = (
            _masked_mean(anchor_error, reliable_anchor > 0)
            if reliable_anchor.any()
            else rotation.new_zeros(())
        )
        if reliable_anchor.any():
            anchor = (
                anchor_error * reliable_anchor
            ).sum() / reliable_anchor.sum().clamp_min(1.0)

        uncertainty = rotation.new_zeros(())
        if "log_variance" in prediction:
            log_variance = prediction["log_variance"].squeeze(-1)
            nll = 0.5 * (
                rotation_error.square() * torch.exp(-log_variance) + log_variance
            )
            uncertainty = _masked_mean(nll, mask)

        total = (
            self.rotation_weight * rotation
            + self.velocity_weight * velocity
            + self.acceleration_weight * acceleration
            + self.anchor_weight * anchor
            + self.uncertainty_weight * uncertainty
        )
        return {
            "total": total,
            "rotation": rotation,
            "velocity": velocity,
            "acceleration": acceleration,
            "anchor": anchor,
            "uncertainty": uncertainty,
        }
