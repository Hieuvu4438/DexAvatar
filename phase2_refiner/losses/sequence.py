"""Complete Phase 2 deterministic-refiner objective with optional geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.losses.geometry import (
    balanced_region_vertex_loss,
    fingertip_loss,
    joint_position_loss,
    masked_mean,
    palm_normal_loss,
)
from phase2_refiner.losses.motion import rotation_motion_losses
from phase2_refiner.losses.uncertainty import (
    heteroscedastic_nll,
    regional_worst_decile_ranking_loss,
)


class RefinerLoss(nn.Module):
    def __init__(
        self,
        rotation_weight: float = 1.0,
        vertex_weight: float = 0.25,
        joint_weight: float = 0.5,
        fingertip_weight: float = 0.5,
        palm_weight: float = 0.1,
        observation_weight: float = 0.1,
        velocity_weight: float = 0.25,
        acceleration_weight: float = 0.1,
        anchor_weight: float = 0.05,
        biomechanical_weight: float = 0.01,
        uncertainty_weight: float = 0.1,
        uncertainty_ranking_weight: float = 0.0,
        benefit_weight: float = 0.0,
        benefit_margin_degrees: float = 0.0,
        target_quality_weighting: bool = False,
        minimum_target_quality: float = 0.1,
        physical_time_motion: bool = False,
        motion_reference_seconds: float = 0.04,
    ) -> None:
        super().__init__()
        self.weights = {
            "rotation": rotation_weight,
            "vertex": vertex_weight,
            "joint": joint_weight,
            "fingertip": fingertip_weight,
            "palm": palm_weight,
            "observation": observation_weight,
            "velocity": velocity_weight,
            "acceleration": acceleration_weight,
            "anchor": anchor_weight,
            "biomechanical": biomechanical_weight,
            "uncertainty": uncertainty_weight,
            "uncertainty_ranking": uncertainty_ranking_weight,
            "benefit": benefit_weight,
        }
        self.benefit_margin = torch.deg2rad(torch.tensor(benefit_margin_degrees)).item()
        self.target_quality_weighting = bool(target_quality_weighting)
        self.minimum_target_quality = float(minimum_target_quality)
        if not 0.0 <= self.minimum_target_quality <= 1.0:
            raise ValueError("minimum_target_quality must be within [0, 1]")
        self.physical_time_motion = bool(physical_time_motion)
        self.motion_reference_seconds = float(motion_reference_seconds)
        if self.motion_reference_seconds <= 0:
            raise ValueError("motion_reference_seconds must be positive")

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        initial_matrix: torch.Tensor,
        target_matrix: torch.Tensor,
        frame_valid: torch.Tensor,
        refine_mask: torch.Tensor,
        observation_confidence: torch.Tensor,
        *,
        target_rotation_valid: torch.Tensor | None = None,
        target_quality: torch.Tensor | None = None,
        target_joint_position: torch.Tensor | None = None,
        target_joint_valid: torch.Tensor | None = None,
        observed_joint_position: torch.Tensor | None = None,
        observed_joint_valid: torch.Tensor | None = None,
        target_palm_normal: torch.Tensor | None = None,
        target_palm_valid: torch.Tensor | None = None,
        predicted_vertices: torch.Tensor | None = None,
        target_vertices: torch.Tensor | None = None,
        vertex_region_masks: dict[str, torch.Tensor] | None = None,
        predicted_keypoints_2d: torch.Tensor | None = None,
        observed_keypoints_2d: torch.Tensor | None = None,
        observed_keypoint_valid: torch.Tensor | None = None,
        time_delta_seconds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = prediction["matrix"]
        if refine_mask.ndim == 1:
            refine_mask = refine_mask[None].expand(output.shape[0], -1)
        mask = frame_valid[:, :, None] & refine_mask[:, None, :]
        if target_rotation_valid is not None:
            mask = mask & target_rotation_valid
        quality = None
        if self.target_quality_weighting:
            if target_quality is None:
                raise ValueError("target_quality_weighting requires target_quality")
            quality = target_quality.clamp(0.0, 1.0)
            mask = mask & (quality >= self.minimum_target_quality)
        rotation_error = geodesic_distance(output, target_matrix)
        if quality is None:
            rotation = masked_mean(rotation_error, mask)
        else:
            rotation_weight = quality * mask
            rotation = (rotation_error * rotation_weight).sum() / rotation_weight.sum().clamp_min(1.0)
        velocity, acceleration = rotation_motion_losses(
            output,
            target_matrix,
            mask,
            time_delta_seconds=time_delta_seconds,
            physical_time_motion=self.physical_time_motion,
            motion_reference_seconds=self.motion_reference_seconds,
        )

        anchor_error = geodesic_distance(output, initial_matrix)
        reliable_anchor = mask.to(output.dtype) * observation_confidence.clamp(0.0, 1.0)
        anchor = (
            anchor_error * reliable_anchor
        ).sum() / reliable_anchor.sum().clamp_min(1.0)

        zero = rotation.new_zeros(())
        vertex = joint = fingertip = palm = observation = zero
        if (
            predicted_vertices is not None
            and target_vertices is not None
            and vertex_region_masks is not None
        ):
            vertex = balanced_region_vertex_loss(
                predicted_vertices,
                target_vertices,
                vertex_region_masks,
                frame_valid,
            )
        predicted_joint = prediction.get("joint_position")
        if (
            predicted_joint is not None
            and target_joint_position is not None
            and target_joint_valid is not None
        ):
            joint_mask = target_joint_valid & mask
            joint = joint_position_loss(
                predicted_joint, target_joint_position, joint_mask
            )
            fingertip = fingertip_loss(
                predicted_joint, target_joint_position, joint_mask
            )
        if (
            predicted_joint is not None
            and observed_joint_position is not None
            and observed_joint_valid is not None
        ):
            observation_mask = observed_joint_valid & frame_valid[:, :, None]
            observation_error = torch.linalg.vector_norm(
                predicted_joint - observed_joint_position, dim=-1
            )
            if "observation_log_variance" in prediction:
                observation = heteroscedastic_nll(
                    observation_error,
                    prediction["observation_log_variance"][..., 1],
                    observation_mask,
                )
            else:
                observation = masked_mean(observation_error, observation_mask)
        if (
            predicted_keypoints_2d is not None
            and observed_keypoints_2d is not None
            and observed_keypoint_valid is not None
        ):
            reprojection_error = torch.linalg.vector_norm(
                predicted_keypoints_2d - observed_keypoints_2d, dim=-1
            )
            reprojection_mask = observed_keypoint_valid & frame_valid[:, :, None]
            if "observation_log_variance" in prediction:
                reprojection = heteroscedastic_nll(
                    reprojection_error,
                    prediction["observation_log_variance"][..., 0],
                    reprojection_mask,
                )
            else:
                reprojection = masked_mean(reprojection_error, reprojection_mask)
            observation = observation + reprojection
        if target_palm_normal is not None and target_palm_valid is not None:
            predicted_palm = prediction.get("palm_normal")
            if predicted_palm is not None:
                palm = palm_normal_loss(
                    predicted_palm, target_palm_normal, target_palm_valid
                )

        biomechanical = torch.relu(
            torch.linalg.vector_norm(prediction["raw_delta"], dim=-1) - 0.5
        )
        biomechanical = masked_mean(biomechanical.square(), mask)

        uncertainty = uncertainty_ranking = benefit = zero
        if "log_variance" in prediction:
            log_variance = prediction["log_variance"].squeeze(-1)
            uncertainty = heteroscedastic_nll(
                rotation_error, log_variance, mask
            )
            uncertainty_ranking = regional_worst_decile_ranking_loss(
                rotation_error, log_variance, mask
            )
        if "benefit_logit" in prediction:
            initial_error = geodesic_distance(initial_matrix, target_matrix).detach()
            group_labels = []
            group_valid = []
            for start, end in ((0, 21), (21, 36), (36, 51)):
                region_mask = mask[..., start:end]
                denominator = region_mask.sum(dim=-1).clamp_min(1)
                initial_region = (
                    initial_error[..., start:end] * region_mask
                ).sum(dim=-1) / denominator
                output_region = (
                    rotation_error[..., start:end].detach() * region_mask
                ).sum(dim=-1) / denominator
                group_labels.append(
                    (initial_region - output_region > self.benefit_margin).float()
                )
                group_valid.append(region_mask.any(dim=-1))
            benefit_label = torch.stack(group_labels, dim=-1)
            benefit_valid = torch.stack(group_valid, dim=-1) & frame_valid[..., None]
            benefit_error = F.binary_cross_entropy_with_logits(
                prediction["benefit_logit"], benefit_label, reduction="none"
            )
            benefit = masked_mean(benefit_error, benefit_valid)

        terms = {
            "rotation": rotation,
            "vertex": vertex,
            "joint": joint,
            "fingertip": fingertip,
            "palm": palm,
            "observation": observation,
            "velocity": velocity,
            "acceleration": acceleration,
            "anchor": anchor,
            "biomechanical": biomechanical,
            "uncertainty": uncertainty,
            "uncertainty_ranking": uncertainty_ranking,
            "benefit": benefit,
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        return {"total": total, **terms}
