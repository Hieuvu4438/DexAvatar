"""Balanced joint, hand landmark, palm, and regional mesh losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


GLOBAL_FINGERTIPS = (23, 26, 29, 32, 35, 38, 41, 44, 47, 50)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def joint_position_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    return masked_mean(torch.linalg.vector_norm(prediction - target, dim=-1), valid)


def fingertip_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    indices = torch.as_tensor(GLOBAL_FINGERTIPS, device=prediction.device)
    return joint_position_loss(
        prediction.index_select(-2, indices),
        target.index_select(-2, indices),
        valid.index_select(-1, indices),
    )


def palm_normal_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=-1, eps=1e-8)
    target = F.normalize(target.float(), dim=-1, eps=1e-8)
    cosine = (prediction * target).sum(dim=-1).clamp(-1.0, 1.0)
    cross_squared = torch.cross(prediction, target, dim=-1).square().sum(dim=-1)
    sine = torch.sqrt(cross_squared.clamp_min(1e-12))
    angle = torch.atan2(sine, cosine)
    angle = torch.where(cross_squared > 1e-12, angle, torch.zeros_like(angle))
    return masked_mean(angle, valid)


def balanced_region_vertex_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    region_masks: dict[str, torch.Tensor],
    frame_valid: torch.Tensor | None = None,
    region_frame_weight: dict[str, torch.Tensor] | None = None,
    translation_centered: bool = False,
) -> torch.Tensor:
    """Give upper body, left hand, and right hand equal loss weight."""
    regions = []
    for name in ("ubody", "lhand", "rhand"):
        if name in region_masks:
            indices = region_masks[name]
            predicted_region = prediction[..., indices, :]
            target_region = target[..., indices, :]
            if translation_centered:
                predicted_region = predicted_region - predicted_region.mean(
                    dim=-2, keepdim=True
                )
                target_region = target_region - target_region.mean(dim=-2, keepdim=True)
            frame_error = torch.linalg.vector_norm(
                predicted_region - target_region, dim=-1
            ).mean(dim=-1)
            weight = None
            if region_frame_weight is not None:
                weight = region_frame_weight.get(name)
            if weight is None:
                weight = torch.ones_like(frame_error)
            if frame_valid is not None:
                weight = weight * frame_valid.to(weight.dtype)
            regions.append((frame_error * weight).sum() / weight.sum().clamp_min(1.0))
    return torch.stack(regions).mean() if regions else prediction.new_zeros(())


def regional_vertex_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    region_masks: dict[str, torch.Tensor],
    *,
    translation_centered: bool = True,
) -> dict[str, torch.Tensor]:
    """Return per-frame regional vertex errors using the release alignment.

    Values retain the input vertex unit. SMPL-X decoding in this package uses
    metres, so callers that report millimetres must multiply by 1,000.
    """
    result = {}
    for name in ("ubody", "lhand", "rhand"):
        if name not in region_masks:
            continue
        indices = region_masks[name]
        predicted_region = prediction[..., indices, :]
        target_region = target[..., indices, :]
        if translation_centered:
            predicted_region = predicted_region - predicted_region.mean(
                dim=-2, keepdim=True
            )
            target_region = target_region - target_region.mean(dim=-2, keepdim=True)
        result[name] = torch.linalg.vector_norm(
            predicted_region - target_region, dim=-1
        ).mean(dim=-1)
    return result
