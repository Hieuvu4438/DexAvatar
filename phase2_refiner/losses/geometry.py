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
) -> torch.Tensor:
    """Give upper body, left hand, and right hand equal loss weight."""
    errors = torch.linalg.vector_norm(prediction - target, dim=-1)
    regions = []
    for name in ("ubody", "lhand", "rhand"):
        if name in region_masks:
            frame_error = errors[..., region_masks[name]].mean(dim=-1)
            regions.append(
                frame_error.mean()
                if frame_valid is None
                else masked_mean(frame_error, frame_valid)
            )
    return torch.stack(regions).mean() if regions else errors.new_zeros(())
