from __future__ import annotations

import torch

from ..geometry.projection import normalized_reprojection_residual, project_points
from .base import FactorResult, pseudo_huber, summarize


def observation_2d_factor(
    joints: torch.Tensor,
    observations: torch.Tensor,
    valid: torch.Tensor,
    camera_k: torch.Tensor,
    image_size_hw: torch.Tensor,
    delta: float = 0.02,
) -> FactorResult:
    if observations.ndim != 4 or observations.shape[-1] != 2:
        raise ValueError("2D observations must have shape [T,S,J,2]")
    if joints.shape != (observations.shape[0], observations.shape[2], 3):
        raise ValueError("joints must have shape [T,J,3]")
    if valid.shape != observations.shape[:-1]:
        raise ValueError("valid must have shape [T,S,J]")
    projected = project_points(joints, camera_k)
    residual = normalized_reprojection_residual(projected[:, None], observations, image_size_hw)
    values = pseudo_huber(torch.linalg.vector_norm(residual, dim=-1), delta)
    masked = torch.where(valid, values, torch.zeros_like(values))
    count = int(valid.sum())
    if not count:
        zero = joints.sum() * 0
        return FactorResult(zero, 0, torch.zeros(joints.shape[0], device=joints.device))
    per_frame = masked.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    return FactorResult(
        masked.sum() / count,
        count,
        per_frame,
        summarize(residual, valid[..., None].expand_as(residual)),
    )
