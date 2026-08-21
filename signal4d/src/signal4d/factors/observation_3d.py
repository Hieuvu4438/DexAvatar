from __future__ import annotations

import torch

from .base import FactorResult, pseudo_huber, summarize


def observation_3d_factor(
    joints: torch.Tensor,
    observations: torch.Tensor,
    valid: torch.Tensor,
    sigma: torch.Tensor,
    delta: float = 2.0,
) -> FactorResult:
    if observations.ndim != 4 or joints.shape != (observations.shape[0], observations.shape[2], 3):
        raise ValueError("joints [T,J,3] and observations [T,S,J,3] are required")
    if valid.shape != observations.shape[:-1] or sigma.shape != observations.shape:
        raise ValueError("valid [T,S,J] and sigma [T,S,J,3] are required")
    residual = (joints[:, None] - observations) / sigma.clamp_min(1e-6)
    robust = pseudo_huber(residual, delta).sum(-1)
    masked = torch.where(valid, robust, torch.zeros_like(robust))
    count = int(valid.sum().item())
    if count == 0:
        zero = joints.sum() * 0
        return FactorResult(
            zero,
            0,
            torch.zeros(joints.shape[0], device=joints.device),
            diagnostics={"warning": 1.0},
        )
    per_frame_count = valid.sum(dim=(1, 2)).clamp_min(1)
    per_frame = masked.sum(dim=(1, 2)) / per_frame_count
    return FactorResult(
        masked.sum() / count,
        count,
        per_frame,
        summarize(residual, valid[..., None].expand_as(residual)),
    )
