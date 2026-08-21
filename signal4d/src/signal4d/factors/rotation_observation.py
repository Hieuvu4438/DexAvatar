from __future__ import annotations

import torch

from ..geometry.so3 import log_map
from .base import FactorResult, pseudo_huber, summarize


def rotation_observation_factor(
    rotations: torch.Tensor,
    observations: torch.Tensor,
    valid: torch.Tensor,
    sigma: torch.Tensor,
    delta: float = 2.0,
) -> FactorResult:
    residual = log_map(observations.transpose(-1, -2) @ rotations[:, None]) / sigma.clamp_min(1e-6)
    robust = pseudo_huber(residual, delta).sum(-1)
    masked = torch.where(valid, robust, torch.zeros_like(robust))
    count = int(valid.sum().item())
    if count == 0:
        zero = rotations.sum() * 0
        return FactorResult(zero, 0, torch.zeros(rotations.shape[0], device=rotations.device))
    per_frame = masked.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    return FactorResult(
        masked.sum() / count,
        count,
        per_frame,
        summarize(residual, valid[..., None].expand_as(residual)),
    )
