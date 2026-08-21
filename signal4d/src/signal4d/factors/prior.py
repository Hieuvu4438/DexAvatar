from __future__ import annotations

import torch

from ..geometry.so3 import log_map
from .base import FactorResult, pseudo_huber, summarize


def pose_prior_factor(
    rotations: torch.Tensor,
    reference: torch.Tensor,
    joint_weights: torch.Tensor | None = None,
    delta: float = 0.5,
) -> FactorResult:
    if rotations.shape != reference.shape or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotation and reference matrices must have matching [...,3,3] shapes")
    residual = torch.linalg.vector_norm(log_map(reference.transpose(-1, -2) @ rotations), dim=-1)
    weights = torch.ones_like(residual) if joint_weights is None else joint_weights.to(residual)
    weights = torch.broadcast_to(weights, residual.shape)
    values = pseudo_huber(residual, delta) * weights
    return FactorResult(values.mean(), values.numel(), values.mean(-1), summarize(residual))
