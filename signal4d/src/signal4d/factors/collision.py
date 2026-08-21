from __future__ import annotations

import torch

from .base import FactorResult, pseudo_huber, summarize


def collision_factor(
    joints: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
    minimum_distance_m: float = 0.008,
) -> FactorResult:
    if not pairs:
        zero = joints.sum() * 0
        return FactorResult(zero, 0, torch.zeros(joints.shape[0], device=joints.device))
    distances = torch.stack(
        [
            torch.linalg.vector_norm(joints[:, first] - joints[:, second], dim=-1)
            for first, second in pairs
        ],
        dim=-1,
    )
    penetration = torch.relu(minimum_distance_m - distances)
    loss_values = pseudo_huber(penetration / minimum_distance_m, 1.0)
    return FactorResult(
        loss_values.mean(), loss_values.numel(), loss_values.mean(-1), summarize(penetration)
    )
