"""Fixed U0 and learned U1 observation reliability."""

from __future__ import annotations

import torch
from torch import nn


class LearnedReliabilityHead(nn.Module):
    """Predict rotation and 2D/3D observation log variance per joint token."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.network(value).clamp(-8.0, 6.0)
        return prediction[..., 0], prediction[..., 1:3]


def effective_reliability(
    fixed: torch.Tensor, rotation_log_variance: torch.Tensor | None
) -> torch.Tensor:
    fixed = fixed.clamp(0.0, 1.0)
    if rotation_log_variance is None:
        return fixed
    learned_precision = torch.sigmoid(-0.5 * rotation_log_variance)
    return fixed * learned_precision
