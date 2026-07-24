"""Identity-safe residual and auxiliary output heads."""

from __future__ import annotations

import torch
from torch import nn


class ResidualHeads(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.delta = nn.Linear(hidden_size, 3)
        self.gate = nn.Linear(hidden_size, 1)
        self.position_delta = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.position_delta.weight)
        nn.init.zeros_(self.position_delta.bias)

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.delta(value),
            torch.sigmoid(self.gate(value)),
            self.position_delta(value),
        )
