from __future__ import annotations

import torch
from torch import Tensor, nn


class SoftGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor, validity: Tensor | None = None) -> Tensor:
        gate = torch.sigmoid(self.network(features))
        if validity is not None:
            gate = gate * validity.to(dtype=gate.dtype).unsqueeze(-1)
        return gate


def uncertainty_weight(log_variance: Tensor, minimum: float = 0.05, maximum: float = 20.0) -> Tensor:
    return torch.exp(-log_variance).clamp(min=minimum, max=maximum)

