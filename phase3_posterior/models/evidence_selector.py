"""GT-free inference-time candidate evidence selector."""

from __future__ import annotations

import torch
from torch import nn


class EvidenceSelector(nn.Module):
    def __init__(self, feature_dim: int, width: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, evidence_features: torch.Tensor) -> torch.Tensor:
        """Score ``(B,K,F)`` evidence only; GT is intentionally absent."""
        if evidence_features.ndim != 3:
            raise ValueError("selector evidence_features must have shape (B,K,F)")
        return self.network(evidence_features).squeeze(-1)

    def select(self, evidence_features: torch.Tensor) -> torch.Tensor:
        return self(evidence_features).argmax(dim=-1)
