"""Contact, persistence, and depth-order prediction heads."""

from __future__ import annotations

import torch
from torch import nn


class ContactHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU()
        )
        self.contact = nn.Linear(width, 1)
        self.persistence = nn.Linear(width, 1)
        self.depth = nn.Linear(width, 3)

    def forward(self, edge_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(edge_tokens)
        return {
            "contact_logits": self.contact(hidden).squeeze(-1),
            "persistence_logits": self.persistence(hidden).squeeze(-1),
            "depth_logits": self.depth(hidden),
        }
