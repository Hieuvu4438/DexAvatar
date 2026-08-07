"""Independent-edge geometry MLP used as the frozen R2 comparator."""

from __future__ import annotations

import torch
from torch import nn

from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    default_edge_index,
)
from phase3_posterior.models.contact_head import ContactHead


class GeometryOnlyRelationMLP(nn.Module):
    """Predict relations per edge without graph message passing."""

    def __init__(self, width: int = 128, predict_distance: bool = True) -> None:
        super().__init__()
        edge_count = default_edge_index().shape[1]
        self.edge_seed = nn.Parameter(torch.zeros(edge_count, width))
        self.encoder = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.head = ContactHead(width, predict_distance=predict_distance)

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del edge_index
        tokens = self.encoder(edge_features) + self.edge_seed.to(edge_features.dtype)
        result = self.head(tokens)
        if "distance" in result:
            result["distance"] = result["distance"] + edge_features[..., 3]
        mask = edge_valid[..., None].to(tokens.dtype)
        result["edge_tokens"] = tokens * mask
        result["relation_token"] = (tokens * mask).sum(dim=-2) / mask.sum(
            dim=-2
        ).clamp_min(1.0)
        return result
