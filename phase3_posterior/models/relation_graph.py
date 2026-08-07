"""Fixed-edge relational graph encoder without optional scatter dependencies."""

from __future__ import annotations

import torch
from torch import nn

from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    NUM_RELATION_NODES,
    default_edge_index,
)
from phase3_posterior.models.contact_head import ContactHead


class RelationGraphEncoder(nn.Module):
    def __init__(
        self,
        width: int = 128,
        layers: int = 3,
        predict_distance: bool = False,
        edge_identity: bool = False,
        input_dim: int = EDGE_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.edge_input = nn.Linear(input_dim, width)
        self.node_seed = nn.Parameter(torch.zeros(NUM_RELATION_NODES, width))
        self.edge_seed = (
            nn.Parameter(torch.zeros(default_edge_index().shape[1], width))
            if edge_identity
            else None
        )
        self.messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, width)
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head = ContactHead(width, predict_distance=predict_distance)

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if edge_index.ndim == 3:
            if not torch.equal(edge_index, edge_index[:1].expand_as(edge_index)):
                raise ValueError(
                    "All batch items must use the fixed Phase 3 edge ordering"
                )
            edge_index = edge_index[0]
        edge = self.edge_input(edge_features)
        if self.edge_seed is not None:
            edge = edge + self.edge_seed.to(edge.dtype)
        batch_shape = edge.shape[:-2]
        nodes = self.node_seed.to(edge.dtype).view(
            *([1] * len(batch_shape)), NUM_RELATION_NODES, -1
        )
        nodes = nodes.expand(*batch_shape, -1, -1)
        source, target = edge_index
        mask = edge_valid[..., None].to(edge.dtype)
        for message, norm in zip(self.messages, self.norms, strict=True):
            update = message(
                torch.cat((edge, nodes[..., source, :], nodes[..., target, :]), dim=-1)
            )
            update = update * mask
            aggregate = torch.zeros_like(nodes)
            count = torch.zeros_like(nodes[..., :1])
            node_update = update.to(aggregate.dtype)
            node_mask = mask.to(count.dtype)
            aggregate.index_add_(-2, source, node_update)
            aggregate.index_add_(-2, target, node_update)
            count.index_add_(-2, source, node_mask)
            count.index_add_(-2, target, node_mask)
            nodes = norm(nodes + aggregate / count.clamp_min(1.0))
            edge = edge + update
        outputs = self.head(edge)
        if "distance" in outputs:
            outputs["distance"] = outputs["distance"] + edge_features[..., 3]
        outputs["edge_tokens"] = edge * mask
        outputs["relation_token"] = (edge * mask).sum(dim=-2) / mask.sum(
            dim=-2
        ).clamp_min(1.0)
        return outputs
