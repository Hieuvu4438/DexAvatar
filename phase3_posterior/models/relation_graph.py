"""Fixed-edge relational graph encoder without optional scatter dependencies."""

from __future__ import annotations

import torch
from torch import nn

from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    NUM_RELATION_NODES,
)
from phase3_posterior.models.contact_head import ContactHead


class RelationGraphEncoder(nn.Module):
    def __init__(self, width: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.edge_input = nn.Linear(EDGE_FEATURE_DIM, width)
        self.node_seed = nn.Parameter(torch.zeros(NUM_RELATION_NODES, width))
        self.messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, width)
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head = ContactHead(width)

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
        batch_shape = edge.shape[:-2]
        nodes = self.node_seed.view(*([1] * len(batch_shape)), NUM_RELATION_NODES, -1)
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
            aggregate.index_add_(-2, source, update)
            aggregate.index_add_(-2, target, update)
            count.index_add_(-2, source, mask)
            count.index_add_(-2, target, mask)
            nodes = norm(nodes + aggregate / count.clamp_min(1.0))
            edge = edge + update
        outputs = self.head(edge)
        outputs["edge_tokens"] = edge * mask
        outputs["relation_token"] = (edge * mask).sum(dim=-2) / mask.sum(
            dim=-2
        ).clamp_min(1.0)
        return outputs
