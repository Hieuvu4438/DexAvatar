"""Frozen spatial prior plus trainable temporal-relational score residual."""

from __future__ import annotations

import torch
from torch import nn

from phase3_posterior.models.dposer_adapter import ZeroSpatialPrior
from phase3_posterior.models.relation_graph import RelationGraphEncoder
from phase3_posterior.models.temporal_score import TemporalScoreNetwork


class RelationalDiffusionPosterior(nn.Module):
    def __init__(self, config: dict, spatial_prior: nn.Module | None = None) -> None:
        super().__init__()
        width = int(config.get("relation_width", 128))
        route = str(config.get("spatial_prior", {}).get("route", "from_scratch"))
        if spatial_prior is None and route != "from_scratch":
            raise ValueError(
                "A pretrained spatial-prior route requires an audited, explicitly injected module"
            )
        self.spatial_prior = spatial_prior or ZeroSpatialPrior()
        self.relation_graph = RelationGraphEncoder(
            width, int(config.get("relation_layers", 3))
        )
        self.residual = TemporalScoreNetwork(
            observation_dim=int(config.get("observation_dim", 45)),
            width=int(config.get("width", 384)),
            blocks=int(config.get("blocks", 8)),
            heads=int(config.get("heads", 8)),
            mlp_ratio=int(config.get("mlp_ratio", 4)),
            dropout=float(config.get("dropout", 0.1)),
            relation_width=width,
            max_frames=int(config.get("max_frames", 64)),
            activation_checkpointing=bool(config.get("activation_checkpointing", True)),
        )

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        observation: torch.Tensor,
        frame_valid: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_valid: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
        residual_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        relation = self.relation_graph(edge_features, edge_index, edge_valid)
        prior = self.spatial_prior(state, time)
        residual = self.residual(
            state,
            time,
            observation,
            relation["relation_token"],
            frame_valid,
            condition_mask,
        )
        outputs = dict(relation)
        outputs.update(
            prior_score=prior,
            residual_score=residual,
            score=prior + residual if residual_enabled else prior,
        )
        return outputs
