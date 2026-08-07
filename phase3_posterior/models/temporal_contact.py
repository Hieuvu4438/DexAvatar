"""Temporal contact refinement on a frozen relative-geometry backbone."""

from __future__ import annotations

import copy

import torch
from torch import nn

from phase3_posterior.models.contact_head import ContactHead
from phase3_posterior.models.relation_graph import RelationGraphEncoder
from phase3_posterior.geometry.relation_anchors import OBSERVATION_EDGE_FEATURE_DIM


class TemporalContactRefiner(nn.Module):
    """Add temporal contact/persistence evidence without drifting geometry outputs.

    The temporal projection is zero initialized.  Before optimization, raw contact
    and persistence logits therefore reproduce the frozen R2 checkpoint exactly.
    Distance, depth, edge tokens, and relation tokens always come directly from the
    frozen backbone.
    """

    def __init__(
        self,
        backbone: RelationGraphEncoder,
        width: int = 128,
        temporal_hidden: int = 128,
        persistence_fusion_weight: float = 2.0,
        train_contact_encoder: bool = False,
        observation_features: bool = False,
        observation_graph_layers: int = 0,
        observation_logit_residual: bool = False,
        observation_only_training: bool = False,
        observation_hand_body_only: bool = False,
    ) -> None:
        super().__init__()
        if temporal_hidden < 2 or temporal_hidden % 2:
            raise ValueError("temporal_hidden must be a positive even integer")
        if persistence_fusion_weight < 0:
            raise ValueError("persistence_fusion_weight cannot be negative")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.contact_encoder = copy.deepcopy(backbone) if train_contact_encoder else None
        if self.contact_encoder is not None:
            # Only the contact representation is adapted. The copied prediction
            # heads are unused and frozen; formal geometry comes from ``backbone``.
            self.contact_encoder.requires_grad_(True)
            self.contact_encoder.head.requires_grad_(False)
        if observation_graph_layers < 0:
            raise ValueError("observation_graph_layers cannot be negative")
        if observation_logit_residual and observation_graph_layers == 0:
            raise ValueError("observation_logit_residual requires an observation graph")
        if observation_hand_body_only and not observation_logit_residual:
            raise ValueError("observation_hand_body_only requires a logit residual")
        self.observation_encoder = (
            RelationGraphEncoder(
                width,
                observation_graph_layers,
                edge_identity=True,
                input_dim=OBSERVATION_EDGE_FEATURE_DIM,
            )
            if observation_features and observation_graph_layers > 0
            else None
        )
        self.observation_graph_gate = (
            nn.Linear(width, width, bias=False)
            if self.observation_encoder is not None and not observation_logit_residual
            else None
        )
        if self.observation_graph_gate is not None:
            nn.init.zeros_(self.observation_graph_gate.weight)
            self.observation_encoder.head.requires_grad_(False)
        self.observation_contact_delta = (
            nn.Linear(width, 1, bias=False)
            if observation_logit_residual
            else None
        )
        if self.observation_contact_delta is not None:
            nn.init.zeros_(self.observation_contact_delta.weight)
        self.observation_projection = (
            nn.Sequential(
                nn.Linear(OBSERVATION_EDGE_FEATURE_DIM, width),
                nn.SiLU(),
                nn.Linear(width, width, bias=False),
            )
            if observation_features and self.observation_encoder is None
            else None
        )
        if self.observation_projection is not None:
            # Preserve the loaded recovery checkpoint at initialization; the
            # observation path is introduced as a learnable residual.
            nn.init.zeros_(self.observation_projection[-1].weight)
        self.temporal = nn.GRU(
            width,
            temporal_hidden // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_projection = nn.Linear(temporal_hidden, width, bias=False)
        nn.init.zeros_(self.temporal_projection.weight)
        self.head = ContactHead(width, predict_distance=False)
        with torch.no_grad():
            self.head.shared.load_state_dict(backbone.head.shared.state_dict())
            self.head.contact.load_state_dict(backbone.head.contact.state_dict())
            self.head.persistence.load_state_dict(
                backbone.head.persistence.state_dict()
            )
        self.persistence_fusion_weight = float(persistence_fusion_weight)
        self.observation_logit_residual = bool(observation_logit_residual)
        self.observation_hand_body_only = bool(observation_hand_body_only)
        if observation_only_training:
            self.requires_grad_(False)
            if self.observation_encoder is None or self.observation_contact_delta is None:
                raise ValueError(
                    "observation_only_training requires observation_logit_residual"
                )
            self.observation_encoder.requires_grad_(True)
            self.observation_encoder.head.requires_grad_(False)
            self.observation_contact_delta.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen geometry provider must remain deterministic even while the
        # temporal contact adapter is optimized.
        self.backbone.eval()
        return self

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_valid: torch.Tensor,
        observation_edge_features: torch.Tensor | None = None,
        observation_edge_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.backbone(edge_features, edge_index, edge_valid)
        contact_encoding = (
            self.contact_encoder(edge_features, edge_index, edge_valid)
            if self.contact_encoder is not None
            else base
        )
        tokens = contact_encoding["edge_tokens"]
        observation_tokens = None
        if self.observation_projection is not None or self.observation_encoder is not None:
            if observation_edge_features is None or observation_edge_valid is None:
                raise ValueError("Observation-aware contact requires 2D edge evidence")
            if self.observation_encoder is not None:
                observation_tokens = self.observation_encoder(
                    observation_edge_features, edge_index, observation_edge_valid
                )["edge_tokens"]
                observation = (
                    None
                    if self.observation_logit_residual
                    else self.observation_graph_gate(observation_tokens)
                )
            else:
                observation = self.observation_projection(observation_edge_features)
            if observation is not None:
                tokens = tokens + observation * observation_edge_valid[..., None].to(
                    observation.dtype
                )
        if tokens.ndim != 4:
            raise ValueError("temporal contact refinement requires shape (B,T,E,W)")
        batch, frames, edges, width = tokens.shape
        sequence = tokens.transpose(1, 2).reshape(batch * edges, frames, width)
        temporal = self.temporal(sequence)[0]
        temporal = self.temporal_projection(temporal)
        temporal = temporal.reshape(batch, edges, frames, width).transpose(1, 2)
        refined = tokens + temporal * edge_valid[..., None].to(temporal.dtype)
        contact = self.head(refined)
        if self.observation_logit_residual:
            if observation_tokens is None or observation_edge_features is None:
                raise AssertionError("Missing observation tokens for logit residual")
            # The last six evidence channels are initializer reprojection
            # residuals. Exact-zero residuals keep generic-domain logits bitwise
            # unchanged, containing the recovery to real residual evidence.
            residual_active = observation_edge_features[..., -6:].abs().sum(dim=-1) > 0
            if self.observation_hand_body_only:
                fixed_edges = edge_index[0] if edge_index.ndim == 3 else edge_index
                source, target = fixed_edges
                hand_body = (source >= 10) ^ (target >= 10)
                residual_active = residual_active & hand_body
            delta = self.observation_contact_delta(observation_tokens).squeeze(-1)
            contact["contact_logits"] = contact["contact_logits"] + torch.where(
                residual_active, delta, torch.zeros_like(delta)
            )
        result = dict(base)
        result["contact_logits"] = contact["contact_logits"]
        result["persistence_logits"] = contact["persistence_logits"]
        result["guided_contact_logits"] = (
            contact["contact_logits"]
            + self.persistence_fusion_weight * contact["persistence_logits"]
        )
        result["temporal_edge_tokens"] = refined
        return result
