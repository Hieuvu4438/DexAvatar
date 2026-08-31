"""Frozen video-pose form consistency scorer with counterfactual margin support."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FormConsistencyScorer(nn.Module):
    """Scores precomputed frozen-video features against pose sequences.

    The implementation deliberately does not claim a SignDINO checkpoint.  The
    video feature provenance is stored alongside each feature cache; an exact
    SignDINO checkpoint can be plugged in when one is publicly available.
    """

    def __init__(
        self, video_feature_dim: int, pose_feature_dim: int, hidden_size: int = 256,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.video = nn.Sequential(
            nn.LayerNorm(video_feature_dim), nn.Linear(video_feature_dim, hidden_size),
            nn.GELU(), nn.Linear(hidden_size, embedding_dim)
        )
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.pose_input = nn.Linear(pose_feature_dim, hidden_size)
        self.pose_encoder = nn.TransformerEncoder(encoder, num_layers=3)
        self.pose_output = nn.Linear(hidden_size, embedding_dim)

    def encode_video(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.video(features.mean(dim=1)), dim=-1)

    def encode_pose(
        self, features: torch.Tensor, frame_valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        value = self.pose_input(features)
        value = self.pose_encoder(value, src_key_padding_mask=None if frame_valid is None else ~frame_valid)
        if frame_valid is None:
            pooled = value.mean(dim=1)
        else:
            weight = frame_valid[..., None].float()
            pooled = (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.pose_output(pooled), dim=-1)

    def forward(
        self, video_features: torch.Tensor, pose_features: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_video(video_features) @ self.encode_pose(pose_features, frame_valid).T

    @staticmethod
    def symmetric_nce(similarity: torch.Tensor, temperature: float) -> torch.Tensor:
        target = torch.arange(similarity.shape[0], device=similarity.device)
        logits = similarity / temperature
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    @staticmethod
    def counterfactual_margin(
        positive: torch.Tensor, negative: torch.Tensor, margin: float
    ) -> torch.Tensor:
        return F.relu(margin - positive + negative).mean()

