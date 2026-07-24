"""Joint, group, time, and observation token embeddings."""

from __future__ import annotations

import torch
from torch import nn


class ObservationTokenEmbedding(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int, max_frames: int, num_joints: int
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.joint_embedding = nn.Embedding(num_joints, hidden_size)
        self.time_embedding = nn.Embedding(max_frames, hidden_size)
        self.group_embedding = nn.Embedding(5, hidden_size)
        self.reliability_embedding = nn.Linear(1, hidden_size, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        _, frames, joints, _ = features.shape
        joint_ids = torch.arange(joints, device=features.device)
        time_ids = torch.arange(frames, device=features.device)
        value = self.input_projection(features)
        value = value + self.joint_embedding(joint_ids)[None, None]
        value = value + self.time_embedding(time_ids)[None, :, None]
        value = value + self.group_embedding(group_ids)[None, None]
        return value + self.reliability_embedding(reliability[..., None])
