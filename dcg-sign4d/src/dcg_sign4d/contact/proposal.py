"""HACO-balanced dynamic contact proposal with temporal edge modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch


@dataclass(frozen=True)
class ContactProposalOutput:
    event_logits: Tensor
    duration_logits: Tensor
    edge_embedding: Tensor


class ContactProposal(nn.Module):
    """Edge-wise temporal Transformer; not the original HACO backbone."""

    def __init__(
        self,
        trajectory_dim: int,
        edge_count: int,
        max_duration: int,
        geometry_dim: int = 5,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.0,
        edge_names: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.edge_count = edge_count
        self.max_duration = max_duration
        if edge_names is None:
            edge_names = tuple(
                (f"source_{index}", f"target_{index}") for index in range(edge_count)
            )
        if len(edge_names) != edge_count:
            raise ValueError("edge-name topology does not match edge_count")
        patch_names = sorted({name for edge in edge_names for name in edge})
        patch_index = {name: index for index, name in enumerate(patch_names)}
        self.register_buffer(
            "source_patch_index",
            torch.tensor([patch_index[source] for source, _ in edge_names], dtype=torch.long),
        )
        self.register_buffer(
            "target_patch_index",
            torch.tensor([patch_index[target] for _, target in edge_names], dtype=torch.long),
        )
        self.pose_projection = nn.Linear(trajectory_dim, hidden_dim)
        self.observation_projection = nn.Linear(12, hidden_dim)
        self.geometry_projection = nn.Linear(geometry_dim, hidden_dim)
        self.patch_embedding = nn.Embedding(len(patch_names), hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.event_head = nn.Linear(hidden_dim, 4)
        self.duration_head = nn.Linear(hidden_dim, max_duration)

    @staticmethod
    def summarize_observations(observations: ObservationBatch) -> Tensor:
        observations.validate()
        valid = observations.keypoint_valid & observations.frame_valid[:, :, None]
        weight = observations.keypoint_reliability * valid
        denominator = weight.sum(-1, keepdim=True).clamp_min(1e-8)
        mean = (observations.keypoints_2d.nan_to_num() * weight[..., None]).sum(-2)
        mean = mean / denominator
        coverage = valid.float().mean(-1, keepdim=True)
        reliability = torch.where(
            valid.any(-1, keepdim=True),
            weight.sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1),
            torch.zeros_like(coverage),
        )
        batch, time = observations.frame_valid.shape
        reference = observations.keypoints_2d

        def zeros(width: int) -> Tensor:
            return reference.new_zeros(batch, time, width)

        if observations.part_masks is None:
            mask_summary = zeros(2)
        else:
            mask_presence = observations.part_masks.float().mean(dim=(-1, -2, -3))[..., None]
            mask_reliability = observations.mask_reliability.mean(-1, keepdim=True)
            mask_summary = torch.cat((mask_presence, mask_reliability), -1)
        if observations.tracks_2d is None:
            track_summary = zeros(4)
        else:
            track_weight = observations.track_reliability
            track_denominator = track_weight.sum(-1, keepdim=True).clamp_min(1e-8)
            track_mean = (observations.tracks_2d * track_weight[..., None]).sum(-2)
            track_mean = track_mean / track_denominator
            track_coverage = (track_weight > 0).float().mean(-1, keepdim=True)
            track_reliability = track_weight.mean(-1, keepdim=True)
            track_summary = torch.cat((track_mean, track_coverage, track_reliability), dim=-1)
        if observations.depth_order is None:
            depth_summary = zeros(2)
        else:
            depth_weight = observations.depth_reliability
            depth_denominator = depth_weight.sum(-1, keepdim=True).clamp_min(1e-8)
            depth_mean = (observations.depth_order * depth_weight).sum(-1, keepdim=True)
            depth_mean = depth_mean / depth_denominator
            depth_reliability = depth_weight.mean(-1, keepdim=True)
            depth_summary = torch.cat((depth_mean, depth_reliability), dim=-1)
        summary = torch.cat(
            (mean, coverage, reliability, mask_summary, track_summary, depth_summary), dim=-1
        )
        return summary * observations.frame_valid[..., None]

    @staticmethod
    def temporal_encoding(time: int, width: int, reference: Tensor) -> Tensor:
        position = torch.arange(time, device=reference.device, dtype=reference.dtype)[:, None]
        half = width // 2
        frequency = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=reference.device, dtype=reference.dtype)
            / max(half - 1, 1)
        )[None]
        angle = position * frequency
        result = torch.cat((angle.sin(), angle.cos()), -1)
        if width % 2:
            result = torch.cat((result, torch.zeros_like(result[:, :1])), -1)
        return result

    def forward(
        self,
        observations: ObservationBatch,
        trajectory: TrajectoryState,
        geometry_features: Tensor,
    ) -> ContactProposalOutput:
        encoded, _ = StateCodec().encode(trajectory)
        batch, time, edges, _ = geometry_features.shape
        if encoded.shape != (batch, time, self.trajectory_dim):
            raise ValueError("trajectory dimension does not match proposal configuration")
        if edges != self.edge_count:
            raise ValueError("edge count mismatch")
        observation = self.summarize_observations(observations)
        pose = self.pose_projection(encoded)[:, :, None, :]
        cue = self.observation_projection(observation)[:, :, None, :]
        geometry = self.geometry_projection(geometry_features)
        source = self.patch_embedding(self.source_patch_index.to(encoded.device))
        target = self.patch_embedding(self.target_patch_index.to(encoded.device))
        edge = (source + target)[None, None, :, :]
        temporal_position = self.temporal_encoding(time, pose.shape[-1], encoded)
        tokens = self.input_norm(pose + cue + geometry + edge + temporal_position[None, :, None, :])
        temporal_input = tokens.permute(0, 2, 1, 3).reshape(batch * edges, time, -1)
        padding = ~trajectory.valid_mask[:, None, :].expand(batch, edges, time).reshape(
            batch * edges, time
        )
        hidden = self.temporal(temporal_input, src_key_padding_mask=padding)
        hidden = hidden.reshape(batch, edges, time, -1).permute(0, 2, 1, 3)
        return ContactProposalOutput(
            event_logits=self.event_head(hidden),
            duration_logits=self.duration_head(hidden),
            edge_embedding=hidden,
        )
