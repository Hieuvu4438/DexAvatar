"""One holistic temporal denoiser with part-aware and graph attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dcg_sign4d.diffusion.dposer_normalizer import DPoserXWholeBodyNormalizer
from dcg_sign4d.diffusion.state_codec import rotation_6d_to_matrix
from dcg_sign4d.geometry.so3 import log_map


def timestep_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class PartAwareTrajectoryDenoiser(nn.Module):
    """Part streams exchange information through equal-capacity attention blocks."""

    def __init__(
        self,
        part_dims: tuple[int, ...],
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.0,
        beta_dim: int | None = None,
    ) -> None:
        super().__init__()
        if len(part_dims) < 4 or any(width <= 0 for width in part_dims):
            raise ValueError("part_dims must cover root/body/left/right[/face]")
        self.part_dims = part_dims
        self.input = nn.ModuleList(nn.Linear(width, hidden_dim) for width in part_dims)
        self.output = nn.ModuleList(nn.Linear(hidden_dim, width) for width in part_dims)
        temporal_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * 4,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        part_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * 4,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, layers, enable_nested_tensor=False)
        self.cross_part = nn.TransformerEncoder(part_layer, layers, enable_nested_tensor=False)
        self.contact_attention = nn.MultiheadAttention(hidden_dim, heads, dropout, batch_first=True)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.reliability_mlp = nn.Linear(1, hidden_dim)
        self.shape_mlp = (
            nn.Linear(beta_dim, len(part_dims) * hidden_dim) if beta_dim is not None else None
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        contact_tokens: Tensor,
        valid_mask: Tensor,
        reliability: Tensor | None = None,
        frame_condition: Tensor | None = None,
        shape: Tensor | None = None,
    ) -> Tensor:
        batch, time, width = noisy.shape
        if sum(self.part_dims) != width:
            raise ValueError("trajectory width does not match part_dims")
        if valid_mask.shape != (batch, time):
            raise ValueError("valid_mask must be [B,T]")
        parts = noisy.split(self.part_dims, dim=-1)
        hidden = torch.stack(
            [projection(part) for projection, part in zip(self.input, parts, strict=True)], dim=2
        )
        part_count = hidden.shape[2]
        if frame_condition is not None:
            if frame_condition.shape != hidden.shape:
                raise ValueError("frame_condition must be [B,T,P,H]")
            hidden = hidden + frame_condition
        if self.shape_mlp is not None:
            if shape is None or shape.shape != (batch, self.shape_mlp.in_features):
                raise ValueError("shape conditioning must be [B,n_beta]")
            shape_condition = self.shape_mlp(shape).reshape(batch, 1, part_count, -1)
            hidden = hidden + shape_condition
        time_token = self.time_mlp(timestep_embedding(timesteps, hidden.shape[-1]))
        hidden = hidden + time_token[:, None, None, :]
        if reliability is not None:
            if reliability.shape != (batch, time):
                raise ValueError("reliability must be [B,T]")
            hidden = hidden + self.reliability_mlp(reliability[..., None])[:, :, None, :]
        temporal = hidden.permute(0, 2, 1, 3).reshape(batch * part_count, time, -1)
        padding = ~valid_mask[:, None, :].expand(batch, part_count, time).reshape(
            batch * part_count, time
        )
        temporal = self.temporal(temporal, src_key_padding_mask=padding)
        hidden = temporal.reshape(batch, part_count, time, -1).permute(0, 2, 1, 3)
        cross = hidden.reshape(batch * time, part_count, -1)
        cross = self.cross_part(cross)
        contact = contact_tokens.reshape(batch * time, contact_tokens.shape[2], -1)
        attended, _ = self.contact_attention(cross, contact, contact, need_weights=False)
        hidden = self.norm(cross + attended).reshape(batch, time, part_count, -1)
        outputs = [projection(hidden[:, :, index]) for index, projection in enumerate(self.output)]
        result = torch.cat(outputs, dim=-1)
        return result * valid_mask[..., None]


class DPoserXConditionedTrajectoryDenoiser(nn.Module):
    """Single holistic DCG denoiser with a frozen official DPoser-X backbone.

    DPoser-X supplies a pretrained per-frame pose feature. A trainable temporal,
    cross-part and contact graph trunk jointly predicts every trajectory channel;
    this is one denoiser, not an independently optimized post-hoc pose prior.
    """

    PRODUCTION_PART_DIMS = (12, 126, 90, 90, 19)

    def __init__(
        self,
        official_bridge: nn.Module,
        *,
        trajectory_steps: int,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if trajectory_steps < 2:
            raise ValueError("trajectory_steps must be at least 2")
        self.official_bridge = official_bridge
        self.trajectory_steps = trajectory_steps
        self.dposer_projection = nn.Linear(
            DPoserXWholeBodyNormalizer.TOTAL_DIMENSION,
            len(self.PRODUCTION_PART_DIMS) * hidden_dim,
        )
        self.trajectory = PartAwareTrajectoryDenoiser(
            self.PRODUCTION_PART_DIMS,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            beta_dim=10,
        )

    @staticmethod
    def _dposer_parts(noisy: Tensor) -> dict[str, Tensor]:
        if noisy.ndim != 3 or noisy.shape[-1] != sum(
            DPoserXConditionedTrajectoryDenoiser.PRODUCTION_PART_DIMS
        ):
            raise ValueError("production DCG trajectory must be [B,T,337]")
        batch, time = noisy.shape[:2]
        _, body, left, right, face = noisy.split(
            DPoserXConditionedTrajectoryDenoiser.PRODUCTION_PART_DIMS, dim=-1
        )

        def axis(value: Tensor, joints: int) -> Tensor:
            rotation = value.reshape(batch * time, joints, 6)
            return log_map(rotation_6d_to_matrix(rotation)).flatten(1)

        expression = noisy.new_zeros(batch * time, 100)
        expression[:, :10] = face.reshape(batch * time, 19)[:, 9:19]
        return {
            "body_pose": axis(body, 21),
            "left_hand_pose": axis(left, 15),
            "right_hand_pose": axis(right, 15),
            "jaw_pose": face.reshape(batch * time, 19)[:, :3],
            "expression": expression,
        }

    def forward(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        contact_tokens: Tensor,
        valid_mask: Tensor,
        reliability: Tensor | None = None,
        shape: Tensor | None = None,
    ) -> Tensor:
        batch, time = noisy.shape[:2]
        parts = self._dposer_parts(noisy)
        normalized = self.official_bridge.normalizer.normalize_parts(parts)
        expanded_time = timesteps[:, None].expand(batch, time).reshape(-1)
        dposer_noise = self.official_bridge.predict_noise(
            normalized, expanded_time, trajectory_steps=self.trajectory_steps
        )
        condition = self.dposer_projection(dposer_noise).reshape(
            batch, time, len(self.PRODUCTION_PART_DIMS), -1
        )
        return self.trajectory(
            noisy,
            timesteps,
            contact_tokens,
            valid_mask,
            reliability,
            frame_condition=condition,
            shape=shape,
        )
