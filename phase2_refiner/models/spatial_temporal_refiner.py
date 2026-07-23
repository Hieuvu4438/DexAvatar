"""Deterministic factorized spatial-temporal residual refiner."""

from __future__ import annotations

import math

import torch
from torch import nn

from phase2_refiner.data.cache_schema import NUM_JOINTS
from phase2_refiner.geometry.rotations import compose_residual


def default_group_ids() -> torch.Tensor:
    """Torso, left arm, right arm, left hand, right hand."""
    groups = torch.zeros(NUM_JOINTS, dtype=torch.long)
    for index in (12, 15, 17, 19):
        groups[index] = 1
    for index in (13, 16, 18, 20):
        groups[index] = 2
    groups[21:36] = 3
    groups[36:51] = 4
    return groups


class FactorizedBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: int, dropout: float
    ) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.spatial_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_size)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.group_norm = nn.LayerNorm(hidden_size)
        self.group_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        frame_valid: torch.Tensor,
        group_ids: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        batch, frames, joints, hidden = value.shape
        spatial = self.spatial_norm(value).reshape(batch * frames, joints, hidden)
        spatial, _ = self.spatial_attention(
            spatial, spatial, spatial, need_weights=False
        )
        value = value + spatial.reshape(batch, frames, joints, hidden)

        temporal = (
            self.temporal_norm(value)
            .permute(0, 2, 1, 3)
            .reshape(batch * joints, frames, hidden)
        )
        padding = (
            (~frame_valid)[:, None, :]
            .expand(batch, joints, frames)
            .reshape(batch * joints, frames)
        )
        attention_mask = None
        if causal:
            attention_mask = torch.ones(
                frames, frames, device=value.device, dtype=torch.bool
            ).triu(1)
        temporal, _ = self.temporal_attention(
            temporal,
            temporal,
            temporal,
            key_padding_mask=padding,
            attn_mask=attention_mask,
            need_weights=False,
        )
        temporal = temporal.reshape(batch, joints, frames, hidden).permute(0, 2, 1, 3)
        value = value + temporal

        normalized = self.group_norm(value)
        group_tokens = []
        for group in range(5):
            group_tokens.append(normalized[:, :, group_ids == group].mean(dim=2))
        group_tokens = torch.stack(group_tokens, dim=2)
        group_flat = group_tokens.reshape(batch * frames, 5, hidden)
        group_flat, _ = self.group_attention(
            group_flat, group_flat, group_flat, need_weights=False
        )
        group_tokens = group_flat.reshape(batch, frames, 5, hidden)
        value = value + group_tokens[:, :, group_ids, :]
        value = value + self.ffn(self.ffn_norm(value))
        return value * frame_valid[:, :, None, None]


class WholeSequenceRefiner(nn.Module):
    """Predict bounded SO(3) residuals while preserving an exact identity initialization."""

    def __init__(
        self,
        input_dim: int = 28,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_frames: int = 64,
        predict_uncertainty: bool = False,
        causal: bool = False,
        body_max_degrees: float = 25.0,
        hand_max_degrees: float = 35.0,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.predict_uncertainty = predict_uncertainty
        self.causal = causal
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.joint_embedding = nn.Embedding(NUM_JOINTS, hidden_size)
        self.time_embedding = nn.Embedding(max_frames, hidden_size)
        self.group_embedding = nn.Embedding(5, hidden_size)
        self.blocks = nn.ModuleList(
            FactorizedBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.delta_head = nn.Linear(hidden_size, 3)
        self.gate_head = nn.Linear(hidden_size, 1)
        self.uncertainty_head = (
            nn.Linear(hidden_size, 1) if predict_uncertainty else None
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        group_ids = default_group_ids()
        max_angles = torch.full((NUM_JOINTS,), math.radians(body_max_degrees))
        max_angles[21:] = math.radians(hand_max_degrees)
        self.register_buffer("group_ids", group_ids, persistent=True)
        self.register_buffer("max_angles", max_angles, persistent=True)

    def forward(
        self,
        features: torch.Tensor,
        initial_matrix: torch.Tensor,
        frame_valid: torch.Tensor,
        refine_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or features.shape[2] != NUM_JOINTS:
            raise ValueError(
                f"Expected features [B,T,{NUM_JOINTS},F], got {features.shape}"
            )
        batch, frames, joints, _ = features.shape
        if frames > self.max_frames:
            raise ValueError(
                f"Sequence length {frames} exceeds max_frames={self.max_frames}"
            )
        joint_ids = torch.arange(joints, device=features.device)
        time_ids = torch.arange(frames, device=features.device)
        value = self.input_projection(features)
        value = value + self.joint_embedding(joint_ids)[None, None]
        value = value + self.time_embedding(time_ids)[None, :, None]
        value = value + self.group_embedding(self.group_ids)[None, None]
        value = value * frame_valid[:, :, None, None]
        for block in self.blocks:
            value = block(value, frame_valid, self.group_ids, self.causal)
        value = self.output_norm(value)
        raw_delta = self.delta_head(value)
        gate = torch.sigmoid(self.gate_head(value))
        if refine_mask.ndim == 1:
            refine_mask = refine_mask[None].expand(batch, -1)
        apply_mask = refine_mask[:, None, :, None] & frame_valid[:, :, None, None]
        raw_delta = raw_delta * apply_mask
        gate = gate * apply_mask
        output_matrix = compose_residual(
            initial_matrix,
            raw_delta,
            gate=gate,
            max_angle=self.max_angles[None, None, :, None],
        )
        result = {
            "matrix": output_matrix,
            "raw_delta": raw_delta,
            "gate": gate,
        }
        if self.uncertainty_head is not None:
            result["log_variance"] = self.uncertainty_head(value).clamp(-8.0, 6.0)
        return result
