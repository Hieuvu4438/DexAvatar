"""Reliability-aware factorized whole-sequence residual refiner."""

from __future__ import annotations

import math

import torch
from torch import nn

from phase2_refiner.data.cache_schema import NUM_JOINTS
from phase2_refiner.data.dataset import (
    REPROJECTION_RESIDUAL_2D,
    TOKEN_FEATURE_DIM,
    TORSO_POSITION,
    U0_RELIABILITY,
)
from phase2_refiner.geometry.rotations import compose_residual
from phase2_refiner.geometry.palm import palm_normal
from phase2_refiner.models.embeddings import ObservationTokenEmbedding
from phase2_refiner.models.heads import ResidualHeads
from phase2_refiner.models.reliability import (
    LearnedReliabilityHead,
    effective_reliability,
)


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
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.spatial_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_size)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.relative_temporal_bias = nn.Embedding(2 * max_frames - 1, 1)
        nn.init.zeros_(self.relative_temporal_bias.weight)
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

    def _temporal_mask(
        self, frames: int, device: torch.device, dtype: torch.dtype, causal: bool
    ) -> torch.Tensor:
        positions = torch.arange(frames, device=device)
        relative = positions[None] - positions[:, None] + self.max_frames - 1
        bias = self.relative_temporal_bias(relative).squeeze(-1).to(dtype)
        if causal:
            bias = bias.masked_fill(
                torch.ones(frames, frames, device=device, dtype=torch.bool).triu(1),
                float("-inf"),
            )
        return bias

    def forward(
        self,
        value: torch.Tensor,
        frame_valid: torch.Tensor,
        group_ids: torch.Tensor,
        reliability: torch.Tensor,
        causal: bool,
        temporal_attention: bool,
    ) -> torch.Tensor:
        batch, frames, joints, hidden = value.shape
        reliable = reliability[..., None].clamp(0.0, 1.0)

        spatial_query = self.spatial_norm(value)
        spatial_key_value = spatial_query * (0.1 + 0.9 * reliable)
        spatial, _ = self.spatial_attention(
            spatial_query.reshape(batch * frames, joints, hidden),
            spatial_key_value.reshape(batch * frames, joints, hidden),
            spatial_key_value.reshape(batch * frames, joints, hidden),
            need_weights=False,
        )
        value = value + spatial.reshape(batch, frames, joints, hidden)

        if temporal_attention:
            temporal_query = self.temporal_norm(value).permute(0, 2, 1, 3)
            temporal_reliable = reliable.permute(0, 2, 1, 3)
            temporal_key_value = temporal_query * (0.1 + 0.9 * temporal_reliable)
            temporal_query = temporal_query.reshape(batch * joints, frames, hidden)
            temporal_key_value = temporal_key_value.reshape(batch * joints, frames, hidden)
            padding_bool = (
                (~frame_valid)[:, None, :]
                .expand(batch, joints, frames)
                .reshape(batch * joints, frames)
            )
            padding = torch.zeros(
                padding_bool.shape, device=value.device, dtype=value.dtype
            ).masked_fill(padding_bool, float("-inf"))
            temporal, _ = self.temporal_attention(
                temporal_query,
                temporal_key_value,
                temporal_key_value,
                key_padding_mask=padding,
                attn_mask=self._temporal_mask(frames, value.device, value.dtype, causal),
                need_weights=False,
            )
            temporal = temporal.reshape(batch, joints, frames, hidden).permute(0, 2, 1, 3)
            value = value + temporal

        normalized = self.group_norm(value)
        group_tokens = []
        for group in range(5):
            group_mask = group_ids == group
            group_weight = reliable[:, :, group_mask]
            group_tokens.append(
                (normalized[:, :, group_mask] * group_weight).sum(dim=2)
                / group_weight.sum(dim=2).clamp_min(1e-4)
            )
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
    """Predict bounded SO(3) residuals with U0/U1 reliability conditioning."""

    def __init__(
        self,
        input_dim: int = TOKEN_FEATURE_DIM,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_frames: int = 64,
        predict_uncertainty: bool = False,
        predict_benefit: bool = False,
        uncertainty_feedback: bool = True,
        use_reprojection_skip: bool = False,
        causal: bool = False,
        temporal_attention: bool = True,
        body_max_degrees: float = 25.0,
        hand_max_degrees: float = 35.0,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.predict_uncertainty = predict_uncertainty
        self.predict_benefit = bool(predict_benefit)
        self.uncertainty_feedback = uncertainty_feedback
        self.causal = causal
        self.temporal_attention = bool(temporal_attention)
        self.token_embedding = ObservationTokenEmbedding(
            input_dim, hidden_size, max_frames, NUM_JOINTS
        )
        self.reprojection_skip = (
            nn.Linear(NUM_JOINTS * 2, NUM_JOINTS * 3)
            if use_reprojection_skip and input_dim >= REPROJECTION_RESIDUAL_2D.stop
            else None
        )
        if self.reprojection_skip is not None:
            nn.init.zeros_(self.reprojection_skip.weight)
            nn.init.zeros_(self.reprojection_skip.bias)
        self.reliability_head = (
            LearnedReliabilityHead(hidden_size) if predict_uncertainty else None
        )
        self.blocks = nn.ModuleList(
            FactorizedBlock(
                hidden_size, num_heads, mlp_ratio, dropout, max_frames=max_frames
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.heads = ResidualHeads(hidden_size)
        self.benefit_head = nn.Linear(hidden_size, 1) if self.predict_benefit else None
        if self.benefit_head is not None:
            nn.init.zeros_(self.benefit_head.weight)
            nn.init.zeros_(self.benefit_head.bias)

        group_ids = default_group_ids()
        max_angles = torch.full((NUM_JOINTS,), math.radians(body_max_degrees))
        max_angles[21:] = math.radians(hand_max_degrees)
        self.register_buffer("group_ids", group_ids, persistent=True)
        self.register_buffer("max_angles", max_angles, persistent=True)

    @property
    def delta_head(self) -> nn.Linear:
        """Backward-compatible access for existing checkpoints/tests."""
        return self.heads.delta

    def forward(
        self,
        features: torch.Tensor,
        initial_matrix: torch.Tensor,
        frame_valid: torch.Tensor,
        refine_mask: torch.Tensor,
        initial_joint_position: torch.Tensor | None = None,
        uncertainty_offset: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or features.shape[2] != NUM_JOINTS:
            raise ValueError(
                f"Expected features [B,T,{NUM_JOINTS},F], got {features.shape}"
            )
        batch, frames, joints, feature_dim = features.shape
        if feature_dim != self.token_embedding.input_projection.in_features:
            raise ValueError(
                f"Feature dimension {feature_dim} does not match model input "
                f"{self.token_embedding.input_projection.in_features}"
            )
        if frames > self.max_frames:
            raise ValueError(
                f"Sequence length {frames} exceeds max_frames={self.max_frames}"
            )
        fixed_reliability = features[..., U0_RELIABILITY].clamp(0.0, 1.0)
        value = self.token_embedding(features, self.group_ids, fixed_reliability)
        value = value * frame_valid[:, :, None, None]
        rotation_log_variance = observation_log_variance = None
        if self.reliability_head is not None:
            rotation_log_variance, observation_log_variance = self.reliability_head(
                value
            )
            rotation_log_variance = rotation_log_variance + uncertainty_offset
            observation_log_variance = observation_log_variance + uncertainty_offset
        feedback_variance = (
            rotation_log_variance if self.uncertainty_feedback else None
        )
        reliability = (
            effective_reliability(fixed_reliability, feedback_variance)
            * frame_valid[:, :, None]
        )
        for block in self.blocks:
            value = block(
                value,
                frame_valid,
                self.group_ids,
                reliability,
                self.causal,
                self.temporal_attention,
            )
        value = self.output_norm(value)
        raw_delta, gate, position_delta = self.heads(value)
        if self.reprojection_skip is not None:
            reprojection_delta = self.reprojection_skip(
                features[..., REPROJECTION_RESIDUAL_2D].flatten(start_dim=2)
            ).reshape(batch, frames, joints, 3)
            raw_delta = raw_delta + reprojection_delta
        if refine_mask.ndim == 1:
            refine_mask = refine_mask[None].expand(batch, -1)
        apply_mask = refine_mask[:, None, :, None] & frame_valid[:, :, None, None]
        raw_delta = raw_delta * apply_mask
        gate = gate * apply_mask
        position_delta = position_delta * apply_mask
        output_matrix = compose_residual(
            initial_matrix,
            raw_delta,
            gate=gate,
            max_angle=self.max_angles[None, None, :, None],
        )
        if initial_joint_position is None:
            initial_joint_position = features[..., TORSO_POSITION]
        joint_position = initial_joint_position + position_delta
        predicted_palm = torch.stack(
            (
                palm_normal(joint_position[..., 21:36, :], "left"),
                palm_normal(joint_position[..., 36:51, :], "right"),
            ),
            dim=-2,
        )
        result = {
            "matrix": output_matrix,
            "raw_delta": raw_delta,
            "gate": gate,
            "reliability": reliability,
            "joint_position": joint_position,
            "position_delta": position_delta,
            "palm_normal": predicted_palm,
        }
        if rotation_log_variance is not None:
            result["log_variance"] = rotation_log_variance[..., None]
            result["observation_log_variance"] = observation_log_variance
        if self.benefit_head is not None:
            joint_benefit = self.benefit_head(value).squeeze(-1)
            result["benefit_logit"] = torch.stack(
                (
                    joint_benefit[..., :21].mean(dim=-1),
                    joint_benefit[..., 21:36].mean(dim=-1),
                    joint_benefit[..., 36:51].mean(dim=-1),
                ),
                dim=-1,
            )
        return result
