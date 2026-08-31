from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from signpk.geometry.rotations import rotation_6d_to_matrix

from .explicit_tokens import ExplicitTokenBatch
from .gates import SoftGate


@dataclass
class PKCOutput:
    upper_rot6d_residual: Tensor
    left_rot6d_residual: Tensor
    right_rot6d_residual: Tensor
    upper_rotmat: Tensor
    left_rotmat: Tensor
    right_rotmat: Tensor
    root_depth_residual: Tensor
    logvar_upper: Tensor
    logvar_left: Tensor
    logvar_right: Tensor
    logvar_palm: Tensor
    phase_gate: Tensor
    interaction_gate: Tensor
    angular_velocity: Tensor
    wrist_velocity: Tensor


class TimestampEncoding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, timestamps: Tensor) -> Tensor:
        relative = (
            timestamps - timestamps[:, timestamps.shape[1] // 2 : timestamps.shape[1] // 2 + 1]
        )
        scale = relative.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        normalized = relative / scale
        features = torch.stack(
            [normalized, torch.sin(torch.pi * normalized), torch.cos(torch.pi * normalized)], dim=-1
        )
        return self.projection(features)


class BiasedCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, query_count: int, context_count: int):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.bias = nn.Parameter(
            _initial_kinematic_bias(query_count, context_count).repeat(heads, 1, 1)
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        batch, queries, hidden = query.shape
        keys = context.shape[1]
        q = self.query(query).view(batch, queries, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch, keys, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch, keys, self.heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax((q @ k.transpose(-1, -2)) * self.scale + self.bias[None], dim=-1)
        result = (attention @ v).transpose(1, 2).reshape(batch, queries, hidden)
        return self.output(result)


def _initial_kinematic_bias(query_count: int, context_count: int) -> Tensor:
    # Queries: root/spines/neck/head/L-clav/L-shoulder/L-elbow/L-wrist and right chain.
    # Context: 15 left fingers, 15 right fingers, one relation token.
    left_distance = torch.tensor([6, 5, 4, 3, 3, 4, 2, 5, 1, 5, 0, 6, 6, 6], dtype=torch.float32)
    right_distance = torch.tensor([6, 5, 4, 3, 3, 4, 5, 2, 5, 1, 6, 0, 6, 6], dtype=torch.float32)
    if query_count != 14 or context_count != 31:
        return torch.zeros(1, query_count, context_count)
    finger_depth = torch.arange(15, dtype=torch.float32).remainder(3) * 0.15
    bias = torch.cat(
        [
            -(left_distance[:, None] + finger_depth[None]),
            -(right_distance[:, None] + finger_depth[None]),
            -torch.minimum(left_distance, right_distance)[:, None] * 0.25,
        ],
        dim=-1,
    )
    return bias[None]


class PalmKinematicCoupler(nn.Module):
    """Bidirectional residual hand-to-upper-body adapter from the proposal."""

    def __init__(
        self,
        body_feature_dim: int = 12,
        hand_feature_dim: int = 54,
        relation_feature_dim: int = 20,
        omni_feature_dim: int = 1024,
        h4w_hand_feature_dim: int = 0,
        body_observer_feature_dim: int = 0,
        hidden_dim: int = 256,
        temporal_layers: int = 4,
        attention_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        upper_body_joints: int = 14,
        hand_joints: int = 15,
        logvar_min: float = -6.0,
        logvar_max: float = 4.0,
        **_: object,
    ):
        super().__init__()
        self.upper_body_joints = upper_body_joints
        self.hand_joints = hand_joints
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.body_projection = nn.Linear(body_feature_dim, hidden_dim)
        self.left_projection = nn.Linear(hand_feature_dim, hidden_dim)
        self.right_projection = nn.Linear(hand_feature_dim, hidden_dim)
        self.relation_projection = nn.Linear(relation_feature_dim, hidden_dim)
        self.left_observer_projection = (
            nn.Sequential(nn.LayerNorm(omni_feature_dim), nn.Linear(omni_feature_dim, hidden_dim))
            if omni_feature_dim > 0
            else None
        )
        self.right_observer_projection = (
            nn.Sequential(nn.LayerNorm(omni_feature_dim), nn.Linear(omni_feature_dim, hidden_dim))
            if omni_feature_dim > 0
            else None
        )
        self.left_h4w_projection = (
            nn.Sequential(
                nn.LayerNorm(h4w_hand_feature_dim),
                nn.Linear(h4w_hand_feature_dim, hidden_dim),
            )
            if h4w_hand_feature_dim > 0
            else None
        )
        self.right_h4w_projection = (
            nn.Sequential(
                nn.LayerNorm(h4w_hand_feature_dim),
                nn.Linear(h4w_hand_feature_dim, hidden_dim),
            )
            if h4w_hand_feature_dim > 0
            else None
        )
        self.body_observer_projection = (
            nn.Sequential(
                nn.LayerNorm(body_observer_feature_dim),
                nn.Linear(body_observer_feature_dim, hidden_dim),
            )
            if body_observer_feature_dim > 0
            else None
        )
        self.body_joint_embedding = nn.Parameter(torch.randn(upper_body_joints, hidden_dim) * 0.02)
        self.hand_joint_embedding = nn.Parameter(torch.randn(hand_joints, hidden_dim) * 0.02)
        self.side_embedding = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)
        self.time_encoding = TimestampEncoding(hidden_dim)
        hand_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            attention_heads,
            hidden_dim * mlp_ratio,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.left_hand_encoder = nn.TransformerEncoder(
            hand_layer, num_layers=2, enable_nested_tensor=False
        )
        self.right_hand_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim,
                attention_heads,
                hidden_dim * mlp_ratio,
                dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.left_to_right = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.right_to_left = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.phase_head = SoftGate(relation_feature_dim)
        self.interaction_head = SoftGate(relation_feature_dim)
        self.hand_to_body = BiasedCrossAttention(
            hidden_dim, attention_heads, upper_body_joints, hand_joints * 2 + 1
        )
        temporal_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            attention_heads,
            hidden_dim * mlp_ratio,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers, enable_nested_tensor=False
        )
        self.upper_rotation_head = nn.Linear(hidden_dim, 6)
        self.left_rotation_head = nn.Linear(hidden_dim, 6)
        self.right_rotation_head = nn.Linear(hidden_dim, 6)
        self.root_depth_head = nn.Linear(hidden_dim, 1)
        self.upper_logvar_head = nn.Linear(hidden_dim, 1)
        self.left_logvar_head = nn.Linear(hidden_dim, 1)
        self.right_logvar_head = nn.Linear(hidden_dim, 1)
        self.palm_logvar_head = nn.Linear(hidden_dim * 2, 2)
        self.angular_velocity_head = nn.Linear(hidden_dim, 3)
        self.wrist_velocity_head = nn.Linear(hidden_dim * 2, 6)
        self._initialize_residual_heads()

    def _initialize_residual_heads(self) -> None:
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        for head in (self.upper_rotation_head, self.left_rotation_head, self.right_rotation_head):
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(identity)
        nn.init.zeros_(self.root_depth_head.weight)
        nn.init.zeros_(self.root_depth_head.bias)

    def _temporal_encode(self, tokens: Tensor, time_encoding: Tensor) -> Tensor:
        batch, time, joints, hidden = tokens.shape
        values = tokens.permute(0, 2, 1, 3).reshape(batch * joints, time, hidden)
        positions = (
            time_encoding[:, None].expand(-1, joints, -1, -1).reshape(batch * joints, time, hidden)
        )
        encoded = self.temporal(values + positions)
        return encoded.reshape(batch, joints, time, hidden).permute(0, 2, 1, 3)

    def forward(self, window: ExplicitTokenBatch) -> PKCOutput:
        window.validate()
        batch, time = window.timestamps.shape
        body = (
            self.body_projection(window.body)
            + self.body_joint_embedding[None, None]
            + self.side_embedding[0]
        )
        left = (
            self.left_projection(window.left)
            + self.hand_joint_embedding[None, None]
            + self.side_embedding[1]
        )
        right = (
            self.right_projection(window.right)
            + self.hand_joint_embedding[None, None]
            + self.side_embedding[2]
        )
        if window.left_observer_feature is not None:
            if self.left_observer_projection is None:
                raise ValueError("left observer features supplied but omni_feature_dim is disabled")
            left = (
                left
                + self.left_observer_projection(window.left_observer_feature.to(left))[:, :, None]
            )
        if window.right_observer_feature is not None:
            if self.right_observer_projection is None:
                raise ValueError(
                    "right observer features supplied but omni_feature_dim is disabled"
                )
            right = (
                right
                + self.right_observer_projection(window.right_observer_feature.to(right))[
                    :, :, None
                ]
            )
        if window.left_h4w_feature is not None:
            if self.left_h4w_projection is None:
                raise ValueError(
                    "left H4W++ features supplied but h4w_hand_feature_dim is disabled"
                )
            left = left + self.left_h4w_projection(window.left_h4w_feature.to(left))[:, :, None]
        if window.right_h4w_feature is not None:
            if self.right_h4w_projection is None:
                raise ValueError(
                    "right H4W++ features supplied but h4w_hand_feature_dim is disabled"
                )
            right = (
                right + self.right_h4w_projection(window.right_h4w_feature.to(right))[:, :, None]
            )
        if window.body_observer_feature is not None:
            if self.body_observer_projection is None:
                raise ValueError(
                    "body observer features supplied but body_observer_feature_dim is disabled"
                )
            observer_body = window.body_observer_feature
            if observer_body.ndim == 3:
                observer_body = observer_body[:, :, None].expand(-1, -1, self.upper_body_joints, -1)
            body = body + self.body_observer_projection(observer_body.to(body))
        relation = self.relation_projection(window.relation) + self.side_embedding[3]
        flat_left = left.reshape(batch * time, self.hand_joints, -1)
        flat_right = right.reshape(batch * time, self.hand_joints, -1)
        flat_left = self.left_hand_encoder(flat_left)
        flat_right = self.right_hand_encoder(flat_right)
        left_cross = self.right_to_left(flat_left, flat_right, flat_right, need_weights=False)[0]
        right_cross = self.left_to_right(flat_right, flat_left, flat_left, need_weights=False)[0]
        both_valid = (window.left_valid & window.right_valid).reshape(batch * time, 1, 1)
        interaction = self.interaction_head(window.relation, window.left_valid & window.right_valid)
        gate = interaction.reshape(batch * time, 1, 1) * both_valid
        flat_left = flat_left + gate * left_cross
        flat_right = flat_right + gate * right_cross
        left = flat_left.reshape(batch, time, self.hand_joints, -1)
        right = flat_right.reshape(batch, time, self.hand_joints, -1)
        context = torch.cat([left, right, relation[:, :, None]], dim=2).reshape(
            batch * time, 31, -1
        )
        body_flat = body.reshape(batch * time, self.upper_body_joints, -1)
        body = (body_flat + self.hand_to_body(body_flat, context)).reshape(
            batch, time, self.upper_body_joints, -1
        )
        time_encoding = self.time_encoding(window.timestamps)
        body = self._temporal_encode(body, time_encoding)
        left = self._temporal_encode(left, time_encoding)
        right = self._temporal_encode(right, time_encoding)
        center = time // 2
        body_center, left_center, right_center = body[:, center], left[:, center], right[:, center]
        upper_residual = self.upper_rotation_head(body_center)
        left_residual = self.left_rotation_head(left_center)
        right_residual = self.right_rotation_head(right_center)
        upper_delta = rotation_6d_to_matrix(upper_residual)
        left_delta = rotation_6d_to_matrix(left_residual)
        right_delta = rotation_6d_to_matrix(right_residual)
        upper_rotmat = upper_delta @ window.upper_base_rotmat[:, center]
        left_rotmat = left_delta @ window.left_base_rotmat[:, center]
        right_rotmat = right_delta @ window.right_base_rotmat[:, center]
        palm_features = torch.cat([left_center.mean(1), right_center.mean(1)], dim=-1)
        angular_features = torch.cat([body_center, left_center, right_center], dim=1)
        return PKCOutput(
            upper_rot6d_residual=upper_residual,
            left_rot6d_residual=left_residual,
            right_rot6d_residual=right_residual,
            upper_rotmat=upper_rotmat,
            left_rotmat=left_rotmat,
            right_rotmat=right_rotmat,
            root_depth_residual=self.root_depth_head(body_center[:, 0]),
            logvar_upper=self.upper_logvar_head(body_center)
            .squeeze(-1)
            .clamp(self.logvar_min, self.logvar_max),
            logvar_left=self.left_logvar_head(left_center)
            .squeeze(-1)
            .clamp(self.logvar_min, self.logvar_max),
            logvar_right=self.right_logvar_head(right_center)
            .squeeze(-1)
            .clamp(self.logvar_min, self.logvar_max),
            logvar_palm=self.palm_logvar_head(palm_features).clamp(
                self.logvar_min, self.logvar_max
            ),
            phase_gate=self.phase_head(window.relation[:, center]),
            interaction_gate=interaction[:, center],
            angular_velocity=self.angular_velocity_head(angular_features),
            wrist_velocity=self.wrist_velocity_head(palm_features).view(batch, 2, 3),
        )
