"""Factorized temporal/kinematic conditional rectified flow."""

from __future__ import annotations

import math

import torch
from torch import nn

from cusp_sl.geometry import compose_right, joint_max_angles


def _time_embedding(time: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequency = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angle = time[..., None] * frequency
    value = torch.cat((angle.sin(), angle.cos()), dim=-1)
    return nn.functional.pad(value, (0, width - value.shape[-1]))


class FactorizedBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(hidden)
        self.joint_norm = nn.LayerNorm(hidden)
        self.temporal = nn.MultiheadAttention(hidden, heads, dropout, batch_first=True)
        self.joint = nn.MultiheadAttention(hidden, heads, dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(hidden * mlp_ratio), hidden), nn.Dropout(dropout)
        )

    def forward(self, value: torch.Tensor, frame_valid: torch.Tensor | None) -> torch.Tensor:
        b, t, j, h = value.shape
        x = self.temporal_norm(value).permute(0, 2, 1, 3).reshape(b * j, t, h)
        padding = None
        if frame_valid is not None:
            padding = (~frame_valid)[:, None].expand(b, j, t).reshape(b * j, t)
        x = self.temporal(x, x, x, key_padding_mask=padding, need_weights=False)[0]
        value = value + x.reshape(b, j, t, h).permute(0, 2, 1, 3)
        x = self.joint_norm(value).reshape(b * t, j, h)
        x = self.joint(x, x, x, need_weights=False)[0]
        value = value + x.reshape(b, t, j, h)
        return value + self.mlp(self.mlp_norm(value))


class SelectiveResidualFlow(nn.Module):
    def __init__(
        self, condition_dim: int, hidden_size: int = 192, layers: int = 4,
        heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1,
        body_max_degrees: float = 25.0, hand_max_degrees: float = 35.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.body_max_degrees = body_max_degrees
        self.hand_max_degrees = hand_max_degrees
        self.state_input = nn.Linear(3, hidden_size)
        self.condition_input = nn.Linear(condition_dim, hidden_size)
        self.time_input = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.joint_embedding = nn.Embedding(51, hidden_size)
        self.blocks = nn.ModuleList(
            [FactorizedBlock(hidden_size, heads, mlp_ratio, dropout) for _ in range(layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, state: torch.Tensor, flow_time: torch.Tensor, condition: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.shape[:-1] != condition.shape[:-1] or state.shape[-2] != 51:
            raise ValueError("state and condition must be [B,T,51,*]")
        b = state.shape[0]
        if flow_time.ndim == 0:
            flow_time = flow_time.expand(b)
        time = self.time_input(_time_embedding(flow_time.float(), self.hidden_size))
        joints = self.joint_embedding(torch.arange(51, device=state.device))
        value = self.state_input(state) + self.condition_input(condition)
        value = value + time[:, None, None] + joints[None, None]
        for block in self.blocks:
            value = block(value, frame_valid)
        return self.output(value)

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, base_rotation: torch.Tensor, gate: torch.Tensor,
        frame_valid: torch.Tensor, steps: int, generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.randn(
            condition.shape[:-1] + (3,), device=condition.device,
            dtype=condition.dtype, generator=generator
        )
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((condition.shape[0],), index * dt, device=condition.device)
            state = state + dt * self(state, time, condition, frame_valid)
        maximum = joint_max_angles(
            state.device, state.dtype, self.body_max_degrees, self.hand_max_degrees
        )
        rotation = compose_right(base_rotation, state, gate=gate, max_angle=maximum)
        rotation = torch.where(frame_valid[:, :, None, None, None], rotation, base_rotation)
        return state, rotation

