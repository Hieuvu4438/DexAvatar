"""Factorized joint and temporal residual score network."""

from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(time: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = time[..., None] * frequencies
    value = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return nn.functional.pad(value, (0, width - value.shape[-1]))


class FactorizedBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.regions = (slice(0, 21), slice(21, 36), slice(36, 51))
        self.heads = heads
        self.max_frames = max_frames
        self.relative_temporal_bias = nn.Parameter(
            torch.zeros(heads, 2 * max_frames - 1)
        )
        self.spatial_norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.regions])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.regions])
        self.spatial = nn.ModuleList(
            [
                nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
                for _ in self.regions
            ]
        )
        self.temporal = nn.ModuleList(
            [
                nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
                for _ in self.regions
            ]
        )
        self.cross_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mlp_ratio, width),
        )

    def forward(self, value: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
        batch, frames, _, width = value.shape
        spatial_update = torch.zeros_like(value)
        for region, norm, attention in zip(
            self.regions, self.spatial_norms, self.spatial, strict=True
        ):
            region_value = norm(value[..., region, :])
            tokens = region_value.reshape(batch * frames, region_value.shape[-2], width)
            tokens = attention(tokens, tokens, tokens, need_weights=False)[0]
            spatial_update[..., region, :] = tokens.reshape(
                batch, frames, region_value.shape[-2], width
            )
        value = value + spatial_update

        groups = torch.stack(
            (
                value[..., :21, :].mean(dim=-2),
                value[..., 21:36, :].mean(dim=-2),
                value[..., 36:51, :].mean(dim=-2),
                value[..., 19, :],
                value[..., 20, :],
                value.mean(dim=-2),
            ),
            dim=-2,
        )
        groups = self.cross_norm(groups).reshape(batch * frames, 6, width)
        groups = self.cross(groups, groups, groups, need_weights=False)[0].reshape(
            batch, frames, 6, width
        )
        cross_update = torch.zeros_like(value)
        cross_update[..., :21, :] = groups[..., 0, None, :] + groups[..., 5, None, :]
        cross_update[..., 21:36, :] = groups[..., 1, None, :] + groups[..., 5, None, :]
        cross_update[..., 36:51, :] = groups[..., 2, None, :] + groups[..., 5, None, :]
        cross_update[..., 19, :] += groups[..., 3, :]
        cross_update[..., 20, :] += groups[..., 4, :]
        value = value + cross_update

        temporal_update = torch.zeros_like(value)
        frame_index = torch.arange(frames, device=value.device)
        relative_index = (
            (frame_index[:, None] - frame_index[None, :]).clamp(
                -self.max_frames + 1, self.max_frames - 1
            )
            + self.max_frames
            - 1
        )
        relative_bias = self.relative_temporal_bias[:, relative_index]
        for region, norm, attention in zip(
            self.regions, self.temporal_norms, self.temporal, strict=True
        ):
            region_value = norm(value[..., region, :])
            region_joints = region_value.shape[-2]
            tokens = region_value.transpose(1, 2).reshape(
                batch * region_joints, frames, width
            )
            padding = ~frame_valid[:, None, :].expand(-1, region_joints, -1).reshape(
                batch * region_joints, frames
            )
            attention_bias = relative_bias[None].expand(
                batch * region_joints, -1, -1, -1
            )
            attention_bias = attention_bias.masked_fill(
                padding[:, None, None, :], torch.finfo(value.dtype).min
            ).reshape(batch * region_joints * self.heads, frames, frames)
            tokens = attention(
                tokens,
                tokens,
                tokens,
                attn_mask=attention_bias,
                need_weights=False,
            )[0]
            temporal_update[..., region, :] = tokens.reshape(
                batch, region_joints, frames, width
            ).transpose(1, 2)
        value = value + temporal_update
        return value + self.mlp(self.mlp_norm(value))


class TemporalScoreNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int = 45,
        width: int = 384,
        blocks: int = 8,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        relation_width: int = 128,
        max_frames: int = 64,
        activation_checkpointing: bool = True,
        masked_rotation_hints: bool = False,
    ) -> None:
        super().__init__()
        self.state = nn.Linear(6, width)
        self.observation = nn.Linear(observation_dim, width)
        self.corruption_observation = (
            nn.Linear(18, width, bias=False) if masked_rotation_hints else None
        )
        if self.corruption_observation is not None:
            nn.init.zeros_(self.corruption_observation.weight)
        self.time = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.relation = nn.Linear(relation_width, width)
        self.activation_checkpointing = activation_checkpointing
        self.blocks = nn.ModuleList(
            [
                FactorizedBlock(width, heads, mlp_ratio, dropout, max_frames)
                for _ in range(blocks)
            ]
        )
        self.output = nn.Linear(width, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        observation: torch.Tensor,
        relation_token: torch.Tensor,
        frame_valid: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
        rotation_hint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hint = None
        if rotation_hint_mask is not None:
            if self.corruption_observation is None:
                raise ValueError(
                    "rotation_hint_mask requires masked_rotation_hints=true"
                )
            if rotation_hint_mask.shape != observation.shape[:-1]:
                raise ValueError("rotation_hint_mask must have shape (B,T,51)")
            hint = observation[..., :18] * rotation_hint_mask[..., None].to(
                observation.dtype
            )
        if condition_mask is not None:
            observation = observation * condition_mask[..., None].to(observation.dtype)
        token = self.state(state) + self.observation(observation)
        if hint is not None:
            token = token + self.corruption_observation(hint)
        token = (
            token
            + self.time(timestep_embedding(time, token.shape[-1]))[:, None, None, :]
        )
        token = token + self.relation(relation_token)[:, :, None, :]
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                token = torch.utils.checkpoint.checkpoint(
                    block, token, frame_valid, use_reentrant=False
                )
            else:
                token = block(token, frame_valid)
        return self.output(token) * frame_valid[..., None, None]
