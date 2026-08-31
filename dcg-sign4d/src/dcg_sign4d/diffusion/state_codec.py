"""SMPL-X trajectory tensor contract and reversible normalized codec."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor


def rotation_6d_to_matrix(rotation: Tensor) -> Tensor:
    if rotation.shape[-1] != 6:
        raise ValueError("rotation representation must end in 6")
    first = rotation[..., :3]
    second = rotation[..., 3:]
    basis1 = functional.normalize(first, dim=-1)
    basis2 = functional.normalize(second - (basis1 * second).sum(-1, keepdim=True) * basis1, dim=-1)
    basis3 = torch.cross(basis1, basis2, dim=-1)
    return torch.stack((basis1, basis2, basis3), dim=-2)


@dataclass(frozen=True)
class TrajectoryState:
    root_rot6d: Tensor
    root_translation: Tensor
    root_velocity: Tensor
    body_rot6d: Tensor
    left_hand_rot6d: Tensor
    right_hand_rot6d: Tensor
    beta: Tensor
    valid_mask: Tensor
    face_state: Tensor | None = None

    def validate(self) -> TrajectoryState:
        if self.root_rot6d.ndim != 3 or self.root_rot6d.shape[-1] != 6:
            raise ValueError("root_rot6d must be [B,T,6]")
        batch, time = self.root_rot6d.shape[:2]
        if self.root_translation.shape != (batch, time, 3):
            raise ValueError("root_translation must be [B,T,3]")
        if self.root_velocity.shape != (batch, time, 3):
            raise ValueError("root_velocity must be [B,T,3]")
        for name in ("body_rot6d", "left_hand_rot6d", "right_hand_rot6d"):
            value = getattr(self, name)
            if value.ndim != 4 or value.shape[:2] != (batch, time) or value.shape[-1] != 6:
                raise ValueError(f"{name} must be [B,T,J,6]")
        if self.face_state is not None and self.face_state.shape[:2] != (batch, time):
            raise ValueError("face_state batch/time mismatch")
        if self.beta.ndim != 2 or self.beta.shape[0] != batch:
            raise ValueError("beta must be clip-shared [B,n_beta]")
        if self.valid_mask.shape != (batch, time) or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,T]")
        for field in (
            self.root_rot6d,
            self.root_translation,
            self.root_velocity,
            self.body_rot6d,
            self.left_hand_rot6d,
            self.right_hand_rot6d,
        ):
            if not torch.isfinite(field[self.valid_mask]).all():
                raise ValueError("valid trajectory contains NaN/Inf")
        return self

    def with_root_translation(self, translation: Tensor) -> TrajectoryState:
        return replace(self, root_translation=translation)


@dataclass(frozen=True)
class CodecContext:
    template: TrajectoryState
    widths: tuple[int, ...]
    root_origin: Tensor


class StateCodec:
    def __init__(self, mean: Tensor | None = None, std: Tensor | None = None):
        self.mean = mean
        self.std = std

    def encode(self, state: TrajectoryState) -> tuple[Tensor, CodecContext]:
        state.validate()
        batch = state.valid_mask.shape[0]
        first_valid = state.valid_mask.float().argmax(dim=1)
        has_valid = state.valid_mask.any(dim=1)
        root_origin = state.root_translation[
            torch.arange(batch, device=state.root_translation.device), first_valid
        ]
        root_origin = torch.where(has_valid[:, None], root_origin, torch.zeros_like(root_origin))
        relative_translation = state.root_translation - root_origin[:, None, :]
        fields = [
            state.root_rot6d,
            relative_translation,
            state.root_velocity,
            state.body_rot6d.flatten(2),
            state.left_hand_rot6d.flatten(2),
            state.right_hand_rot6d.flatten(2),
        ]
        if state.face_state is not None:
            fields.append(state.face_state.flatten(2))
        widths = tuple(field.shape[-1] for field in fields)
        encoded = torch.cat(fields, dim=-1)
        if self.mean is not None or self.std is not None:
            if self.mean is None or self.std is None or self.mean.shape != encoded.shape[-1:]:
                raise ValueError("normalizer must contain matching mean and std")
            encoded = (encoded - self.mean.to(encoded)) / self.std.to(encoded).clamp_min(1e-8)
        return encoded, CodecContext(state, widths, root_origin)

    def decode(self, encoded: Tensor, context: CodecContext) -> TrajectoryState:
        if self.mean is not None:
            encoded = encoded * self.std.to(encoded).clamp_min(1e-8) + self.mean.to(encoded)
        fields = list(encoded.split(context.widths, dim=-1))
        template = context.template
        face = None
        if template.face_state is not None:
            face = fields.pop().reshape_as(template.face_state)
        return TrajectoryState(
            root_rot6d=fields[0],
            root_translation=fields[1] + context.root_origin[:, None, :].to(fields[1]),
            root_velocity=fields[2],
            body_rot6d=fields[3].reshape_as(template.body_rot6d),
            left_hand_rot6d=fields[4].reshape_as(template.left_hand_rot6d),
            right_hand_rot6d=fields[5].reshape_as(template.right_hand_rot6d),
            face_state=face,
            beta=template.beta,
            valid_mask=template.valid_mask,
        ).validate()

    @classmethod
    def fit(
        cls,
        state: TrajectoryState,
        supervision_mask: Tensor | None = None,
    ) -> StateCodec:
        """Fit channel statistics on training-only valid supervised entries."""

        encoded, _ = cls().encode(state)
        active = state.valid_mask[..., None].expand_as(encoded)
        if supervision_mask is not None:
            if supervision_mask.shape != encoded.shape or supervision_mask.dtype != torch.bool:
                raise ValueError("supervision mask must be bool [B,T,D]")
            active = active & supervision_mask
        support = active.sum(dim=(0, 1))
        safe_support = support.clamp_min(1)
        mean = (encoded * active).sum(dim=(0, 1)) / safe_support
        centered = (encoded - mean) * active
        variance = centered.square().sum(dim=(0, 1)) / safe_support
        std = variance.sqrt().clamp_min(1e-6)
        mean = torch.where(support > 0, mean, torch.zeros_like(mean))
        std = torch.where(support > 0, std, torch.ones_like(std))
        return cls(mean.detach(), std.detach())

    def to_payload(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            raise ValueError("cannot serialize an unfitted trajectory normalizer")
        return {
            "schema_version": "dcg_trajectory_normalizer_v1",
            "coordinate_policy": "translation_relative_to_first_valid_root",
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StateCodec:
        if payload.get("schema_version") != "dcg_trajectory_normalizer_v1":
            raise ValueError("unknown trajectory normalizer schema")
        if payload.get("coordinate_policy") != "translation_relative_to_first_valid_root":
            raise ValueError("unknown trajectory coordinate policy")
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        if mean.ndim != 1 or std.shape != mean.shape or not torch.isfinite(mean).all():
            raise ValueError("invalid trajectory normalizer tensors")
        if not torch.isfinite(std).all() or bool((std <= 0).any()):
            raise ValueError("trajectory normalizer std must be finite and positive")
        return cls(mean, std)
