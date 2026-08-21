from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry.so3 import rotation_6d_to_matrix


@dataclass
class SequenceState:
    global_rot6d: torch.Tensor
    body_rot6d: torch.Tensor
    left_hand_rot6d: torch.Tensor
    right_hand_rot6d: torch.Tensor
    translation: torch.Tensor
    betas: torch.Tensor
    expression: torch.Tensor | None = None
    contact_logits: torch.Tensor | None = None

    def validate(self) -> None:
        t = self.global_rot6d.shape[0]
        expected = {
            "global_rot6d": (t, 6),
            "body_rot6d": (t, 21, 6),
            "left_hand_rot6d": (t, 15, 6),
            "right_hand_rot6d": (t, 15, 6),
            "translation": (t, 3),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")
        if self.betas.ndim != 2 or self.betas.shape[0] not in {1, t}:
            raise ValueError("betas must have shape [1,B] or [T,B]")

    def rotations(self) -> dict[str, torch.Tensor]:
        self.validate()
        return {
            "global_orient": rotation_6d_to_matrix(self.global_rot6d),
            "body_pose": rotation_6d_to_matrix(self.body_rot6d),
            "left_hand_pose": rotation_6d_to_matrix(self.left_hand_rot6d),
            "right_hand_pose": rotation_6d_to_matrix(self.right_hand_rot6d),
        }

    def detached_clone(self) -> SequenceState:
        return SequenceState(
            **{
                name: value.detach().clone() if value is not None else None
                for name, value in self.__dict__.items()
            }
        )


@dataclass
class TrajectoryState:
    joints: torch.Tensor
    rotations: torch.Tensor | None
    translation: torch.Tensor
    contact_logits: torch.Tensor | None = None

    def validate(self) -> None:
        if self.joints.ndim != 3 or self.joints.shape[-1] != 3:
            raise ValueError("trajectory joints must be [T,J,3]")
        if self.translation.shape != (self.joints.shape[0], 3):
            raise ValueError("trajectory translation must be [T,3]")
        if self.rotations is not None and self.rotations.shape != (*self.joints.shape[:2], 3, 3):
            raise ValueError("trajectory rotations must be [T,J,3,3]")
        for value in (self.joints, self.translation, self.rotations, self.contact_logits):
            if value is not None and not torch.isfinite(value).all():
                raise ValueError("trajectory state contains non-finite values")
