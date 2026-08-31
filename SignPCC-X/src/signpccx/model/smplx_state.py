from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import torch
from torch import nn


UPPER_BODY_SLOTS = (2, 5, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
ARM_SLOTS = {
    "left": (12, 15, 17, 19),
    "right": (13, 16, 18, 20),
}
WRIST_SLOT = {"left": 19, "right": 20}

# Names used by smplx 0.1.28.  Runtime validation keeps the numeric slots from
# silently drifting when a different model implementation is installed.
SMPLX_BODY_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)


def validate_body_slots(names: tuple[str, ...] = SMPLX_BODY_NAMES) -> None:
    expected = {
        13: "left_collar", 14: "right_collar", 16: "left_shoulder",
        17: "right_shoulder", 18: "left_elbow", 19: "right_elbow",
        20: "left_wrist", 21: "right_wrist",
    }
    for joint, name in expected.items():
        if joint >= len(names) or names[joint].lower() != name:
            raise RuntimeError(
                f"SMPL-X joint contract mismatch at {joint}: "
                f"{names[joint] if joint < len(names) else '<missing>'} != {name}"
            )


class SharedSignedCamera(nn.Module):
    """Shared full-image camera preserving H4W++'s signed-axis convention."""

    def __init__(
        self,
        focal_magnitude: float,
        principal: tuple[float, float],
        image_wh: tuple[int, int],
        focal_signs: tuple[float, float] = (-1.0, 1.0),
        focal_bounds: tuple[float, float] = (0.5, 2.0),
        max_principal_shift_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        if focal_magnitude <= 0:
            raise ValueError("focal magnitude must be positive")
        self.log_f = nn.Parameter(torch.tensor(float(focal_magnitude)).log())
        self.delta_c = nn.Parameter(torch.zeros(2))
        self.register_buffer("center0", torch.tensor(principal, dtype=torch.float32))
        self.register_buffer("signs", torch.tensor(focal_signs, dtype=torch.float32))
        self.register_buffer(
            "max_delta",
            torch.tensor(image_wh, dtype=torch.float32) * float(max_principal_shift_fraction),
        )
        self.register_buffer(
            "focal_limits",
            torch.tensor(focal_bounds, dtype=torch.float32) * float(focal_magnitude),
        )

    def matrix(self) -> torch.Tensor:
        magnitude = self.log_f.exp().clamp(self.focal_limits[0], self.focal_limits[1])
        center = self.center0 + self.max_delta * torch.tanh(self.delta_c)
        matrix = torch.eye(3, dtype=magnitude.dtype, device=magnitude.device)
        matrix[0, 0] = self.signs[0] * magnitude
        matrix[1, 1] = self.signs[1] * magnitude
        matrix[0, 2] = center[0]
        matrix[1, 2] = center[1]
        return matrix


class FrameState(nn.Module):
    PARAMETER_NAMES = (
        "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
        "jaw_pose", "expression", "transl",
    )

    def __init__(self, initial: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        shapes = {
            "global_orient": (1, 3), "body_pose": (1, 21, 3),
            "left_hand_pose": (1, 15, 3), "right_hand_pose": (1, 15, 3),
            "jaw_pose": (1, 3), "expression": (1, 10), "transl": (1, 3),
        }
        for name, shape in shapes.items():
            default = torch.zeros(shape, dtype=torch.float32)
            value = initial.get(name, default)
            setattr(self, name, nn.Parameter(value.detach().clone().reshape(shape)))

    def clone(self) -> "FrameState":
        result = FrameState({name: getattr(self, name).detach() for name in self.PARAMETER_NAMES})
        return result.to(self.transl.device)

    def smplx_kwargs(self, beta: torch.Tensor) -> dict[str, torch.Tensor]:
        zero_eye = torch.zeros_like(self.jaw_pose)
        return {
            "global_orient": self.global_orient.reshape(1, 3),
            "body_pose": self.body_pose.reshape(1, 63),
            "left_hand_pose": self.left_hand_pose.reshape(1, 45),
            "right_hand_pose": self.right_hand_pose.reshape(1, 45),
            "jaw_pose": self.jaw_pose.reshape(1, 3),
            "leye_pose": zero_eye,
            "reye_pose": zero_eye,
            "expression": self.expression.reshape(1, 10),
            "betas": beta.reshape(1, 10),
            "transl": self.transl.reshape(1, 3),
            "return_verts": True,
        }

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}

    def restore(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict(deepcopy(dict(snapshot)))


def mask_body_gradient(state: FrameState, active_slots: tuple[int, ...]) -> None:
    if state.body_pose.grad is None:
        return
    mask = torch.zeros_like(state.body_pose.grad)
    mask[:, list(active_slots)] = 1
    state.body_pose.grad.mul_(mask)

