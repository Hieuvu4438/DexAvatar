from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from signpk.geometry.rotations import compose_residual
from signpk.models.explicit_tokens import UPPER_BODY_INDICES


@dataclass
class SequenceRotations:
    root: Tensor
    body: Tensor
    left_hand: Tensor
    right_hand: Tensor
    upper: Tensor


class SequenceState(nn.Module):
    """Residual SO(3) state whose zero value exactly reproduces PKC."""

    def __init__(
        self,
        base_upper: Tensor,
        base_body: Tensor,
        base_left_hand: Tensor,
        base_right_hand: Tensor,
        shape: Tensor,
        translation: Tensor,
        expression: Tensor | None = None,
    ):
        super().__init__()
        frames = base_upper.shape[0]
        if base_upper.shape != (frames, 14, 3, 3):
            raise ValueError("base_upper must be [T,14,3,3]")
        if base_body.shape != (frames, 21, 3, 3):
            raise ValueError("base_body must be [T,21,3,3]")
        for name, value in (("left", base_left_hand), ("right", base_right_hand)):
            if value.shape != (frames, 15, 3, 3):
                raise ValueError(f"base_{name}_hand must be [T,15,3,3]")
        self.register_buffer("base_upper", base_upper.detach().clone())
        self.register_buffer("base_body", base_body.detach().clone())
        self.register_buffer("base_left_hand", base_left_hand.detach().clone())
        self.register_buffer("base_right_hand", base_right_hand.detach().clone())
        self.root_delta = nn.Parameter(torch.zeros(frames, 1, 3, device=base_upper.device))
        self.upper_delta = nn.Parameter(torch.zeros(frames, 11, 3, device=base_upper.device))
        self.wrist_delta = nn.Parameter(torch.zeros(frames, 2, 3, device=base_upper.device))
        self.left_hand_delta = nn.Parameter(torch.zeros(frames, 15, 3, device=base_upper.device))
        self.right_hand_delta = nn.Parameter(torch.zeros(frames, 15, 3, device=base_upper.device))
        self.beta = nn.Parameter(shape.detach().reshape(1, -1).clone())
        self.translation = nn.Parameter(translation.detach().clone())
        self.register_buffer(
            "expression",
            torch.zeros(frames, 10, device=base_upper.device) if expression is None else expression.detach().clone(),
        )

    @property
    def num_frames(self) -> int:
        return self.translation.shape[0]

    def rotations(self) -> SequenceRotations:
        delta = torch.cat([self.root_delta, self.upper_delta, self.wrist_delta], dim=1)
        upper = compose_residual(self.base_upper, delta)
        combined = torch.cat(
            [
                self.base_upper[:, :1],
                self.base_body,
            ],
            dim=1,
        ).clone()
        combined[:, UPPER_BODY_INDICES] = upper
        left = compose_residual(self.base_left_hand, self.left_hand_delta)
        right = compose_residual(self.base_right_hand, self.right_hand_delta)
        return SequenceRotations(
            root=combined[:, 0],
            body=combined[:, 1:],
            left_hand=left,
            right_hand=right,
            upper=upper,
        )

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "body_shape": (self.beta,),
            "translation": (self.translation,),
            "root": (self.root_delta,),
            "upper_body": (self.upper_delta, self.wrist_delta),
            "spine": (self.upper_delta,),
            "neck": (self.upper_delta,),
            "clavicles": (self.upper_delta,),
            "shoulders": (self.upper_delta,),
            "elbows": (self.upper_delta,),
            "wrists": (self.wrist_delta,),
            "left_hand": (self.left_hand_delta,),
            "right_hand": (self.right_hand_delta,),
        }

    def set_trainable(self, names: list[str]) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        selected: list[nn.Parameter] = []
        groups = self.parameter_groups()
        for name in names:
            if name not in groups:
                raise KeyError(f"unknown BA variable group {name!r}")
            for parameter in groups[name]:
                parameter.requires_grad_(True)
                if all(parameter is not existing for existing in selected):
                    selected.append(parameter)
        return selected

    def residual_radians(self) -> dict[str, Tensor]:
        return {
            "root": torch.linalg.vector_norm(self.root_delta, dim=-1),
            "upper_body": torch.linalg.vector_norm(self.upper_delta, dim=-1),
            "wrists": torch.linalg.vector_norm(self.wrist_delta, dim=-1),
            "left_hand": torch.linalg.vector_norm(self.left_hand_delta, dim=-1),
            "right_hand": torch.linalg.vector_norm(self.right_hand_delta, dim=-1),
        }

    def snapshot(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.state_dict().items()}

    def restore(self, snapshot: dict[str, Tensor]) -> None:
        self.load_state_dict(snapshot, strict=True)


def initialize_state(
    pkc_upper: Tensor,
    h4w_body: Tensor,
    pkc_left: Tensor,
    pkc_right: Tensor,
    h4w_shape: Tensor,
    h4w_translation: Tensor,
) -> SequenceState:
    robust_shape = h4w_shape.median(dim=0).values
    return SequenceState(
        base_upper=pkc_upper,
        base_body=h4w_body,
        base_left_hand=pkc_left,
        base_right_hand=pkc_right,
        shape=robust_shape,
        translation=h4w_translation,
    )

