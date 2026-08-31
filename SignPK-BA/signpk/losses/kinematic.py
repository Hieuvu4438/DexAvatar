from __future__ import annotations

from torch import Tensor

from signpk.geometry.robustifiers import charbonnier, masked_mean
from signpk.geometry.rotations import so3_distance


def forward_kinematic_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    centered: bool = False,
) -> Tensor:
    if centered:
        prediction = prediction - prediction.mean(-2, keepdim=True)
        target = target - target.mean(-2, keepdim=True)
    return masked_mean(charbonnier(prediction - target), valid)


def palm_frame_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    normal_weight: float = 1.0,
) -> Tensor:
    geodesic = so3_distance(prediction, target, squared=True)
    normal = 1 - (prediction[..., :, 2] * target[..., :, 2]).sum(-1)
    return masked_mean(geodesic + normal_weight * normal, valid)


def relation_loss(
    predicted_wrist_delta: Tensor,
    target_wrist_delta: Tensor,
    predicted_relative_palm: Tensor | None = None,
    target_relative_palm: Tensor | None = None,
    interaction_gate: Tensor | None = None,
    valid: Tensor | None = None,
    rotation_weight: float = 1.0,
) -> Tensor:
    residual = charbonnier(predicted_wrist_delta - target_wrist_delta).mean(-1)
    if predicted_relative_palm is not None and target_relative_palm is not None:
        residual = residual + rotation_weight * so3_distance(
            predicted_relative_palm, target_relative_palm, squared=True
        )
    if interaction_gate is not None:
        residual = residual * interaction_gate.squeeze(-1)
    return masked_mean(residual, valid)

