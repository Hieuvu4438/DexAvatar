from __future__ import annotations

from torch import Tensor

from signpk.geometry.robustifiers import masked_mean
from signpk.geometry.rotations import so3_distance


def geodesic_rotation_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor | None = None,
    valid: Tensor | None = None,
) -> Tensor:
    distance = so3_distance(prediction, target, squared=True)
    if weights is not None:
        distance = distance * weights
    return masked_mean(distance, valid)


def residual_magnitude_loss(residual_rotvec: Tensor, valid: Tensor | None = None) -> Tensor:
    return masked_mean(residual_rotvec.square().sum(-1), valid)

