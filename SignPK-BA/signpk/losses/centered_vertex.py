from __future__ import annotations

from torch import Tensor

from signpk.geometry.robustifiers import charbonnier, masked_mean


def center_region(vertices: Tensor, indices: Tensor | list[int]) -> Tensor:
    region = vertices[..., indices, :]
    return region - region.mean(dim=-2, keepdim=True)


def centered_vertex_loss(
    prediction: Tensor,
    target: Tensor,
    indices: Tensor | list[int],
    valid: Tensor | None = None,
    epsilon: float = 1e-6,
) -> Tensor:
    residual = center_region(prediction, indices) - center_region(target, indices)
    return masked_mean(charbonnier(residual, epsilon), valid)
