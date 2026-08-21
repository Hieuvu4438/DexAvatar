from __future__ import annotations

import torch


def mirror_points_x(
    points: torch.Tensor, image_width: float | torch.Tensor | None = None
) -> torch.Tensor:
    mirrored = points.clone()
    if image_width is None:
        mirrored[..., 0] = -mirrored[..., 0]
    else:
        mirrored[..., 0] = (
            torch.as_tensor(image_width, device=points.device, dtype=points.dtype)
            - 1
            - points[..., 0]
        )
    return mirrored


def validate_handedness_probability(value: torch.Tensor) -> None:
    if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("handedness probability must be finite in [0,1]")
