from __future__ import annotations

import torch
from torch import Tensor

from signpk.geometry.robustifiers import masked_mean


def penetration_loss(
    signed_distances: Tensor,
    interaction_gate: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    penetration = (-signed_distances).clamp_min(0).square()
    while interaction_gate.ndim < penetration.ndim:
        interaction_gate = interaction_gate.unsqueeze(-1)
    return masked_mean(penetration * interaction_gate, valid)


def vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Differentiable area-weighted vertex normals for a fixed mesh topology."""

    triangle = vertices[:, faces]
    face_normals = torch.cross(
        triangle[:, :, 1] - triangle[:, :, 0],
        triangle[:, :, 2] - triangle[:, :, 0],
        dim=-1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        index = faces[:, corner][None, :, None].expand(vertices.shape[0], -1, 3)
        normals.scatter_add_(1, index, face_normals)
    return torch.nn.functional.normalize(normals, dim=-1, eps=1e-8)


def hand_penetration_loss(
    vertices: Tensor,
    faces: Tensor,
    left_indices: Tensor,
    right_indices: Tensor,
    interaction_gate: Tensor,
    valid: Tensor | None = None,
    sample_stride: int = 4,
) -> Tensor:
    """Approximate symmetric hand penetration from nearest surface normals.

    This is a prevention term only: positive separation is never attracted to
    a contact distance. The interaction gate decides when the factor is active.
    """

    normals = vertex_normals(vertices, faces)
    left_indices = left_indices[::sample_stride]
    right_indices = right_indices[::sample_stride]
    left = vertices[:, left_indices]
    right = vertices[:, right_indices]
    left_normals = normals[:, left_indices]
    right_normals = normals[:, right_indices]
    distances = torch.cdist(left, right)
    nearest_right = distances.argmin(-1)
    nearest_left = distances.argmin(-2)
    right_points = torch.gather(right, 1, nearest_right[..., None].expand(-1, -1, 3))
    right_normal = torch.gather(right_normals, 1, nearest_right[..., None].expand(-1, -1, 3))
    left_points = torch.gather(left, 1, nearest_left[..., None].expand(-1, -1, 3))
    left_normal = torch.gather(left_normals, 1, nearest_left[..., None].expand(-1, -1, 3))
    signed_left_to_right = ((left - right_points) * right_normal).sum(-1)
    signed_right_to_left = ((right - left_points) * left_normal).sum(-1)
    signed_distances = torch.cat([signed_left_to_right, signed_right_to_left], dim=-1)
    return penetration_loss(signed_distances, interaction_gate, valid)
