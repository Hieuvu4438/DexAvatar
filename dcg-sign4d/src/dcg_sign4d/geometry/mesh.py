"""Small differentiable mesh utilities used by contact geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Area-weighted vertex normals for vertices [...,V,3] and faces [F,3]."""
    if vertices.ndim < 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must end in [V,3]")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be long [F,3]")
    if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[-2]):
        raise ValueError("face index outside vertex topology")
    first, second, third = (vertices[..., faces[:, index], :] for index in range(3))
    face_normals = torch.linalg.cross(second - first, third - first, dim=-1)
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        index = faces[:, corner]
        expanded = index.reshape(*((1,) * (vertices.ndim - 2)), -1, 1).expand_as(face_normals)
        normals.scatter_add_(-2, expanded, face_normals)
    return functional.normalize(normals, dim=-1, eps=1e-12)


def vertex_areas(vertices: Tensor, faces: Tensor) -> Tensor:
    """Barycentric surface area assigned to vertices, preserving batch dimensions."""

    if vertices.ndim < 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must end in [V,3]")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be long [F,3]")
    if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[-2]):
        raise ValueError("face index outside vertex topology")
    first, second, third = (vertices[..., faces[:, index], :] for index in range(3))
    face_area = 0.5 * torch.linalg.vector_norm(
        torch.linalg.cross(second - first, third - first, dim=-1), dim=-1
    )
    result = vertices.new_zeros(*vertices.shape[:-1])
    share = face_area / 3
    for corner in range(3):
        index = faces[:, corner]
        expanded = index.reshape(*((1,) * (vertices.ndim - 2)), -1).expand_as(share)
        result.scatter_add_(-1, expanded, share)
    return result
