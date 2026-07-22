"""Differentiable rotation conversions with no PyTorch3D dependency."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    x, y, z = vectors.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(
        vectors.shape[:-1] + (3, 3)
    )


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors (..., 3) to matrices (..., 3, 3)."""
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3), got {tuple(axis_angle.shape)}")
    theta2 = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    safe_theta2 = theta2.clamp_min(1e-8)
    theta = torch.sqrt(safe_theta2)
    small = theta2 < 1e-8
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / safe_theta2,
    )
    k = _skew(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(axis_angle.shape[:-1] + (3, 3))
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert matrices (..., 3, 3) to real-first unit quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tuple(matrix.shape)}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    denominators = (2.0 * q_abs).clamp_min(0.1)[..., :, None]
    candidates = candidates / denominators
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(best.shape + (1, 4))
    quaternion = torch.gather(candidates, -2, gather_index).squeeze(-2)
    quaternion = F.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    sin_half = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quaternion[..., :1].clamp_min(1e-12))
    scale = torch.where(
        sin_half > 1e-8,
        angle / sin_half,
        2.0 + angle * angle / 12.0,
    )
    return vector * scale


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions (..., 4) to rotation matrices."""
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Zhou et al. continuous 6D representation using the first two rows."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6), got {tuple(rotation_6d.shape)}")
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tuple(matrix.shape)}")
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


def geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest angular distance between rotation matrices, in radians."""
    relative = a @ b.transpose(-1, -2)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    skew_vector = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1)
    return torch.atan2(sine, cosine.clamp(-1.0, 1.0))


def bound_rotation_vector(
    vector: torch.Tensor, max_angle: torch.Tensor | float
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    max_tensor = torch.as_tensor(max_angle, dtype=vector.dtype, device=vector.device)
    while max_tensor.ndim < vector.ndim:
        max_tensor = max_tensor.unsqueeze(0)
    bounded_norm = max_tensor * torch.tanh(norm / max_tensor.clamp_min(1e-8))
    scale = torch.where(
        norm > 1e-8, bounded_norm / norm.clamp_min(1e-8), torch.ones_like(norm)
    )
    return vector * scale


def compose_residual(
    initial: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor | None = None,
    max_angle: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Left-compose a bounded local rotation residual with initial matrices."""
    if max_angle is not None:
        residual = bound_rotation_vector(residual, max_angle)
    if gate is not None:
        residual = residual * gate
    return axis_angle_to_matrix(residual) @ initial
