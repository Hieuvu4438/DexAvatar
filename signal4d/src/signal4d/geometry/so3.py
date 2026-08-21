from __future__ import annotations

import torch
import torch.nn.functional as functional


def skew(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("skew expects [...,3]")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def vee(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("vee expects [...,3,3]")
    return torch.stack((matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]), dim=-1)


def exp_map(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Stable Rodrigues exponential map for tangent vectors in radians."""
    if vector.shape[-1] != 3:
        raise ValueError("exp_map expects [...,3]")
    theta2 = (vector * vector).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(0))
    small = theta2 < 1e-8
    a_regular = torch.sin(theta) / theta.clamp_min(eps)
    b_regular = (1 - torch.cos(theta)) / theta2.clamp_min(eps)
    a_series = 1 - theta2 / 6 + theta2 * theta2 / 120
    b_series = 0.5 - theta2 / 24 + theta2 * theta2 / 720
    a = torch.where(small, a_series, a_regular)[..., None]
    b = torch.where(small, b_series, b_regular)[..., None]
    k = skew(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(k.shape)
    return identity + a * k + b * (k @ k)


def log_map(rotation: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Stable SO(3) logarithm, including a dedicated branch close to pi."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("log_map expects [...,3,3]")
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    # Keep acos away from both singular endpoints so gradients remain finite.
    cosine = ((trace - 1) * 0.5).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    raw = 0.5 * vee(rotation - rotation.transpose(-1, -2))
    scale = theta / sine.clamp_min(eps)
    vector = raw * scale[..., None]

    small = theta < 1e-4
    vector = torch.where(small[..., None], raw * (1 + theta[..., None] ** 2 / 6), vector)

    near_pi = theta > torch.pi - 1e-3
    if near_pi.any():
        diagonal = rotation.diagonal(dim1=-2, dim2=-1)
        axis_abs = torch.sqrt(((diagonal + 1) * 0.5).clamp_min(0))
        signs = torch.stack(
            (
                torch.ones_like(axis_abs[..., 0]),
                torch.sign(rotation[..., 0, 1] + rotation[..., 1, 0] + eps),
                torch.sign(rotation[..., 0, 2] + rotation[..., 2, 0] + eps),
            ),
            dim=-1,
        )
        axis = functional.normalize(axis_abs * signs, dim=-1, eps=eps)
        pi_vector = axis * theta[..., None]
        vector = torch.where(near_pi[..., None], pi_vector, vector)
    return vector


def geodesic_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(log_map(first.transpose(-1, -2) @ second), dim=-1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d_to_matrix expects [...,6]")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    basis1 = functional.normalize(first, dim=-1)
    basis2 = functional.normalize(second - (basis1 * second).sum(-1, keepdim=True) * basis1, dim=-1)
    basis3 = torch.cross(basis1, basis2, dim=-1)
    return torch.stack((basis1, basis2, basis3), dim=-1)


def matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("matrix_to_rotation_6d expects [...,3,3]")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def slerp(
    first: torch.Tensor, second: torch.Tensor, fraction: torch.Tensor | float
) -> torch.Tensor:
    delta = log_map(first.transpose(-1, -2) @ second)
    weight = torch.as_tensor(fraction, dtype=delta.dtype, device=delta.device)
    while weight.ndim < delta.ndim:
        weight = weight.unsqueeze(-1)
    return first @ exp_map(weight * delta)
