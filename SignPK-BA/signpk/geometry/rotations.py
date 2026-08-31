from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def hat(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def vee(matrix: Tensor) -> Tensor:
    return torch.stack(
        [matrix[..., 2, 1] - matrix[..., 1, 2], matrix[..., 0, 2] - matrix[..., 2, 0], matrix[..., 1, 0] - matrix[..., 0, 1]],
        dim=-1,
    ) * 0.5


def so3_exp(vector: Tensor, eps: float = 1e-7) -> Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("SO(3) tangent vectors must end in dimension 3")
    theta2 = (vector * vector).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(eps * eps))
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6 + theta2 * theta2 / 120, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24 + theta2 * theta2 / 720, (1 - torch.cos(theta)) / theta2.clamp_min(eps * eps))
    skew = hat(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(vector.shape[:-1] + (3, 3))
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


axis_angle_to_matrix = so3_exp


def so3_log(matrix: Tensor, eps: float = 1e-7) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape [...,3,3]")
    cos_theta = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    # ``vee`` already takes half of the antisymmetric difference. Passing
    # ``R - R^T`` would apply that difference twice and double every log map.
    skew_vector = vee(matrix)
    scale = theta / sin_theta.clamp_min(eps)
    scale = torch.where(theta < 1e-4, 1 + theta.square() / 6, scale)
    result = skew_vector * scale[..., None]

    # acos is poorly conditioned close to pi. Recover the axis from the
    # symmetric part there, retaining signs from the skew part.
    near_pi = cos_theta < -0.999
    if near_pi.any():
        diagonal = ((matrix.diagonal(dim1=-2, dim2=-1) + 1) * 0.5).clamp_min(0)
        axis = torch.sqrt(diagonal)
        signs = torch.sign(skew_vector + eps)
        axis = F.normalize(axis * signs, dim=-1)
        result = torch.where(near_pi[..., None], axis * theta[..., None], result)
    return result


matrix_to_axis_angle = so3_log


def compose_residual(base: Tensor, residual: Tensor) -> Tensor:
    return so3_exp(residual) @ base


def so3_distance(first: Tensor, second: Tensor, squared: bool = False) -> Tensor:
    distance = torch.linalg.vector_norm(so3_log(first.transpose(-1, -2) @ second), dim=-1)
    return distance.square() if squared else distance


def rotation_6d_to_matrix(rotation_6d: Tensor, eps: float = 1e-8) -> Tensor:
    if rotation_6d.shape[-1] != 6:
        raise ValueError("6D rotations must end in dimension 6")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(first, dim=-1, eps=eps)
    b2 = F.normalize(second - (b1 * second).sum(-1, keepdim=True) * b1, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape [...,3,3]")
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def project_to_so3(matrix: Tensor) -> Tensor:
    u, _, vh = torch.linalg.svd(matrix)
    correction = torch.ones(matrix.shape[:-2] + (3,), dtype=matrix.dtype, device=matrix.device)
    correction[..., -1] = torch.det(u @ vh)
    return u @ torch.diag_embed(correction) @ vh


def angular_velocity(rotations: Tensor, timestamps: Tensor) -> Tensor:
    if rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must end in [3,3]")
    if rotations.shape[-3] != timestamps.shape[-1]:
        raise ValueError("time dimension does not match timestamps")
    delta_t = (timestamps[..., 1:] - timestamps[..., :-1]).clamp_min(1e-6)
    relative = rotations[..., :-1, :, :].transpose(-1, -2) @ rotations[..., 1:, :, :]
    return so3_log(relative) / delta_t[..., None]
