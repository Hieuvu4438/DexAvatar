from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping


def _axis_angle_to_matrix(axis_angle):
    import torch

    vector = axis_angle.reshape(-1, 3)
    angle = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    axis = vector / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=1).reshape(-1, 3, 3)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(len(vector), -1, -1)
    sin = torch.sin(angle).reshape(-1, 1, 1)
    cos = torch.cos(angle).reshape(-1, 1, 1)
    result = identity + sin * skew + (1.0 - cos) * (skew @ skew)
    small = (angle.reshape(-1) < 1e-7).reshape(-1, 1, 1)
    result = torch.where(small, identity + skew * angle.reshape(-1, 1, 1), result)
    return result.reshape(*axis_angle.shape[:-1], 3, 3)


def _matrix_to_axis_angle(matrix):
    import torch

    flat = matrix.reshape(-1, 3, 3)
    r00, r01, r02 = flat[:, 0, 0], flat[:, 0, 1], flat[:, 0, 2]
    r10, r11, r12 = flat[:, 1, 0], flat[:, 1, 1], flat[:, 1, 2]
    r20, r21, r22 = flat[:, 2, 0], flat[:, 2, 1], flat[:, 2, 2]
    q_abs = torch.sqrt(torch.clamp(torch.stack((
        1 + r00 + r11 + r22,
        1 + r00 - r11 - r22,
        1 - r00 + r11 - r22,
        1 - r00 - r11 + r22,
    ), dim=1), min=0.0))
    candidates = torch.stack((
        torch.stack((q_abs[:, 0].square(), r21 - r12, r02 - r20, r10 - r01), dim=1),
        torch.stack((r21 - r12, q_abs[:, 1].square(), r10 + r01, r02 + r20), dim=1),
        torch.stack((r02 - r20, r10 + r01, q_abs[:, 2].square(), r12 + r21), dim=1),
        torch.stack((r10 - r01, r02 + r20, r12 + r21, q_abs[:, 3].square()), dim=1),
    ), dim=1)
    candidates = candidates / (2.0 * q_abs.clamp_min(0.1).unsqueeze(-1))
    quaternion = candidates[torch.arange(len(flat), device=flat.device), q_abs.argmax(dim=1)]
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-8)
    quaternion = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    vector = quaternion[:, 1:]
    sin_half = vector.norm(dim=1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quaternion[:, :1].clamp_min(0.0))
    scale = torch.where(sin_half > 1e-7, angle / sin_half, 2.0 * torch.ones_like(sin_half))
    result = vector * scale
    return result.reshape(*matrix.shape[:-2], 3)


def twist_wrist(wrist_axis_angle, local_axis, degrees: float):
    axis = local_axis / local_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    delta = axis * (float(degrees) * math.pi / 180.0)
    return _matrix_to_axis_angle(
        _axis_angle_to_matrix(wrist_axis_angle) @ _axis_angle_to_matrix(delta)
    )


def signed_area_2d(wrist, index_mcp, pinky_mcp):
    a = index_mcp - wrist
    b = pinky_mcp - wrist
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def chirality_loss(predicted_xy, observed_xy, ids: tuple[int, int, int], margin: float = 1e-4):
    import torch

    wrist, index_mcp, pinky_mcp = ids
    predicted = signed_area_2d(
        predicted_xy[:, wrist], predicted_xy[:, index_mcp], predicted_xy[:, pinky_mcp]
    )
    observed = signed_area_2d(
        observed_xy[:, wrist], observed_xy[:, index_mcp], observed_xy[:, pinky_mcp]
    ).detach()
    sign = torch.sign(observed)
    valid = (sign != 0).to(predicted.dtype)
    return (
        valid * torch.nn.functional.softplus(-(sign * predicted) / float(margin))
    ).sum() / valid.sum().clamp_min(1.0)


@dataclass(frozen=True)
class HandHypothesis:
    name: str
    wrist_axis_angle: object
    hand_pose: object
    source: str
    scores: Mapping[str, float] | None = None

    @property
    def total_score(self) -> float:
        if self.scores is None:
            return float("inf")
        return float(sum(self.scores.values()))


def wrist_twist_hypotheses(
    base: HandHypothesis,
    local_axis,
    degrees: tuple[float, ...] = (-30.0, 0.0, 30.0),
) -> list[HandHypothesis]:
    return [
        replace(
            base,
            name=f"{base.name}_twist_{value:+g}",
            wrist_axis_angle=twist_wrist(base.wrist_axis_angle, local_axis, value),
        )
        for value in degrees
    ]


def rank_hypotheses(candidates: list[HandHypothesis], keep: int) -> list[HandHypothesis]:
    finite = [item for item in candidates if math.isfinite(item.total_score)]
    # Deterministic tie-break: penetration, teacher disagreement, joint limit, name.
    def key(item: HandHypothesis):
        scores = item.scores or {}
        return (
            item.total_score,
            float(scores.get("penetration", 0.0)),
            float(scores.get("teacher_disagreement", 0.0)),
            float(scores.get("joint_limit", 0.0)),
            item.name,
        )
    return sorted(finite, key=key)[:max(0, int(keep))]
