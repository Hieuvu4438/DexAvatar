"""Observation-guided, arm-only SMPL-X bundle adjustment utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from phase2_refiner.data.refine_how2sign_targets import _decode_joints


ARM_JOINTS = np.asarray((15, 16, 17, 18, 19, 20), dtype=np.int64)
OBSERVATION_JOINTS = np.arange(15, 51, dtype=np.int64)


def bounded_arm_delta(raw: torch.Tensor, max_degrees: float) -> torch.Tensor:
    limit = math.radians(max_degrees)
    norm = torch.sqrt(raw.square().sum(dim=-1, keepdim=True) + 1e-8)
    delta = limit * torch.tanh(norm) * raw / norm
    mask = torch.zeros((1, 1, 51, 1), dtype=raw.dtype, device=raw.device)
    mask[..., ARM_JOINTS, :] = 1.0
    return delta * mask


def project_how2sign(
    joints: torch.Tensor, bboxes: torch.Tensor, image_sizes: torch.Tensor
) -> torch.Tensor:
    x, y, width, height = bboxes.unbind(-1)
    focal_x = 5000.0 / 192.0 * width
    focal_y = 5000.0 / 256.0 * height
    principal_x = x + width * 0.5
    principal_y = y + height * 0.5
    z = joints[..., 2].clamp_min(1e-5)
    pixel_x = joints[..., 0] / z * focal_x[..., None] + principal_x[..., None]
    pixel_y = joints[..., 1] / z * focal_y[..., None] + principal_y[..., None]
    return torch.stack((pixel_x, pixel_y), dim=-1) / image_sizes[:, None, None, :]


def project_intrinsics(
    joints: torch.Tensor, intrinsics: torch.Tensor, image_sizes: torch.Tensor
) -> torch.Tensor:
    z = joints[..., 2].clamp_min(1e-5)
    pixel_x = (
        joints[..., 0] / z * intrinsics[..., None, 0, 0]
        + intrinsics[..., None, 0, 2]
    )
    pixel_y = (
        joints[..., 1] / z * intrinsics[..., None, 1, 1]
        + intrinsics[..., None, 1, 2]
    )
    height = image_sizes[..., None, 0]
    width = image_sizes[..., None, 1]
    return torch.stack(
        (pixel_x / width * 2.0 - 1.0, pixel_y / height * 2.0 - 1.0), dim=-1
    )


def source_intrinsics(clip: Any) -> np.ndarray:
    import pickle

    matrices = []
    for source in clip.source_paths:
        with Path(source).open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        matrix = np.asarray(payload.get("K"), dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"Missing 3x3 K in {source}")
        matrices.append(matrix)
    return np.stack(matrices)


def _loss_metrics(
    initial: torch.Tensor,
    final: torch.Tensor,
    observed: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    def error(value: torch.Tensor) -> float:
        distance = torch.linalg.vector_norm(value - observed, dim=-1)
        return float((distance * weight).sum() / weight.sum().clamp_min(1.0))

    before = error(initial)
    after = error(final)
    return {
        "initial_reprojection": before,
        "final_reprojection": after,
        "relative_gain": (before - after) / max(before, 1e-8),
    }


def fit_arm_batch(
    model: Any,
    clips: list[Any],
    observed: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    projection: str,
    projection_aux: tuple[np.ndarray, np.ndarray],
    iterations: int = 30,
    learning_rate: float = 0.03,
    max_degrees: float = 12.0,
    anchor_weight: float = 0.02,
    velocity_weight: float = 0.10,
    acceleration_weight: float = 0.05,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    initial = torch.as_tensor(
        np.stack([clip.init_axis_angle for clip in clips]),
        dtype=torch.float32,
        device=device,
    )
    observed_t = torch.as_tensor(observed, dtype=torch.float32, device=device)
    confidence_t = torch.as_tensor(confidence, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    observation_mask = torch.zeros((1, 1, 51), dtype=torch.bool, device=device)
    observation_mask[..., OBSERVATION_JOINTS] = True
    weight = confidence_t * valid_t * observation_mask
    first_aux = torch.as_tensor(projection_aux[0], dtype=torch.float32, device=device)
    image_sizes = torch.as_tensor(
        projection_aux[1], dtype=torch.float32, device=device
    )

    def project(pose: torch.Tensor) -> torch.Tensor:
        joints = _decode_joints(model, pose, clips, device)
        if projection == "how2sign":
            return project_how2sign(joints, first_aux, image_sizes)
        if projection == "intrinsics":
            return project_intrinsics(joints, first_aux, image_sizes)
        raise ValueError(f"Unknown projection: {projection}")

    raw_delta = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam((raw_delta,), lr=learning_rate)
    with torch.no_grad():
        initial_projection = project(initial)
    for _ in range(iterations):
        delta = bounded_arm_delta(raw_delta, max_degrees)
        current_projection = project(initial + delta)
        coordinate = F.smooth_l1_loss(
            current_projection, observed_t, reduction="none", beta=0.01
        ).sum(dim=-1)
        reprojection = (coordinate * weight).sum() / weight.sum().clamp_min(1.0)
        arm_delta = delta[..., ARM_JOINTS, :]
        anchor = arm_delta.square().mean()
        velocity = (arm_delta[:, 1:] - arm_delta[:, :-1]).square().mean()
        acceleration = (
            arm_delta[:, 2:] - 2.0 * arm_delta[:, 1:-1] + arm_delta[:, :-2]
        ).square().mean()
        loss = (
            reprojection
            + anchor_weight * anchor
            + velocity_weight * velocity
            + acceleration_weight * acceleration
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_delta,), 1.0)
        optimizer.step()
    with torch.no_grad():
        result = initial + bounded_arm_delta(raw_delta, max_degrees)
        final_projection = project(result)
        reports = [
            _loss_metrics(
                initial_projection[index],
                final_projection[index],
                observed_t[index],
                weight[index],
            )
            for index in range(len(clips))
        ]
    return result.cpu().numpy(), reports
