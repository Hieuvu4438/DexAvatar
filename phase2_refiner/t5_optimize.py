"""Observation-only T5 sequence optimization with fail-closed group safety.

The optimizer never reads a training or evaluation target.  It starts from the
direct UAWSR output and uses the cache's frozen 2D observations, camera model,
reliable-output anchor, and temporal regularization for at most 20 Adam steps.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from phase2_refiner.data.cache_schema import CacheClip
from phase2_refiner.data.refine_how2sign_targets import (
    _decode_joints,
    _project,
    _teacher_observations,
)
from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    bound_rotation_vector,
    matrix_to_axis_angle,
)


REGIONS = {
    "ubody": slice(0, 21),
    "lhand": slice(21, 36),
    "rhand": slice(36, 51),
}


def validate_t5_config(config: dict[str, Any]) -> None:
    if not config.get("enabled", False):
        return
    steps = int(config.get("steps", 15))
    if not 1 <= steps <= 20:
        raise ValueError("T5 steps must be within the proposal bound [1, 20]")
    positive = (
        "learning_rate",
        "body_max_degrees",
        "hand_max_degrees",
    )
    for name in positive:
        if float(config.get(name, 0.0)) <= 0.0:
            raise ValueError(f"T5 {name} must be positive")
    nonnegative = (
        "anchor_weight",
        "velocity_weight",
        "acceleration_weight",
        "minimum_relative_reprojection_gain",
        "reprojection_worsening_tolerance",
    )
    for name in nonnegative:
        if float(config.get(name, 0.0)) < 0.0:
            raise ValueError(f"T5 {name} must be non-negative")


def _lane_camera(clip: CacheClip, device: torch.device) -> torch.Tensor:
    matrices = []
    for source in clip.source_paths:
        with Path(str(source)).open("rb") as handle:
            params = pickle.load(handle, encoding="latin1")
        matrix = np.asarray(params.get("K"), dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"Missing 3x3 K in {source}")
        matrices.append(matrix)
    return torch.as_tensor(np.stack(matrices), dtype=torch.float32, device=device)


def _projection_context(
    clip: CacheClip, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str]]:
    metadata = json.loads(clip.metadata_json)
    dataset = str(metadata.get("dataset", "")).lower()
    if dataset == "how2sign":
        keypoints, confidence, valid, _, bboxes, image_sizes = _teacher_observations(
            clip
        )
        observed = keypoints * 2.0 - 1.0
        context: dict[str, torch.Tensor | str] = {
            "kind": "how2sign_crop_camera",
            "bboxes": torch.as_tensor(bboxes, dtype=torch.float32, device=device),
            "image_sizes": torch.as_tensor(
                image_sizes, dtype=torch.float32, device=device
            ),
        }
    else:
        observed = clip.keypoints_2d
        confidence = clip.observation_features[..., 0]
        valid = clip.keypoint_valid
        context = {
            "kind": "perspective_K",
            "camera": _lane_camera(clip, device),
            "image_size": torch.as_tensor(
                clip.image_size, dtype=torch.float32, device=device
            ),
        }
    refine = clip.refine_mask[None]
    valid = valid & refine
    reliability = np.clip(clip.u0_reliability, 0.0, 1.0)
    weight = np.clip(confidence, 0.0, 1.0) * reliability * valid
    return (
        torch.as_tensor(observed, dtype=torch.float32, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
        torch.as_tensor(weight, dtype=torch.float32, device=device),
        context,
    )


def _project_pose(
    body_model,
    pose_matrix: torch.Tensor,
    clip: CacheClip,
    context: dict[str, torch.Tensor | str],
    device: torch.device,
) -> torch.Tensor:
    pose = matrix_to_axis_angle(pose_matrix)[None]
    joints = _decode_joints(body_model, pose, [clip], device)[0]
    if context["kind"] == "how2sign_crop_camera":
        projected = _project(
            joints[None],
            context["bboxes"][None],
            context["image_sizes"][None],
        )[0]
        return projected * 2.0 - 1.0
    camera = context["camera"]
    image_size = context["image_size"]
    z = joints[..., 2].clamp_min(1e-5)
    pixel_x = joints[..., 0] / z * camera[:, None, 0, 0] + camera[:, None, 0, 2]
    pixel_y = joints[..., 1] / z * camera[:, None, 1, 1] + camera[:, None, 1, 2]
    height = image_size[:, None, 0]
    width = image_size[:, None, 1]
    return torch.stack(
        (pixel_x / width * 2.0 - 1.0, pixel_y / height * 2.0 - 1.0),
        dim=-1,
    )


def _regional_reprojection(
    projected: torch.Tensor,
    observed: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float | None]:
    distance = torch.linalg.vector_norm(projected - observed, dim=-1)
    result: dict[str, float | None] = {}
    for name, indices in REGIONS.items():
        selected_weight = weight[:, indices]
        if float(selected_weight.sum()) <= 0.0:
            result[name] = None
        else:
            result[name] = float(
                (distance[:, indices] * selected_weight).sum()
                / selected_weight.sum().clamp_min(1e-8)
            )
    return result


def _accepted_regions(
    before: dict[str, float | None],
    after: dict[str, float | None],
    minimum_gain: float,
    worsening_tolerance: float,
) -> dict[str, bool]:
    accepted = {}
    for name in REGIONS:
        initial = before[name]
        final = after[name]
        if initial is None or final is None or initial <= 0.0:
            accepted[name] = False
            continue
        relative_gain = (initial - final) / initial
        accepted[name] = (
            relative_gain >= minimum_gain
            and final <= initial * (1.0 + worsening_tolerance)
        )
    return accepted


def optimize_t5_sequence(
    clip: CacheClip,
    direct_matrix: torch.Tensor,
    body_model,
    config: dict[str, Any],
    device: torch.device,
    initializer_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Refine one complete clip without reading any target or benchmark GT."""
    validate_t5_config(config)
    if not config.get("enabled", False):
        return direct_matrix, {"enabled": False}
    direct = direct_matrix.detach().to(device=device, dtype=torch.float32)
    initializer = (
        initializer_matrix.detach().to(device=device, dtype=torch.float32)
        if initializer_matrix is not None
        else None
    )
    observed, valid, weight, projection_context = _projection_context(clip, device)
    if not valid.any() or float(weight.sum()) <= 0.0:
        return direct, {
            "enabled": True,
            "accepted_regions": {name: False for name in REGIONS},
            "reason": "no_valid_weighted_observations",
        }

    steps = int(config.get("steps", 15))
    learning_rate = float(config.get("learning_rate", 0.03))
    anchor_weight = float(config.get("anchor_weight", 0.05))
    velocity_weight = float(config.get("velocity_weight", 0.10))
    acceleration_weight = float(config.get("acceleration_weight", 0.05))
    max_angles = torch.full(
        (1, 51, 1),
        math.radians(float(config.get("hand_max_degrees", 5.0))),
        dtype=torch.float32,
        device=device,
    )
    max_angles[:, :21] = math.radians(
        float(config.get("body_max_degrees", 3.0))
    )
    refine = torch.as_tensor(
        clip.refine_mask, dtype=torch.bool, device=device
    )[None, :, None]
    anchor_reliability = torch.as_tensor(
        clip.u0_reliability, dtype=torch.float32, device=device
    ).clamp(0.0, 1.0)
    anchor_mask = refine.squeeze(-1).expand(len(direct), -1)
    raw_delta = torch.zeros(
        len(direct), 51, 3, dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam((raw_delta,), lr=learning_rate)

    with torch.no_grad():
        initial_projection = _project_pose(
            body_model, direct, clip, projection_context, device
        )
        frozen_projection = (
            _project_pose(body_model, initializer, clip, projection_context, device)
            if initializer is not None
            else None
        )
    best_loss = float("inf")
    best_delta = torch.zeros_like(raw_delta)
    history = []
    with torch.enable_grad():
        for step in range(steps):
            bounded = bound_rotation_vector(raw_delta, max_angles) * refine
            candidate = axis_angle_to_matrix(bounded) @ direct
            projection = _project_pose(
                body_model, candidate, clip, projection_context, device
            )
            coordinate = F.smooth_l1_loss(
                projection, observed, reduction="none", beta=0.01
            ).sum(dim=-1)
            reprojection = (coordinate * weight).sum() / weight.sum().clamp_min(1.0)
            anchor_weights = anchor_reliability * anchor_mask
            anchor = (bounded.square().sum(dim=-1) * anchor_weights).sum() / (
                anchor_weights.sum().clamp_min(1.0)
            )
            velocity = (
                (bounded[1:] - bounded[:-1]).square().mean()
                if len(bounded) > 1
                else bounded.new_zeros(())
            )
            acceleration = (
                (bounded[2:] - 2.0 * bounded[1:-1] + bounded[:-2])
                .square()
                .mean()
                if len(bounded) > 2
                else bounded.new_zeros(())
            )
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
            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best_loss = value
                best_delta = raw_delta.detach().clone()

    with torch.no_grad():
        bounded = bound_rotation_vector(best_delta, max_angles) * refine
        candidate = axis_angle_to_matrix(bounded) @ direct
        final_projection = _project_pose(
            body_model, candidate, clip, projection_context, device
        )
        before = _regional_reprojection(initial_projection, observed, weight)
        after = _regional_reprojection(final_projection, observed, weight)
        accepted = _accepted_regions(
            before,
            after,
            float(config.get("minimum_relative_reprojection_gain", 0.002)),
            float(config.get("reprojection_worsening_tolerance", 0.001)),
        )
        output = candidate.clone()
        for name, indices in REGIONS.items():
            if not accepted[name]:
                output[:, indices] = direct[:, indices]
        initializer_reprojection = (
            _regional_reprojection(frozen_projection, observed, weight)
            if frozen_projection is not None
            else {name: None for name in REGIONS}
        )
        output_projection = _project_pose(body_model, output, clip, projection_context, device)
        output_reprojection = _regional_reprojection(output_projection, observed, weight)
        fallback_to_initializer = {name: False for name in REGIONS}
        tolerance = float(config.get("reprojection_worsening_tolerance", 0.001))
        if initializer is not None:
            # Body replacement can move both hands kinematically. Reproject
            # after every newly rejected group until the decision is stable.
            for _ in range(len(REGIONS)):
                changed = False
                for name, indices in REGIONS.items():
                    frozen_error = initializer_reprojection[name]
                    final_error = output_reprojection[name]
                    if (
                        not fallback_to_initializer[name]
                        and frozen_error is not None
                        and final_error is not None
                        and final_error > frozen_error * (1.0 + tolerance)
                    ):
                        output[:, indices] = initializer[:, indices]
                        fallback_to_initializer[name] = True
                        changed = True
                if not changed:
                    break
                output_projection = _project_pose(
                    body_model, output, clip, projection_context, device
                )
                output_reprojection = _regional_reprojection(
                    output_projection, observed, weight
                )
        applied_delta = torch.rad2deg(
            torch.linalg.vector_norm(bounded, dim=-1)
        )
    return output, {
        "enabled": True,
        "steps": steps,
        "camera_kind": projection_context["kind"],
        "objective_initial": history[0] if history else None,
        "objective_best": best_loss,
        "reprojection_before": before,
        "reprojection_candidate": after,
        "reprojection_initializer": initializer_reprojection,
        "reprojection_selected_before_initializer_safety": output_reprojection,
        "accepted_regions": accepted,
        "fallback_to_initializer_regions": fallback_to_initializer,
        "max_applied_delta_degrees": {
            name: float(applied_delta[:, indices].max()) if accepted[name] else 0.0
            for name, indices in REGIONS.items()
        },
    }
