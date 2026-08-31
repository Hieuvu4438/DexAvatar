"""GT-free candidate evidence computed in original-image coordinates."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch

from cusp_sl.geometry import matrix_to_axis_angle
from phase2_refiner.data.refine_how2sign_targets import (
    _decode_joints, _project, _teacher_observations,
)


# Only body tokens with a direct COCO whole-body observation are scored.  The
# remaining refined torso/collar rotations are still constrained indirectly by
# their observed descendants and by the temporal/physical terms.
DIRECT_BODY_OBSERVATIONS = (15, 16, 17, 18, 19, 20)


def _dataset(clip) -> str:
    metadata = json.loads(clip.metadata_json)
    return str(
        metadata.get("dataset")
        or metadata.get("provenance", {}).get("dataset")
        or ""
    ).lower()


def _sgnify_intrinsics(clip, device: torch.device) -> torch.Tensor:
    matrices = []
    for source in clip.source_paths:
        with Path(str(source)).open("rb") as handle:
            record = pickle.load(handle, encoding="latin1")
        matrix = np.asarray(record.get("K"), dtype=np.float32)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"Missing finite 3x3 camera K in {source}")
        matrices.append(matrix)
    return torch.as_tensor(np.stack(matrices), device=device)


def project_candidates(
    model, rotations: torch.Tensor, clip, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized projection, observation, mask, confidence and 3D joints."""
    pose = matrix_to_axis_angle(rotations)
    candidates, frames = pose.shape[:2]
    joints = _decode_joints(model, pose, [clip] * candidates, device)
    dataset = _dataset(clip)
    if dataset == "how2sign":
        _, confidence_np, valid_np, _, bboxes_np, image_sizes_np = _teacher_observations(clip)
        bboxes = torch.as_tensor(bboxes_np, dtype=joints.dtype, device=device)
        sizes = torch.as_tensor(image_sizes_np, dtype=joints.dtype, device=device)
        if sizes.ndim == 2:
            if not torch.equal(sizes, sizes[:1].expand_as(sizes)):
                raise ValueError("How2Sign projection expects a constant image size per clip")
            sizes = sizes[0]
        if sizes.shape != (2,):
            raise ValueError(f"Invalid How2Sign image size shape: {tuple(sizes.shape)}")
        bboxes = bboxes[None].expand(candidates, -1, -1)
        sizes = sizes[None].expand(candidates, -1)
        projected = _project(joints, bboxes, sizes) * 2.0 - 1.0
        observed = torch.as_tensor(clip.keypoints_2d * 2.0 - 1.0, device=device)
        valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
        confidence = torch.as_tensor(confidence_np, device=device)
    elif dataset == "sgnify":
        camera = _sgnify_intrinsics(clip, device).to(joints.dtype)
        z = joints[..., 2].clamp_min(1e-5)
        pixel_x = joints[..., 0] / z * camera[None, :, None, 0, 0] + camera[None, :, None, 0, 2]
        pixel_y = joints[..., 1] / z * camera[None, :, None, 1, 1] + camera[None, :, None, 1, 2]
        height = torch.as_tensor(clip.image_size[:, 0], device=device, dtype=joints.dtype)
        width = torch.as_tensor(clip.image_size[:, 1], device=device, dtype=joints.dtype)
        projected = torch.stack(
            (pixel_x / width[None, :, None] * 2.0 - 1.0,
             pixel_y / height[None, :, None] * 2.0 - 1.0), dim=-1,
        )
        observed = torch.as_tensor(clip.keypoints_2d, device=device)
        valid = torch.as_tensor(clip.keypoint_valid, dtype=torch.bool, device=device)
        confidence = torch.as_tensor(clip.raw_confidence, device=device)
    else:
        raise ValueError(f"No locked projection contract for dataset={dataset!r}")
    observable = torch.zeros(51, dtype=torch.bool, device=device)
    observable[list(DIRECT_BODY_OBSERVATIONS)] = True
    observable[21:] = True
    refine = torch.as_tensor(clip.refine_mask, dtype=torch.bool, device=device)
    valid = valid & observable[None] & refine[None]
    confidence = confidence.clamp(0.0, 1.0) * valid
    return projected, observed, valid, confidence, joints


def candidate_evidence_terms(
    model, rotations: torch.Tensor, clip, device: torch.device,
    *, huber_delta: float, rom_threshold_degrees: float,
) -> torch.Tensor:
    """Compute observation, visible-motion, ROM and optional-form terms."""
    projected, observed, valid, confidence, _ = project_candidates(
        model, rotations, clip, device
    )
    residual = torch.linalg.vector_norm(projected - observed[None], dim=-1)
    delta = torch.as_tensor(huber_delta, device=device, dtype=residual.dtype)
    robust = torch.where(
        residual <= delta, 0.5 * residual.square() / delta.clamp_min(1e-8),
        residual - 0.5 * delta,
    )
    weight = confidence[None].expand_as(robust)
    observation = (robust * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)
    if rotations.shape[1] > 1:
        predicted_velocity = projected[:, 1:] - projected[:, :-1]
        observed_velocity = observed[1:] - observed[:-1]
        motion_residual = torch.linalg.vector_norm(
            predicted_velocity - observed_velocity[None], dim=-1
        )
        motion_weight = torch.minimum(confidence[1:], confidence[:-1])[None]
        motion = (motion_residual * motion_weight).sum(dim=(1, 2)) / motion_weight.sum(dim=(1, 2)).clamp_min(1.0)
    else:
        motion = torch.zeros(rotations.shape[0], device=device, dtype=rotations.dtype)
    angle = matrix_to_axis_angle(rotations).norm(dim=-1)
    excess = (angle - math.radians(rom_threshold_degrees)).clamp_min(0.0)
    physical = excess.square().mean(dim=(1, 2))
    form = torch.zeros_like(physical)
    return torch.stack((observation, motion, physical, form), dim=-1)
