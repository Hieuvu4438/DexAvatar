from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from signpk.data.cache_schema import CACHE_SCHEMA_VERSION, HandObservation
from signpk.data.frame_manifest import SignManifest
from signpk.geometry.palm_frame import make_palm_frame
from signpk.geometry.rotations import axis_angle_to_matrix, rotation_6d_to_matrix


REQUIRED_OUTPUT_KEYS = {
    "mano_pose_left",
    "mano_pose_right",
    "mano_shape_left",
    "mano_shape_right",
    "verts3d_left",
    "verts3d_right",
    "joints3d_left",
    "joints3d_right",
    "verts3d_world_left",
    "verts3d_world_right",
    "joints3d_world_left",
    "joints3d_world_right",
    "root_rel",
    "cam_aligned_left",
    "cam_aligned_right",
}


def _cpu(value: Tensor | np.ndarray) -> Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).contiguous()


def export_omnihands_output(
    output: Mapping[str, Tensor],
    temporal_token: Tensor,
    manifest: SignManifest,
    windows: list[Mapping[str, Any]],
    bboxes: Mapping[str, Tensor],
    valid: Mapping[str, Tensor],
    output_path: str | Path,
    metadata: Mapping[str, Any],
) -> None:
    """Persist OmniHands temporal results before any render/smoothing step."""

    missing = REQUIRED_OUTPUT_KEYS - set(output)
    if missing:
        raise KeyError(f"OmniHands output missing keys: {sorted(missing)}")
    length = len(manifest.records)
    if len(windows) != length:
        raise ValueError("one temporal-window record is required per manifest frame")
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manifest_frame_ids": torch.tensor(manifest.frame_ids, dtype=torch.int64),
        "manifest_gt_ids": torch.tensor(manifest.gt_ids, dtype=torch.int64),
        "temporal_token": _cpu(temporal_token),
        "bbox_left": _cpu(bboxes["left"]),
        "bbox_right": _cpu(bboxes["right"]),
        "valid_left": torch.as_tensor(valid["left"], dtype=torch.bool).cpu(),
        "valid_right": torch.as_tensor(valid["right"], dtype=torch.bool).cpu(),
        "windows_json": json.dumps(windows, sort_keys=True),
        "metadata_json": json.dumps(dict(metadata), sort_keys=True),
    }
    for key in sorted(REQUIRED_OUTPUT_KEYS | {"mano_pose6d_left", "mano_pose6d_right"}):
        if key in output:
            value = _cpu(output[key])
            if value.shape[0] != length:
                raise ValueError(f"{key} first dimension {value.shape[0]} != {length}")
            payload[key] = value
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_omnihands_cache(
    path: str | Path,
    manifest: SignManifest,
) -> tuple[HandObservation, HandObservation, Tensor, dict[str, Any]]:
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported OmniHands cache schema")
    frame_ids = tuple(int(value) for value in data["manifest_frame_ids"].tolist())
    if frame_ids != manifest.frame_ids:
        raise ValueError("OmniHands cache frame IDs do not match the manifest")
    windows = json.loads(data["windows_json"])
    padding_ratio = torch.tensor(
        [sum(bool(x) for x in item["padded"]) / len(item["padded"]) for item in windows]
    )
    hands: dict[str, HandObservation] = {}
    for side in ("left", "right"):
        if f"mano_pose6d_{side}" in data:
            pose = rotation_6d_to_matrix(data[f"mano_pose6d_{side}"].view(-1, 16, 6))
        else:
            pose = axis_angle_to_matrix(data[f"mano_pose_{side}"].view(-1, 16, 3))
        # OmniHands decodes both local streams with its right MANO layer. Its
        # published temporal head then mirrors left-x exactly once when it
        # builds the world-relative output. Use those world tensors for the
        # topology-safe local factor so we do not apply a second point mirror.
        vertices_world = data[f"verts3d_world_{side}"]
        joints_world = data[f"joints3d_world_{side}"]
        vertices = vertices_world - joints_world[:, :1]
        joints = joints_world - joints_world[:, :1]
        if side == "left":
            reflection = pose.new_tensor([-1.0, 1.0, 1.0])
            pose = pose * reflection[None, None, :, None] * reflection[None, None, None, :]
        palm, wrist, palm_valid = make_palm_frame(joints, side)
        valid = data[f"valid_{side}"].bool() & palm_valid
        confidence = valid.float()
        hands[side] = HandObservation(
            pose_rotmat=pose,
            shape=data[f"mano_shape_{side}"],
            vertices_local=vertices,
            joints_local=joints,
            palm_rotmat=palm,
            wrist_world_rel=joints_world[:, 0],
            bbox_xyxy=data[f"bbox_{side}"],
            keypoints2d=torch.zeros(len(frame_ids), 21, 2),
            keypoint_confidence=torch.zeros(len(frame_ids), 21),
            confidence=confidence,
            valid=valid,
            temporal_token=data["temporal_token"][:, 0 if side == "right" else 1],
            vertices_world_rel=vertices_world,
            joints_world_rel=joints_world,
            reprojection_error=torch.zeros(len(frame_ids)),
            padding_ratio=padding_ratio,
        )
        hands[side].validate()
    metadata = json.loads(data["metadata_json"])
    return hands["left"], hands["right"], data["root_rel"], metadata
