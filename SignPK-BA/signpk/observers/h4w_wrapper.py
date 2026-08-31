from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from signpk.data.cache_schema import BodyObservation, CACHE_SCHEMA_VERSION, HandObservation
from signpk.data.frame_manifest import SignManifest
from signpk.geometry.palm_frame import make_palm_frame
from signpk.geometry.rotations import axis_angle_to_matrix

from .h4w_feature_hook import load_h4w_feature_cache


def _tensor(values: list[np.ndarray] | np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.stack(values) if isinstance(values, list) else values, dtype=dtype)


def _reprojection_error(
    pred: torch.Tensor, observed: torch.Tensor, confidence: torch.Tensor
) -> torch.Tensor:
    valid = confidence > 0
    distance = torch.linalg.vector_norm(pred - observed, dim=-1)
    numerator = (distance * confidence).sum(-1)
    denominator = (confidence * valid).sum(-1).clamp_min(1e-8)
    return numerator / denominator


def _pool_hand_feature(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or value.ndim == 2:
        return value
    if value.ndim < 2:
        raise ValueError("H4W++ hand feature must preserve frame and channel dimensions")
    return value.flatten(2).mean(-1)


def _upper_body_feature_tokens(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or value.ndim == 2:
        return value
    if value.ndim != 3 or value.shape[1] < 25:
        raise ValueError("H4W++ body pose tokens must have shape [T,25,F]")
    pelvis, neck = value[:, 0], value[:, 7]
    spine = neck - pelvis
    left_shoulder, right_shoulder = value[:, 8], value[:, 9]
    return torch.stack(
        [
            pelvis,
            pelvis + 0.25 * spine,
            pelvis + 0.50 * spine,
            pelvis + 0.75 * spine,
            neck,
            value[:, 24],
            0.5 * (neck + left_shoulder),
            0.5 * (neck + right_shoulder),
            left_shoulder,
            right_shoulder,
            value[:, 10],
            value[:, 11],
            value[:, 12],
            value[:, 13],
        ],
        dim=1,
    )


def load_h4w_cache(
    cache_root: str | Path,
    manifest: SignManifest,
    *,
    expected_commit: str | None = None,
) -> tuple[BodyObservation, HandObservation, HandObservation, dict[str, Any]]:
    """Load the existing deterministic H4W++/WiLoR NPZ cache.

    The wrapper is deliberately read-only and checks every manifest frame ID.
    H4W++ feature tensors are optional because the current 1493-frame cache
    stores parameters/geometry but not the large ViT activations.
    """

    sign_root = Path(cache_root) / manifest.sign_name
    upstream_manifest_path = sign_root / "manifest.json"
    if not upstream_manifest_path.is_file():
        raise FileNotFoundError(upstream_manifest_path)
    upstream = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
    if expected_commit and upstream.get("h4wpp_git_sha") != expected_commit:
        raise ValueError(
            f"H4W++ revision mismatch: {upstream.get('h4wpp_git_sha')} != {expected_commit}"
        )
    lookup = {int(item["frame_id"]): item["cache"] for item in upstream["frames"]}
    missing = sorted(set(manifest.frame_ids) - set(lookup))
    if missing:
        raise FileNotFoundError(f"H4W++ cache missing frame IDs {missing[:8]}")
    rows = [
        np.load(sign_root / lookup[frame_id], allow_pickle=False) for frame_id in manifest.frame_ids
    ]

    body_pose = axis_angle_to_matrix(
        _tensor([row["smplx_body_pose_aa"] for row in rows]).view(-1, 21, 3)
    )
    root_pose = axis_angle_to_matrix(_tensor([row["smplx_root_pose_aa"] for row in rows]))
    joints3d = _tensor([row["smplx_keypoints_3d"] for row in rows])
    keypoints2d = torch.cat(
        [
            _tensor([row["right_keypoints_2d"] for row in rows]),
            _tensor([row["left_keypoints_2d"] for row in rows]),
        ],
        dim=1,
    )
    keypoint_confidence = torch.cat(
        [
            _tensor([row["right_keypoint_confidence"] for row in rows]),
            _tensor([row["left_keypoint_confidence"] for row in rows]),
        ],
        dim=1,
    )
    feature_path = sign_root / "features.pt"
    features: dict[str, torch.Tensor] = {}
    feature_metadata: dict[str, Any] | None = None
    if feature_path.is_file():
        features, feature_metadata = load_h4w_feature_cache(feature_path, manifest.frame_ids)
    body = BodyObservation(
        root_rotmat=root_pose,
        body_rotmat=body_pose,
        shape=_tensor([row["smplx_shape"] for row in rows]),
        vertices=_tensor([row["smplx_vertices"] for row in rows]),
        joints3d=joints3d,
        keypoints2d=keypoints2d,
        keypoint_confidence=keypoint_confidence,
        translation=_tensor([row["smplx_trans"] for row in rows]),
        focal_length=_tensor([row["camera_focal"] for row in rows]),
        principal_point=_tensor([row["camera_principal_point"] for row in rows]),
        body_features=_upper_body_feature_tokens(features.get("body_pose_token")),
    )

    hands: dict[str, HandObservation] = {}
    for side in ("left", "right"):
        local_rotmat = _tensor([row[f"{side}_hand_rotmat"] for row in rows])
        global_rotmat = _tensor([row[f"{side}_global_rotmat"] for row in rows])
        pose = torch.cat([global_rotmat[:, None], local_rotmat], dim=1)
        vertices = _tensor([row[f"{side}_vertices"] for row in rows])
        joints = _tensor([row[f"{side}_keypoints_3d"] for row in rows])
        palm, wrist, palm_valid = make_palm_frame(joints, side)
        valid = torch.as_tensor([bool(row[f"{side}_exists"]) for row in rows]) & palm_valid
        confidence = _tensor([row[f"{side}_keypoint_confidence"] for row in rows]).mean(-1)
        hands[side] = HandObservation(
            pose_rotmat=pose,
            shape=_tensor([row[f"{side}_betas"] for row in rows]),
            vertices_local=vertices - joints[:, :1],
            joints_local=joints - joints[:, :1],
            palm_rotmat=palm,
            wrist_world_rel=wrist,
            bbox_xyxy=_tensor([row[f"{side}_bbox_xyxy"] for row in rows]),
            keypoints2d=_tensor([row[f"{side}_keypoints_2d"] for row in rows]),
            keypoint_confidence=_tensor([row[f"{side}_keypoint_confidence"] for row in rows]),
            confidence=confidence,
            valid=valid,
            reprojection_error=torch.zeros(len(rows)),
            padding_ratio=torch.zeros(len(rows)),
            temporal_token=_pool_hand_feature(features.get(f"{side}_wilor_feature")),
        )
        hands[side].validate()
    body.validate()
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "observer": "hand4whole_pp",
        "repository_commit": upstream.get("h4wpp_git_sha"),
        "checkpoint": upstream.get("checkpoint"),
        "checkpoint_sha256": upstream.get("checkpoint_sha256"),
        "source_schema_version": upstream.get("schema_version"),
        "units": "meters",
        "coordinates": "x_right_y_down_z_forward",
        "frame_ids": list(manifest.frame_ids),
        "feature_cache": feature_metadata,
    }
    return body, hands["left"], hands["right"], metadata
