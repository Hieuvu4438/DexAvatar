"""Versioned Phase 2 observation-cache contract.

Schema v2 expands the executable v1 vertical slice with the coordinate,
provenance, reliability, and optional geometry fields required by the Phase 2
design.  Loading v1 remains supported so existing smoke caches are reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
NUM_JOINTS = 51
NUM_OBSERVATION_FEATURES = 8
NUM_HANDS = 2


def _identity_transforms(shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.eye(4, dtype=np.float32), shape + (4, 4)).copy()


@dataclass
class CacheClip:
    clip_id: str
    frame_names: np.ndarray
    init_axis_angle: np.ndarray
    observation_features: np.ndarray
    keypoints_2d: np.ndarray
    keypoint_valid: np.ndarray
    refine_mask: np.ndarray
    betas: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    expression: np.ndarray
    source_paths: np.ndarray
    target_axis_angle: np.ndarray | None = None
    target_rotation_valid: np.ndarray | None = None
    frame_numbers: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    fps: float = 0.0
    image_size: np.ndarray | None = None
    frame_sha256: np.ndarray | None = None
    source_sha256: np.ndarray | None = None
    keypoints_3d: np.ndarray | None = None
    keypoint_3d_valid: np.ndarray | None = None
    torso_positions: np.ndarray | None = None
    torso_position_valid: np.ndarray | None = None
    wrist_local_positions: np.ndarray | None = None
    wrist_local_valid: np.ndarray | None = None
    palm_normals: np.ndarray | None = None
    palm_valid: np.ndarray | None = None
    torso_to_camera: np.ndarray | None = None
    wrist_to_torso: np.ndarray | None = None
    u0_reliability: np.ndarray | None = None
    target_joint_positions: np.ndarray | None = None
    target_joint_valid: np.ndarray | None = None
    metadata_json: str = "{}"

    def _fill_optional_defaults(self) -> None:
        t = len(self.frame_names)
        if self.frame_numbers is None:
            self.frame_numbers = np.arange(t, dtype=np.int64)
        if self.timestamps is None:
            denominator = self.fps if self.fps > 0 else 1.0
            self.timestamps = self.frame_numbers.astype(np.float64) / denominator
        if self.image_size is None:
            self.image_size = np.ones((t, 2), dtype=np.int32)
        if self.frame_sha256 is None:
            self.frame_sha256 = np.full(t, "", dtype=str)
        if self.source_sha256 is None:
            self.source_sha256 = np.full(t, "", dtype=str)
        if self.keypoints_3d is None:
            self.keypoints_3d = np.zeros((t, NUM_JOINTS, 3), dtype=np.float32)
        if self.keypoint_3d_valid is None:
            self.keypoint_3d_valid = np.zeros((t, NUM_JOINTS), dtype=bool)
        if self.torso_positions is None:
            self.torso_positions = np.zeros((t, NUM_JOINTS, 3), dtype=np.float32)
        if self.torso_position_valid is None:
            self.torso_position_valid = np.zeros((t, NUM_JOINTS), dtype=bool)
        if self.wrist_local_positions is None:
            self.wrist_local_positions = np.zeros((t, NUM_JOINTS, 3), dtype=np.float32)
        if self.wrist_local_valid is None:
            self.wrist_local_valid = np.zeros((t, NUM_JOINTS), dtype=bool)
        if self.palm_normals is None:
            self.palm_normals = np.zeros((t, NUM_HANDS, 3), dtype=np.float32)
        if self.palm_valid is None:
            self.palm_valid = np.zeros((t, NUM_HANDS), dtype=bool)
        if self.torso_to_camera is None:
            self.torso_to_camera = _identity_transforms((t,))
        if self.wrist_to_torso is None:
            self.wrist_to_torso = _identity_transforms((t, NUM_HANDS))
        if self.u0_reliability is None:
            confidence = np.clip(self.observation_features[..., 0], 0.0, 1.0)
            presence = np.clip(self.observation_features[..., 1], 0.0, 1.0)
            missing = np.clip(self.observation_features[..., 2], 0.0, 1.0)
            truncation = np.clip(self.observation_features[..., 4], 0.0, 1.0)
            innovation = np.clip(self.observation_features[..., 5], 0.0, 1.0)
            duplicate = np.clip(self.observation_features[..., 6], 0.0, 1.0)
            self.u0_reliability = (
                confidence
                * presence
                * (1.0 - missing)
                * (1.0 - truncation)
                * (1.0 - 0.5 * duplicate)
                * np.exp(-2.0 * innovation)
            ).astype(np.float32)
        if self.target_axis_angle is not None and self.target_rotation_valid is None:
            # Backward compatibility: caches written before partial supervision
            # was introduced represented complete rotation targets.
            self.target_rotation_valid = np.ones((t, NUM_JOINTS), dtype=bool)

    def validate(self) -> None:
        self._fill_optional_defaults()
        t = len(self.frame_names)
        expected = {
            "init_axis_angle": (t, NUM_JOINTS, 3),
            "observation_features": (t, NUM_JOINTS, NUM_OBSERVATION_FEATURES),
            "keypoints_2d": (t, NUM_JOINTS, 2),
            "keypoint_valid": (t, NUM_JOINTS),
            "refine_mask": (NUM_JOINTS,),
            "global_orient": (t, 3),
            "transl": (t, 3),
            "jaw_pose": (t, 3),
            "leye_pose": (t, 3),
            "reye_pose": (t, 3),
            "expression": (t, 10),
            "source_paths": (t,),
            "frame_numbers": (t,),
            "timestamps": (t,),
            "image_size": (t, 2),
            "frame_sha256": (t,),
            "source_sha256": (t,),
            "keypoints_3d": (t, NUM_JOINTS, 3),
            "keypoint_3d_valid": (t, NUM_JOINTS),
            "torso_positions": (t, NUM_JOINTS, 3),
            "torso_position_valid": (t, NUM_JOINTS),
            "wrist_local_positions": (t, NUM_JOINTS, 3),
            "wrist_local_valid": (t, NUM_JOINTS),
            "palm_normals": (t, NUM_HANDS, 3),
            "palm_valid": (t, NUM_HANDS),
            "torso_to_camera": (t, 4, 4),
            "wrist_to_torso": (t, NUM_HANDS, 4, 4),
            "u0_reliability": (t, NUM_JOINTS),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value is None or value.shape != shape:
                actual = None if value is None else value.shape
                raise ValueError(f"{name}: expected {shape}, got {actual}")
        if self.betas.shape != (10,):
            raise ValueError(f"betas: expected (10,), got {self.betas.shape}")
        optional_targets = {
            "target_axis_angle": (t, NUM_JOINTS, 3),
            "target_rotation_valid": (t, NUM_JOINTS),
            "target_joint_positions": (t, NUM_JOINTS, 3),
            "target_joint_valid": (t, NUM_JOINTS),
        }
        for name, shape in optional_targets.items():
            value = getattr(self, name)
            if value is not None and value.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {value.shape}")
        if t == 0:
            raise ValueError("A cache clip must contain at least one frame")
        if len(np.unique(self.frame_names)) != t:
            raise ValueError("frame_names must be unique within a clip")
        if np.any(np.diff(self.frame_numbers) <= 0):
            raise ValueError("frame_numbers must be strictly increasing")
        finite_fields = (
            "init_axis_angle",
            "observation_features",
            "keypoints_2d",
            "keypoints_3d",
            "torso_positions",
            "wrist_local_positions",
            "palm_normals",
            "torso_to_camera",
            "wrist_to_torso",
            "u0_reliability",
        )
        for name in finite_fields:
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if (
            self.target_axis_angle is not None
            and not np.isfinite(self.target_axis_angle).all()
        ):
            raise ValueError("target_axis_angle contains NaN or Inf")
        if not np.all((self.u0_reliability >= 0.0) & (self.u0_reliability <= 1.0)):
            raise ValueError("u0_reliability must be within [0, 1]")


def save_cache_clip(path: str | Path, clip: CacheClip) -> None:
    clip.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "clip_id": np.asarray(clip.clip_id),
        "frame_names": clip.frame_names.astype(str),
        "init_axis_angle": clip.init_axis_angle.astype(np.float32),
        "observation_features": clip.observation_features.astype(np.float32),
        "keypoints_2d": clip.keypoints_2d.astype(np.float32),
        "keypoint_valid": clip.keypoint_valid.astype(bool),
        "refine_mask": clip.refine_mask.astype(bool),
        "betas": clip.betas.astype(np.float32),
        "global_orient": clip.global_orient.astype(np.float32),
        "transl": clip.transl.astype(np.float32),
        "jaw_pose": clip.jaw_pose.astype(np.float32),
        "leye_pose": clip.leye_pose.astype(np.float32),
        "reye_pose": clip.reye_pose.astype(np.float32),
        "expression": clip.expression.astype(np.float32),
        "source_paths": clip.source_paths.astype(str),
        "frame_numbers": clip.frame_numbers.astype(np.int64),
        "timestamps": clip.timestamps.astype(np.float64),
        "fps": np.asarray(clip.fps, dtype=np.float32),
        "image_size": clip.image_size.astype(np.int32),
        "frame_sha256": clip.frame_sha256.astype(str),
        "source_sha256": clip.source_sha256.astype(str),
        "keypoints_3d": clip.keypoints_3d.astype(np.float32),
        "keypoint_3d_valid": clip.keypoint_3d_valid.astype(bool),
        "torso_positions": clip.torso_positions.astype(np.float32),
        "torso_position_valid": clip.torso_position_valid.astype(bool),
        "wrist_local_positions": clip.wrist_local_positions.astype(np.float32),
        "wrist_local_valid": clip.wrist_local_valid.astype(bool),
        "palm_normals": clip.palm_normals.astype(np.float32),
        "palm_valid": clip.palm_valid.astype(bool),
        "torso_to_camera": clip.torso_to_camera.astype(np.float32),
        "wrist_to_torso": clip.wrist_to_torso.astype(np.float32),
        "u0_reliability": clip.u0_reliability.astype(np.float32),
        "metadata_json": np.asarray(clip.metadata_json),
        "has_target": np.asarray(clip.target_axis_angle is not None, dtype=bool),
        "has_target_joints": np.asarray(
            clip.target_joint_positions is not None, dtype=bool
        ),
    }
    if clip.target_axis_angle is not None:
        payload["target_axis_angle"] = clip.target_axis_angle.astype(np.float32)
        payload["target_rotation_valid"] = clip.target_rotation_valid.astype(bool)
    if clip.target_joint_positions is not None:
        payload["target_joint_positions"] = clip.target_joint_positions.astype(
            np.float32
        )
        valid = clip.target_joint_valid
        if valid is None:
            valid = np.ones(clip.target_joint_positions.shape[:2], dtype=bool)
        payload["target_joint_valid"] = valid.astype(bool)
    np.savez_compressed(output, **payload)


def _get(data: np.lib.npyio.NpzFile, key: str, default):
    return data[key] if key in data.files else default


def load_cache_clip(path: str | Path) -> CacheClip:
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported cache schema {version}; supported {SUPPORTED_SCHEMA_VERSIONS}"
            )
        has_target = bool(_get(data, "has_target", False))
        has_target_joints = bool(_get(data, "has_target_joints", False))
        clip = CacheClip(
            clip_id=str(data["clip_id"]),
            frame_names=data["frame_names"].astype(str),
            init_axis_angle=data["init_axis_angle"],
            observation_features=data["observation_features"],
            keypoints_2d=data["keypoints_2d"],
            keypoint_valid=data["keypoint_valid"],
            refine_mask=data["refine_mask"],
            betas=data["betas"],
            global_orient=data["global_orient"],
            transl=data["transl"],
            jaw_pose=data["jaw_pose"],
            leye_pose=data["leye_pose"],
            reye_pose=data["reye_pose"],
            expression=data["expression"],
            source_paths=data["source_paths"].astype(str),
            target_axis_angle=data["target_axis_angle"] if has_target else None,
            target_rotation_valid=(
                data["target_rotation_valid"]
                if has_target and "target_rotation_valid" in data.files
                else None
            ),
            frame_numbers=_get(data, "frame_numbers", None),
            timestamps=_get(data, "timestamps", None),
            fps=float(_get(data, "fps", 0.0)),
            image_size=_get(data, "image_size", None),
            frame_sha256=(
                data["frame_sha256"].astype(str)
                if "frame_sha256" in data.files
                else None
            ),
            source_sha256=(
                data["source_sha256"].astype(str)
                if "source_sha256" in data.files
                else None
            ),
            keypoints_3d=_get(data, "keypoints_3d", None),
            keypoint_3d_valid=_get(data, "keypoint_3d_valid", None),
            torso_positions=_get(data, "torso_positions", None),
            torso_position_valid=_get(data, "torso_position_valid", None),
            wrist_local_positions=_get(data, "wrist_local_positions", None),
            wrist_local_valid=_get(data, "wrist_local_valid", None),
            palm_normals=_get(data, "palm_normals", None),
            palm_valid=_get(data, "palm_valid", None),
            torso_to_camera=_get(data, "torso_to_camera", None),
            wrist_to_torso=_get(data, "wrist_to_torso", None),
            u0_reliability=_get(data, "u0_reliability", None),
            target_joint_positions=(
                data["target_joint_positions"] if has_target_joints else None
            ),
            target_joint_valid=(
                data["target_joint_valid"] if has_target_joints else None
            ),
            metadata_json=str(_get(data, "metadata_json", "{}")),
        )
    clip.validate()
    return clip
