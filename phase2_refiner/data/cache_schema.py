"""Versioned Phase 2 observation-cache contract.

Schema v2 expands the executable v1 vertical slice with the coordinate,
provenance, reliability, and optional geometry fields required by the Phase 2
design.  Loading v1 remains supported so existing smoke caches are reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4)
PHASE2R_SEMANTIC_CONTRACT = "phase2r-v1"
NUM_JOINTS = 51
NUM_OBSERVATION_FEATURES = 8
NUM_HANDS = 2


def _identity_transforms(shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.eye(4, dtype=np.float32), shape + (4, 4)).copy()


def _filled_strings(length: int, value: str) -> np.ndarray:
    """Create a fixed-width string vector without NumPy's U1 truncation."""
    return np.asarray([value] * length, dtype=str)


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
    reprojection_residual_2d: np.ndarray | None = None
    target_joint_positions: np.ndarray | None = None
    target_joint_valid: np.ndarray | None = None
    raw_confidence: np.ndarray | None = None
    calibrated_confidence: np.ndarray | None = None
    detector_present: np.ndarray | None = None
    track_valid: np.ndarray | None = None
    in_frame: np.ndarray | None = None
    copied_observation: np.ndarray | None = None
    interpolated_observation: np.ndarray | None = None
    target_quality: np.ndarray | None = None
    initializer_component: np.ndarray | None = None
    fallback_reason: np.ndarray | None = None
    camera_model: np.ndarray | None = None
    camera_intrinsics: np.ndarray | None = None
    crop_transform: np.ndarray | None = None
    hand_activity: np.ndarray | None = None
    semantic_contract_version: str = "legacy"
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
            self.frame_sha256 = _filled_strings(t, "")
        if self.source_sha256 is None:
            self.source_sha256 = _filled_strings(t, "")
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
        if self.reprojection_residual_2d is None:
            self.reprojection_residual_2d = np.zeros(
                (t, NUM_JOINTS, 2), dtype=np.float32
            )
        confidence = np.clip(self.observation_features[..., 0], 0.0, 1.0)
        presence = self.observation_features[..., 1] > 0.5
        missing = self.observation_features[..., 2] > 0.5
        truncation = self.observation_features[..., 4] > 0.5
        duplicate = self.observation_features[..., 6] > 0.5
        if self.raw_confidence is None:
            self.raw_confidence = confidence.astype(np.float32)
        if self.calibrated_confidence is None:
            self.calibrated_confidence = confidence.astype(np.float32)
        if self.detector_present is None:
            self.detector_present = presence
        if self.track_valid is None:
            self.track_valid = self.keypoint_valid.copy()
        if self.in_frame is None:
            self.in_frame = ~truncation
        if self.copied_observation is None:
            self.copied_observation = duplicate
        if self.interpolated_observation is None:
            self.interpolated_observation = np.zeros((t, NUM_JOINTS), dtype=bool)
        if self.target_quality is None:
            self.target_quality = np.zeros((t, NUM_JOINTS), dtype=np.float32)
        if self.initializer_component is None:
            self.initializer_component = _filled_strings(t, "unknown")
        if self.fallback_reason is None:
            self.fallback_reason = _filled_strings(t, "")
        if self.camera_model is None:
            self.camera_model = _filled_strings(t, "unknown")
        if self.camera_intrinsics is None:
            self.camera_intrinsics = np.broadcast_to(
                np.eye(3, dtype=np.float32), (t, 3, 3)
            ).copy()
        if self.crop_transform is None:
            self.crop_transform = np.broadcast_to(
                np.eye(3, dtype=np.float32), (t, 3, 3)
            ).copy()
        if self.hand_activity is None:
            self.hand_activity = np.stack(
                (
                    self.track_valid[:, 21:36].mean(axis=1),
                    self.track_valid[:, 36:51].mean(axis=1),
                ),
                axis=-1,
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
            "reprojection_residual_2d": (t, NUM_JOINTS, 2),
            "raw_confidence": (t, NUM_JOINTS),
            "calibrated_confidence": (t, NUM_JOINTS),
            "detector_present": (t, NUM_JOINTS),
            "track_valid": (t, NUM_JOINTS),
            "in_frame": (t, NUM_JOINTS),
            "copied_observation": (t, NUM_JOINTS),
            "interpolated_observation": (t, NUM_JOINTS),
            "target_quality": (t, NUM_JOINTS),
            "initializer_component": (t,),
            "fallback_reason": (t,),
            "camera_model": (t,),
            "camera_intrinsics": (t, 3, 3),
            "crop_transform": (t, 3, 3),
            "hand_activity": (t, NUM_HANDS),
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
            "reprojection_residual_2d",
            "raw_confidence",
            "calibrated_confidence",
            "target_quality",
            "camera_intrinsics",
            "crop_transform",
            "hand_activity",
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
        for name in (
            "raw_confidence",
            "calibrated_confidence",
            "target_quality",
            "hand_activity",
        ):
            value = getattr(self, name)
            if not np.all((value >= 0.0) & (value <= 1.0)):
                raise ValueError(f"{name} must be within [0, 1]")


def validate_phase2r_semantics(clip: CacheClip) -> None:
    """Reject legacy/ambiguous caches before a Phase 2R experiment starts."""
    clip.validate()
    if clip.semantic_contract_version != PHASE2R_SEMANTIC_CONTRACT:
        raise ValueError(
            "Phase 2R requires semantic_contract_version="
            f"{PHASE2R_SEMANTIC_CONTRACT!r}; got {clip.semantic_contract_version!r}"
        )
    metadata = json.loads(clip.metadata_json)
    policy = metadata.get("coordinate_policy")
    if not isinstance(policy, dict) or not policy.get("keypoints_2d"):
        raise ValueError(
            "Phase 2R cache metadata requires coordinate_policy.keypoints_2d"
        )
    if np.any(clip.track_valid & ~clip.detector_present):
        raise ValueError("track_valid cannot be true when detector_present is false")
    if np.any(clip.interpolated_observation & clip.detector_present):
        raise ValueError("interpolated observations cannot be detector-present")


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
        "reprojection_residual_2d": clip.reprojection_residual_2d.astype(np.float32),
        "raw_confidence": clip.raw_confidence.astype(np.float32),
        "calibrated_confidence": clip.calibrated_confidence.astype(np.float32),
        "detector_present": clip.detector_present.astype(bool),
        "track_valid": clip.track_valid.astype(bool),
        "in_frame": clip.in_frame.astype(bool),
        "copied_observation": clip.copied_observation.astype(bool),
        "interpolated_observation": clip.interpolated_observation.astype(bool),
        "target_quality": clip.target_quality.astype(np.float32),
        "initializer_component": clip.initializer_component.astype(str),
        "fallback_reason": clip.fallback_reason.astype(str),
        "camera_model": clip.camera_model.astype(str),
        "camera_intrinsics": clip.camera_intrinsics.astype(np.float32),
        "crop_transform": clip.crop_transform.astype(np.float32),
        "hand_activity": clip.hand_activity.astype(np.float32),
        "semantic_contract_version": np.asarray(clip.semantic_contract_version),
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
            reprojection_residual_2d=_get(data, "reprojection_residual_2d", None),
            target_joint_positions=(
                data["target_joint_positions"] if has_target_joints else None
            ),
            target_joint_valid=(
                data["target_joint_valid"] if has_target_joints else None
            ),
            raw_confidence=_get(data, "raw_confidence", None),
            calibrated_confidence=_get(data, "calibrated_confidence", None),
            detector_present=_get(data, "detector_present", None),
            track_valid=_get(data, "track_valid", None),
            in_frame=_get(data, "in_frame", None),
            copied_observation=_get(data, "copied_observation", None),
            interpolated_observation=_get(data, "interpolated_observation", None),
            target_quality=_get(data, "target_quality", None),
            initializer_component=(
                data["initializer_component"].astype(str)
                if "initializer_component" in data.files
                else None
            ),
            fallback_reason=(
                data["fallback_reason"].astype(str)
                if "fallback_reason" in data.files
                else None
            ),
            camera_model=(
                data["camera_model"].astype(str)
                if "camera_model" in data.files
                else None
            ),
            camera_intrinsics=_get(data, "camera_intrinsics", None),
            crop_transform=_get(data, "crop_transform", None),
            hand_activity=_get(data, "hand_activity", None),
            semantic_contract_version=str(
                _get(data, "semantic_contract_version", "legacy")
            ),
            metadata_json=str(_get(data, "metadata_json", "{}")),
        )
    clip.validate()
    return clip
