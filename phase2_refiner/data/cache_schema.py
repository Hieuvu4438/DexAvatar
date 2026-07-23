"""Versioned Phase 2 observation-cache contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
NUM_JOINTS = 51
NUM_OBSERVATION_FEATURES = 8


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

    def validate(self) -> None:
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
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {value.shape}")
        if self.betas.shape != (10,):
            raise ValueError(f"betas: expected (10,), got {self.betas.shape}")
        if self.target_axis_angle is not None and self.target_axis_angle.shape != (
            t,
            NUM_JOINTS,
            3,
        ):
            raise ValueError(
                f"target_axis_angle: expected {(t, NUM_JOINTS, 3)}, got {self.target_axis_angle.shape}"
            )
        if t == 0:
            raise ValueError("A cache clip must contain at least one frame")
        for name in ("init_axis_angle", "observation_features", "keypoints_2d"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains NaN or Inf")


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
        "has_target": np.asarray(clip.target_axis_angle is not None, dtype=bool),
    }
    if clip.target_axis_angle is not None:
        payload["target_axis_angle"] = clip.target_axis_angle.astype(np.float32)
    np.savez_compressed(output, **payload)


def load_cache_clip(path: str | Path) -> CacheClip:
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cache schema {version}; expected {SCHEMA_VERSION}"
            )
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
            target_axis_angle=data["target_axis_angle"]
            if bool(data["has_target"])
            else None,
        )
    clip.validate()
    return clip
