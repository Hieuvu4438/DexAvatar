from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from signeft.data.manifest import ManifestRecord
from signeft.observations.nlf import TTA_NAMES
from signeft.observations.validate import POSE_CORE, validate_nlf_cache, validate_pose_cache


def _record(tmp_path: Path) -> tuple[ManifestRecord, Path]:
    record = ManifestRecord(
        record_id="S/1", sign_id="S", sign_class="S", frame_index=0,
        source_frame_id=1, rgb_path="unused.png", a3f_state_path="unused.npz",
        a3f_obj_path="unused.obj", width=192, height=256, sha256_rgb="rgb",
        sha256_a3f_state="state", sha256_a3f_obj="obj",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(asdict(record)) + "\n", encoding="utf-8")
    return record, manifest


def _pose_payload(*, swapped: bool = False, uniform: bool = False) -> dict[str, np.ndarray]:
    names = np.asarray(POSE_CORE)
    heatmap = np.zeros((len(names), 64, 48), dtype=np.uint8)
    if uniform:
        heatmap.fill(255)
    else:
        heatmap[:, 32, 24] = 255
    probability = heatmap.astype(np.float64)
    probability /= probability.sum((1, 2), keepdims=True)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum((1, 2))
    coordinates = np.tile(np.asarray([96.0, 128.0], dtype=np.float32), (len(names), 1))
    left = POSE_CORE.index("left_shoulder")
    right = POSE_CORE.index("right_shoulder")
    coordinates[left, 0], coordinates[right, 0] = ((70.0, 120.0) if swapped else (120.0, 70.0))
    return {
        "joint_names": names,
        "heatmap_q": heatmap,
        "heatmap_scale": np.ones(len(names), dtype=np.float32) / 255.0,
        "heatmap_zero": np.zeros(len(names), dtype=np.float32),
        "valid": np.ones(len(names), dtype=bool),
        "entropy": entropy.astype(np.float32),
        "cov2d": np.zeros((len(names), 2, 2), dtype=np.float32),
        "crop_to_full": np.eye(3, dtype=np.float32),
        "coords_full": coordinates,
        "rgb_sha256": np.asarray("rgb"),
    }


@pytest.mark.parametrize("payload", [
    _pose_payload(uniform=True),
    _pose_payload(swapped=True),
])
def test_corrupt_pose_observation_fails_closed(tmp_path: Path, payload: dict[str, np.ndarray]):
    _, manifest = _record(tmp_path)
    destination = tmp_path / "pose" / "S" / "000001.npz"
    destination.parent.mkdir(parents=True)
    np.savez_compressed(destination, **payload)
    with pytest.raises(RuntimeError, match="pose cache validation failed"):
        validate_pose_cache(manifest, tmp_path / "pose", tmp_path / "pose_report.json")


def test_nlf_scale_times_1000_fails_closed(tmp_path: Path):
    _, manifest = _record(tmp_path)
    required = (
        "pelvis", "neck", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist",
    )
    names = required + tuple(f"joint_{i}" for i in range(55 - len(required)))
    joints = np.zeros((5, 55, 3), dtype=np.float32)
    index = {name: i for i, name in enumerate(names)}
    values = {
        "neck": (0.0, 300.0, 0.0), "left_shoulder": (200.0, 300.0, 0.0),
        "right_shoulder": (-200.0, 300.0, 0.0), "left_elbow": (500.0, 300.0, 0.0),
        "right_elbow": (-500.0, 300.0, 0.0), "left_wrist": (800.0, 300.0, 0.0),
        "right_wrist": (-800.0, 300.0, 0.0),
    }
    for name, value in values.items():
        joints[:, index[name]] = value
    destination = tmp_path / "nlf" / "S" / "000001.npz"
    destination.parent.mkdir(parents=True)
    np.savez_compressed(
        destination, joint_names=np.asarray(names), joints3d=joints,
        valid=np.ones((5, 55), dtype=bool), cov3d=np.zeros((55, 3, 3), dtype=np.float32),
        tta_names=np.asarray(TTA_NAMES), unit=np.asarray("meter"),
        coord_frame=np.asarray("evaluator_camera_centered"), rgb_sha256=np.asarray("rgb"),
    )
    with pytest.raises(RuntimeError, match="NLF cache validation failed"):
        validate_nlf_cache(manifest, tmp_path / "nlf", tmp_path / "nlf_report.json")
