"""Immutable complete initialization artifacts including camera and provenance."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

from .camera import CameraTrajectory


def save_initialization_artifact(
    destination: str | Path,
    trajectory: TrajectoryState,
    camera: CameraTrajectory,
    *,
    metadata: dict[str, Any],
    source_hashes: dict[str, str],
    smplx_forward: dict[str, np.ndarray] | None = None,
) -> Path:
    trajectory.validate()
    camera.validate()
    if trajectory.valid_mask.shape != camera.valid_mask.shape:
        raise ValueError("trajectory/camera batch-time mismatch")
    if not source_hashes or any(len(value) != 64 for value in source_hashes.values()):
        raise ValueError("initialization source hashes must be non-empty SHA-256 values")
    required_metadata = {
        "clip_id",
        "dexavatar_commit",
        "config_sha256",
        "checkpoint_sha256",
        "runtime",
        "development_only",
    }
    if not required_metadata <= metadata.keys():
        raise ValueError("initialization metadata lacks required provenance fields")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"immutable initialization artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        trajectory_path = temporary / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            **{
                name: value.detach().cpu().numpy()
                for name in trajectory.__dataclass_fields__
                if isinstance((value := getattr(trajectory, name)), torch.Tensor)
            },
        )
        camera_path = temporary / "camera.npz"
        np.savez_compressed(
            camera_path,
            intrinsics=camera.intrinsics.detach().cpu().numpy(),
            world_to_camera=camera.world_to_camera.detach().cpu().numpy(),
            image_size_wh=camera.image_size_wh.detach().cpu().numpy(),
            valid_mask=camera.valid_mask.detach().cpu().numpy(),
        )
        source_path = temporary / "source_hashes.json"
        source_path.write_text(json.dumps(source_hashes, sort_keys=True, indent=2) + "\n", "utf-8")
        payload = {
            "schema_version": "dcg_initialization_v1",
            **metadata,
            "coordinate_convention": camera.coordinate_convention,
            "trajectory_sha256": file_sha256(trajectory_path),
            "camera_sha256": file_sha256(camera_path),
            "source_hashes_sha256": file_sha256(source_path),
            "frames": int(trajectory.valid_mask.shape[1]),
        }
        if smplx_forward is not None:
            required = {"vertices", "joints", "frame_ids"}
            if set(smplx_forward) != required:
                raise ValueError("SMPL-X forward replay requires vertices/joints/frame_ids")
            forward_path = temporary / "smplx_forward.npz"
            np.savez_compressed(forward_path, **smplx_forward)
            payload["smplx_forward_sha256"] = file_sha256(forward_path)
        payload["metadata_identity_sha256"] = canonical_hash(payload)
        (temporary / "metadata.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", "utf-8"
        )
        (temporary / "INITIALIZATION_COMPLETE").write_text("complete\n", "utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_initialization_artifact(
    source: str | Path,
) -> tuple[TrajectoryState, CameraTrajectory, dict[str, Any]]:
    source = Path(source)
    if not (source / "INITIALIZATION_COMPLETE").is_file():
        raise ValueError("initialization artifact has no completion marker")
    metadata = json.loads((source / "metadata.json").read_text("utf-8"))
    identity = metadata.pop("metadata_identity_sha256", None)
    if identity != canonical_hash(metadata):
        raise ValueError("initialization metadata identity mismatch")
    metadata["metadata_identity_sha256"] = identity
    for filename, field in (
        ("trajectory.npz", "trajectory_sha256"),
        ("camera.npz", "camera_sha256"),
        ("source_hashes.json", "source_hashes_sha256"),
    ):
        if file_sha256(source / filename) != metadata.get(field):
            raise ValueError(f"initialization {filename} hash mismatch")
    with np.load(source / "trajectory.npz", allow_pickle=False) as arrays:
        values = {name: torch.from_numpy(arrays[name]) for name in arrays.files}
    trajectory = TrajectoryState(
        root_rot6d=values["root_rot6d"],
        root_translation=values["root_translation"],
        root_velocity=values["root_velocity"],
        body_rot6d=values["body_rot6d"],
        left_hand_rot6d=values["left_hand_rot6d"],
        right_hand_rot6d=values["right_hand_rot6d"],
        beta=values["beta"],
        valid_mask=values["valid_mask"].bool(),
        face_state=values.get("face_state"),
    ).validate()
    with np.load(source / "camera.npz", allow_pickle=False) as arrays:
        camera = CameraTrajectory(
            intrinsics=torch.from_numpy(arrays["intrinsics"]),
            world_to_camera=torch.from_numpy(arrays["world_to_camera"]),
            image_size_wh=torch.from_numpy(arrays["image_size_wh"]),
            valid_mask=torch.from_numpy(arrays["valid_mask"]).bool(),
            coordinate_convention=metadata["coordinate_convention"],
        ).validate()
    if trajectory.valid_mask.shape != camera.valid_mask.shape:
        raise ValueError("loaded trajectory/camera time mismatch")
    return trajectory, camera, metadata
