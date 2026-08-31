"""Explicit trusted conversion boundary for existing DexAvatar fit parameters."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.utils.hashing import file_sha256


def axis_angle_to_matrix(rotation: Tensor) -> Tensor:
    if rotation.shape[-1] != 3:
        raise ValueError("axis-angle rotation must end in 3")
    x, y, z = rotation.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(
        *rotation.shape[:-1], 3, 3
    )
    theta = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    first = torch.sinc(theta / torch.pi)[..., None]
    theta_squared = theta.square()
    second_scalar = torch.where(
        theta_squared > 1e-8,
        (1 - torch.cos(theta)) / theta_squared,
        0.5 - theta_squared / 24,
    )
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    return identity + first * skew + second_scalar[..., None] * (skew @ skew)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def axis_angle_to_6d(rotation: Tensor) -> Tensor:
    return matrix_to_rotation_6d(axis_angle_to_matrix(rotation))


class DexAvatarPklInitializer:
    """Convert existing per-frame fitting PKLs once, then replay safe NPZ.

    Pickle can execute code. Loading requires both `trusted=True` and a mapping
    from every file to an expected SHA-256. The resulting metadata preserves all
    input hashes and quantifies per-frame shape variation before the required
    clip-shared mean is applied.
    """

    def __init__(self, fps: float):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps

    def reconstruct_from_directory(
        self,
        results_dir: str | Path,
        *,
        expected_hashes: dict[str, str],
        trusted: bool = False,
        include_names: set[str] | None = None,
    ) -> tuple[TrajectoryState, dict[str, object]]:
        if not trusted:
            raise PermissionError("legacy DexAvatar pickle conversion requires trusted=True")
        paths = sorted(
            (
                path
                for path in Path(results_dir).glob("low_*.pkl")
                if include_names is None or path.name in include_names
            ),
            key=lambda path: int(re.search(r"low_(\d+)", path.stem).group(1)),
        )
        if not paths:
            raise FileNotFoundError(f"no low_*.pkl under {results_dir}")
        records: list[dict[str, np.ndarray]] = []
        source_hashes: dict[str, str] = {}
        frame_ids: list[int] = []
        for path in paths:
            digest = file_sha256(path)
            if expected_hashes.get(path.name) != digest:
                raise ValueError(f"missing/mismatched trusted SHA-256 for {path}")
            with path.open("rb") as handle:
                record = pickle.load(handle, encoding="latin1")
            if not isinstance(record, dict):
                raise ValueError(f"unexpected DexAvatar record: {path}")
            records.append(record)
            source_hashes[path.name] = digest
            frame_ids.append(int(re.search(r"low_(\d+)", path.stem).group(1)))

        def stack(name: str, width: int) -> Tensor:
            values = []
            for record in records:
                value = np.asarray(record[name], dtype=np.float32).reshape(-1)
                if value.size != width:
                    raise ValueError(f"{name} expected {width}, got {value.size}")
                values.append(value)
            return torch.from_numpy(np.stack(values))[None]

        beta_frames = stack("betas", 10)
        camera_intrinsics = stack("K", 9).reshape(1, len(paths), 3, 3)
        beta = beta_frames.mean(dim=1)
        translation = stack("transl", 3)
        velocity = torch.zeros_like(translation)
        velocity[:, 1:] = (translation[:, 1:] - translation[:, :-1]) * self.fps
        face_fields = []
        for name, width in (
            ("jaw_pose", 3),
            ("leye_pose", 3),
            ("reye_pose", 3),
            ("expression", 10),
        ):
            face_fields.append(stack(name, width))
        state = TrajectoryState(
            root_rot6d=axis_angle_to_6d(stack("global_orient", 3)),
            root_translation=translation,
            root_velocity=velocity,
            body_rot6d=axis_angle_to_6d(stack("body_pose", 63).reshape(1, len(paths), 21, 3)),
            left_hand_rot6d=axis_angle_to_6d(
                stack("left_hand_pose", 45).reshape(1, len(paths), 15, 3)
            ),
            right_hand_rot6d=axis_angle_to_6d(
                stack("right_hand_pose", 45).reshape(1, len(paths), 15, 3)
            ),
            face_state=torch.cat(face_fields, dim=-1),
            beta=beta,
            valid_mask=torch.ones(1, len(paths), dtype=torch.bool),
        ).validate()
        metadata: dict[str, object] = {
            "backend": "dexavatar_existing_pkl_conversion",
            "frame_ids": frame_ids,
            "fps": self.fps,
            "source_hashes": source_hashes,
            "shape_policy": "clip_mean",
            "max_beta_deviation": float((beta_frames - beta[:, None]).abs().max()),
            "camera_intrinsics": camera_intrinsics.numpy().tolist(),
        }
        return state, metadata
