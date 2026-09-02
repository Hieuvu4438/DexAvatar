from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


STATE_KEYS = (
    "betas",
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
    "transl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class H1State:
    path: Path
    arrays: dict[str, np.ndarray]
    vertices_evaluator: np.ndarray
    K_evaluator: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "H1State":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["coord_frame"]) != "evaluator_camera":
                raise RuntimeError(f"unexpected H1 coordinate frame: {path}")
            if str(archive["unit"]) != "meter":
                raise RuntimeError(f"unexpected H1 unit: {path}")
            arrays = {
                key: np.asarray(archive[key], dtype=np.float32).reshape(1, -1)
                for key in STATE_KEYS
            }
            vertices = np.asarray(archive["vertices"], dtype=np.float32)
            K = np.asarray(archive["K"], dtype=np.float32).reshape(3, 3)
        if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
            raise RuntimeError(f"invalid H1 vertices: {path}")
        return cls(path=path, arrays=arrays, vertices_evaluator=vertices, K_evaluator=K)


def state_path(root: Path, record: dict) -> Path:
    return root / record["sign_id"] / f"{int(record['source_frame_id']):06d}.npz"

