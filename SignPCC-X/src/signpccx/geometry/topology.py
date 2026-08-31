from __future__ import annotations

from pathlib import Path

import numpy as np

from signpccx.io import array_sha256


def load_canonical_faces(model_path: Path) -> np.ndarray:
    with np.load(model_path, allow_pickle=True) as model:
        if "f" not in model.files:
            raise KeyError(f"No faces 'f' in {model_path}")
        faces = np.asarray(model["f"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(faces.shape)
    return np.ascontiguousarray(faces)


def validate_faces_lock(faces: np.ndarray, expected_hash: str, expected_count: int | None = None) -> str:
    if expected_count is not None and len(faces) != expected_count:
        raise RuntimeError(f"face count {len(faces)} != {expected_count}")
    digest = array_sha256(np.asarray(faces, dtype=np.int64))
    if digest != expected_hash:
        raise RuntimeError(f"faces hash {digest} != {expected_hash}")
    return digest

