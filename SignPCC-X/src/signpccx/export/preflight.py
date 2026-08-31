from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from signpccx.export.obj import validate_mesh


def load_obj_minimal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("v "):
            vertices.append([float(value) for value in raw.split()[1:4]])
        elif raw.startswith("f "):
            faces.append([int(value.split("/")[0]) - 1 for value in raw.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def first_stem_int(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if match is None:
        raise ValueError(path)
    return int(match.group())


def preflight_sign(
    mesh_dir: Path,
    expected_count: int,
    canonical_faces: np.ndarray,
    require_contiguous_names: bool = True,
) -> dict[str, object]:
    paths = sorted(mesh_dir.glob("*.obj"), key=first_stem_int)
    if len(paths) != expected_count:
        raise RuntimeError(f"{mesh_dir}: OBJ count {len(paths)} != {expected_count}")
    names = [f"{index:03d}.obj" for index in range(expected_count)]
    if require_contiguous_names and [path.name for path in paths] != names:
        raise RuntimeError(f"{mesh_dir}: names are not contiguous 000.obj..N.obj")
    for path in paths:
        vertices, faces = load_obj_minimal(path)
        validate_mesh(vertices, faces)
        if not np.array_equal(faces, canonical_faces):
            raise RuntimeError(f"{path}: face topology/order mismatch")
    return {
        "mesh_dir": str(mesh_dir), "count": len(paths), "status": "ok",
        "contiguous_names_required": require_contiguous_names,
    }
