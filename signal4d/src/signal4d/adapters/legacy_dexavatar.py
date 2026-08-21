from __future__ import annotations

import re
from pathlib import Path

import torch


def load_obj_vertices(path: str | Path) -> torch.Tensor:
    vertices: list[list[float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    if not vertices:
        raise ValueError(f"No vertices in {path}")
    return torch.tensor(vertices, dtype=torch.float32)


def collect_legacy_meshes(root: str | Path, clip_id: str) -> tuple[list[int], torch.Tensor]:
    mesh_root = Path(root) / clip_id / "smplifyx" / "meshes"
    paths = sorted(
        mesh_root.glob("low_*.obj"),
        key=lambda path: int(re.search(r"low_(\d+)", path.stem).group(1)),  # type: ignore[union-attr]
    )
    if not paths:
        raise FileNotFoundError(f"No legacy meshes under {mesh_root}")
    frame_ids = [int(re.search(r"low_(\d+)", path.stem).group(1)) for path in paths]  # type: ignore[union-attr]
    vertices = torch.stack([load_obj_vertices(path) for path in paths])
    return frame_ids, vertices
