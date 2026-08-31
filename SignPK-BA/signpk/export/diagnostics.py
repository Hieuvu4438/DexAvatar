from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )


def render_front_side_points(
    vertices: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "SignPK-BA diagnostic",
    sample_stride: int = 8,
    y_up: bool = True,
) -> None:
    """Dependency-light front/side geometry check; not a paper renderer.

    Exported SGNify meshes are y-up after the benchmark-only x-axis rotation.
    Set ``y_up=False`` only when inspecting canonical camera-space vertices.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for diagnostic rendering") from exc
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be [V,3]")
    points = vertices[::sample_stride]
    vertical = points[:, 1] if y_up else -points[:, 1]
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(points[:, 0], vertical, s=0.3)
    axes[0].set_title("front")
    axes[1].scatter(points[:, 2], vertical, s=0.3)
    axes[1].set_title("side")
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
    figure.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
