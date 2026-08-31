from __future__ import annotations

from pathlib import Path

import numpy as np

from signpccx.io import atomic_write_json, sha256_file


def nearest_region(
    template_vertices: np.ndarray,
    landmark: np.ndarray,
    allowed_vertex_ids: np.ndarray,
    k: int = 24,
) -> np.ndarray:
    vertices = np.asarray(template_vertices, dtype=np.float64)
    point = np.asarray(landmark, dtype=np.float64).reshape(3)
    allowed = np.asarray(allowed_vertex_ids, dtype=np.int64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"template vertices {vertices.shape}")
    if not len(allowed) or allowed.min() < 0 or allowed.max() >= len(vertices):
        raise IndexError("invalid allowed vertex IDs")
    if k <= 0 or k > len(allowed):
        raise ValueError(f"k={k} for {len(allowed)} allowed vertices")
    squared_distance = np.square(vertices[allowed] - point).sum(axis=1)
    # Stable sort makes the cache deterministic when distances tie.
    return allowed[np.argsort(squared_distance, kind="stable")[:k]]


def write_contact_region_cache(
    output: Path,
    regions: dict[str, np.ndarray],
    model_path: Path,
) -> dict[str, object]:
    normalized = {}
    for name, indices in sorted(regions.items()):
        value = np.asarray(indices, dtype=np.int64).reshape(-1)
        if not len(value) or len(np.unique(value)) != len(value):
            raise ValueError(f"invalid region {name}")
        normalized[name] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **normalized)
    report = {
        "schema_version": "signpccx.contact-regions.v1",
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "regions": {name: value.tolist() for name, value in normalized.items()},
        "cache_sha256": sha256_file(output),
    }
    atomic_write_json(output.with_suffix(".json"), report)
    return report
