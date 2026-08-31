from __future__ import annotations

from pathlib import Path

import numpy as np

from signpccx.data.manifest import FrameRecord, read_jsonl
from signpccx.export.obj import write_obj_atomic
from signpccx.io import atomic_write_json, sha256_file


VERTEX_KEYS = ("vertices", "verts", "v", "mesh_parametric")


def load_parametric_sequence(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        vertex_key = next((key for key in VERTEX_KEYS if key in archive.files), None)
        if vertex_key is None:
            raise KeyError(f"{path}: no parametric vertex sequence")
        vertices = np.asarray(archive[vertex_key], dtype=np.float32)
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
    if vertices.ndim != 3 or vertices.shape[1:] != (10475, 3):
        raise ValueError(f"{path}: vertices {vertices.shape}")
    if frame_ids.shape != (len(vertices),):
        raise ValueError(f"{path}: frame IDs {frame_ids.shape}")
    if not np.isfinite(vertices).all():
        raise FloatingPointError(f"{path}: NaN/Inf")
    return vertices, frame_ids


def materialize_h4wpp_sequences(
    sequence_root: Path,
    manifest_root: Path,
    output_root: Path,
    canonical_faces: np.ndarray,
    export_transform: str,
    source_label: str = "H4WPP",
    schema_version: str = "signpccx.materialized-h4wpp.v1",
    signs: set[str] | None = None,
) -> dict[str, object]:
    if output_root.exists() and any(output_root.rglob("*.obj")):
        raise FileExistsError(f"Refusing to overwrite existing OBJ layout: {output_root}")
    items = []
    for manifest_path in sorted(manifest_root.glob("*.jsonl")):
        if signs is not None and manifest_path.stem not in signs:
            continue
        records = read_jsonl(manifest_path)
        sign = manifest_path.stem
        source = sequence_root / "clips" / sign / "mesh_parametric_final.npz"
        vertices, frame_ids = load_parametric_sequence(source)
        expected_ids = np.asarray([record.source_frame_id for record in records], dtype=np.int64)
        if len(vertices) != len(records) or not np.array_equal(frame_ids, expected_ids):
            raise RuntimeError(
                f"{sign}: H4W sequence IDs/count do not match manifest: "
                f"{frame_ids.tolist()[:5]} vs {expected_ids.tolist()[:5]}"
            )
        mesh_dir = output_root / sign / "smplifyx" / "meshes"
        for index, mesh in enumerate(vertices):
            write_obj_atomic(mesh_dir / f"{index:03d}.obj", mesh, canonical_faces, transform=export_transform)
        sidecar = {
            "schema_version": schema_version,
            "sign": sign,
            "frames": len(records),
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "source_coordinate_frame": "dexavatar_evaluator_obj",
            "export_transform": export_transform,
            "frame_ids": frame_ids.tolist(),
            "statuses": [f"OK_{source_label}" for _ in records],
        }
        atomic_write_json(output_root / sign / "materialization.json", sidecar)
        items.append({"sign": sign, "frames": len(records), "source_sha256": sidecar["source_sha256"]})
    if signs is not None:
        materialized = {str(item["sign"]) for item in items}
        missing = sorted(signs - materialized)
        if missing:
            raise FileNotFoundError(f"No manifest/sequence materialized for signs: {missing}")
    summary = {"schema_version": "signpccx.materialization-summary.v1", "source_label": source_label, "signs": len(items), "frames": sum(item["frames"] for item in items), "items": items}
    atomic_write_json(output_root / "materialization_summary.json", summary)
    return summary
