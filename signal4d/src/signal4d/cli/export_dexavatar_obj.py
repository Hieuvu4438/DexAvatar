from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..data.manifest import load_manifest
from ..io.obj import format_face_block, read_simple_obj, write_dexavatar_obj
from ..io.predictions import PredictionArtifact
from ..utils.hashing import sha256_file


def run(
    manifest_path: str,
    prediction_root: str,
    model_path: str,
    output_root: str,
    method_name: str,
    decimals: int = 8,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    source_root = Path(prediction_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OBJ export root: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".export_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")

    model = np.load(model_path, allow_pickle=True)
    faces = np.asarray(model["f"], dtype=np.int64)
    face_block = format_face_block(faces)
    model_hash = sha256_file(model_path)
    rows: list[dict[str, Any]] = []
    maximum_roundtrip_error_mm = 0.0
    frames = 0
    for item in manifest:
        prediction, metadata = PredictionArtifact.load(source_root / item.clip_id)
        if prediction.frame_ids.tolist() != item.frame_ids:
            raise ValueError(f"prediction frame mismatch for {item.clip_id}")
        if prediction.vertices is None:
            raise ValueError(f"prediction has no vertices for {item.clip_id}")
        if metadata.get("smplx_model_sha256") != model_hash:
            raise ValueError(f"SMPL-X model mismatch for {item.clip_id}")
        if metadata.get("coordinate_convention") != "opencv_x_right_y_down_z_forward":
            raise ValueError(f"coordinate mismatch for {item.clip_id}")

        mesh_root = output / item.clip_id / "smplifyx" / "meshes"
        for frame_index, frame_id in enumerate(item.frame_ids):
            vertices = prediction.vertices[frame_index].detach().cpu().numpy()
            target = mesh_root / f"low_{frame_id}.obj"
            write_dexavatar_obj(
                target, vertices, faces, decimals=decimals, face_block=face_block
            )
            roundtrip_vertices, roundtrip_faces = read_simple_obj(target)
            np.testing.assert_array_equal(roundtrip_faces, faces)
            if roundtrip_vertices.shape != vertices.shape:
                raise ValueError(f"OBJ vertex count mismatch after export: {target}")
            roundtrip_error_mm = float(
                np.max(np.abs(roundtrip_vertices - vertices)) * 1000.0
            )
            maximum_roundtrip_error_mm = max(maximum_roundtrip_error_mm, roundtrip_error_mm)
            rows.append(
                {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "obj_relpath": str(target.relative_to(output)),
                    "obj_sha256": sha256_file(target),
                    "source_artifact_sha256": metadata["artifact_sha256"],
                    "max_roundtrip_error_mm": roundtrip_error_mm,
                }
            )
            frames += 1

    report = {
        "schema_version": "1.0",
        "method_name": method_name,
        "format": "dexavatar_trimesh_obj",
        "header": "# https://github.com/mikedh/trimesh",
        "vertex_format": f"v %.{decimals}f %.{decimals}f %.{decimals}f",
        "face_format": "f %d %d %d (one-indexed)",
        "coordinate_convention": "opencv_x_right_y_down_z_forward",
        "length_unit": "meter",
        "vertices_per_mesh": int(faces.max()) + 1,
        "faces_per_mesh": int(len(faces)),
        "clips": len(manifest),
        "frames": frames,
        "manifest_sha256": sha256_file(manifest_path),
        "smplx_model_sha256": model_hash,
        "prediction_root": str(source_root),
        "max_roundtrip_error_mm": maximum_roundtrip_error_mm,
        "files": rows,
    }
    (output / "export_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report
