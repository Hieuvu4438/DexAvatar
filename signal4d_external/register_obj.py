"""Register an existing DexAvatar OBJ tree for the strict author evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signal4d.data.manifest import load_manifest
from signal4d.io.obj import read_simple_obj
from signal4d.utils.hashing import sha256_file


def register(
    manifest_path: Path,
    source_root: Path,
    model_path: Path,
    output_root: Path,
    method_name: str,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite OBJ registry: {output_root}")
    manifest = load_manifest(manifest_path)
    faces = np.asarray(np.load(model_path, allow_pickle=True)["f"], dtype=np.int64)
    rows = []
    output_root.mkdir(parents=True)
    incomplete = output_root / ".register_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    for item in manifest:
        source_clip = (source_root / item.clip_id).resolve()
        mesh_root = source_clip / "smplifyx" / "meshes"
        expected = {f"low_{frame_id}.obj" for frame_id in item.frame_ids}
        actual = {path.name for path in mesh_root.glob("*.obj")}
        if actual != expected:
            raise ValueError(
                f"OBJ coverage mismatch for {item.clip_id}: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        (output_root / item.clip_id).symlink_to(source_clip, target_is_directory=True)
        for frame_id in item.frame_ids:
            source = mesh_root / f"low_{frame_id}.obj"
            vertices, obj_faces = read_simple_obj(source)
            if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
                raise ValueError(f"invalid SMPL-X vertices: {source}")
            np.testing.assert_array_equal(obj_faces, faces)
            rows.append(
                {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "obj_relpath": str(
                        Path(item.clip_id) / "smplifyx" / "meshes" / source.name
                    ),
                    "obj_sha256": sha256_file(source),
                    "source_obj_path": str(source),
                }
            )
    report = {
        "schema_version": "1.0",
        "method_name": method_name,
        "format": "dexavatar_trimesh_obj",
        "registration": "validated_absolute_clip_symlinks",
        "header": "# https://github.com/mikedh/trimesh",
        "coordinate_convention": "opencv_x_right_y_down_z_forward",
        "length_unit": "meter",
        "vertices_per_mesh": 10475,
        "faces_per_mesh": int(len(faces)),
        "clips": len(manifest),
        "frames": len(rows),
        "manifest_sha256": sha256_file(manifest_path),
        "smplx_model_sha256": sha256_file(model_path),
        "source_root": str(source_root.resolve()),
        "files": rows,
    }
    (output_root / "export_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    args = parser.parse_args()
    report = register(
        manifest_path=args.manifest,
        source_root=args.source_root,
        model_path=args.model_path,
        output_root=args.output_root,
        method_name=args.method_name,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
