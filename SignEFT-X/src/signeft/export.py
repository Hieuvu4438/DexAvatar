from __future__ import annotations

from pathlib import Path
import shutil

from signeft.data.manifest import read_manifest
from signeft.io.obj import load_obj, validate_mesh
from signeft.io_utils import array_sha256, atomic_write_json, sha256_file


def materialize(manifest: Path, run_root: Path, output_root: Path) -> dict[str, object]:
    records = read_manifest(manifest)
    by_sign: dict[str, list] = {}
    for record in records:
        by_sign.setdefault(record.sign_id, []).append(record)
    items = []
    for sign, sign_records in sorted(by_sign.items()):
        hashes = []
        for record in sorted(sign_records, key=lambda item: item.frame_index):
            source = run_root / "frames" / sign / f"{record.source_frame_id:06d}.obj"
            destination = output_root / sign / "smplifyx" / "meshes" / f"{record.frame_index:03d}.obj"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                if sha256_file(destination) != sha256_file(source):
                    raise RuntimeError(f"refusing to overwrite different materialized OBJ: {destination}")
            else:
                shutil.copyfile(source, destination)
            hashes.append(sha256_file(destination))
        items.append({"sign": sign, "frames": len(sign_records), "obj_sha256": hashes})
    report = {
        "schema_version": "signeft.materialization.v1",
        "status": "ok",
        "signs": len(items),
        "frames": len(records),
        "output_root": str(output_root.resolve()),
        "items": items,
    }
    atomic_write_json(output_root / "materialization_summary.json", report)
    return report


def preflight(
    manifest: Path,
    prediction_root: Path,
    expected_faces_sha256: str,
    output: Path,
) -> dict[str, object]:
    records = read_manifest(manifest)
    by_sign: dict[str, list] = {}
    for record in records:
        by_sign.setdefault(record.sign_id, []).append(record)
    items = []
    for sign, sign_records in sorted(by_sign.items()):
        mesh_dir = prediction_root / sign / "smplifyx" / "meshes"
        paths = sorted(mesh_dir.glob("*.obj"))
        expected_names = [f"{index:03d}.obj" for index in range(len(sign_records))]
        if [path.name for path in paths] != expected_names:
            raise RuntimeError(f"non-contiguous/missing predictions: {mesh_dir}")
        for path in paths:
            vertices, faces = load_obj(path)
            validate_mesh(vertices, faces)
            if array_sha256(faces) != expected_faces_sha256:
                raise RuntimeError(f"face topology/order mismatch: {path}")
        items.append({"sign": sign, "frames": len(paths), "status": "ok"})
    report = {
        "schema_version": "signeft.preflight.v1",
        "status": "ok",
        "signs": len(items),
        "frames": sum(item["frames"] for item in items),
        "items": items,
    }
    atomic_write_json(output, report)
    return report
