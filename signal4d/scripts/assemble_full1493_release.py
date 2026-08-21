#!/usr/bin/env python3
"""Assemble split-disjoint SIGNAL4D predictions and register DexAvatar OBJ files.

The script never rewrites predictions or meshes.  It validates exact manifest
coverage, model/topology compatibility, and then creates an immutable-style
release tree whose clip directories are absolute symlinks to the source runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal4d.data.manifest import load_manifest
from signal4d.io.obj import read_simple_obj
from signal4d.io.predictions import PredictionArtifact
from signal4d.utils.hashing import sha256_file


def _parse_split(value: str) -> tuple[Path, Path]:
    try:
        manifest, predictions = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "split must be MANIFEST=PREDICTION_ROOT"
        ) from error
    return Path(manifest), Path(predictions)


def _frame_keys(manifest_path: Path) -> set[tuple[str, int]]:
    return {
        (item.clip_id, frame_id)
        for item in load_manifest(manifest_path)
        for frame_id in item.frame_ids
    }


def _assemble_predictions(
    manifest_path: Path,
    splits: list[tuple[Path, Path]],
    model_path: Path,
    output_run: Path,
) -> dict[str, Any]:
    if output_run.exists():
        raise FileExistsError(f"refusing to overwrite release run: {output_run}")
    expected = _frame_keys(manifest_path)
    split_keys: list[set[tuple[str, int]]] = []
    clip_sources: dict[str, Path] = {}
    source_rows: list[dict[str, Any]] = []
    model_hash = sha256_file(model_path)

    for split_manifest, prediction_root in splits:
        keys = _frame_keys(split_manifest)
        if any(keys & previous for previous in split_keys):
            raise ValueError(f"split frame overlap: {split_manifest}")
        split_keys.append(keys)
        items = load_manifest(split_manifest)
        for item in items:
            if item.clip_id in clip_sources:
                raise ValueError(f"clip appears in multiple splits: {item.clip_id}")
            source = (prediction_root / item.clip_id).resolve()
            prediction, metadata = PredictionArtifact.load(source)
            if prediction.frame_ids.tolist() != item.frame_ids:
                raise ValueError(f"prediction frame mismatch: {source}")
            if prediction.vertices is None:
                raise ValueError(f"prediction has no vertices: {source}")
            if metadata.get("smplx_model_sha256") != model_hash:
                raise ValueError(f"SMPL-X model mismatch: {source}")
            if metadata.get("coordinate_convention") != (
                "opencv_x_right_y_down_z_forward"
            ):
                raise ValueError(f"coordinate convention mismatch: {source}")
            clip_sources[item.clip_id] = source
        source_rows.append(
            {
                "manifest_path": str(split_manifest.resolve()),
                "manifest_sha256": sha256_file(split_manifest),
                "prediction_root": str(prediction_root.resolve()),
                "clips": len(items),
                "frames": len(keys),
            }
        )

    union = set().union(*split_keys)
    if union != expected:
        raise ValueError(
            f"split union mismatch: missing={len(expected - union)} "
            f"extra={len(union - expected)}"
        )
    expected_clips = {item.clip_id for item in load_manifest(manifest_path)}
    if set(clip_sources) != expected_clips:
        raise ValueError("split clip union does not match full manifest")

    predictions = output_run / "predictions"
    predictions.mkdir(parents=True)
    incomplete = output_run / ".assemble_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    for clip_id in sorted(clip_sources):
        (predictions / clip_id).symlink_to(clip_sources[clip_id], target_is_directory=True)

    report = {
        "schema_version": "1.0",
        "status": "success",
        "method_name": "signal4d_m1_multiscale_gt_free_gate_v5_full1493",
        "assembly": "clip_disjoint_prediction_symlinks",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "smplx_model_sha256": model_hash,
        "coordinate_convention": "opencv_x_right_y_down_z_forward",
        "clips": len(expected_clips),
        "frames": len(expected),
        "sources": source_rows,
    }
    (output_run / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report


def _register_baseline_obj(
    manifest_path: Path,
    source_root: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite OBJ registry: {output_root}")
    manifest = load_manifest(manifest_path)
    model = np.load(model_path, allow_pickle=True)
    faces = np.asarray(model["f"], dtype=np.int64)
    rows: list[dict[str, Any]] = []

    output_root.mkdir(parents=True)
    incomplete = output_root / ".register_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    for item in manifest:
        source_clip = (source_root / item.clip_id).resolve()
        mesh_root = source_clip / "smplifyx" / "meshes"
        expected_names = {f"low_{frame_id}.obj" for frame_id in item.frame_ids}
        actual_names = {path.name for path in mesh_root.glob("*.obj")}
        if actual_names != expected_names:
            raise ValueError(
                f"DexAvatar OBJ coverage mismatch for {item.clip_id}: "
                f"missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}"
            )
        (output_root / item.clip_id).symlink_to(source_clip, target_is_directory=True)
        for frame_id in item.frame_ids:
            source = mesh_root / f"low_{frame_id}.obj"
            vertices, obj_faces = read_simple_obj(source)
            if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
                raise ValueError(f"invalid SMPL-X vertices: {source}")
            np.testing.assert_array_equal(obj_faces, faces)
            relative = Path(item.clip_id) / "smplifyx" / "meshes" / source.name
            rows.append(
                {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "obj_relpath": str(relative),
                    "obj_sha256": sha256_file(source),
                    "source_obj_path": str(source),
                }
            )

    report = {
        "schema_version": "1.0",
        "method_name": "DexAvatar_HaMeR_SignBPoser_SignHPoser",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", action="append", type=_parse_split, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--baseline-obj-root", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    args = parser.parse_args()

    prediction_report = _assemble_predictions(
        args.manifest, args.split, args.model_path, args.output_run
    )
    baseline_report = _register_baseline_obj(
        args.manifest, args.baseline_obj_root, args.model_path, args.baseline_output
    )
    print(
        json.dumps(
            {"prediction_release": prediction_report, "baseline_obj": baseline_report},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
