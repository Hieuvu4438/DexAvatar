from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..data.manifest import load_manifest
from ..io.predictions import PredictionArtifact
from ..utils.hashing import sha256_file

AUTHOR_REGIONS = (
    ("all", "tr_all_mm"),
    ("left hand", "tr_left_hand_mm"),
    ("right hand", "tr_right_hand_mm"),
    ("above pelvis upper body", "tr_upper_body_mm"),
    ("above pelvis minus head", "tr_upper_body_minus_head_mm"),
    ("above pelvis minus face", "tr_upper_body_minus_face_mm"),
)
WRIST_METRICS = {
    "left hand": "v2v_left_wrist_mm",
    "right hand": "v2v_right_wrist_mm",
}
PRIMARY_COLUMNS = (
    "tr_all_mm",
    "tr_upper_body_mm",
    "tr_left_hand_mm",
    "tr_right_hand_mm",
)
METHOD_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _load_author_module(source_path: str | Path) -> ModuleType:
    path = Path(source_path).resolve()
    spec = importlib.util.spec_from_file_location("signal4d_author_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import author evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("read_verts_and_faces", "transl_point_error", "point_error_common_center")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(f"author evaluator is missing required functions: {missing}")
    return module


def _read_sign_classes(path: str | Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 2 or tokens[1] not in {"0", "~0"}:
            raise ValueError(f"invalid author sign row {line_number}: {line!r}")
        if tokens[0] in classes:
            raise ValueError(f"duplicate author sign: {tokens[0]}")
        classes[tokens[0]] = tokens[1]
    return classes


def _load_author_assets(
    asset_root: str | Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    root = Path(asset_root)
    import pickle

    with (root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle, encoding="latin1")
    model = np.load(root / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    left = np.asarray(mano["left_hand"], dtype=np.int64)
    right = np.asarray(mano["right_hand"], dtype=np.int64)
    region_root = root / "sgnify_part_segm_above_pelvis_joint"
    regions = {
        "all": np.arange(10475, dtype=np.int64),
        "left hand": left,
        "right hand": right,
        "above pelvis upper body": np.load(region_root / "upper_body.npy"),
        "above pelvis minus head": np.load(region_root / "upper_body_minus_head.npy"),
        "above pelvis minus face": np.load(region_root / "upper_body_minus_face.npy"),
    }
    return regions, np.asarray(model["J_regressor"]), np.asarray(model["f"], dtype=np.int64)


def _expected_central_frame_ids(
    clip_id: str, gt_root: Path, segments: Mapping[str, list[int]]
) -> list[int]:
    if clip_id not in segments:
        raise ValueError(f"clip is absent from author segment file: {clip_id}")
    start, end = segments[clip_id]
    numbered = {
        int(path.stem): path
        for path in (gt_root / clip_id).glob("*.obj")
        if path.stem.isdigit()
    }
    gt_ids = sorted(frame for frame in numbered if start * 2 <= frame <= end * 2)
    if not gt_ids or any(frame % 2 for frame in gt_ids):
        raise ValueError(f"invalid author central GT frame IDs for {clip_id}")
    return [frame // 2 for frame in gt_ids]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _format_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _comparison_markdown(rows: list[dict[str, Any]], baseline: str, protocol: str) -> str:
    labels = {
        "tr_all_mm": "TR all",
        "tr_upper_body_mm": "TR upper body",
        "tr_left_hand_mm": "TR left hand",
        "tr_right_hand_mm": "TR right hand",
    }
    lines = [
        "# Author SGNify evaluator comparison",
        "",
        f"Protocol: `{protocol}`. Values are the author's vertex-micro means in mm; "
        "lower is better.",
        "",
        "| Method | " + " | ".join(labels[column] for column in PRIMARY_COLUMNS) + " |",
        "|---|" + "---:|" * len(PRIMARY_COLUMNS),
    ]
    for row in rows:
        values = " | ".join(_format_value(row.get(column)) for column in PRIMARY_COLUMNS)
        lines.append(f"| {row['method']} | {values} |")
    lines.extend(
        [
            "",
            f"Deltas in `comparison.csv` are method minus `{baseline}`; negative is better.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_author_sgnify(
    *,
    manifest_path: str,
    methods: Mapping[str, str],
    baseline: str,
    gt_root: str,
    author_source: str,
    author_asset_root: str,
    author_sign_file: str,
    author_segment_file: str,
    frame_policy: str,
    prediction_format: str,
    output_root: str,
) -> dict[str, Any]:
    if not methods or baseline not in methods:
        raise ValueError("methods must be non-empty and contain the declared baseline")
    invalid_labels = [label for label in methods if not METHOD_PATTERN.fullmatch(label)]
    if invalid_labels:
        raise ValueError(f"invalid method labels: {invalid_labels}")
    if frame_policy not in {"author-central", "manifest"}:
        raise ValueError(f"unsupported frame policy: {frame_policy}")
    if prediction_format not in {"safetensors", "dexavatar-obj"}:
        raise ValueError(f"unsupported prediction format: {prediction_format}")

    author = _load_author_module(author_source)
    manifest = load_manifest(manifest_path)
    gt_root_path = Path(gt_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    sign_classes = _read_sign_classes(author_sign_file)
    segments = json.loads(Path(author_segment_file).read_text(encoding="utf-8"))
    regions, joint_regressor, model_faces = _load_author_assets(author_asset_root)
    model_path = Path(author_asset_root) / "SMPLX_NEUTRAL.npz"
    model_hash = sha256_file(model_path)
    obj_hashes: dict[str, dict[str, str]] = {}
    if prediction_format == "dexavatar-obj":
        for label, root in methods.items():
            export_manifest_path = Path(root) / "export_manifest.json"
            export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
            if export_manifest.get("manifest_sha256") != sha256_file(manifest_path):
                raise ValueError(f"OBJ export manifest mismatch for {label}")
            if export_manifest.get("smplx_model_sha256") != model_hash:
                raise ValueError(f"OBJ export SMPL-X model mismatch for {label}")
            if export_manifest.get("format") != "dexavatar_trimesh_obj":
                raise ValueError(f"OBJ export format mismatch for {label}")
            obj_hashes[label] = {
                row["obj_relpath"]: row["obj_sha256"] for row in export_manifest["files"]
            }

    vectors: dict[str, dict[str, list[np.ndarray]]] = {
        label: {metric: [] for _, metric in AUTHOR_REGIONS}
        | {metric: [] for metric in WRIST_METRICS.values()}
        for label in methods
    }
    frame_rows: dict[str, list[dict[str, Any]]] = {label: [] for label in methods}
    clip_rows: dict[str, list[dict[str, Any]]] = {label: [] for label in methods}
    topology_checked = False

    for item in manifest:
        if item.clip_id not in sign_classes:
            raise ValueError(f"manifest clip is absent from author sign file: {item.clip_id}")
        if frame_policy == "author-central":
            expected = _expected_central_frame_ids(item.clip_id, gt_root_path, segments)
            if item.frame_ids != expected:
                raise ValueError(
                    f"author central frame mismatch for {item.clip_id}: "
                    f"manifest={item.frame_ids} author={expected}"
                )

        predictions: dict[str, np.ndarray] = {}
        for label, root in methods.items():
            if prediction_format == "safetensors":
                prediction, metadata = PredictionArtifact.load(Path(root) / item.clip_id)
                if prediction.frame_ids.tolist() != item.frame_ids:
                    raise ValueError(f"prediction frame mismatch for {label}/{item.clip_id}")
                if prediction.vertices is None:
                    raise ValueError(f"prediction has no vertices for {label}/{item.clip_id}")
                if metadata.get("smplx_model_sha256") != model_hash:
                    raise ValueError(f"SMPL-X model mismatch for {label}/{item.clip_id}")
                if metadata.get("coordinate_convention") != "opencv_x_right_y_down_z_forward":
                    raise ValueError(f"coordinate mismatch for {label}/{item.clip_id}")
                values = prediction.vertices.detach().cpu().numpy()
                if values.shape != (len(item.frame_ids), 10475, 3):
                    raise ValueError(
                        f"vertex shape mismatch for {label}/{item.clip_id}: {values.shape}"
                    )
                predictions[label] = values
            else:
                mesh_root = Path(root) / item.clip_id / "smplifyx" / "meshes"
                expected_names = {f"low_{frame_id}.obj" for frame_id in item.frame_ids}
                actual_names = {path.name for path in mesh_root.glob("*.obj")}
                if actual_names != expected_names:
                    raise ValueError(
                        f"strict OBJ coverage mismatch for {label}/{item.clip_id}: "
                        f"missing={sorted(expected_names - actual_names)} "
                        f"extra={sorted(actual_names - expected_names)}"
                    )
                clip_vertices = []
                for frame_id in item.frame_ids:
                    obj_path = mesh_root / f"low_{frame_id}.obj"
                    relative = str(obj_path.relative_to(Path(root)))
                    if obj_hashes[label].get(relative) != sha256_file(obj_path):
                        raise ValueError(f"strict OBJ hash mismatch: {label}/{relative}")
                    vertex_list, face_list = author.read_verts_and_faces(obj_path, "slrt")
                    vertices = np.asarray(vertex_list)
                    faces = np.asarray(face_list, dtype=np.int64).reshape(-1, 3)
                    if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
                        raise ValueError(f"strict OBJ vertex failure: {obj_path}")
                    np.testing.assert_array_equal(faces, model_faces)
                    clip_vertices.append(vertices)
                predictions[label] = np.stack(clip_vertices)

        clip_frame_values: dict[str, dict[str, list[float]]] = {
            label: {metric: [] for _, metric in AUTHOR_REGIONS}
            | {metric: [] for metric in WRIST_METRICS.values()}
            for label in methods
        }
        one_handed = sign_classes[item.clip_id] == "0"
        for frame_index, frame_id in enumerate(item.frame_ids):
            gt_path = gt_root_path / item.clip_id / f"{frame_id * 2:05d}.obj"
            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)
            target_list, faces_list = author.read_verts_and_faces(gt_path, "soma")
            target = np.asarray(target_list)
            faces = np.asarray(faces_list, dtype=np.int64).reshape(-1, 3)
            if target.shape != (10475, 3) or not np.isfinite(target).all():
                raise ValueError(f"invalid GT vertices: {gt_path}")
            if not topology_checked:
                np.testing.assert_array_equal(faces, model_faces)
                topology_checked = True
            elif faces.shape != model_faces.shape or not np.array_equal(faces, model_faces):
                raise ValueError(f"GT topology mismatch: {gt_path}")
            target_joints = joint_regressor.dot(target)

            for label, prediction in predictions.items():
                source = prediction[frame_index]
                if not np.isfinite(source).all():
                    raise ValueError(f"non-finite prediction: {label}/{item.clip_id}/{frame_id}")
                row: dict[str, Any] = {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "method": label,
                    "one_handed_class_0": one_handed,
                }
                for region_name, metric in AUTHOR_REGIONS:
                    if region_name == "left hand" and one_handed:
                        row[metric] = None
                        continue
                    ids = regions[region_name]
                    if region_name != "left hand" and one_handed:
                        ids = np.setdiff1d(ids, regions["left hand"])
                    error = np.asarray(author.transl_point_error(source[ids], target[ids]))
                    vectors[label][metric].append(error)
                    value = float(error.mean() * 1000.0)
                    row[metric] = value
                    clip_frame_values[label][metric].append(value)

                    wrist_metric = WRIST_METRICS.get(region_name)
                    if wrist_metric is not None:
                        wrist_index = 20 if region_name == "left hand" else 21
                        wrist_error = np.asarray(
                            author.point_error_common_center(
                                source[ids], target[ids], target_joints[wrist_index].reshape(-1, 3)
                            )
                        )
                        vectors[label][wrist_metric].append(wrist_error)
                        wrist_value = float(wrist_error.mean() * 1000.0)
                        row[wrist_metric] = wrist_value
                        clip_frame_values[label][wrist_metric].append(wrist_value)
                frame_rows[label].append(row)

        for label in methods:
            clip_row: dict[str, Any] = {
                "clip_id": item.clip_id,
                "frames": len(item.frame_ids),
                "method": label,
                "one_handed_class_0": one_handed,
            }
            for metric, values in clip_frame_values[label].items():
                clip_row[metric] = _mean_or_none(values)
            clip_rows[label].append(clip_row)

    summaries: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for label, root in methods.items():
        summary: dict[str, Any] = {
            "method": label,
            "prediction_root": str(Path(root)),
            "clips": len(manifest),
            "frames": sum(len(item.frame_ids) for item in manifest),
            "coverage": 1.0,
            "aggregation": "author_vertex_micro",
            "prediction_format": prediction_format,
        }
        for metric, values in vectors[label].items():
            summary[metric] = float(np.concatenate(values).mean() * 1000.0) if values else None
            per_clip = [row[metric] for row in clip_rows[label] if row[metric] is not None]
            summary[f"clip_macro_{metric}"] = _mean_or_none(per_clip)
        method_output = output / "methods" / label
        method_output.mkdir(parents=True, exist_ok=True)
        _write_csv(frame_rows[label], method_output / "per_frame.csv")
        _write_csv(clip_rows[label], method_output / "per_clip.csv")
        (method_output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries[label] = summary

    baseline_summary = summaries[baseline]
    for summary in summaries.values():
        row = dict(summary)
        for _, metric in AUTHOR_REGIONS:
            value = summary[metric]
            baseline_value = baseline_summary[metric]
            row[f"delta_{metric}"] = (
                None if value is None or baseline_value is None else value - baseline_value
            )
        comparison_rows.append(row)

    _write_csv(comparison_rows, output / "comparison.csv")
    comparison = {
        "schema_version": "1.0",
        "baseline": baseline,
        "frame_policy": frame_policy,
        "prediction_format": prediction_format,
        "manifest_sha256": sha256_file(manifest_path),
        "author_source_sha256": sha256_file(author_source),
        "author_asset_hashes": {
            "model": model_hash,
            "mano_vertex_ids": sha256_file(Path(author_asset_root) / "MANO_SMPLX_vertex_ids.pkl"),
            "upper_body": sha256_file(
                Path(author_asset_root)
                / "sgnify_part_segm_above_pelvis_joint"
                / "upper_body.npy"
            ),
            "upper_body_minus_head": sha256_file(
                Path(author_asset_root)
                / "sgnify_part_segm_above_pelvis_joint"
                / "upper_body_minus_head.npy"
            ),
            "upper_body_minus_face": sha256_file(
                Path(author_asset_root)
                / "sgnify_part_segm_above_pelvis_joint"
                / "upper_body_minus_face.npy"
            ),
        },
        "author_sign_file_sha256": sha256_file(author_sign_file),
        "author_segment_file_sha256": sha256_file(author_segment_file),
        "metrics": (
            "functions imported from author evaluator; exact author regions and class-0 rule"
        ),
        "obj_export_manifest_sha256": {
            label: sha256_file(Path(root) / "export_manifest.json")
            for label, root in methods.items()
        }
        if prediction_format == "dexavatar-obj"
        else None,
        "rows": comparison_rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "comparison.md").write_text(
        _comparison_markdown(comparison_rows, baseline, frame_policy), encoding="utf-8"
    )
    return comparison
