"""Strict exact-frame SGNify evaluator for root- and wrist-aligned endpoints."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dcg_sign4d.utils.hashing import file_sha256

from .hand_metrics import HandPlacementMetrics
from .temporal import temporal_motion_metrics


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                tokens = line.split()
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    if not vertices or not faces:
        raise ValueError(f"OBJ is empty/incomplete: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _sign_classes(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, category = line.split()
            result[name] = category
    return result


def _load_assets(root: str | Path, *, trusted: bool) -> tuple[HandPlacementMetrics, np.ndarray]:
    if not trusted:
        raise PermissionError("author MANO pickle requires --trusted-author-assets")
    asset_root = Path(root)
    with (asset_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle, encoding="latin1")
    model = np.load(asset_root / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    region_root = asset_root / "sgnify_part_segm_above_pelvis_joint"
    body_ids = np.load(region_root / "upper_body.npy")
    metrics = HandPlacementMetrics(
        np.asarray(model["J_regressor"]),
        np.asarray(mano["left_hand"], dtype=np.int64),
        np.asarray(mano["right_hand"], dtype=np.int64),
        np.asarray(body_ids, dtype=np.int64),
    )
    return metrics, np.asarray(model["f"], dtype=np.int64)


def evaluate_sgnify_obj(
    *,
    manifest_path: str | Path,
    prediction_root: str | Path,
    gt_root: str | Path,
    author_asset_root: str | Path,
    author_sign_file: str | Path,
    output_root: str | Path,
    trusted_author_assets: bool,
) -> dict[str, Any]:
    manifest = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not manifest:
        raise ValueError("empty SGNify manifest")
    metrics, expected_faces = _load_assets(author_asset_root, trusted=trusted_author_assets)
    classes = _sign_classes(author_sign_file)
    prediction_root = Path(prediction_root)
    gt_root = Path(gt_root)
    rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    seen_clips: set[str] = set()
    for item in manifest:
        clip_id = item["clip_id"]
        if clip_id in seen_clips:
            raise ValueError(f"duplicate clip: {clip_id}")
        seen_clips.add(clip_id)
        frame_ids = item["frame_ids"]
        if len(frame_ids) != len(set(frame_ids)) or frame_ids != sorted(frame_ids):
            raise ValueError(f"invalid frame order: {clip_id}")
        mesh_root = prediction_root / clip_id / "smplifyx" / "meshes"
        actual = {path.name for path in mesh_root.glob("low_*.obj")}
        expected = {f"low_{frame_id}.obj" for frame_id in frame_ids}
        if actual != expected:
            raise ValueError(
                f"prediction coverage mismatch {clip_id}: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        clip_metrics: dict[str, list[float]] = {}
        temporal_source: dict[str, list[np.ndarray]] = {
            "left_hand": [],
            "right_hand": [],
            "body": [],
        }
        temporal_target: dict[str, list[np.ndarray]] = {
            "left_hand": [],
            "right_hand": [],
            "body": [],
        }
        for frame_id in frame_ids:
            prediction_path = mesh_root / f"low_{frame_id}.obj"
            gt_path = gt_root / clip_id / f"{frame_id * 2:05d}.obj"
            prediction, prediction_faces = read_obj(prediction_path)
            target, target_faces = read_obj(gt_path)
            if not np.array_equal(prediction_faces, expected_faces):
                raise ValueError(f"prediction topology mismatch: {prediction_path}")
            if not np.array_equal(target_faces, expected_faces):
                raise ValueError(f"GT topology mismatch: {gt_path}")
            result = metrics.evaluate_frame(prediction, target)
            prediction_joints = metrics.joint_regressor @ prediction
            target_joints = metrics.joint_regressor @ target
            prediction_pelvis = prediction_joints[list(metrics.pelvis_indices)].mean(
                0, keepdims=True
            )
            target_pelvis = target_joints[list(metrics.pelvis_indices)].mean(0, keepdims=True)
            for region, indices in (
                ("left_hand", metrics.left_hand_ids),
                ("right_hand", metrics.right_hand_ids),
                ("body", metrics.body_ids),
            ):
                temporal_source[region].append(prediction[indices] - prediction_pelvis)
                temporal_target[region].append(target[indices] - target_pelvis)
            row: dict[str, Any] = {"clip_id": clip_id, "frame_id": frame_id, **result}
            if classes.get(clip_id) == "0":
                for key in tuple(result):
                    if "left_hand" in key:
                        row[key] = None
            for prefix in ("root_aligned", "wrist_aligned", "legacy_region_tr"):
                side_values = [
                    row.get(f"{prefix}_{side}_hand_pve_mm") for side in ("left", "right")
                ]
                available = [value for value in side_values if value is not None]
                row[f"{prefix}_hand_pve_mm"] = float(np.mean(available))
            rows.append(row)
            for name, value in row.items():
                if name not in {"clip_id", "frame_id"} and value is not None:
                    clip_metrics.setdefault(name, []).append(value)
        clip_row: dict[str, Any] = {
            "clip_id": clip_id,
            "signer_id": item.get("signer_id", "unknown"),
            "sign_type": item.get("sign_type", "unknown"),
            **{name: float(np.mean(values)) for name, values in clip_metrics.items()},
        }
        fps = float(item.get("fps_effective", item.get("fps", 0)))
        if fps <= 0:
            raise ValueError(f"manifest lacks positive effective fps: {clip_id}")
        temporal_by_region: dict[str, dict[str, float]] = {}
        for region in temporal_source:
            if region == "left_hand" and classes.get(clip_id) == "0":
                continue
            temporal_by_region[region] = temporal_motion_metrics(
                np.stack(temporal_source[region]),
                np.stack(temporal_target[region]),
                fps=fps,
            )
            for name, value in temporal_by_region[region].items():
                clip_row[f"temporal_{region}_{name}"] = value
        hand_regions = [
            temporal_by_region[region]
            for region in ("left_hand", "right_hand")
            if region in temporal_by_region
        ]
        for name in hand_regions[0]:
            clip_row[f"temporal_hand_{name}"] = float(
                np.mean([region_metrics[name] for region_metrics in hand_regions])
            )
        clip_rows.append(clip_row)

    metric_names = sorted(
        set().union(*(row.keys() for row in clip_rows)) - {"clip_id", "signer_id", "sign_type"}
    )
    signer_ids = [str(row["signer_id"]) for row in clip_rows]
    signer_ready = all(value and value.lower() != "unknown" for value in signer_ids)
    signer_ready &= len(set(signer_ids)) >= 2
    summary: dict[str, Any] = {
        "clips": len(clip_rows),
        "frames": len(rows),
        "coverage": 1.0,
        "aggregation": "frame_to_clip_then_equal_weight_clip_macro",
        "primary_endpoint": "root_aligned_hand_pve",
        "evaluator_schema_version": "dcg_sgnify_v3_geometry_temporal",
        "temporal_alignment": "per_frame_pelvis_root_before_physical_time_differences",
        "signer_cluster_bootstrap_status": (
            "READY" if signer_ready else "BLOCKED_SIGNER_IDS_UNAVAILABLE"
        ),
        "manifest_sha256": file_sha256(manifest_path),
        "prediction_root": str(prediction_root),
    }
    for name in metric_names:
        values = [row[name] for row in clip_rows if name in row]
        summary[name] = float(np.mean(values))
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"immutable evaluation output exists: {output}")
    output.mkdir(parents=True)
    for name, values in (("per_frame.csv", rows), ("per_clip.csv", clip_rows)):
        columns = sorted(set().union(*(row.keys() for row in values)))
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output / "EVALUATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary
