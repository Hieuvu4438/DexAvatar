#!/usr/bin/env python3
"""Evaluate native NLF SMPL-X meshes with the SGNify author TR-V2V rule.

This is a diagnostic replacement test, not a proposed final SIGNAL-4D method.
It follows the author's translation-only, region-wise centering and vertex-micro
aggregation. NLF camera coordinates are converted by a fixed 180-degree
rotation around X: ``(x, y, z) -> (x, -y, -z)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from signal4d_v7_nlf_fusion.extract_nlf_observations import load_manifest_frames


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    result = np.asarray(vertices, dtype=np.float64)
    if result.shape != (10475, 3):
        raise ValueError(f"{path}: expected (10475, 3), got {result.shape}")
    return result


def translation_relative_errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    return np.linalg.norm(pred_centered - target_centered, axis=-1)


def sign_classes(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields:
                result[fields[0]] = fields[1]
    return result


def rows_from_index(path: Path) -> Dict[tuple[str, int], Dict[str, object]]:
    result: Dict[tuple[str, int], Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[(str(row["clip_id"]), int(row["frame_id"]))] = row
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--observation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("data/evaluation_from_author/data/data"),
    )
    parser.add_argument(
        "--sign-file",
        type=Path,
        default=Path("data/evaluation_from_author/signs.txt"),
    )
    parser.add_argument("--require-frames", type=int, default=1493)
    return parser.parse_args()


def concatenate(values: Iterable[np.ndarray]) -> np.ndarray:
    arrays = list(values)
    if not arrays:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(arrays)


def main() -> None:
    args = parse_args()
    frames = load_manifest_frames(args.manifest, args.data_root.resolve())
    if args.require_frames and len(frames) != args.require_frames:
        raise ValueError(f"expected {args.require_frames} manifest frames, got {len(frames)}")
    index = rows_from_index(args.observation_root / "index.jsonl")
    classes = sign_classes(args.sign_file)

    with (args.asset_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle)
    left_hand = np.asarray(mano["left_hand"], dtype=np.int64)
    right_hand = np.asarray(mano["right_hand"], dtype=np.int64)
    upper_body = np.load(
        args.asset_root
        / "sgnify_part_segm_above_pelvis_joint"
        / "upper_body.npy"
    ).astype(np.int64)
    upper_body_minus_face = np.load(
        args.asset_root
        / "sgnify_part_segm_above_pelvis_joint"
        / "upper_body_minus_face.npy"
    ).astype(np.int64)
    all_vertices = np.arange(10475, dtype=np.int64)
    regions = {
        "all": all_vertices,
        "upper_body": upper_body,
        "upper_body_minus_face": upper_body_minus_face,
        "left_hand": left_hand,
        "right_hand": right_hand,
    }

    per_region: Dict[str, List[np.ndarray]] = {name: [] for name in regions}
    per_frame: List[Dict[str, object]] = []
    rotation_x_pi = np.asarray([1.0, -1.0, -1.0], dtype=np.float64)

    for frame in frames:
        key = (frame["clip_id"], frame["frame_id"])
        record = index.get(key)
        if record is None or record.get("status") not in {"ok", "existing"}:
            raise RuntimeError(f"missing successful NLF observation for {key}: {record}")
        observation_path = args.observation_root / str(record["output_relpath"])
        prediction = (
            np.load(observation_path)["vertices3d"].astype(np.float64)
            / 1000.0
            * rotation_x_pi
        )
        gt_path = args.data_root / "smplx_gt" / frame["clip_id"] / f"{frame['frame_id'] * 2:05d}.obj"
        target = load_obj_vertices(gt_path)
        one_handed = classes[frame["clip_id"]] == "0"

        row: Dict[str, object] = {
            "split": frame["split"],
            "sign": frame["clip_id"],
            "frame": frame["frame_id"],
            "one_handed": one_handed,
            "mean_joint_uncertainty_mm": record.get("mean_joint_uncertainty_mm"),
            "mean_vertex_uncertainty_mm": record.get("mean_vertex_uncertainty_mm"),
        }
        for name, base_indices in regions.items():
            if name == "left_hand" and one_handed:
                row[f"tr_{name}_mm"] = ""
                continue
            indices = base_indices
            if one_handed and name != "left_hand":
                indices = np.setdiff1d(indices, left_hand, assume_unique=False)
            errors = translation_relative_errors(prediction[indices], target[indices])
            per_region[name].append(errors)
            row[f"tr_{name}_mm"] = float(errors.mean() * 1000.0)
        per_frame.append(row)

    summary = {
        "method": "NLF_v0.3.2_direct",
        "protocol": "author_translation_relative_vertex_micro",
        "coordinate_conversion": "Rx(pi): [x,-y,-z]",
        "frames": len(per_frame),
        "clips": len({row["sign"] for row in per_frame}),
        "coverage": len(per_frame) / len(frames),
    }
    for name, values in per_region.items():
        errors = concatenate(values)
        summary[f"tr_{name}_mm"] = float(errors.mean() * 1000.0)
        frame_values = [
            float(row[f"tr_{name}_mm"])
            for row in per_frame
            if row[f"tr_{name}_mm"] != ""
        ]
        summary[f"frame_macro_tr_{name}_mm"] = float(np.mean(frame_values))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
