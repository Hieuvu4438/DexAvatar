"""Evaluate frozen CUSP selection artifacts against development targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cusp_sl.evaluate_development import clustered_delta_interval
from cusp_sl.geometry import axis_angle_to_matrix, geodesic_distance
from phase2_refiner.data.cache_schema import load_cache_clip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation_group_metrics(
    base: torch.Tensor,
    selected: torch.Tensor,
    candidates: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    refine_mask: torch.Tensor,
    candidate_valid: torch.Tensor | None = None,
) -> dict[str, float | int]:
    valid = valid.bool() & refine_mask[None].bool()
    joint_index = torch.arange(base.shape[1], device=base.device)[None]
    masks = {
        "overall": valid,
        "body": valid & (joint_index < 21),
        "hands": valid & (joint_index >= 21),
        "left": valid & ((joint_index >= 21) & (joint_index < 36)),
        "right": valid & (joint_index >= 36),
    }
    base_error = torch.rad2deg(geodesic_distance(base, target))
    selected_error = torch.rad2deg(geodesic_distance(selected, target))
    candidate_error = torch.rad2deg(
        geodesic_distance(candidates, target[None])
    )
    if candidate_valid is None:
        candidate_valid = torch.ones(
            candidates.shape[0], dtype=torch.bool, device=candidates.device
        )
    if candidate_valid.shape != (candidates.shape[0],) or not candidate_valid.any():
        raise ValueError("Candidate-valid mask is invalid or empty")
    result: dict[str, float | int] = {}
    for group, mask in masks.items():
        tokens = int(mask.sum())
        if tokens == 0:
            raise ValueError(f"Development clip has no valid {group} tokens")
        per_candidate = (candidate_error * mask[None]).sum((1, 2)) / tokens
        per_candidate = torch.where(
            candidate_valid, per_candidate, torch.full_like(per_candidate, float("inf"))
        )
        result[f"{group}_tokens"] = tokens
        result[f"base_{group}_degrees"] = float(base_error[mask].mean())
        result[f"selected_{group}_degrees"] = float(selected_error[mask].mean())
        result[f"oracle_{group}_degrees"] = float(per_candidate.min())
    return result


def weighted(records: list[dict], value: str, weights: str) -> float:
    return float(np.average(
        [record[value] for record in records],
        weights=[record[weights] for record in records],
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction-input-manifest",
        type=Path,
        help="Targetless manifest consumed by inference; defaults to --manifest",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    args.output.mkdir(parents=True)
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    inference_manifest_path = args.predictions / "manifest.json"
    inference_manifest = json.loads(
        inference_manifest_path.read_text(encoding="utf-8")
    )
    if inference_manifest.get("protocol_role") != "development_validation":
        raise ValueError("Predictions are not labeled development_validation")
    prediction_input_manifest = args.prediction_input_manifest or args.manifest
    if inference_manifest.get("input_manifest_sha256") != sha256(
        prediction_input_manifest
    ):
        raise ValueError("Prediction/input-manifest hash mismatch")
    input_source = json.loads(
        prediction_input_manifest.read_text(encoding="utf-8")
    )
    if args.prediction_input_manifest is not None:
        if input_source.get("role") != "development_targetless_inference":
            raise ValueError("Prediction input is not targetless development data")
        input_clips = {}
        for entry in input_source["clips"]:
            relative = entry["cache"] if isinstance(entry, dict) else entry
            path = Path(relative)
            if not path.is_absolute():
                path = prediction_input_manifest.parent / path
            clip = load_cache_clip(path)
            if clip.target_axis_angle is not None or clip.target_joint_positions is not None:
                raise ValueError(f"Prediction-input cache retains targets: {path}")
            input_clips[clip.clip_id] = clip
    else:
        input_clips = None
    declared = {
        str(item["clip_id"]): str(item["prediction_sha256"])
        for item in inference_manifest["summaries"]
    }
    records = []
    for entry in source["clips"]:
        relative = entry["cache"] if isinstance(entry, dict) else entry
        cache_path = Path(relative)
        if not cache_path.is_absolute():
            cache_path = args.manifest.parent / cache_path
        clip = load_cache_clip(cache_path)
        if clip.target_axis_angle is None or clip.target_rotation_valid is None:
            raise ValueError(f"Development clip lacks rotation target: {cache_path}")
        metadata = json.loads(clip.metadata_json)
        source_group = str(metadata.get("source_group", ""))
        if not source_group:
            raise ValueError(f"Development clip lacks source_group: {cache_path}")
        if input_clips is not None:
            input_clip = input_clips.get(clip.clip_id)
            if input_clip is None or not np.array_equal(
                input_clip.frame_names.astype(str), clip.frame_names.astype(str)
            ):
                raise ValueError(f"Targetless/target clip mismatch: {clip.clip_id}")
        prediction_path = args.predictions / "clips" / f"{clip.clip_id}.npz"
        if clip.clip_id not in declared:
            raise ValueError(f"Inference manifest omits {clip.clip_id}")
        if sha256(prediction_path) != declared[clip.clip_id]:
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as prediction:
            if str(prediction["clip_id"].item()) != clip.clip_id:
                raise ValueError(f"Prediction clip ID mismatch: {prediction_path}")
            if not np.array_equal(
                prediction["frame_names"].astype(str), clip.frame_names.astype(str)
            ):
                raise ValueError(f"Prediction frame order mismatch: {prediction_path}")
            selected = torch.from_numpy(
                prediction["selected_rotation"].astype(np.float32)
            )
            candidates = torch.from_numpy(
                prediction["candidate_rotation"].astype(np.float32)
            )
            candidate_valid = torch.from_numpy(
                prediction["candidate_valid"].astype(bool)
            )
            selected_index = int(prediction["selected_index"])
        base = axis_angle_to_matrix(torch.from_numpy(clip.init_axis_angle).float())
        target = axis_angle_to_matrix(
            torch.from_numpy(clip.target_axis_angle).float()
        )
        metrics = rotation_group_metrics(
            base,
            selected,
            candidates,
            target,
            torch.from_numpy(clip.target_rotation_valid),
            torch.from_numpy(clip.refine_mask),
            candidate_valid,
        )
        record = {
            "clip_id": clip.clip_id,
            "source_group": source_group,
            "frames": len(clip.frame_names),
            "selected_index": selected_index,
            **metrics,
        }
        record["tokens"] = record["overall_tokens"]
        record["base_degrees"] = record["base_overall_degrees"]
        record["selected_degrees"] = record["selected_overall_degrees"]
        record["oracle_degrees"] = record["oracle_overall_degrees"]
        record["selection_regret_degrees"] = (
            record["selected_degrees"] - record["oracle_degrees"]
        )
        records.append(record)

    if set(declared) != {str(record["clip_id"]) for record in records}:
        raise ValueError("Inference/source clip sets differ")
    if input_clips is not None and set(input_clips) != set(declared):
        raise ValueError("Targetless/target/prediction clip sets differ")
    csv_path = args.output / "per_clip.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary: dict[str, object] = {
        "role": "development_selection_evaluation",
        "variant": inference_manifest["variant"],
        "clips": len(records),
        "frames": int(sum(record["frames"] for record in records)),
        "manifest_sha256": sha256(args.manifest),
        "prediction_input_manifest_sha256": sha256(prediction_input_manifest),
        "inference_manifest_sha256": sha256(inference_manifest_path),
        "selected_base_fraction": float(np.mean([
            record["selected_index"] == 0 for record in records
        ])),
    }
    for group in ("overall", "body", "hands", "left", "right"):
        weights = f"{group}_tokens"
        summary[f"base_{group}_degrees"] = weighted(
            records, f"base_{group}_degrees", weights
        )
        summary[f"selected_{group}_degrees"] = weighted(
            records, f"selected_{group}_degrees", weights
        )
        summary[f"oracle_{group}_degrees"] = weighted(
            records, f"oracle_{group}_degrees", weights
        )
    summary["selection_regret_degrees"] = weighted(
        records, "selection_regret_degrees", "tokens"
    )
    summary["clustered_selected_minus_base"] = clustered_delta_interval(
        records,
        "selected_degrees",
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary["clustered_oracle_minus_base"] = clustered_delta_interval(
        records,
        "oracle_degrees",
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )
    summary["clustered_selected_minus_base_by_group"] = {
        group: clustered_delta_interval(
            records,
            f"selected_{group}_degrees",
            replicates=args.bootstrap_replicates,
            seed=args.seed + 100 + index,
            weight_key=f"{group}_tokens",
            base_key=f"base_{group}_degrees",
        )
        for index, group in enumerate(("body", "hands", "left", "right"))
    }
    summary["clustered_oracle_minus_base_by_group"] = {
        group: clustered_delta_interval(
            records,
            f"oracle_{group}_degrees",
            replicates=args.bootstrap_replicates,
            seed=args.seed + 200 + index,
            weight_key=f"{group}_tokens",
            base_key=f"base_{group}_degrees",
        )
        for index, group in enumerate(("body", "hands", "left", "right"))
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
