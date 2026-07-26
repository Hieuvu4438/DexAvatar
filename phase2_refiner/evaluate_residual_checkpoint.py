"""Evaluate a Phase-2 checkpoint on source-disjoint real-residual caches.

This evaluator reports equal-region SO(3) error against the cached clean target,
the frozen observation-difficulty hard subset, safety fallback, and clip-level
bootstrap intervals.  It also keeps proxy and exact-locked-initializer evidence
separate when constructing the machine-readable G4 payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase2_refiner.config import load_config
from phase2_refiner.data.dataset import (
    OBSERVATION_FEATURES,
    U0_RELIABILITY,
    SequenceCacheDataset,
    collate_sequences,
)
from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.infer import _apply_safety_fallback, _load_model
from phase2_refiner.provenance import sha256_file


REGIONS = {
    "ubody": slice(0, 21),
    "lhand": slice(21, 36),
    "rhand": slice(36, 51),
}
HARD_SUBSET_CONTRACT = {
    "version": "observation_difficulty_v1",
    "definition": (
        "mean U0 reliability <0.35 OR missing fraction >0.25 OR "
        "truncation >0.25 OR duplicate/disagreement flag"
    ),
}


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _bootstrap_clip_delta(
    rows: list[dict], region: str, samples: int, seed: int
) -> dict:
    deltas = np.asarray(
        [row[region]["prediction"] - row[region]["baseline"] for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    distribution = deltas[indices].mean(axis=1)
    return {
        "clips": len(deltas),
        "mean_delta_degrees": float(deltas.mean()),
        "ci95_low_degrees": float(np.quantile(distribution, 0.025)),
        "ci95_high_degrees": float(np.quantile(distribution, 0.975)),
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model = _load_model(config, args.checkpoint, device)
    dataset = SequenceCacheDataset(
        str(args.manifest.resolve()),
        max_frames=model.max_frames,
        training=False,
        input_dim=model.token_embedding.input_projection.in_features,
        reprojection_residual_scale=float(
            config.get("data", {}).get("reprojection_residual_scale", 10.0)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with args.real_residual_audit.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    validation = audit.get("validation") or {}
    if Path(validation.get("manifest", "")).resolve() != args.manifest.resolve():
        raise ValueError("Manifest differs from residual-audit validation split")

    values = {
        region: {"baseline": [], "prediction": [], "hard_baseline": [], "hard_prediction": []}
        for region in REGIONS
    }
    per_clip: list[dict] = []
    frames = fallback_group_frames = total_group_frames = 0
    for batch in loader:
        batch = _to_device(batch, device)
        prediction = model(
            batch["features"],
            batch["initial_matrix"],
            batch["frame_valid"],
            batch["refine_mask"],
            batch["initial_joint_position"],
        )
        output = prediction["matrix"]
        fallback = torch.zeros(
            output.shape[0], output.shape[1], 3, dtype=torch.bool, device=device
        )
        body_limit = float(torch.rad2deg(model.max_angles[:21].max()).cpu())
        hand_limit = float(torch.rad2deg(model.max_angles[21:].max()).cpu())
        for batch_index in range(output.shape[0]):
            output[batch_index], fallback[batch_index] = _apply_safety_fallback(
                output[batch_index],
                batch["initial_matrix"][batch_index],
                body_limit,
                hand_limit,
                prediction.get("log_variance", None)[batch_index]
                if "log_variance" in prediction
                else None,
            )
        baseline_error = torch.rad2deg(
            geodesic_distance(batch["initial_matrix"], batch["target_matrix"])
        )
        prediction_error = torch.rad2deg(
            geodesic_distance(output, batch["target_matrix"])
        )
        observations = batch["features"][..., OBSERVATION_FEATURES]
        reliability = batch["features"][..., U0_RELIABILITY]
        frames += int(batch["frame_valid"].sum())
        fallback_group_frames += int(
            (fallback & batch["frame_valid"][..., None]).sum()
        )
        total_group_frames += int(batch["frame_valid"].sum()) * 3

        for batch_index, clip_id in enumerate(batch["clip_id"]):
            row: dict = {"clip_id": clip_id}
            frame_mask = batch["frame_valid"][batch_index]
            for region, indices in REGIONS.items():
                valid = (
                    frame_mask[:, None]
                    & batch["target_rotation_valid"][batch_index, :, indices]
                    & batch["refine_mask"][batch_index, None, indices]
                )
                baseline_region = baseline_error[batch_index, :, indices]
                prediction_region = prediction_error[batch_index, :, indices]
                baseline_mean = float(baseline_region[valid].mean())
                prediction_mean = float(prediction_region[valid].mean())
                values[region]["baseline"].append(baseline_mean)
                values[region]["prediction"].append(prediction_mean)

                group_observation = observations[batch_index, :, indices]
                group_reliability = reliability[batch_index, :, indices]
                hard_frames = (
                    (group_reliability.mean(dim=-1) < 0.35)
                    | (group_observation[..., 2].mean(dim=-1) > 0.25)
                    | (group_observation[..., 4].amax(dim=-1) > 0.25)
                    | (group_observation[..., 6:8].amax(dim=(-1, -2)) > 0)
                ) & frame_mask
                hard_valid = valid & hard_frames[:, None]
                hard_count = int(hard_valid.sum())
                if hard_count:
                    values[region]["hard_baseline"].extend(
                        baseline_region[hard_valid].cpu().tolist()
                    )
                    values[region]["hard_prediction"].extend(
                        prediction_region[hard_valid].cpu().tolist()
                    )
                row[region] = {
                    "baseline": baseline_mean,
                    "prediction": prediction_mean,
                    "hard_joint_frames": hard_count,
                }
            per_clip.append(row)

    baseline = {
        region: float(np.mean(region_values["baseline"]))
        for region, region_values in values.items()
    }
    prediction = {
        region: float(np.mean(region_values["prediction"]))
        for region, region_values in values.items()
    }
    gains = {
        region: (baseline[region] - prediction[region]) / baseline[region]
        for region in REGIONS
    }
    hard_gains = {}
    hard_counts = {}
    for region, region_values in values.items():
        hard_counts[region] = len(region_values["hard_baseline"])
        hard_gains[region] = (
            (
                float(np.mean(region_values["hard_baseline"]))
                - float(np.mean(region_values["hard_prediction"]))
            )
            / float(np.mean(region_values["hard_baseline"]))
            if region_values["hard_baseline"]
            else None
        )
    complete_hard = all(value is not None for value in hard_gains.values())
    exact_audit = bool(audit.get("passed")) and all(
        bool((audit.get(split) or {}).get("locked_initializer_required"))
        for split in ("train", "validation", "calibration")
    )
    report = {
        "schema_version": 1,
        "metric": "mean regional joint-rotation geodesic error (degrees)",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "real_residual_audit": str(args.real_residual_audit.resolve()),
        "real_residual_audit_sha256": sha256_file(args.real_residual_audit),
        "frames": frames,
        "expected_frames": int(validation.get("frames", -1)),
        "source_disjoint_verified": bool(audit.get("split_disjoint_verified")),
        "proxy_residual_audit_passed": bool(audit.get("passed")),
        "real_residual_audit_passed": exact_audit,
        "baseline": baseline,
        "prediction": prediction,
        "regional_relative_gain": gains,
        "equal_weight_relative_gain": float(np.mean(list(gains.values()))),
        "hard_subset_contract": HARD_SUBSET_CONTRACT,
        "hard_subset_joint_frames": hard_counts,
        "hard_subset_regional_relative_gain": hard_gains,
        "hard_subset_relative_gain": (
            float(np.mean(list(hard_gains.values()))) if complete_hard else -1.0
        ),
        "hard_subset_complete": complete_hard,
        "fallback_group_frames": fallback_group_frames,
        "total_group_frames": total_group_frames,
        "group_frame_fallback_fraction": fallback_group_frames
        / max(total_group_frames, 1),
        "paired_clip_bootstrap": {
            region: _bootstrap_clip_delta(per_clip, region, args.bootstrap_samples, args.seed)
            for region in REGIONS
        },
        "scope": (
            "formal_exact-A1" if exact_audit else "H32 Tier-C proxy; formal G4 exact-A1 check is false"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with args.per_clip_output.open("x", encoding="utf-8") as handle:
        for row in per_clip:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-residual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-clip-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.per_clip_output.exists():
        raise FileExistsError("Refusing to overwrite residual-evaluation artifacts")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
