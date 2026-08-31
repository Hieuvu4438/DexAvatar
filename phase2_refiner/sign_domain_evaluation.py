"""Calibration and rotation evaluation for the source-domain refiner.

The module deliberately keeps calibration and test evaluation as separate CLI
commands.  Thresholds are selected on the calibration split only and are then
treated as immutable input to evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase2_refiner.config import load_config, validate_config
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import SequenceCacheDataset, collate_sequences
from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.infer import _apply_safety_fallback, _load_model
from phase2_refiner.provenance import sha256_file


REGIONS = {
    "ubody": (slice(0, 21), 0),
    "lhand": (slice(21, 36), 1),
    "rhand": (slice(36, 51), 2),
}


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _lineage_guard(
    path: str | Path, *, require_test: bool = False
) -> dict[str, Any]:
    lineage_path = Path(path)
    with lineage_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    allowed_decisions = {
        "PASS",
        "PASS_WITH_REPORTED_OFFICIAL_TEST_SOURCE_OVERLAP",
    }
    if report.get("decision") not in allowed_decisions:
        raise ValueError("Lineage audit did not pass")
    if int(report.get("sgnify_training_or_selection_reads", -1)) != 0:
        raise ValueError("Lineage audit reports SGNify training/selection reads")
    if require_test and "test" not in report.get("manifests", {}):
        raise ValueError("Test evaluation requires a lineage audit containing test")
    overlaps = report.get("source_group_overlaps", {})
    illegal_overlaps = {
        pair: values
        for pair, values in overlaps.items()
        if values and "test" not in pair.split("__")
    }
    if illegal_overlaps:
        raise ValueError("Lineage audit reports development-split overlap")
    test_overlaps = {
        pair: values
        for pair, values in overlaps.items()
        if values and "test" in pair.split("__")
    }
    if test_overlaps and report.get("test_protocol") != (
        "official-held-out-clips-not-signer-disjoint"
    ):
        raise ValueError("Test source overlap is not explicitly disclosed")
    return {
        "path": str(lineage_path.resolve()),
        "sha256": sha256_file(lineage_path),
        "decision": report["decision"],
        "sgnify_training_or_selection_reads": 0,
        "test_protocol": report.get("test_protocol"),
        "reported_test_source_overlaps": test_overlaps,
    }


def _dataset_from_path(path: Path) -> str:
    metadata = json.loads(load_cache_clip(path).metadata_json)
    dataset = str(metadata.get("dataset", ""))
    if not dataset:
        raise ValueError(f"Cache has no dataset metadata: {path}")
    return dataset


@torch.no_grad()
def collect_records(
    config: dict[str, Any],
    checkpoint: Path,
    split_manifest: str | Path,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect per-clip errors and benefit probabilities without aggregation."""
    model_config = config["model"]
    data_config = config["data"]
    dataset = SequenceCacheDataset(
        str(split_manifest),
        max_frames=int(model_config["max_frames"]),
        training=False,
        input_dim=int(model_config["input_dim"]),
        reprojection_residual_scale=float(
            data_config.get("reprojection_residual_scale", 10.0)
        ),
        physical_time_motion=bool(data_config.get("physical_time_motion", False)),
        motion_reference_seconds=float(
            data_config.get("motion_reference_seconds", 0.04)
        ),
        require_phase2r_semantics=bool(
            data_config.get("require_phase2r_semantics", False)
        ),
    )
    domains = [_dataset_from_path(path) for path in dataset.paths]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequences,
    )
    model = _load_model(config, checkpoint, device, use_ema=True)
    records: list[dict[str, Any]] = []
    offset = 0
    safety_counts = defaultdict(int)
    for batch in loader:
        tensor_batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        prediction = model(
            tensor_batch["features"],
            tensor_batch["initial_matrix"],
            tensor_batch["frame_valid"],
            tensor_batch["refine_mask"],
            tensor_batch["initial_joint_position"],
        )
        if "benefit_logit" not in prediction:
            raise ValueError("Checkpoint has no benefit logits")
        for index, clip_id in enumerate(batch["clip_id"]):
            length = int(batch["length"][index])
            initial = tensor_batch["initial_matrix"][index, :length]
            candidate, fallback = _apply_safety_fallback(
                prediction["matrix"][index, :length],
                initial,
                body_limit_degrees=float(model_config["body_max_degrees"]),
                hand_limit_degrees=float(model_config["hand_max_degrees"]),
                log_variance=(
                    prediction["log_variance"][index, :length]
                    if "log_variance" in prediction
                    else None
                ),
            )
            target = tensor_batch["target_matrix"][index, :length]
            valid = (
                tensor_batch["target_rotation_valid"][index, :length]
                & tensor_batch["refine_mask"][index, None]
                & tensor_batch["frame_valid"][index, :length, None]
            )
            for region, (_, group_index) in REGIONS.items():
                safety_counts[region] += int(fallback[:, group_index].sum().cpu())
            records.append(
                {
                    "clip_id": str(clip_id),
                    "dataset": domains[offset + index],
                    "baseline_error": torch.rad2deg(
                        geodesic_distance(initial.float(), target.float())
                    ).cpu().numpy(),
                    "candidate_error": torch.rad2deg(
                        geodesic_distance(candidate.float(), target.float())
                    ).cpu().numpy(),
                    "valid": valid.cpu().numpy(),
                    "benefit_probability": prediction["benefit_logit"][
                        index, :length
                    ].sigmoid().cpu().numpy(),
                }
            )
        offset += len(batch["clip_id"])
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    diagnostics = {
        "clips": len(records),
        "frames": int(sum(len(record["valid"]) for record in records)),
        "checkpoint_step": int(checkpoint_data.get("step", -1)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "safety_fallback_group_frames": dict(safety_counts),
    }
    return records, diagnostics


def clip_region_error(
    record: dict[str, Any], region: str, threshold: float | None
) -> tuple[float, float, int, int] | None:
    joint_slice, group_index = REGIONS[region]
    valid = record["valid"][:, joint_slice]
    count = int(valid.sum())
    if count == 0:
        return None
    baseline = record["baseline_error"][:, joint_slice]
    candidate = record["candidate_error"][:, joint_slice]
    if threshold is None:
        selected = candidate
        accepted_frames = int(valid.any(axis=1).sum())
    else:
        accept = record["benefit_probability"][:, group_index] >= threshold
        selected = np.where(accept[:, None], candidate, baseline)
        accepted_frames = int((accept & valid.any(axis=1)).sum())
    return (
        float(baseline[valid].mean()),
        float(selected[valid].mean()),
        count,
        accepted_frames,
    )


def summarize(
    records: list[dict[str, Any]], thresholds: dict[str, float] | None
) -> dict[str, Any]:
    domains = sorted({record["dataset"] for record in records})
    result: dict[str, Any] = {}
    for domain in domains + ["pooled"]:
        subset = (
            records
            if domain == "pooled"
            else [record for record in records if record["dataset"] == domain]
        )
        domain_result: dict[str, Any] = {"clips": len(subset)}
        ratios = []
        for region in REGIONS:
            rows = [
                value
                for record in subset
                if (
                    value := clip_region_error(
                        record,
                        region,
                        None if thresholds is None else thresholds[region],
                    )
                )
                is not None
            ]
            if not rows:
                raise ValueError(f"No eligible {domain}/{region} targets")
            baseline = float(np.mean([row[0] for row in rows]))
            prediction = float(np.mean([row[1] for row in rows]))
            ratio = prediction / baseline
            ratios.append(ratio)
            domain_result[region] = {
                "eligible_clips": len(rows),
                "valid_joint_frames": int(sum(row[2] for row in rows)),
                "baseline_macro_clip_deg": baseline,
                "prediction_macro_clip_deg": prediction,
                "delta_deg": prediction - baseline,
                "prediction_over_baseline": ratio,
                "accepted_group_frames": int(sum(row[3] for row in rows)),
            }
        domain_result["equal_region_ratio"] = float(np.mean(ratios))
        result[domain] = domain_result
    return result


def select_thresholds(
    records: list[dict[str, Any]], grid: list[float]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Minimize worst-domain ratio independently for each anatomical region."""
    domains = sorted({record["dataset"] for record in records})
    selected: dict[str, float] = {}
    audit: dict[str, Any] = {}
    for region in REGIONS:
        candidates = []
        for threshold in grid:
            summary = summarize(records, {name: threshold for name in REGIONS})
            ratios = {
                domain: summary[domain][region]["prediction_over_baseline"]
                for domain in domains
            }
            pooled_ratio = summary["pooled"][region]["prediction_over_baseline"]
            coverage = summary["pooled"][region]["accepted_group_frames"]
            candidates.append(
                {
                    "threshold": float(threshold),
                    "worst_domain_ratio": float(max(ratios.values())),
                    "pooled_ratio": float(pooled_ratio),
                    "accepted_group_frames": int(coverage),
                    "domain_ratios": ratios,
                }
            )
        # Conservative final tie-break: higher threshold abstains more often.
        best = min(
            candidates,
            key=lambda row: (
                round(row["worst_domain_ratio"], 10),
                round(row["pooled_ratio"], 10),
                -row["threshold"],
            ),
        )
        selected[region] = best["threshold"]
        audit[region] = {"selected": best, "grid": candidates}
    return selected, audit


def paired_bootstrap(
    records: list[dict[str, Any]],
    region: str,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    deltas = []
    for record in records:
        value = clip_region_error(record, region, threshold)
        if value is not None:
            deltas.append(value[1] - value[0])
    array = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    draws = array[indices].mean(axis=1)
    return {
        "eligible_clips": len(array),
        "mean_delta_deg": float(array.mean()),
        "ci95_low_deg": float(np.quantile(draws, 0.025)),
        "ci95_high_deg": float(np.quantile(draws, 0.975)),
        "probability_improved": float((draws < 0).mean()),
    }


def calibrate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    validate_config(config)
    lineage_path = args.lineage_report or Path(config["data"]["lineage_report"])
    lineage = _lineage_guard(lineage_path)
    manifest = Path(config["data"]["calibration_glob"])
    records, diagnostics = collect_records(
        config, args.checkpoint, manifest, torch.device(args.device), args.batch_size
    )
    grid = [float(value) for value in np.linspace(0.0, 1.0, args.grid_steps)]
    thresholds, threshold_audit = select_thresholds(records, grid)
    report = {
        "schema_version": 1,
        "kind": "sign-domain-benefit-calibration",
        "split": "calibration",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "lineage": lineage,
        "diagnostics": diagnostics,
        "thresholds": thresholds,
        "selection_rule": "minimum worst source-dataset macro-clip ratio",
        "threshold_audit": threshold_audit,
        "selected_metrics": summarize(records, thresholds),
    }
    _write_new_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def preview(args: argparse.Namespace) -> None:
    """Report ungated validation metrics without touching calibration or test."""
    config = load_config(args.config)
    validate_config(config)
    lineage_path = args.lineage_report or Path(config["data"]["lineage_report"])
    lineage = _lineage_guard(lineage_path)
    manifest = Path(config["data"]["val_glob"])
    records, diagnostics = collect_records(
        config, args.checkpoint, manifest, torch.device(args.device), args.batch_size
    )
    report = {
        "schema_version": 1,
        "kind": "sign-domain-ungated-validation-preview",
        "split": "validation",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "lineage": lineage,
        "diagnostics": diagnostics,
        "metrics": summarize(records, thresholds=None),
    }
    _write_new_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    validate_config(config)
    lineage_path = args.lineage_report or Path(config["data"]["lineage_report"])
    lineage = _lineage_guard(lineage_path, require_test=args.split == "test")
    with args.calibration.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("split") != "calibration":
        raise ValueError("Threshold file was not produced on calibration split")
    if calibration["diagnostics"]["checkpoint_sha256"] != sha256_file(args.checkpoint):
        raise ValueError("Calibration checkpoint does not match evaluation checkpoint")
    thresholds = {
        region: float(calibration["thresholds"][region]) for region in REGIONS
    }
    manifest = Path(config["data"][f"{args.split}_glob"])
    records, diagnostics = collect_records(
        config, args.checkpoint, manifest, torch.device(args.device), args.batch_size
    )
    metrics = summarize(records, thresholds)
    domains = sorted({record["dataset"] for record in records})
    bootstrap: dict[str, Any] = {}
    for domain in domains + ["pooled"]:
        subset = (
            records
            if domain == "pooled"
            else [record for record in records if record["dataset"] == domain]
        )
        bootstrap[domain] = {
            region: paired_bootstrap(
                subset,
                region,
                thresholds[region],
                args.bootstrap_samples,
                args.seed + group_index,
            )
            for group_index, region in enumerate(REGIONS)
        }
    report = {
        "schema_version": 1,
        "kind": "sign-domain-checkpoint-evaluation",
        "split": args.split,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "lineage": lineage,
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "thresholds": thresholds,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
    }
    _write_new_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument(
        "--lineage-report",
        type=Path,
        help="Override the immutable training config's lineage report",
    )
    common.add_argument("--device", default="cuda")
    common.add_argument("--batch-size", type=int, default=16)

    calibration = subparsers.add_parser("calibrate", parents=[common])
    calibration.add_argument("--grid-steps", type=int, default=21)
    calibration.set_defaults(function=calibrate)

    validation_preview = subparsers.add_parser("preview", parents=[common])
    validation_preview.set_defaults(function=preview)

    evaluation = subparsers.add_parser("evaluate", parents=[common])
    evaluation.add_argument("--calibration", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), required=True)
    evaluation.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.set_defaults(function=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
