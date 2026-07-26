"""Executable Go/No-Go decisions for Phase 2 G4, G5, and G6 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REGIONS = ("ubody", "lhand", "rhand")


def _relative_gain(baseline: float, prediction: float) -> float:
    if baseline <= 0:
        raise ValueError("Baseline regional errors must be positive")
    return (baseline - prediction) / baseline


def g4_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the external, source-disjoint real-validation contract."""
    gains = {
        region: _relative_gain(
            float(payload["baseline"][region]), float(payload["prediction"][region])
        )
        for region in REGIONS
    }
    equal_weight_gain = sum(gains.values()) / len(gains)
    checks = {
        "source_disjoint_validation": bool(payload.get("source_disjoint_verified", False)),
        "exact_expert_real_residual_audit": bool(
            payload.get("real_residual_audit_passed", False)
        ),
        "identical_full_coverage": int(payload.get("frames", 0))
        == int(payload.get("expected_frames", -1)),
        "weighted_regional_gain_at_least_3pct": equal_weight_gain >= 0.03,
        "no_region_regresses_over_1pct": all(gain >= -0.01 for gain in gains.values()),
        "hard_subset_gain_at_least_8pct": float(
            payload.get("hard_subset_relative_gain", float("-inf"))
        )
        >= 0.08,
    }
    return {
        "gate": "G4",
        "passed": all(checks.values()),
        "checks": checks,
        "regional_relative_gain": gains,
        "equal_weight_relative_gain": equal_weight_gain,
    }


def g6_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every numerical criterion in the proposal's G6 contract."""
    seeds = payload.get("seeds", [])
    if not seeds:
        return {
            "gate": "G6",
            "passed": False,
            "checks": {"exactly_three_seeds": False},
            "reason": "No seed results were provided",
        }
    for item in seeds:
        for key in ("prediction", "baseline", "paired_bootstrap"):
            if key not in item:
                raise ValueError(f"Seed result lacks {key}")

    per_seed_gain = []
    regional_values = {region: [] for region in REGIONS}
    ci_improvements = []
    coverage = []
    for item in seeds:
        gains = []
        improved_with_ci = 0
        for region in REGIONS:
            prediction = float(item["prediction"][region])
            baseline = float(item["baseline"][region])
            regional_values[region].append(prediction)
            gains.append(_relative_gain(baseline, prediction))
            ci = item["paired_bootstrap"][region]
            if float(ci["ci95_high_mm"]) < 0.0:
                improved_with_ci += 1
        per_seed_gain.append(sum(gains) / len(gains))
        ci_improvements.append(improved_with_ci)
        coverage.append(int(item["frames"]))

    diagnostics = payload.get("diagnostics", {})
    hard_value = diagnostics.get("hard_subset_relative_gain")
    hard_gain = float(hard_value) if hard_value is not None else float("-inf")
    clean_regression = diagnostics.get("clean_regression_fraction", {})
    fallback_fraction = float(diagnostics.get("group_frame_fallback_fraction", float("inf")))
    checks = {
        "exactly_three_seeds": len(seeds) == 3,
        "identical_full_coverage": len(set(coverage)) == 1
        and coverage[0] == int(payload.get("expected_frames", coverage[0])),
        "no_region_regresses_over_0.20_mm": all(
            float(item["prediction"][region]) - float(item["baseline"][region]) <= 0.20
            for item in seeds
            for region in REGIONS
        ),
        "two_regions_improve_with_ci": all(count >= 2 for count in ci_improvements),
        "equal_weight_relative_gain_at_least_3pct": all(gain >= 0.03 for gain in per_seed_gain),
        "hard_subset_gain_at_least_8pct": hard_gain >= 0.08,
        "clean_regression_below_1pct": all(
            region in clean_regression
            and clean_regression[region] is not None
            and float(clean_regression[region]) < 0.01
            for region in REGIONS
        ),
        "fallback_below_1pct": fallback_fraction < 0.01,
        "regional_seed_sd_below_0.20_mm": all(
            _population_sd(values) < 0.20 for values in regional_values.values()
        ),
    }
    return {
        "gate": "G6",
        "passed": all(checks.values()),
        "checks": checks,
        "equal_weight_relative_gain": per_seed_gain,
        "ci_improved_regions": ci_improvements,
        "regional_seed_sd_mm": {
            region: _population_sd(values) for region, values in regional_values.items()
        },
    }


def _population_sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g6", type=Path, help="JSON containing three evaluator summaries and diagnostics")
    parser.add_argument("--g4", type=Path, help="JSON containing an external real-validation summary")
    parser.add_argument(
        "--g6-seed",
        type=Path,
        action="append",
        help="Evaluator summary for one seed; repeat exactly three times",
    )
    parser.add_argument("--diagnostics", type=Path, help="Frozen G6 hard/clean/fallback diagnostics")
    parser.add_argument("--expected-frames", type=int, default=1493)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = sum((args.g4 is not None, args.g6 is not None, args.g6_seed is not None))
    if modes != 1:
        raise ValueError("Select exactly one gate input: --g4, --g6, or --g6-seed")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.g6_seed is not None:
        seeds = []
        for path in args.g6_seed:
            with path.open("r", encoding="utf-8") as handle:
                seeds.append(json.load(handle))
        diagnostics = {}
        if args.diagnostics:
            with args.diagnostics.open("r", encoding="utf-8") as handle:
                diagnostics = json.load(handle)
        payload = {
            "expected_frames": args.expected_frames,
            "seeds": seeds,
            "diagnostics": diagnostics,
        }
        decision = g6_decision(payload)
    else:
        selected = args.g4 or args.g6
        with selected.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        decision = g4_decision(payload) if args.g4 else g6_decision(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
