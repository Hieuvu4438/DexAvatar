"""Fail-closed numerical Phase 3 GO/NO-GO decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase3_posterior.provenance import atomic_json


REGIONS = ("ubody", "lhand", "rhand")


def _check(value: bool, actual: Any, requirement: str) -> dict[str, Any]:
    return {"passed": bool(value), "actual": actual, "requirement": requirement}


def stage_decision(gate: str, metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    if gate == "g0":
        checks = {
            "no_leakage": _check(
                bool(metrics.get("no_leakage")), metrics.get("no_leakage"), "true"
            ),
            "disjoint": _check(
                bool(metrics.get("disjoint")), metrics.get("disjoint"), "true"
            ),
            "hashes_licenses": _check(
                bool(metrics.get("hashes_licenses")),
                metrics.get("hashes_licenses"),
                "true",
            ),
            "manual_failure": _check(
                float(metrics.get("manual_failure_rate", 1.0)) < 0.10,
                metrics.get("manual_failure_rate"),
                "<0.10",
            ),
        }
    elif gate == "g1":
        checks = {
            "adapter_equivalence": _check(
                bool(metrics.get("adapter_equivalence")),
                metrics.get("adapter_equivalence"),
                "true",
            ),
            "pretrained_gain": _check(
                float(metrics.get("pretrained_gain", -np.inf)) >= 0.05,
                metrics.get("pretrained_gain"),
                ">=0.05",
            ),
        }
    elif gate == "g2":
        checks = {
            "relation_mae_gain": _check(
                float(metrics.get("relation_mae_gain", -np.inf)) >= 0.10,
                metrics.get("relation_mae_gain"),
                ">=0.10",
            ),
            "contact_f1": _check(
                float(metrics.get("contact_f1", -np.inf)) >= 0.65,
                metrics.get("contact_f1"),
                ">=0.65",
            ),
            "sign_contact_f1": _check(
                float(metrics.get("sign_contact_f1", -np.inf)) >= 0.60,
                metrics.get("sign_contact_f1"),
                ">=0.60",
            ),
            "max_regression": _check(
                float(metrics.get("max_region_regression", np.inf)) <= 0.01,
                metrics.get("max_region_regression"),
                "<=0.01",
            ),
        }
    elif gate in {"g3", "g4"}:
        threshold = 0.30 if gate == "g3" else 0.35
        for region in REGIONS:
            checks[f"{region}_recovery"] = _check(
                float(metrics.get("recovery", {}).get(region, -np.inf)) >= threshold,
                metrics.get("recovery", {}).get(region),
                f">={threshold}",
            )
        checks["clean_regression"] = _check(
            float(metrics.get("max_clean_regression", np.inf)) < 0.01,
            metrics.get("max_clean_regression"),
            "<0.01",
        )
        if gate == "g4":
            checks["interaction_gain"] = _check(
                float(metrics.get("interaction_gain", -np.inf)) >= 0.05,
                metrics.get("interaction_gain"),
                ">=0.05",
            )
    elif gate in {"g5", "g6"}:
        checks = {
            "equal_region_gain": _check(
                float(metrics.get("equal_region_gain", -np.inf)) >= 0.03,
                metrics.get("equal_region_gain"),
                ">=0.03",
            ),
            "max_regression": _check(
                float(metrics.get("max_region_regression", np.inf)) <= 0.01,
                metrics.get("max_region_regression"),
                "<=0.01",
            ),
            "hard_gain": _check(
                float(metrics.get("hard_gain", -np.inf)) >= 0.08,
                metrics.get("hard_gain"),
                ">=0.08",
            ),
        }
        if gate == "g6":
            checks["additional_hard_gain"] = _check(
                float(metrics.get("additional_hard_gain", -np.inf)) >= 0.02,
                metrics.get("additional_hard_gain"),
                ">=0.02",
            )
    elif gate == "g7":
        hard = float(metrics.get("hard_gain_over_k1", -np.inf)) >= 0.02
        gap = float(metrics.get("oracle_gap_closed", -np.inf)) >= 0.25
        checks = {
            "selection_gain": _check(
                hard or gap,
                {
                    "hard_gain": metrics.get("hard_gain_over_k1"),
                    "oracle_gap_closed": metrics.get("oracle_gap_closed"),
                },
                "hard>=0.02 or oracle-gap>=0.25",
            ),
            "clean_regression": _check(
                float(metrics.get("max_clean_regression", np.inf)) <= 0.005,
                metrics.get("max_clean_regression"),
                "<=0.005",
            ),
            "full_coverage": _check(
                bool(metrics.get("full_coverage")), metrics.get("full_coverage"), "true"
            ),
        }
    else:
        raise ValueError(f"Unsupported stage gate: {gate}")
    return {
        "gate": f"P3-{gate.upper()}",
        "passed": bool(checks) and all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def g8_decision(
    summaries: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    expected_frames: int = 1493,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    checks["exactly_three_seeds"] = _check(len(summaries) == 3, len(summaries), "3")
    frames = [item.get("frames") for item in summaries]
    checks["identical_full_coverage"] = _check(
        all(value == expected_frames for value in frames), frames, str(expected_frames)
    )
    gains = []
    regional_values: dict[str, list[float]] = {region: [] for region in REGIONS}
    safe = True
    ci_counts = []
    for summary in summaries:
        ratios = []
        ci_count = 0
        for region in REGIONS:
            baseline = float(summary["baseline"][region])
            prediction = float(summary["prediction"][region])
            regional_values[region].append(prediction)
            safe &= prediction - baseline <= 0.20
            ratios.append((baseline - prediction) / baseline)
            ci_count += int(
                float(summary["paired_bootstrap"][region]["ci95_high_mm"]) < 0
            )
        gains.append(float(np.mean(ratios)))
        ci_counts.append(ci_count)
    checks["no_region_regresses_over_0.20_mm"] = _check(safe, None, "all delta<=0.20mm")
    checks["two_regions_improve_with_ci"] = _check(
        all(value >= 2 for value in ci_counts), ci_counts, ">=2 each seed"
    )
    checks["equal_region_gain"] = _check(
        all(value >= 0.03 for value in gains), gains, ">=0.03 each seed"
    )
    checks["hard_subset_gain"] = _check(
        float(diagnostics.get("hard_subset_gain", -np.inf)) >= 0.08,
        diagnostics.get("hard_subset_gain"),
        ">=0.08",
    )
    checks["clean_regression"] = _check(
        float(diagnostics.get("max_clean_regression", np.inf)) < 0.01,
        diagnostics.get("max_clean_regression"),
        "<0.01",
    )
    checks["fallback"] = _check(
        float(diagnostics.get("fallback_rate", np.inf)) < 0.01,
        diagnostics.get("fallback_rate"),
        "<0.01",
    )
    standard_deviations = {
        key: float(np.std(value)) for key, value in regional_values.items()
    }
    checks["regional_seed_sd"] = _check(
        all(value < 0.20 for value in standard_deviations.values()),
        standard_deviations,
        "<0.20mm",
    )
    return {
        "gate": "P3-G8",
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=[f"g{i}" for i in range(9)], required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--seed-summary", type=Path, action="append", default=[])
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--expected-frames", type=int, default=1493)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gate == "g8":
        summaries = [json.loads(path.read_text()) for path in args.seed_summary]
        diagnostics = (
            json.loads(args.diagnostics.read_text()) if args.diagnostics else {}
        )
        decision = g8_decision(summaries, diagnostics, args.expected_frames)
    else:
        if args.metrics is None:
            raise ValueError("--metrics is required for P3-G0 through P3-G7")
        decision = stage_decision(args.gate, json.loads(args.metrics.read_text()))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    atomic_json(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
