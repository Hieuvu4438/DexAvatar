from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GEOMETRY = (
    "tr_v2v_left_hand_mm",
    "tr_v2v_upper_body_mm",
    "tr_v2v_right_hand_mm",
)
DYNAMICS = ("velocity_error", "acceleration_error", "jerk_error")


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run(
    candidate_summary: str,
    baseline_summary: str,
    comparison_root: str,
    reproducibility: str,
    output: str,
    expected_clips: int,
    expected_frames: int,
) -> dict[str, Any]:
    candidate = _read(candidate_summary)
    baseline = _read(baseline_summary)
    comparisons = {
        metric: _read(Path(comparison_root) / f"{metric}.json")
        for metric in (*GEOMETRY, *DYNAMICS)
    }
    repro = _read(reproducibility)

    completeness = {
        "candidate_clips": candidate.get("clips"),
        "candidate_frames": candidate.get("frames"),
        "candidate_coverage": candidate.get("coverage"),
        "baseline_clips": baseline.get("clips"),
        "baseline_frames": baseline.get("frames"),
        "baseline_coverage": baseline.get("coverage"),
    }
    completeness_pass = (
        candidate.get("clips") == baseline.get("clips") == expected_clips
        and candidate.get("frames") == baseline.get("frames") == expected_frames
        and candidate.get("coverage") == baseline.get("coverage") == 1.0
    )
    left = comparisons[GEOMETRY[0]]
    left_pass = (
        float(left["candidate_minus_baseline"]) <= -0.5
        and float(left["ci_high"]) < 0.0
    )
    noninferiority = {
        metric: float(comparisons[metric]["ci_high"]) < 0.5
        for metric in GEOMETRY[1:]
    }
    dynamics: dict[str, dict[str, float | bool]] = {}
    for metric in DYNAMICS:
        delta = float(comparisons[metric]["candidate_minus_baseline"])
        baseline_value = float(baseline[metric])
        relative = delta / baseline_value if baseline_value != 0 else float("inf")
        dynamics[metric] = {
            "candidate_minus_baseline": delta,
            "baseline": baseline_value,
            "relative_regression": relative,
            "passed": relative <= 0.02,
        }
    checks = {
        "complete_56_clips_769_frames": completeness_pass,
        "left_superiority_effect_and_ci": left_pass,
        "upper_body_noninferiority": noninferiority[GEOMETRY[1]],
        "right_hand_noninferiority": noninferiority[GEOMETRY[2]],
        "dynamics_no_more_than_2pct_regression": all(
            bool(value["passed"]) for value in dynamics.values()
        ),
        "byte_exact_gate_reproducibility": bool(repro.get("passed", False)),
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "completeness": completeness,
        "comparisons": comparisons,
        "dynamics_relative": dynamics,
        "reproducibility": repro,
        "claim_scope_if_pass": (
            "prospective SIGNAL-4D extended-post SGNify endpoint versus the "
            "pre-frozen recomputed same-protocol A1 baseline"
        ),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
