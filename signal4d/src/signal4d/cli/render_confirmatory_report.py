from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric_row(
    label: str,
    metric: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    return (
        f"| {label} | {float(baseline[metric]):.4f} | "
        f"{float(candidate[metric]):.4f} | "
        f"{float(comparison['candidate_minus_baseline']):+.4f} | "
        f"[{float(comparison['ci_low']):+.4f}, "
        f"{float(comparison['ci_high']):+.4f}] |"
    )


def run(
    decision_path: str,
    baseline_summary_path: str,
    candidate_summary_path: str,
    gate_metadata_path: str,
    release_path: str,
    output: str,
) -> dict[str, Any]:
    decision = _read(decision_path)
    baseline = _read(baseline_summary_path)
    candidate = _read(candidate_summary_path)
    gate = _read(gate_metadata_path)
    release = _read(release_path)
    comparisons = decision["comparisons"]
    lines = [
        "# SIGNAL-4D extended-post confirmatory report",
        "",
        f"**Decision: {decision['decision']}**",
        "",
        "This report is generated from the frozen evaluator, paired-bootstrap, gate, "
        "and release artifacts. The endpoint contains 56 clips and 769 declared frames.",
        "",
        "## Confirmatory results",
        "",
        "All geometry values are equal-weight clip-macro TR-V2V in mm. Dynamics use "
        "the evaluator's joint-error units. Deltas are candidate minus A1; lower is better.",
        "",
        "| Metric | A1 baseline | SIGNAL-4D | Delta | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
        _metric_row(
            "Left hand",
            "tr_v2v_left_hand_mm",
            baseline,
            candidate,
            comparisons["tr_v2v_left_hand_mm"],
        ),
        _metric_row(
            "Upper body",
            "tr_v2v_upper_body_mm",
            baseline,
            candidate,
            comparisons["tr_v2v_upper_body_mm"],
        ),
        _metric_row(
            "Right hand",
            "tr_v2v_right_hand_mm",
            baseline,
            candidate,
            comparisons["tr_v2v_right_hand_mm"],
        ),
        _metric_row(
            "Velocity error",
            "velocity_error",
            baseline,
            candidate,
            comparisons["velocity_error"],
        ),
        _metric_row(
            "Acceleration error",
            "acceleration_error",
            baseline,
            candidate,
            comparisons["acceleration_error"],
        ),
        _metric_row(
            "Jerk error",
            "jerk_error",
            baseline,
            candidate,
            comparisons["jerk_error"],
        ),
        "",
        "## Preregistered gates",
        "",
    ]
    for name, passed in decision["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Label-blind selection evidence",
            "",
            f"The grouped historical OOF gate delta was "
            f"{float(gate['oof_clip_macro_delta_mm']):+.4f} mm "
            f"(95% CI [{float(gate['oof_ci95_clip_bootstrap_mm'][0]):+.4f}, "
            f"{float(gate['oof_ci95_clip_bootstrap_mm'][1]):+.4f}]), with "
            f"{int(gate['oof_switches'])} within-clip switches. Prospective GT was not "
            "available to the gate.",
            "",
            "## Integrity and claim boundary",
            "",
            f"Release status: `{release['status']}`. The release file hashes source, "
            "configs, manifest, calibration, gate artifacts, inputs, baseline and "
            "candidate predictions before the prospective GT cache is created.",
            "",
            "This is a temporally disjoint post-segment endpoint with overlapping "
            "clip/sign identities and unavailable signer IDs. It is not an external "
            "published leaderboard or unseen-signer evaluation. Contact and semantic "
            "claims remain blocked because trustworthy annotations/evaluators are absent.",
            "",
        ]
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return {
        "decision": decision["decision"],
        "output": str(destination),
        "clips": candidate["clips"],
        "frames": candidate["frames"],
    }
