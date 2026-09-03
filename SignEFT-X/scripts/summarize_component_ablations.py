#!/usr/bin/env python3
"""Summarize canonical-component and wrist-locking ablations with paired CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signeft.io_utils import atomic_write_json, atomic_write_text, sha256_file


PA_KEYS = ("pa_mpvpe_all", "pa_mpvpe_body_only", "pa_mpvpe_hand")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(before: dict, after: dict, key: str, rng: np.random.Generator,
           samples: int) -> dict:
    signs = sorted(set(before["per_sign"]) & set(after["per_sign"]))
    delta = np.asarray([
        after["per_sign"][sign]["metrics"][key]
        - before["per_sign"][sign]["metrics"][key]
        for sign in signs
    ], dtype=np.float64)
    boot = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(1)
    return {
        "before_micro_mm": before["micro_average_mm"][key],
        "after_micro_mm": after["micro_average_mm"][key],
        "delta_micro_mm": after["micro_average_mm"][key] - before["micro_average_mm"][key],
        "paired_sign_mean_delta_mm": float(delta.mean()),
        "paired_sign_bootstrap_95ci_mm": [
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        ],
        "after_better_signs": int((delta < -1e-9).sum()),
        "before_better_signs": int((delta > 1e-9).sum()),
        "ties": int((np.abs(delta) <= 1e-9).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ablation_canonical_components"))
    parser.add_argument("--optimized-refit-pa", type=Path, default=Path("outputs/ablation_lbfgs/evaluation/canonical_adam.json"))
    parser.add_argument("--optimized-refit-official", type=Path, default=Path("outputs/ablation_lbfgs/evaluation/canonical_adam_official/official_result.json"))
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    paths = {
        "without_beta_without_pose": args.root / "evaluation/beta_robust_beta__pose_without.json",
        "with_beta_without_pose": args.root / "evaluation/beta_beta__pose_without.json",
        "without_beta_with_pose": args.root / "evaluation/beta_robust_beta__pose_with.json",
        "with_beta_with_pose": args.optimized_refit_pa,
    }
    official_paths = {
        "without_beta_without_pose": args.root / "evaluation/beta_robust_beta__pose_without_official/official_result.json",
        "with_beta_without_pose": args.root / "evaluation/beta_beta__pose_without_official/official_result.json",
        "without_beta_with_pose": args.root / "evaluation/beta_robust_beta__pose_with_official/official_result.json",
        "with_beta_with_pose": args.optimized_refit_official,
    }
    values = {key: load(value) for key, value in paths.items()}
    official = {key: load(value)["metrics_mm"] for key, value in official_paths.items()}
    comparisons = {
        "beta_effect_without_pose_refit": ("without_beta_without_pose", "with_beta_without_pose"),
        "beta_effect_with_pose_refit": ("without_beta_with_pose", "with_beta_with_pose"),
        "pose_refit_effect_without_beta_refinement": ("without_beta_without_pose", "without_beta_with_pose"),
        "pose_refit_effect_with_beta_refinement": ("with_beta_without_pose", "with_beta_with_pose"),
    }
    rng = np.random.default_rng(args.seed)
    stats = {
        name: {key: paired(values[a], values[b], key, rng, args.samples) for key in PA_KEYS}
        for name, (a, b) in comparisons.items()
    }
    result = {
        "schema_version": "signeft.canonical-component-ablation-report.v1",
        "signs": 57, "frames": 1493, "bootstrap_unit": "sign",
        "bootstrap_samples": args.samples, "seed": args.seed,
        "rows": {
            name: {
                "official_mm": official[name],
                "pa_mm": values[name]["micro_average_mm"],
                "pa_sha256": sha256_file(paths[name]),
                "official_sha256": sha256_file(official_paths[name]),
            } for name in paths
        },
        "paired_effects": stats,
    }
    report = args.root / "report"
    atomic_write_json(report / "comparison.json", result)
    lines = [
        "# Canonical component ablation", "",
        "All rows use 57 signs / 1,493 frames. W/o beta refinement uses the robust Huber estimate; w/ beta refinement uses beta*. Post-beta pose refit denotes the released hand plus shoulder/elbow/wrist canonical refit.", "",
        "| Beta refinement | Post-beta pose refit | All | UBody | UBody (-F) | UBody (-H) | LHand | RHand | PA hands |",
        "|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        "without_beta_without_pose", "with_beta_without_pose",
        "without_beta_with_pose", "with_beta_with_pose",
    )
    for name in order:
        o, p = official[name], values[name]["micro_average_mm"]
        lines.append(
            f"| {'w/' if name.startswith('with_beta') else 'w/o'} | "
            f"{'w/' if name.endswith('with_pose') else 'w/o'} | "
            f"{o['tr all']:.4f} | {o['tr above pelvis upper body']:.4f} | "
            f"{o['tr above pelvis minus face']:.4f} | {o['tr above pelvis minus head']:.4f} | "
            f"{o['tr left hand']:.4f} | {o['tr right hand']:.4f} | {p['pa_mpvpe_hand']:.4f} |"
        )
    lines += ["", "Paired effects (after minus before; negative is better):", ""]
    for name, metrics in stats.items():
        v = metrics["pa_mpvpe_hand"]
        lo, hi = v["paired_sign_bootstrap_95ci_mm"]
        lines.append(
            f"- {name}: PA hands {v['delta_micro_mm']:+.4f} mm; paired sign 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}]; wins/losses {v['after_better_signs']}/{v['before_better_signs']}."
        )
    atomic_write_text(report / "REPORT.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
