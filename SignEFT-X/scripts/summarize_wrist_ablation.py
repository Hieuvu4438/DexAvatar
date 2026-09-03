#!/usr/bin/env python3
"""Summarize locked, one-degree, and unrestricted-wrist ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signeft.io_utils import atomic_write_json, atomic_write_text, sha256_file


ROOT = Path("outputs/ablation_wrist")
RUNS = Path("_archive/research_history/runs")
PA = {name: ROOT / "evaluation" / f"{name}.json" for name in ("locked", "one_degree", "free")}
OFFICIAL = {
    "locked": RUNS / "paper_ablation_native_radius12_full57/metrics/official_result.json",
    "one_degree": RUNS / "paper_ablation_native_unlocked_wrist_full57/metrics/official_result.json",
    "free": RUNS / "paper_ablation_native_free_wrist_full57/metrics/official_result.json",
}


def main() -> None:
    pa = {key: json.loads(value.read_text()) for key, value in PA.items()}
    official = {key: json.loads(value.read_text())["metrics_mm"] for key, value in OFFICIAL.items()}
    rng = np.random.default_rng(20260904)
    stats = {}
    for variant in ("one_degree", "free"):
        signs = sorted(pa["locked"]["per_sign"])
        delta = np.asarray([
            pa[variant]["per_sign"][sign]["metrics"]["pa_mpvpe_hand"]
            - pa["locked"]["per_sign"][sign]["metrics"]["pa_mpvpe_hand"]
            for sign in signs
        ])
        boot = delta[rng.integers(0, len(signs), size=(100_000, len(signs)))].mean(1)
        stats[variant] = {
            "delta_pa_hand_micro_mm": pa[variant]["micro_average_mm"]["pa_mpvpe_hand"] - pa["locked"]["micro_average_mm"]["pa_mpvpe_hand"],
            "paired_sign_mean_delta_mm": float(delta.mean()),
            "paired_sign_bootstrap_95ci_mm": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "variant_better_signs": int((delta < -1e-9).sum()),
            "locked_better_signs": int((delta > 1e-9).sum()),
        }
    summaries = {
        "locked": RUNS / "paper_ablation_native_radius12_full57/refinement_summary.json",
        "one_degree": RUNS / "paper_ablation_native_unlocked_wrist_full57/refinement_summary.json",
        "free": RUNS / "paper_ablation_native_free_wrist_full57/refinement_summary.json",
    }
    acceptance = {key: json.loads(value.read_text()) for key, value in summaries.items()}
    result = {
        "schema_version": "signeft.wrist-ablation-report.v1",
        "signs": 57, "frames": 1493, "bootstrap_samples": 100_000,
        "bootstrap_unit": "sign", "seed": 20260904,
        "rows": {
            name: {
                "official_mm": official[name],
                "pa_mm": pa[name]["micro_average_mm"],
                "accepted_frames": acceptance[name]["accepted"],
                "acceptance_rate": acceptance[name]["acceptance_rate"],
                "pa_sha256": sha256_file(PA[name]),
                "official_sha256": sha256_file(OFFICIAL[name]),
            } for name in PA
        },
        "paired_vs_locked": stats,
    }
    atomic_write_json(ROOT / "report/comparison.json", result)
    lines = [
        "# Wrist-locking ablation", "",
        "All rows use 57 signs / 1,493 frames. Free uses a 180-degree SO(3) radius, which spans the complete geodesic range; all other settings equal the one-degree run.", "",
        "| Wrist state | Accepted frames | All | UBody | UBody (-F) | UBody (-H) | LHand | RHand | PA hands |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"locked": "Locked", "one_degree": "Up to 1 degree", "free": "Free (180 degrees)"}
    for name in ("free", "one_degree", "locked"):
        o, p = official[name], pa[name]["micro_average_mm"]
        lines.append(
            f"| {labels[name]} | {acceptance[name]['accepted']}/1493 | {o['tr all']:.4f} | "
            f"{o['tr above pelvis upper body']:.4f} | {o['tr above pelvis minus face']:.4f} | "
            f"{o['tr above pelvis minus head']:.4f} | {o['tr left hand']:.4f} | "
            f"{o['tr right hand']:.4f} | {p['pa_mpvpe_hand']:.4f} |"
        )
    lines += ["", "Paired PA-hand effects versus locked (variant minus locked; positive is worse):", ""]
    for name in ("one_degree", "free"):
        value = stats[name]
        lo, hi = value["paired_sign_bootstrap_95ci_mm"]
        lines.append(
            f"- {labels[name]}: {value['delta_pa_hand_micro_mm']:+.4f} mm; paired sign 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}]; variant/locked wins "
            f"{value['variant_better_signs']}/{value['locked_better_signs']}."
        )
    atomic_write_text(ROOT / "report/REPORT.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
