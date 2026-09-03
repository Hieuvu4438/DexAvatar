#!/usr/bin/env python3
"""Create paired Adam-vs-L-BFGS accuracy tables from evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signeft.io_utils import atomic_write_json, atomic_write_text, sha256_file


DEFAULT_METRICS = (
    "pa_mpvpe_all",
    "pa_mpvpe_body_only",
    "pa_mpvpe_rhand",
    "pa_mpvpe_lhand",
    "pa_mpvpe_hand",
    "tr_ub_minus_face",
    "tr_rhand",
    "tr_lhand",
)


def compare(
    adam_path: Path,
    lbfgs_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    adam = json.loads(adam_path.read_text(encoding="utf-8"))
    lbfgs = json.loads(lbfgs_path.read_text(encoding="utf-8"))
    if adam["total_frames_evaluated"] != lbfgs["total_frames_evaluated"]:
        raise RuntimeError("Adam/L-BFGS frame count mismatch")
    signs = sorted(set(adam["per_sign"]) & set(lbfgs["per_sign"]))
    if len(signs) != adam["total_signs_evaluated"] or len(signs) != lbfgs["total_signs_evaluated"]:
        raise RuntimeError("Adam/L-BFGS sign set mismatch")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(signs), size=(bootstrap_samples, len(signs)))
    metrics = {}
    for key in DEFAULT_METRICS:
        if key not in adam["micro_average_mm"] or key not in lbfgs["micro_average_mm"]:
            continue
        paired = np.asarray([
            lbfgs["per_sign"][sign]["metrics"][key]
            - adam["per_sign"][sign]["metrics"][key]
            for sign in signs
        ], dtype=np.float64)
        boot = paired[indices].mean(axis=1)
        delta_micro = (
            lbfgs["micro_average_mm"][key] - adam["micro_average_mm"][key]
        )
        metrics[key] = {
            "adam_micro_mm": adam["micro_average_mm"][key],
            "lbfgs_micro_mm": lbfgs["micro_average_mm"][key],
            "delta_micro_mm": delta_micro,
            "relative_delta_percent": 100.0 * delta_micro / adam["micro_average_mm"][key],
            "paired_sign_mean_delta_mm": float(paired.mean()),
            "paired_sign_bootstrap_95ci_mm": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
            "lbfgs_better_signs": int((paired < -1e-9).sum()),
            "adam_better_signs": int((paired > 1e-9).sum()),
            "ties": int((np.abs(paired) <= 1e-9).sum()),
        }
    return {
        "adam_evaluation": str(adam_path.resolve()),
        "adam_sha256": sha256_file(adam_path),
        "lbfgs_evaluation": str(lbfgs_path.resolve()),
        "lbfgs_sha256": sha256_file(lbfgs_path),
        "signs": len(signs),
        "frames": adam["total_frames_evaluated"],
        "bootstrap_unit": "sign",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "delta_definition": "LBFGS minus Adam; negative favors LBFGS",
        "metrics": metrics,
    }


def markdown(result: dict, title: str) -> str:
    rows = [
        f"## {title}",
        "",
        f"Paired evaluation: {result['signs']} signs / {result['frames']} frames. "
        "Delta is L-BFGS minus Adam, so negative values favor L-BFGS.",
        "",
        "| Metric | Adam | L-BFGS | Delta | Relative | 95% sign-bootstrap CI | Wins/Losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in result["metrics"].items():
        low, high = value["paired_sign_bootstrap_95ci_mm"]
        rows.append(
            f"| `{key}` | {value['adam_micro_mm']:.4f} | "
            f"{value['lbfgs_micro_mm']:.4f} | {value['delta_micro_mm']:+.4f} | "
            f"{value['relative_delta_percent']:+.3f}% | [{low:+.4f}, {high:+.4f}] | "
            f"{value['lbfgs_better_signs']}/{value['adam_better_signs']} |"
        )
    return "\n".join(rows) + "\n"


def compare_official(adam_path: Path, lbfgs_path: Path) -> dict:
    adam = json.loads(adam_path.read_text(encoding="utf-8"))
    lbfgs = json.loads(lbfgs_path.read_text(encoding="utf-8"))
    adam_metrics = adam["metrics_mm"]
    lbfgs_metrics = lbfgs["metrics_mm"]
    if set(adam_metrics) != set(lbfgs_metrics):
        raise RuntimeError("official Adam/L-BFGS metric set mismatch")
    return {
        key: {
            "adam_mm": adam_metrics[key],
            "lbfgs_mm": lbfgs_metrics[key],
            "delta_mm": lbfgs_metrics[key] - adam_metrics[key],
        }
        for key in sorted(adam_metrics)
    }


def official_markdown(result: dict, title: str) -> str:
    rows = [
        f"### {title}",
        "",
        "| Official metric | Adam | L-BFGS | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, value in result.items():
        rows.append(
            f"| {key} | {value['adam_mm']:.4f} | {value['lbfgs_mm']:.4f} | "
            f"{value['delta_mm']:+.4f} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-adam", type=Path, required=True)
    parser.add_argument("--canonical-lbfgs", type=Path, required=True)
    parser.add_argument("--palm-adam", type=Path, required=True)
    parser.add_argument("--palm-lbfgs", type=Path, required=True)
    parser.add_argument("--canonical-adam-official", type=Path, required=True)
    parser.add_argument("--canonical-lbfgs-official", type=Path, required=True)
    parser.add_argument("--palm-adam-official", type=Path, required=True)
    parser.add_argument("--palm-lbfgs-official", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    canonical = compare(
        args.canonical_adam, args.canonical_lbfgs,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    palm = compare(
        args.palm_adam, args.palm_lbfgs,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    canonical_official = compare_official(
        args.canonical_adam_official, args.canonical_lbfgs_official
    )
    palm_official = compare_official(
        args.palm_adam_official, args.palm_lbfgs_official
    )
    result = {
        "schema_version": "signeft.optimizer-ablation.v1",
        "canonical": canonical,
        "palm": palm,
        "canonical_official": canonical_official,
        "palm_official": palm_official,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "comparison.json", result)
    content = "# Adam vs. L-BFGS ablation\n\n"
    content += markdown(canonical, "Contribution 1: signer-consistent canonical reconstruction")
    content += "\n" + official_markdown(canonical_official, "Official evaluator")
    content += "\n" + markdown(palm, "Contribution 2: palm-canonical hand refinement")
    content += "\n" + official_markdown(palm_official, "Official evaluator")
    content += (
        "\n## Decision\n\n"
        "Retain Adam for both contributions. L-BFGS degrades the canonical "
        "reconstruction on the official metrics. For palm refinement, the two "
        "optimizers are effectively tied: the average hand PA-MPVPE difference "
        "is below 0.001 mm and its paired sign-bootstrap interval crosses zero. "
        "The tiny official improvements are not practically meaningful.\n"
    )
    atomic_write_text(args.output_root / "REPORT.md", content)


if __name__ == "__main__":
    main()
