from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load(path: str | Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip_id = row["clip_id"]
            if clip_id in rows:
                raise ValueError(f"duplicate clip_id in {path}: {clip_id}")
            numeric: dict[str, float] = {}
            for key, value in row.items():
                if key == "clip_id":
                    continue
                if value in (None, ""):
                    numeric[key] = float("nan")
                    continue
                try:
                    numeric[key] = float(value)
                except ValueError:
                    continue
            rows[clip_id] = numeric
    return rows


def paired_sign_bootstrap(
    candidate: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    metric: str,
    *,
    replicates: int = 10_000,
    seed: int = 12_345,
) -> dict[str, object]:
    if candidate.keys() != baseline.keys():
        raise ValueError("candidate and baseline clip IDs differ")
    clip_ids = sorted(candidate)
    delta = np.asarray(
        [candidate[clip][metric] - baseline[clip][metric] for clip in clip_ids],
        dtype=np.float64,
    )
    eligible = np.isfinite(delta)
    delta = delta[eligible]
    if not len(delta):
        raise ValueError(f"metric has no paired finite signs: {metric}")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(replicates, len(delta)), replace=True).mean(axis=1)
    return {
        "metric": metric,
        "unit": "mm",
        "estimand": "equal_weight_paired_sign_macro_delta_candidate_minus_baseline",
        "eligible_signs": int(len(delta)),
        "mean_delta_mm": float(delta.mean()),
        "median_delta_mm": float(np.median(delta)),
        "ci95_percentile_mm": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "bootstrap_probability_nonnegative": float(np.mean(samples >= 0)),
        "improved_signs": int(np.sum(delta < 0)),
        "worse_signs": int(np.sum(delta > 0)),
        "equal_signs": int(np.sum(delta == 0)),
        "replicates": replicates,
        "seed": seed,
    }


def run(
    candidate_csv: str,
    baseline_csv: str,
    metrics: list[str],
    output: str,
    *,
    replicates: int = 10_000,
    seed: int = 12_345,
) -> dict[str, object]:
    candidate = _load(candidate_csv)
    baseline = _load(baseline_csv)
    report = {
        "schema_version": "1.0",
        "candidate_csv": str(Path(candidate_csv)),
        "baseline_csv": str(Path(baseline_csv)),
        "results": [
            paired_sign_bootstrap(
                candidate,
                baseline,
                metric,
                replicates=replicates,
                seed=seed,
            )
            for metric in metrics
        ],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="signal4d-v6-bootstrap-author")
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12_345)
    args = parser.parse_args()
    report = run(
        args.candidate_csv,
        args.baseline_csv,
        args.metric,
        args.output,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
