"""Calibrate and audit U1 log variance on a disjoint residual dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def calibration_metrics(error: np.ndarray, log_variance: np.ndarray) -> dict:
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    log_variance = np.asarray(log_variance, dtype=np.float64).reshape(-1)
    valid = np.isfinite(error) & np.isfinite(log_variance) & (error >= 0)
    error, log_variance = error[valid], log_variance[valid]
    if len(error) < 20:
        raise ValueError(
            f"Calibration needs at least 20 valid observations, got {len(error)}"
        )
    variance = np.exp(np.clip(log_variance, -20, 20))
    squared_error = np.square(error)
    variance_scale = float(np.mean(squared_error / np.maximum(variance, 1e-12)))
    variance_scale = max(variance_scale, 1e-12)
    calibrated_log_variance = log_variance + np.log(variance_scale)
    uncertainty = np.sqrt(np.exp(calibrated_log_variance))
    correlation = float(spearmanr(uncertainty, error).statistic)
    worst_threshold = float(np.quantile(error, 0.9))
    auc = float(roc_auc(error >= worst_threshold, uncertainty))
    order = np.argsort(uncertainty)
    risk_coverage = {}
    for coverage in (1.0, 0.9, 0.8, 0.7):
        retained = max(1, int(round(len(error) * coverage)))
        risk_coverage[f"{coverage:.1f}"] = float(error[order[:retained]].mean())
    nll_before = float(np.mean(0.5 * (squared_error / variance + log_variance)))
    calibrated_variance = np.exp(calibrated_log_variance)
    nll_after = float(
        np.mean(0.5 * (squared_error / calibrated_variance + calibrated_log_variance))
    )
    risks = list(risk_coverage.values())
    monotonic = all(
        risks[index + 1] <= risks[index] + 1e-12 for index in range(len(risks) - 1)
    )
    quantile_edges = np.quantile(uncertainty, np.linspace(0.0, 1.0, 11))
    calibration_gaps = []
    for low, high in zip(quantile_edges[:-1], quantile_edges[1:]):
        selected = (uncertainty >= low) & (uncertainty <= high)
        if selected.any():
            calibration_gaps.append(
                abs(float(uncertainty[selected].mean() - error[selected].mean()))
            )
    expected_calibration_error = float(np.mean(calibration_gaps))
    return {
        "observations": int(len(error)),
        "log_variance_offset": float(np.log(variance_scale)),
        "spearman": correlation,
        "worst_decile_auc": auc,
        "risk_coverage": risk_coverage,
        "risk_monotonic": monotonic,
        "expected_calibration_error": expected_calibration_error,
        "nll_before": nll_before,
        "nll_after": nll_after,
    }


def calibration_gate(metrics: dict, group: str = "all") -> dict:
    auc_threshold = 0.75 if "hand" in group.lower() else 0.70
    checks = {
        "spearman_at_least_0.35": metrics["spearman"] >= 0.35,
        f"worst_decile_auc_at_least_{auc_threshold:.2f}": metrics["worst_decile_auc"]
        >= auc_threshold,
        "risk_monotonic": bool(metrics["risk_monotonic"]),
        "calibrated_nll_improves": metrics["nll_after"] < metrics["nll_before"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        type=Path,
        required=True,
        help="NPZ containing error, log_variance, and optional group arrays",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    with np.load(args.residuals, allow_pickle=False) as data:
        if "error" not in data or "log_variance" not in data:
            raise ValueError("Residual NPZ must contain error and log_variance")
        error = data["error"]
        log_variance = data["log_variance"]
        all_metrics = calibration_metrics(error, log_variance)
        report = {"all": all_metrics, "gate": calibration_gate(all_metrics)}
        if "group" in data:
            groups = data["group"]
            group_metrics = {
                str(group): calibration_metrics(
                    error[groups == group], log_variance[groups == group]
                )
                for group in np.unique(groups)
            }
            report["groups"] = group_metrics
            report["group_gates"] = {
                group: calibration_gate(metrics, group)
                for group, metrics in group_metrics.items()
            }
            report["gate"]["passed"] = report["gate"]["passed"] and all(
                gate["passed"] for gate in report["group_gates"].values()
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
