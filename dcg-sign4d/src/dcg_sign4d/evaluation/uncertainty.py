"""GT-evaluation-only ranking and risk-coverage diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

FloatArray = NDArray[np.floating]


def risk_coverage_metrics(errors: FloatArray, uncertainty: FloatArray) -> dict[str, object]:
    if errors.ndim != 1 or uncertainty.shape != errors.shape or len(errors) < 2:
        raise ValueError("errors and uncertainty must be equal nontrivial vectors")
    if not np.isfinite(errors).all() or not np.isfinite(uncertainty).all():
        raise ValueError("risk-coverage values must be finite")
    order = np.argsort(uncertainty, kind="stable")
    ordered_error = errors[order]
    coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
    risk = np.cumsum(ordered_error) / np.arange(1, len(errors) + 1)
    correlation = spearmanr(errors, uncertainty).statistic
    return {
        "aurc": float(np.trapezoid(risk, coverage)),
        "error_uncertainty_spearman": float(correlation),
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }


def top1_oracle_metrics(
    error_by_hypothesis: FloatArray, ranking_score: FloatArray
) -> dict[str, float]:
    if error_by_hypothesis.ndim != 2:
        raise ValueError("hypothesis errors must be [clips,K]")
    if ranking_score.shape != error_by_hypothesis.shape:
        raise ValueError("ranking score must match hypothesis errors")
    selected = ranking_score.argmax(axis=1)
    top1 = error_by_hypothesis[np.arange(len(selected)), selected]
    oracle = error_by_hypothesis.min(axis=1)
    return {
        "top1_error": float(top1.mean()),
        "oracle_best_of_k_error": float(oracle.mean()),
        "selection_regret": float((top1 - oracle).mean()),
    }
