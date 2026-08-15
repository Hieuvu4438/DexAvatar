import numpy as np
import torch

from phase2_refiner.calibrate import calibration_gate, calibration_metrics
from phase2_refiner.losses.uncertainty import regional_worst_decile_ranking_loss


def test_calibration_recovers_variance_scale() -> None:
    rng = np.random.default_rng(5)
    expected_sigma = np.linspace(0.1, 1.0, 1000)
    error = np.abs(rng.normal(scale=expected_sigma))
    log_variance = np.log(np.square(expected_sigma)) - np.log(4.0)
    metrics = calibration_metrics(error, log_variance)
    assert metrics["observations"] == 1000
    assert metrics["spearman"] > 0.35
    assert metrics["worst_decile_auc"] > 0.7
    assert metrics["nll_after"] < metrics["nll_before"]


def test_gate_fails_without_real_residual_and_u0_comparator() -> None:
    metrics = {
        "spearman": 0.9,
        "worst_decile_auc": 0.9,
        "risk_monotonic": True,
        "nll_before": 1.0,
        "nll_after": 0.5,
    }
    gate = calibration_gate(metrics)
    assert not gate["passed"]
    assert not gate["checks"]["source_and_signer_disjoint_real_residual"]


def test_gate_requires_a_causal_uncertainty_feedback_gain() -> None:
    metrics = {
        "spearman": 0.9,
        "worst_decile_auc": 0.9,
        "risk_monotonic": True,
        "nll_before": 1.0,
        "nll_after": 0.5,
    }
    reconstruction = {
        "u1_corrupt": 0.8,
        "u0_corrupt": 1.0,
        "u1_clean": 0.9,
        "u0_clean": 0.9,
    }
    no_effect = {
        "feedback_corrupt": 0.8,
        "no_feedback_corrupt": 0.8,
        "feedback_clean": 0.9,
        "no_feedback_clean": 0.9,
    }
    effect = {**no_effect, "feedback_corrupt": 0.7}

    failed = calibration_gate(
        metrics,
        u0_nll=0.6,
        reconstruction=reconstruction,
        intervention=no_effect,
        valid_real_residual=True,
    )
    passed = calibration_gate(
        metrics,
        u0_nll=0.6,
        reconstruction=reconstruction,
        intervention=effect,
        valid_real_residual=True,
    )

    assert not failed["passed"]
    assert passed["passed"]


def test_worst_decile_ranking_loss_drives_high_error_variance_up() -> None:
    error = torch.linspace(0.0, 1.0, 2 * 3 * 51).reshape(2, 3, 51)
    log_variance = torch.zeros_like(error, requires_grad=True)
    valid = torch.ones_like(error, dtype=torch.bool)
    loss = regional_worst_decile_ranking_loss(error, log_variance, valid)
    loss.backward()
    assert torch.isfinite(loss)
    worst = error >= torch.quantile(error, 0.9)
    assert log_variance.grad[worst].mean() < 0.0
