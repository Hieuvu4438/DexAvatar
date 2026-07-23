import numpy as np

from phase2_refiner.calibrate import calibration_metrics


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
