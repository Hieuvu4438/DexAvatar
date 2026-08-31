import numpy as np
import pytest

from dcg_sign4d.evaluation.temporal import temporal_motion_metrics
from dcg_sign4d.evaluation.uncertainty import risk_coverage_metrics, top1_oracle_metrics


def test_temporal_identical_motion_has_zero_error_and_unit_ratios():
    time = np.linspace(0, 1, 16)
    value = np.stack((time, time**2, np.sin(time)), axis=-1)[:, None]
    result = temporal_motion_metrics(value, value, fps=15)
    assert result["velocity_error_mm_per_s"] == 0
    assert result["acceleration_error_mm_per_s2"] == 0
    assert result["jerk_error_mm_per_s3"] == 0
    assert result["spectral_l1_distance"] == 0
    assert result["motion_amplitude_ratio"] == pytest.approx(1)
    assert result["high_frequency_energy_ratio"] == pytest.approx(1)


def test_uncertainty_ranking_and_oracle_are_explicit():
    risk = risk_coverage_metrics(np.array([1.0, 3.0, 2.0]), np.array([0.1, 0.9, 0.5]))
    assert risk["risk"] == pytest.approx([1.0, 1.5, 2.0])
    result = top1_oracle_metrics(
        np.array([[2.0, 1.0], [3.0, 4.0]]), np.array([[0.1, 0.2], [0.5, 0.4]])
    )
    assert result["top1_error"] == pytest.approx(2.0)
    assert result["oracle_best_of_k_error"] == pytest.approx(2.0)
