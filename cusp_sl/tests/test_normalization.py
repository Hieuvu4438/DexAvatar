import numpy as np
import torch

from cusp_sl.normalization import ResidualNormalizer
from cusp_sl.training import binary_calibration_metrics


def test_residual_normalizer_round_trip(tmp_path):
    path = tmp_path / "statistics.npz"
    mean = np.linspace(-0.1, 0.1, 51 * 3, dtype=np.float32).reshape(51, 3)
    std = np.linspace(0.01, 0.2, 51 * 3, dtype=np.float32).reshape(51, 3)
    np.savez(path, mean=mean, std=std)
    normalizer = ResidualNormalizer.from_path(path)
    value = torch.randn(2, 4, 51, 3)
    torch.testing.assert_close(
        normalizer.denormalize(normalizer.normalize(value)), value
    )
    assert len(normalizer.sha256) == 64


def test_identity_statistics_preserve_unsupported_coordinates():
    normalizer = ResidualNormalizer()
    value = torch.randn(1, 2, 51, 3)
    torch.testing.assert_close(normalizer.denormalize(value), value)


def test_calibration_metrics_report_discrimination_and_prevalence():
    metrics = binary_calibration_metrics(
        np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0, 0, 1, 1])
    )
    assert metrics["positive_prevalence"] == 0.5
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["balanced_accuracy_at_0.5"] == 1.0
    assert metrics["log_loss"] < 0.2
