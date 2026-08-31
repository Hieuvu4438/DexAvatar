import pytest
import torch

from cusp_sl.summarize_deterministic_restarts import (
    distribution,
    require_equal,
    validate_common_checkpoint,
)


def checkpoint(seed=43, step=10000):
    return {
        "training_seed": seed,
        "step": step,
        "model_kind": "deterministic_residual",
        "model": {"weight": torch.zeros(2, 3)},
        "best_validation_metric": 0.25,
        "config_sha256": "config",
        "reliability_checkpoint_sha256": "q",
        "residual_statistics_sha256": "stats",
    }


def test_validate_restart_accepts_legacy_declared_seed42():
    value = checkpoint(seed=None)
    result = validate_common_checkpoint(value, 42)
    assert result["seed"] == 42
    assert result["parameter_count"] == 6


def test_validate_restart_rejects_seed_and_budget_mismatch():
    with pytest.raises(ValueError, match="seed mismatch"):
        validate_common_checkpoint(checkpoint(seed=44), 43)
    with pytest.raises(ValueError, match="10,000"):
        validate_common_checkpoint(checkpoint(step=9999), 43)


def test_matched_invariants_and_distribution():
    assert require_equal([{"x": "same"}, {"x": "same"}], "x") == "same"
    with pytest.raises(ValueError, match="differs"):
        require_equal([{"x": 1}, {"x": 2}], "x")
    result = distribution([1.0, 2.0, 3.0])
    assert result == {"mean": 2.0, "sample_std": 1.0, "minimum": 1.0, "maximum": 3.0}
