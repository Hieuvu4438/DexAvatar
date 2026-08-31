import pytest

from cusp_sl.evaluate_development import clustered_delta_interval


def test_clustered_delta_interval_uses_paired_token_weighting():
    records = [
        {
            "source_group": "a",
            "tokens": 1,
            "base_degrees": 5.0,
            "method_degrees": 4.0,
        },
        {
            "source_group": "a",
            "tokens": 3,
            "base_degrees": 5.0,
            "method_degrees": 3.0,
        },
        {
            "source_group": "b",
            "tokens": 2,
            "base_degrees": 4.0,
            "method_degrees": 3.0,
        },
    ]
    result = clustered_delta_interval(
        records, "method_degrees", replicates=500, seed=42
    )
    assert result["clusters"] == 2
    assert result["delta_degrees"] == pytest.approx(-1.5)
    assert result["ci95_high_degrees"] < 0.0
    assert result["bootstrap_probability_improvement"] == 1.0


def test_clustered_delta_interval_requires_independent_groups():
    records = [
        {
            "source_group": "only",
            "tokens": 1,
            "base_degrees": 2.0,
            "method_degrees": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="at least two"):
        clustered_delta_interval(
            records, "method_degrees", replicates=10, seed=42
        )
