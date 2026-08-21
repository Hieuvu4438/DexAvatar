from __future__ import annotations

import pytest

from signal4d.extensions.v6_uqdiff.bootstrap_author import paired_sign_bootstrap


def test_paired_sign_bootstrap_is_deterministic_and_ignores_unpaired_nan() -> None:
    candidate = {
        "a": {"metric": 1.0},
        "b": {"metric": 2.0},
        "c": {"metric": float("nan")},
    }
    baseline = {
        "a": {"metric": 2.0},
        "b": {"metric": 4.0},
        "c": {"metric": 1.0},
    }
    first = paired_sign_bootstrap(candidate, baseline, "metric", replicates=100, seed=7)
    second = paired_sign_bootstrap(candidate, baseline, "metric", replicates=100, seed=7)
    assert first == second
    assert first["eligible_signs"] == 2
    assert first["mean_delta_mm"] == pytest.approx(-1.5)
    assert first["ci95_percentile_mm"] == [-2.0, -1.0]
