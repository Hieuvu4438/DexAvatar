import pytest

from dcg_sign4d.evaluation.bootstrap import cluster_bootstrap, paired_cluster_bootstrap


def test_cluster_bootstrap_is_paired_and_deterministic():
    baseline = {"a": 3.0, "b": 4.0, "c": 5.0, "d": 8.0}
    candidate = {"a": 2.0, "b": 3.0, "c": 4.0, "d": 7.0}
    clusters = {"a": "s1", "b": "s1", "c": "s2", "d": "s2"}
    first = paired_cluster_bootstrap(baseline, candidate, clusters, replicates=100, seed=9)
    second = paired_cluster_bootstrap(baseline, candidate, clusters, replicates=100, seed=9)
    assert first == second
    assert first["mean_delta"] == -1.0
    assert first["ci95_low"] == first["ci95_high"] == -1.0


def test_unknown_signer_fails_closed():
    with pytest.raises(ValueError, match="unknown"):
        paired_cluster_bootstrap(
            {"a": 1.0, "b": 2.0},
            {"a": 0.0, "b": 1.0},
            {"a": "unknown", "b": "unknown"},
            replicates=100,
        )


def test_unpaired_point_estimate_remains_clip_macro_with_unequal_cluster_sizes():
    values = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 10.0}
    clusters = {"a": "s1", "b": "s1", "c": "s1", "d": "s2"}
    result = cluster_bootstrap(values, clusters, replicates=100, seed=3)
    assert result["mean"] == 2.5
    assert result["clusters"] == 2
    assert result["items"] == 4
