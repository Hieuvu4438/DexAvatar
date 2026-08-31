import csv

import pytest

from dcg_sign4d.evaluation.compare import compare_per_clip


def write(path, values):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "metric"])
        writer.writeheader()
        for clip_id, value in values.items():
            writer.writerow({"clip_id": clip_id, "metric": value})


def test_comparison_is_explicitly_clip_sensitivity_without_signers(tmp_path):
    baseline, candidate = tmp_path / "b.csv", tmp_path / "c.csv"
    write(baseline, {"a": 2, "b": 3})
    write(candidate, {"a": 1, "b": 2})
    result = compare_per_clip(
        baseline,
        candidate,
        metrics=("metric",),
        output_path=tmp_path / "comparison.json",
        replicates=100,
    )
    assert result["cluster_unit"].startswith("clip_sensitivity")
    assert result["metrics"]["metric"]["mean_delta"] == -1


def test_comparison_rejects_missing_metric_before_bootstrap(tmp_path):
    baseline, candidate = tmp_path / "b.csv", tmp_path / "c.csv"
    write(baseline, {"a": 2, "b": 3})
    write(candidate, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="requested metric columns are missing: absent"):
        compare_per_clip(
            baseline,
            candidate,
            metrics=("absent",),
            output_path=tmp_path / "comparison.json",
            replicates=100,
        )


def test_comparison_rejects_unpaired_missing_values(tmp_path):
    baseline, candidate = tmp_path / "b.csv", tmp_path / "c.csv"
    write(baseline, {"a": 2, "b": 3})
    write(candidate, {"a": 1, "b": ""})
    with pytest.raises(ValueError, match="valid coverage differs"):
        compare_per_clip(
            baseline,
            candidate,
            metrics=("metric",),
            output_path=tmp_path / "comparison.json",
            replicates=100,
        )
