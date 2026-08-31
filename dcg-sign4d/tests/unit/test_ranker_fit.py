from __future__ import annotations

import csv

import pytest

from dcg_sign4d.inference.ranker_fit import fit_ranker, load_frozen_ranker


def test_validation_ranker_fit_and_strict_load(tmp_path):
    source = tmp_path / "ranker.csv"
    rows = []
    for clip in ("a", "b", "c"):
        for hypothesis, observation in enumerate((-3.0, -2.0, -1.0)):
            rows.append(
                {
                    "clip_id": clip,
                    "hypothesis_id": hypothesis,
                    "split": "validation",
                    "validation_error": 3 - hypothesis,
                    "observation": observation,
                    "contact": 0,
                    "event": 0,
                    "motion": 0,
                }
            )
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "ranker.json"
    report = fit_ranker(
        source,
        output,
        steps=50,
        learning_rate=0.1,
        l2_weight=0,
        minimum_pair_accuracy=1,
        seed=4,
        development_only=True,
        config_sha256="a" * 64,
    )
    assert report["pairwise_accuracy"] == 1
    with pytest.raises(PermissionError, match="development ranker"):
        load_frozen_ranker(output)
    weights, _ = load_frozen_ranker(output, allow_development=True)
    assert weights.observation > 0


def test_ranker_rejects_test_rows(tmp_path):
    source = tmp_path / "ranker.csv"
    source.write_text(
        "clip_id,hypothesis_id,split,validation_error,observation,contact,event,motion\n"
        "a,0,test,1,-1,0,0,0\n"
        "a,1,test,2,-2,0,0,0\n",
        "utf-8",
    )
    with pytest.raises(ValueError, match="validation rows only"):
        fit_ranker(
            source,
            tmp_path / "ranker.json",
            steps=1,
            learning_rate=0.1,
            l2_weight=0,
            minimum_pair_accuracy=0,
            seed=1,
            development_only=True,
            config_sha256="a" * 64,
        )
