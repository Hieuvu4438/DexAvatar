import json

import numpy as np
import pytest

from phase2_refiner.sign_domain_evaluation import (
    _lineage_guard,
    clip_region_error,
    select_thresholds,
    summarize,
)


def _record(dataset: str, baseline: float, candidate: float, probability: float):
    return {
        "clip_id": f"{dataset}-{baseline}",
        "dataset": dataset,
        "baseline_error": np.full((2, 51), baseline, dtype=np.float32),
        "candidate_error": np.full((2, 51), candidate, dtype=np.float32),
        "valid": np.ones((2, 51), dtype=bool),
        "benefit_probability": np.full((2, 3), probability, dtype=np.float32),
    }


def test_clip_region_error_skips_empty_region() -> None:
    record = _record("SOKE", 10.0, 5.0, 0.9)
    record["valid"][:, 21:36] = False
    assert clip_region_error(record, "lhand", 0.5) is None


def test_summarize_reports_eligible_clips_and_abstention() -> None:
    records = [_record("SOKE", 10.0, 5.0, 0.4)]
    summary = summarize(records, {name: 0.5 for name in ("ubody", "lhand", "rhand")})
    assert summary["SOKE"]["ubody"]["eligible_clips"] == 1
    assert summary["SOKE"]["ubody"]["prediction_macro_clip_deg"] == 10.0
    assert summary["SOKE"]["ubody"]["accepted_group_frames"] == 0


def test_threshold_selection_protects_worst_domain() -> None:
    records = [
        _record("SOKE", 10.0, 5.0, 0.8),
        _record("SignAvatars", 10.0, 20.0, 0.6),
    ]
    selected, audit = select_thresholds(records, [0.0, 0.7, 1.0])
    assert selected == {"ubody": 0.7, "lhand": 0.7, "rhand": 0.7}
    assert audit["ubody"]["selected"]["worst_domain_ratio"] == 1.0


def test_test_evaluation_requires_test_in_lineage(tmp_path) -> None:
    path = tmp_path / "lineage.json"
    path.write_text(
        json.dumps(
            {
                "decision": "PASS",
                "sgnify_training_or_selection_reads": 0,
                "source_group_overlaps": {},
                "manifests": {"train": {}, "validation": {}, "calibration": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="containing test"):
        _lineage_guard(path, require_test=True)
