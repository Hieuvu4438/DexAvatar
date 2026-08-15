import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from phase2_refiner.data.materialize_signavatars_targets import (
    attach_target,
    validate_license_acceptance,
    validate_target_audit,
)
from phase2_refiner.tests.test_cache import make_clip


def test_signavatars_target_uses_exact_bound_source_frames(tmp_path: Path) -> None:
    clip = make_clip(3)
    clip.source_paths = np.asarray(
        [f"/videos/example.mp4#frame={index}" for index in (1, 3, 5)]
    )
    parameters = np.zeros((6, 169), dtype=np.float32)
    for index in range(6):
        parameters[index, 3:156] = index
    annotation = tmp_path / "annotation.pkl"
    with annotation.open("wb") as handle:
        pickle.dump(
            {
                "unsmooth_smplx": parameters,
                "total_valid_index": np.ones(6, dtype=bool),
                "left_valid": np.asarray([1, 1, 1, 0, 1, 1], dtype=bool),
                "right_valid": np.ones(6, dtype=bool),
            },
            handle,
        )
    audit = tmp_path / "audit.json"
    audit.write_text('{"passed":true}\n')
    license_record = tmp_path / "license.json"
    license_record.write_text('{"accepted":true}\n')

    updated = attach_target(clip, annotation, "unsmooth_smplx", audit, license_record)

    np.testing.assert_array_equal(updated.target_axis_angle[:, 0, 0], [1, 3, 5])
    assert updated.target_axis_angle.shape == (3, 51, 3)
    assert updated.target_rotation_valid[0].all()
    assert not updated.target_rotation_valid[1, 21:36].any()
    metadata = json.loads(updated.metadata_json)
    assert metadata["target_contract"]["geometry"] == "released_3d_smplx_pose"
    assert metadata["target_contract"]["independent_from_initializer"]
    assert metadata["target_contract"]["audit_passed"] is True
    assert metadata["target_contract"]["shared_geometry_decode"] is True
    assert metadata["target_contract"]["exact_frame_count_match"] is False


def test_signavatars_audit_candidate_is_explicitly_ineligible(
    tmp_path: Path,
) -> None:
    clip = make_clip(2)
    clip.source_paths = np.asarray(
        [f"/videos/example.mp4#frame={index}" for index in (0, 1)]
    )
    parameters = np.zeros((2, 169), dtype=np.float32)
    annotation = tmp_path / "annotation.pkl"
    with annotation.open("wb") as handle:
        pickle.dump(
            {
                "unsmooth_smplx": parameters,
                "total_valid_index": np.ones(2, dtype=bool),
                "left_valid": np.ones(2, dtype=bool),
                "right_valid": np.ones(2, dtype=bool),
            },
            handle,
        )
    license_record = tmp_path / "license.json"
    license_record.write_text('{"accepted":true}\n')

    updated = attach_target(
        clip,
        annotation,
        "unsmooth_smplx",
        None,
        license_record,
        audit_candidate=True,
        source_video_frame_count=2,
        source_video_fps=24.0,
        signer_id="signer-1",
    )

    metadata = json.loads(updated.metadata_json)
    assert metadata["target_contract"]["audit_passed"] is False
    assert metadata["target_contract"]["audit_report_sha256"] == ""
    assert metadata["target_contract"]["exact_frame_count_match"] is True
    assert metadata["signer_id"] == "signer-1"
    assert metadata["target_type"].endswith("audit_candidate")

    with pytest.raises(ValueError, match="frame-count mismatch"):
        attach_target(
            clip,
            annotation,
            "unsmooth_smplx",
            None,
            license_record,
            audit_candidate=True,
            source_video_frame_count=3,
            source_video_fps=24.0,
        )


def test_signavatars_license_record_is_fail_closed(tmp_path: Path) -> None:
    record = tmp_path / "license.json"
    record.write_text(
        json.dumps(
            {
                "dataset": "SignAvatars",
                "non_commercial_research_terms_accepted": False,
                "registrant_name": "Researcher",
                "registrant_email": "researcher@example.org",
                "access_granted_at": "2026-08-12",
            }
        )
    )
    with pytest.raises(ValueError, match="not accepted"):
        validate_license_acceptance(record)


def _valid_target_audit() -> dict:
    return {
        "audit_schema": "signavatars-target-audit-v1",
        "passed": True,
        "audited_clips": 100,
        "sample_seed": 20260812,
        "sample_manifest_sha256": "a" * 64,
        "stratified_by": [
            "signer",
            "hand_activity",
            "hand_size",
            "truncation",
            "motion",
        ],
        "catastrophic_failure_fraction": 0.03,
        "catastrophic_failure_fraction_by_region": {
            "body": 0.01,
            "left_hand": 0.04,
            "right_hand": 0.03,
        },
        "reviewer": "Researcher",
        "completed_at": "2026-08-12T20:00:00+07:00",
    }


def test_signavatars_target_audit_requires_all_regional_rates(tmp_path: Path) -> None:
    audit = _valid_target_audit()
    audit["catastrophic_failure_fraction_by_region"]["right_hand"] = 0.10
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="right_hand"):
        validate_target_audit(path)


def test_signavatars_target_audit_requires_declared_strata(tmp_path: Path) -> None:
    audit = _valid_target_audit()
    audit["stratified_by"].remove("truncation")
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="truncation"):
        validate_target_audit(path)
