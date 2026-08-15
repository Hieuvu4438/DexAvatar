import json
from pathlib import Path

import numpy as np
import pytest

from phase2_refiner.data.audit_formal_phase2r import (
    FORMAL_CONTRACT_VERSION,
    audit_formal_manifests,
    validate_formal_audit_report,
)
from phase2_refiner.data.cache_schema import (
    PHASE2R_SEMANTIC_CONTRACT,
    save_cache_clip,
)
from phase2_refiner.tests.test_cache import make_clip


DIGEST = "a" * 64


def _formal_clip(group: str):
    clip = make_clip()
    clip.clip_id = group
    clip.semantic_contract_version = PHASE2R_SEMANTIC_CONTRACT
    clip.target_axis_angle = clip.init_axis_angle.copy()
    clip.target_rotation_valid = np.ones((5, 51), dtype=bool)
    clip.target_quality = np.ones((5, 51), dtype=np.float32)
    clip.detector_present = np.ones((5, 51), dtype=bool)
    clip.track_valid = np.ones((5, 51), dtype=bool)
    clip.initializer_component = np.asarray(["portable_whole_body"] * 5)
    clip.source_sha256 = np.asarray([DIGEST] * 5)
    clip.metadata_json = json.dumps(
        {
            "source_group": group,
            "signer_id": group,
            "coordinate_policy": {"keypoints_2d": "normalized_image_0_to_1"},
            "initializer_contract": {
                "portable": True,
                "frozen": True,
                "benchmark_conditioning": False,
                "provider_id": "portable-a1r-v1",
                "weights_sha256": DIGEST,
                "configuration_sha256": DIGEST,
                "provider_code_sha256": DIGEST,
            },
            "target_contract": {
                "independent_from_initializer": True,
                "not_same_view_2d_only": True,
                "initializer_outputs_used_false": True,
                "release_benchmark_excluded": True,
                "audit_passed": True,
                "license_verified": True,
                "exact_frame_count_match": True,
                "shared_geometry_decode": True,
                "geometry": "multiview_3d",
                "provider_id": "held-out-multiview-v1",
                "source_sha256": [DIGEST],
                "provider_code_sha256": DIGEST,
                "audit_report_sha256": DIGEST,
            },
        }
    )
    return clip


def _write_split(root: Path, split: str, group: str) -> Path:
    clip_path = root / f"{split}.npz"
    save_cache_clip(clip_path, _formal_clip(group))
    manifest = root / f"{split}.json"
    manifest.write_text(json.dumps({"clips": [str(clip_path)]}) + "\n")
    return manifest


def test_formal_audit_passes_and_binds_exact_manifests(tmp_path: Path) -> None:
    manifests = {
        "train": _write_split(tmp_path, "train", "signer-a"),
        "validation": _write_split(tmp_path, "validation", "signer-b"),
        "calibration": _write_split(tmp_path, "calibration", "signer-c"),
    }
    report = audit_formal_manifests(manifests)
    assert report["contract_version"] == FORMAL_CONTRACT_VERSION
    assert report["passed"]
    report_path = tmp_path / "formal-audit.json"
    report_path.write_text(json.dumps(report) + "\n")

    validated = validate_formal_audit_report(report_path, manifests)

    assert validated["passed"]


def test_formal_audit_rejects_proxy_contract_and_split_overlap(
    tmp_path: Path,
) -> None:
    train = _write_split(tmp_path, "train", "same-signer")
    validation = _write_split(tmp_path, "validation", "same-signer")
    clip_path = tmp_path / "train.npz"
    clip = _formal_clip("same-signer")
    metadata = json.loads(clip.metadata_json)
    metadata.pop("target_contract")
    metadata["phase2r_adapter"] = {"target_independence": "NO: proxy"}
    clip.metadata_json = json.dumps(metadata)
    save_cache_clip(clip_path, clip)

    report = audit_formal_manifests({"train": train, "validation": validation})

    assert not report["passed"]
    assert not report["checks"]["all_clips_satisfy_formal_contract"]
    assert not report["checks"]["source_group_disjoint_splits"]
    report_path = tmp_path / "failed.json"
    report_path.write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="did not pass"):
        validate_formal_audit_report(report_path, {"train": train})


def test_formal_audit_rejects_target_audit_candidate(tmp_path: Path) -> None:
    manifest = _write_split(tmp_path, "train", "signer-a")
    clip_path = tmp_path / "train.npz"
    clip = _formal_clip("signer-a")
    metadata = json.loads(clip.metadata_json)
    metadata["target_contract"]["audit_passed"] = False
    metadata["target_contract"]["audit_report_sha256"] = ""
    clip.metadata_json = json.dumps(metadata)
    save_cache_clip(clip_path, clip)

    report = audit_formal_manifests({"train": manifest})

    assert report["passed"] is False
    failures = report["failures"][0]["failures"]
    assert "target_contract.audit_passed must be true" in failures


def test_formal_audit_rejects_signer_overlap_with_distinct_sources(
    tmp_path: Path,
) -> None:
    train = _write_split(tmp_path, "train", "source-a")
    validation = _write_split(tmp_path, "validation", "source-b")
    for clip_name in ("train", "validation"):
        path = tmp_path / f"{clip_name}.npz"
        clip = _formal_clip(f"source-{'a' if clip_name == 'train' else 'b'}")
        metadata = json.loads(clip.metadata_json)
        metadata["signer_id"] = "same-signer"
        clip.metadata_json = json.dumps(metadata)
        save_cache_clip(path, clip)

    report = audit_formal_manifests({"train": train, "validation": validation})

    assert report["checks"]["source_group_disjoint_splits"] is True
    assert report["checks"]["signer_disjoint_splits"] is False
    assert report["signer_overlaps"]["train__validation"] == ["same-signer"]


def test_formal_audit_consolidates_multiple_failures_per_clip(tmp_path: Path) -> None:
    clip = _formal_clip("source-a")
    clip.metadata_json = json.dumps(
        {"coordinate_policy": {"keypoints_2d": "normalized_image_0_to_1"}}
    )
    path = tmp_path / "broken.npz"
    save_cache_clip(path, clip)
    manifest = tmp_path / "broken.json"
    manifest.write_text(json.dumps({"clips": [str(path)]}))

    report = audit_formal_manifests({"train": manifest})

    assert report["failure_count"] == 1
    failures = report["failures"][0]["failures"]
    assert "missing initializer_contract" in failures
    assert "missing target_contract" in failures
    assert "metadata.source_group is missing" in failures
    assert "metadata.signer_id is missing" in failures
