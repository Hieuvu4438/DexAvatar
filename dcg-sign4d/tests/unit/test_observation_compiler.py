from __future__ import annotations

import json

import numpy as np

from dcg_sign4d.observations.cache import ObservationCache
from dcg_sign4d.observations.compiler import compile_calibrated_keypoint_caches
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def test_frozen_raw_scores_compile_to_calibrated_immutable_cache(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture video identity")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip",
                "video_path": str(video),
                "fps_native": 20,
                "frame_count": 2,
                "width": 16,
                "height": 16,
                "signer_id": "signer",
                "split": "train",
                "camera_id": "cam",
                "dataset_name": "fixture",
                "dataset_version": "v1",
                "license_id": "fixture-license",
            }
        )
        + "\n",
        "utf-8",
    )
    extractor_hash = "a" * 64
    raw_root = tmp_path / "raw"
    clip_root = raw_root / "clip"
    clip_root.mkdir(parents=True)
    raw_path = clip_root / "raw_keypoints.npz"
    np.savez_compressed(
        raw_path,
        frame_ids=np.array([0, 1], dtype=np.int64),
        timestamps_sec=np.array([0.0, 0.05], dtype=np.float64),
        keypoints_2d=np.array([[[1.0, 2.0]], [[0.0, 0.0]]], dtype=np.float32),
        raw_score=np.array([[2.0], [-3.0]], dtype=np.float32),
        keypoint_available=np.array([[True], [False]]),
        frame_available=np.array([True, False]),
    )
    (clip_root / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_sha256": file_sha256(raw_path),
                "extractor_checkpoint_sha256": extractor_hash,
            }
        ),
        "utf-8",
    )
    report_path = raw_root / "compilation_report.json"
    report_path.write_text(json.dumps({"development_only": False}), "utf-8")
    (raw_root / "COMPILATION_COMPLETE").write_text("complete\n", "utf-8")
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text(
        json.dumps(
            {
                "schema_version": "dcg_temperature_calibration_v1",
                "development_only": False,
                "temperature": 2.0,
                "gate_status": "PASS",
                "fit_split": "calibration",
                "input_transform": "scalar_as_binary_logit",
                "calibration_model_sha256": canonical_hash({"fixture": "calibrator"}),
            }
        ),
        "utf-8",
    )
    output = tmp_path / "calibrated"
    report = compile_calibrated_keypoint_caches(
        raw_root=raw_root,
        source_report_sha256=file_sha256(report_path),
        manifest_path=manifest,
        calibrator_path=calibrator,
        extractor={
            "name": "fixture",
            "version": "v1",
            "checkpoint_sha256": extractor_hash,
        },
        preprocessing={"topology": "fixture"},
        output=output,
        development_only=False,
    )
    row = report["per_clip"][0]
    restored = ObservationCache(output / "caches").load(row["cache_id"])
    assert 0 < float(restored.keypoint_reliability[0, 0, 0]) < 1
    assert float(restored.keypoint_reliability[0, 1, 0]) == 0
    assert not bool(restored.keypoint_valid[0, 1, 0])
    assert report["clips"] == 1 and report["frames"] == 2
    assert (output / "CALIBRATED_OBSERVATIONS_COMPLETE").is_file()
