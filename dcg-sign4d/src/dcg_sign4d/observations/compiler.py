"""Compile calibrated immutable observations from frozen raw detector artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dcg_sign4d.data.manifest import load_manifest
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

from .cache import ObservationCache
from .calibration import load_frozen_temperature
from .raw import RawKeypointBatch
from .schema import ObservationBatch


def calibrate_raw_keypoints(
    raw: RawKeypointBatch,
    calibrator_path: str | Path,
    *,
    allow_development: bool,
    metadata: dict[str, Any],
) -> ObservationBatch:
    """Map scalar detector evidence to correctness probability via a frozen calibrator."""

    raw.validate()
    calibrator_path = Path(calibrator_path)
    payload = json.loads(calibrator_path.read_text("utf-8"))
    if payload.get("input_transform") != "scalar_as_binary_logit":
        raise ValueError("raw scalar calibration requires input_transform=scalar_as_binary_logit")
    scaler = load_frozen_temperature(calibrator_path, allow_development=allow_development)
    logits = torch.stack((torch.zeros_like(raw.raw_score), raw.raw_score), -1)
    with torch.no_grad():
        reliability = scaler(logits).softmax(-1)[..., 1]
    valid = raw.keypoint_available & raw.frame_available[:, None]
    reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
    keypoints = torch.where(
        valid[..., None], raw.keypoints_2d, torch.full_like(raw.keypoints_2d, float("nan"))
    )
    item_metadata = {
        **metadata,
        "frame_ids": raw.frame_ids.tolist(),
        "timestamps_sec": raw.timestamps_sec.tolist(),
        "calibration_model_sha256": payload["calibration_model_sha256"],
        "calibrator_file_sha256": file_sha256(calibrator_path),
        "calibration_fit_split": payload["fit_split"],
        "calibration_input_transform": payload["input_transform"],
    }
    return ObservationBatch(
        keypoints_2d=keypoints[None],
        keypoint_reliability=reliability[None],
        keypoint_valid=valid[None],
        frame_valid=raw.frame_available[None],
        metadata=(item_metadata,),
    ).validate()


def compile_calibrated_keypoint_caches(
    *,
    raw_root: str | Path,
    source_report_sha256: str,
    manifest_path: str | Path,
    calibrator_path: str | Path,
    extractor: dict[str, str],
    preprocessing: dict[str, Any],
    output: str | Path,
    development_only: bool,
    allow_incomplete_extractor_provenance: bool = False,
    use_raw_frame_source_identity: bool = False,
) -> dict[str, Any]:
    source = Path(raw_root)
    report_path = source / "compilation_report.json"
    if not (source / "COMPILATION_COMPLETE").is_file():
        raise ValueError("raw observation source is incomplete")
    if file_sha256(report_path) != source_report_sha256:
        raise ValueError("raw observation source report hash mismatch")
    required_extractor = {"name", "version", "checkpoint_sha256"}
    if set(extractor) != required_extractor or len(extractor["checkpoint_sha256"]) != 64:
        raise ValueError("extractor identity requires name/version/checkpoint SHA-256")
    if use_raw_frame_source_identity and not development_only:
        raise PermissionError("raw frame source identity fallback is development-only")
    items = load_manifest(manifest_path, require_existing_video=not use_raw_frame_source_identity)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"immutable calibrated observation root exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    incomplete = temporary / ".compilation_incomplete"
    incomplete.write_text("incomplete\n", "utf-8")
    try:
        cache = ObservationCache(temporary / "caches")
        rows = []
        for item in items:
            clip_root = source / item.clip_id
            metadata_path = clip_root / "metadata.json"
            raw_path = clip_root / "raw_keypoints.npz"
            source_metadata = json.loads(metadata_path.read_text("utf-8"))
            if file_sha256(raw_path) != source_metadata.get("artifact_sha256"):
                raise ValueError(f"raw artifact hash mismatch: {item.clip_id}")
            source_checkpoint = source_metadata.get("extractor_checkpoint_sha256")
            incomplete_provenance = source_checkpoint is None
            if incomplete_provenance and not (
                development_only and allow_incomplete_extractor_provenance
            ):
                raise ValueError(f"extractor checkpoint provenance is incomplete: {item.clip_id}")
            if not incomplete_provenance and source_checkpoint != extractor["checkpoint_sha256"]:
                raise ValueError(f"extractor checkpoint mismatch: {item.clip_id}")
            with np.load(raw_path, allow_pickle=False) as arrays:
                raw = RawKeypointBatch(
                    frame_ids=torch.from_numpy(arrays["frame_ids"]).long(),
                    timestamps_sec=torch.from_numpy(arrays["timestamps_sec"]).double(),
                    keypoints_2d=torch.from_numpy(arrays["keypoints_2d"]),
                    raw_score=torch.from_numpy(arrays["raw_score"]),
                    keypoint_available=torch.from_numpy(arrays["keypoint_available"]).bool(),
                    frame_available=torch.from_numpy(arrays["frame_available"]).bool(),
                ).validate()
            expected_frames = (
                list(item.frame_mapping)
                if item.frame_mapping is not None
                else list(range(item.frame_count))
            )
            if raw.frame_ids.tolist() != expected_frames:
                raise ValueError(f"manifest/raw frame mapping mismatch: {item.clip_id}")
            expected_timestamps = torch.tensor(
                [item.timestamp_sec(index) for index in range(item.effective_frame_count)],
                dtype=torch.float64,
            )
            if not torch.allclose(raw.timestamps_sec, expected_timestamps, atol=1e-9, rtol=0):
                raise ValueError(f"manifest/raw timestamps mismatch: {item.clip_id}")
            clip_preprocessing = {
                **preprocessing,
                "frame_ids": expected_frames,
                "fps_native": item.fps_native,
                "fps_effective": item.fps_effective or item.fps_native,
            }
            observation = calibrate_raw_keypoints(
                raw,
                calibrator_path,
                allow_development=development_only,
                metadata={
                    "clip_id": item.clip_id,
                    "development_only": development_only,
                    "extractor": extractor,
                    "extractor_provenance_status": (
                        "INCOMPLETE_ACCEPTED_FOR_DEVELOPMENT"
                        if incomplete_provenance
                        else "HASH_VERIFIED"
                    ),
                    "preprocessing": clip_preprocessing,
                    "raw_artifact_sha256": file_sha256(raw_path),
                },
            )
            video_hash = (
                source_metadata["source_identity_sha256"]
                if use_raw_frame_source_identity
                else file_sha256(item.video_path)
            )
            cache_id = ObservationCache.identity(
                video_hash=video_hash,
                extractor=extractor,
                preprocessing=clip_preprocessing,
                calibration_hash=file_sha256(calibrator_path),
            )
            cache.save(cache_id, observation)
            rows.append(
                {
                    "clip_id": item.clip_id,
                    "cache_id": cache_id,
                    "frames": item.effective_frame_count,
                    "video_sha256": video_hash,
                    "raw_artifact_sha256": file_sha256(raw_path),
                }
            )
        payload = {
            "schema_version": "dcg_calibrated_observation_index_v1",
            "development_only": development_only,
            "scientific_status": (
                "DEVELOPMENT_INCOMPLETE_EXTRACTOR_PROVENANCE"
                if allow_incomplete_extractor_provenance
                else "CALIBRATED"
            ),
            "incomplete_extractor_provenance_accepted": (allow_incomplete_extractor_provenance),
            "source_identity_policy": (
                "frozen_raw_frame_hash_registry"
                if use_raw_frame_source_identity
                else "video_file_sha256"
            ),
            "manifest_sha256": file_sha256(manifest_path),
            "raw_source_report_sha256": source_report_sha256,
            "calibrator_sha256": file_sha256(calibrator_path),
            "extractor": extractor,
            "preprocessing": preprocessing,
            "clips": len(rows),
            "frames": sum(row["frames"] for row in rows),
            "per_clip": rows,
        }
        payload["index_identity_sha256"] = canonical_hash(payload)
        (temporary / "index.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", "utf-8"
        )
        os.replace(incomplete, temporary / "CALIBRATED_OBSERVATIONS_COMPLETE")
        os.replace(temporary, destination)
        return payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
