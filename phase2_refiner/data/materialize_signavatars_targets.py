"""Attach licensed SignAvatars SMPL-X targets to Phase 2R observation caches."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any

import cv2
import numpy as np

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip, save_cache_clip
from phase2_refiner.provenance import sha256_file


NUM_JOINTS = 51
AUDIT_SCHEMA = "signavatars-target-audit-v1"
AUDIT_STRATA = {
    "signer",
    "hand_activity",
    "hand_size",
    "truncation",
    "motion",
}


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_license_acceptance(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("dataset") != "SignAvatars":
        raise ValueError("License record dataset must be SignAvatars")
    if payload.get("non_commercial_research_terms_accepted") is not True:
        raise ValueError("SignAvatars non-commercial research terms are not accepted")
    for key in ("registrant_name", "registrant_email", "access_granted_at"):
        if not str(payload.get(key, "")).strip():
            raise ValueError(f"License record lacks {key}")
    return payload


def validate_target_audit(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("audit_schema") != AUDIT_SCHEMA:
        raise ValueError(f"SignAvatars target audit must use {AUDIT_SCHEMA}")
    if payload.get("passed") is not True:
        raise ValueError("SignAvatars target-quality audit did not pass")
    if int(payload.get("audited_clips", 0)) < 100:
        raise ValueError("SignAvatars target audit must cover at least 100 clips")
    if not isinstance(payload.get("sample_seed"), int):
        raise ValueError("SignAvatars target audit lacks an integer sample_seed")
    if not _is_sha256(payload.get("sample_manifest_sha256")):
        raise ValueError("SignAvatars target audit lacks sample_manifest_sha256")
    strata = set(payload.get("stratified_by", ()))
    missing_strata = sorted(AUDIT_STRATA - strata)
    if missing_strata:
        raise ValueError(
            f"SignAvatars target audit lacks required strata: {missing_strata}"
        )
    for field in ("reviewer", "completed_at"):
        if not str(payload.get(field, "")).strip():
            raise ValueError(f"SignAvatars target audit lacks {field}")
    fractions = payload.get("catastrophic_failure_fraction_by_region")
    if not isinstance(fractions, dict):
        raise ValueError("SignAvatars target audit lacks regional failure fractions")
    for region in ("body", "left_hand", "right_hand"):
        fraction = float(fractions.get(region, 1.0))
        if not 0.0 <= fraction < 0.10:
            raise ValueError(
                "SignAvatars catastrophic target failure must be below 10% "
                f"for {region}: {fraction}"
            )
    aggregate = float(payload.get("catastrophic_failure_fraction", 1.0))
    if not 0.0 <= aggregate < 0.10:
        raise ValueError(
            "SignAvatars aggregate catastrophic target failure must be below 10%"
        )
    return payload


def _annotation_path(
    clip,
    annotations_root: Path,
    annotation_map: dict[str, str],
) -> Path:
    metadata = json.loads(clip.metadata_json)
    source_clip = str(metadata.get("source_clip", clip.clip_id))
    if source_clip in annotation_map:
        candidate = Path(annotation_map[source_clip])
        return candidate if candidate.is_absolute() else annotations_root / candidate
    candidates = [
        annotations_root / f"{source_clip}.pkl",
        annotations_root / f"{source_clip.removesuffix('-rgb_front')}.pkl",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one SignAvatars annotation for {source_clip}; "
            f"found={existing}. Supply --annotation-map when names differ."
        )
    return existing[0]


def _load_annotation(path: Path, target_key: str) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict) or target_key not in payload:
        raise ValueError(f"{path} lacks SignAvatars target array {target_key!r}")
    parameters = np.asarray(payload[target_key], dtype=np.float32)
    minimum_width = 169 if target_key == "unsmooth_smplx" else 182
    if parameters.ndim != 2 or parameters.shape[1] < minimum_width:
        raise ValueError(f"Invalid {target_key} shape in {path}: {parameters.shape}")
    if not np.isfinite(parameters).all():
        raise ValueError(f"Non-finite SignAvatars parameters in {path}")
    return {**payload, target_key: parameters}


def _source_frame_indices(clip) -> np.ndarray:
    indices = []
    for reference in clip.source_paths.astype(str):
        marker = "#frame="
        if reference.count(marker) != 1:
            raise ValueError(
                f"Cache source is not an exact video-frame binding: {reference}"
            )
        indices.append(int(reference.rsplit(marker, 1)[1]))
    result = np.asarray(indices, dtype=np.int64)
    if np.any(result < 0) or np.any(np.diff(result) <= 0):
        raise ValueError(f"Invalid ordered source frames for {clip.clip_id}")
    return result


def _source_video_contract(clip) -> tuple[int, float]:
    videos = {
        str(reference).rsplit("#frame=", 1)[0]
        for reference in clip.source_paths.astype(str)
    }
    if len(videos) != 1:
        raise ValueError(f"Cache does not bind exactly one source video: {videos}")
    video = Path(videos.pop())
    if not video.is_file():
        raise FileNotFoundError(f"Bound source video is missing: {video}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"Cannot inspect source video: {video}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"Invalid source video timing: frames={frame_count} fps={fps} {video}"
        )
    if not np.isclose(fps, clip.fps, rtol=0.0, atol=1e-3):
        raise ValueError(
            f"Cache/source FPS mismatch for {clip.clip_id}: {clip.fps} vs {fps}"
        )
    return frame_count, fps


def _validity(payload: dict[str, Any], name: str, frames: np.ndarray) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"SignAvatars annotation lacks {name}")
    value = np.asarray(payload[name])
    if value.ndim != 1 or frames.max() >= len(value):
        raise ValueError(f"Invalid {name} coverage: shape={value.shape}")
    return value[frames].astype(bool)


def attach_target(
    clip,
    annotation_path: Path,
    target_key: str,
    target_audit: Path | None,
    license_record: Path,
    *,
    audit_candidate: bool = False,
    source_video_frame_count: int | None = None,
    source_video_fps: float | None = None,
    signer_id: str | None = None,
) -> Any:
    payload = _load_annotation(annotation_path, target_key)
    frames = _source_frame_indices(clip)
    parameters = payload[target_key]
    if (
        source_video_frame_count is not None
        and len(parameters) != source_video_frame_count
    ):
        raise ValueError(
            "SignAvatars/source video frame-count mismatch: "
            f"annotation={len(parameters)} video={source_video_frame_count}"
        )
    if frames.max() >= len(parameters):
        raise ValueError(
            f"SignAvatars annotation is shorter than requested frame {frames.max()}: "
            f"{annotation_path} has {len(parameters)} frames"
        )
    selected = parameters[frames]
    target = np.concatenate(
        (selected[:, 3:66], selected[:, 66:111], selected[:, 111:156]), axis=1
    ).reshape(-1, NUM_JOINTS, 3)
    body_valid = _validity(payload, "total_valid_index", frames)
    left_valid = _validity(payload, "left_valid", frames)
    right_valid = _validity(payload, "right_valid", frames)
    target_valid = np.zeros((len(frames), NUM_JOINTS), dtype=bool)
    target_valid[:, :21] = body_valid[:, None]
    target_valid[:, 21:36] = left_valid[:, None]
    target_valid[:, 36:51] = right_valid[:, None]
    target_quality = target_valid.astype(np.float32)
    metadata = json.loads(clip.metadata_json)
    if signer_id is not None:
        if not str(signer_id).strip():
            raise ValueError("signer_id must be non-empty when supplied")
        metadata["signer_id"] = str(signer_id)
    metadata["target_provider"] = f"SignAvatars {target_key} SMPL-X"
    metadata["target_type"] = (
        "licensed_released_3d_smplx_pose_audit_candidate"
        if audit_candidate
        else "licensed_released_3d_smplx_pose"
    )
    metadata["target_scope"] = (
        "released SMPL-X body/both-hand pose decoded in common initializer "
        "shape/root geometry with released validity"
    )
    metadata["target_contract"] = {
        "independent_from_initializer": True,
        "not_same_view_2d_only": True,
        "initializer_outputs_used_false": True,
        "release_benchmark_excluded": True,
        "geometry": "released_3d_smplx_pose",
        "shared_geometry_decode": True,
        "exact_frame_count_match": source_video_frame_count == len(parameters),
        "source_video_frame_count": source_video_frame_count,
        "annotation_frame_count": len(parameters),
        "source_video_fps": source_video_fps,
        "provider_id": f"signavatars-eccv2024-{target_key}-v1",
        "source_sha256": [sha256_file(annotation_path)]
        + ([sha256_file(target_audit)] if target_audit is not None else []),
        "provider_code_sha256": sha256_file(Path(__file__)),
        "audit_passed": not audit_candidate,
        "audit_report_sha256": (
            sha256_file(target_audit) if target_audit is not None else ""
        ),
        "license_verified": True,
        "license_record_sha256": sha256_file(license_record),
        "annotation_path": str(annotation_path.resolve()),
    }
    metadata.pop("phase2r_adapter", None)
    return replace(
        clip,
        target_axis_angle=target.astype(np.float32),
        target_rotation_valid=target_valid,
        target_quality=target_quality,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    template_root = args.template_root.resolve()
    annotations_root = args.annotations_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Append-only output exists: {output_root}")
    validate_license_acceptance(args.license_record.resolve())
    target_audit = args.target_audit.resolve() if args.target_audit else None
    if args.audit_candidate:
        if target_audit is not None:
            raise ValueError("Audit-candidate materialization must not claim an audit")
    else:
        if target_audit is None:
            raise ValueError("Formal materialization requires --target-audit")
        validate_target_audit(target_audit)
    annotation_map = (
        _load_json(args.annotation_map.resolve()) if args.annotation_map else {}
    )
    signer_map = _load_json(args.signer_map.resolve())
    report: dict[str, Any] = {
        "provider": f"signavatars-eccv2024-{args.target_key}-v1",
        "template_root": str(template_root),
        "annotations_root": str(annotations_root),
        "evidence_tier": "audit_candidate" if args.audit_candidate else "formal",
        "license_record_sha256": sha256_file(args.license_record),
        "target_audit_sha256": (
            sha256_file(target_audit) if target_audit is not None else None
        ),
        "splits": {},
    }
    output_root.mkdir(parents=True)
    try:
        (output_root / "splits").mkdir()
        for split in ("train", "val", "calibration"):
            manifest = template_root / "splits" / f"{split}.json"
            clip_dir = output_root / "clips" / split
            clip_dir.mkdir(parents=True)
            entries = []
            frames = supervised = 0
            for source in _manifest_paths(manifest):
                clip = load_cache_clip(source)
                source_metadata = json.loads(clip.metadata_json)
                source_clip = str(source_metadata.get("source_clip", clip.clip_id))
                signer_id = str(signer_map.get(source_clip, "")).strip()
                if not signer_id:
                    raise ValueError(f"Signer map lacks {source_clip}")
                annotation = _annotation_path(
                    clip, annotations_root, annotation_map
                ).resolve()
                video_frame_count, video_fps = _source_video_contract(clip)
                updated = attach_target(
                    clip,
                    annotation,
                    args.target_key,
                    target_audit,
                    args.license_record.resolve(),
                    audit_candidate=args.audit_candidate,
                    source_video_frame_count=video_frame_count,
                    source_video_fps=video_fps,
                    signer_id=signer_id,
                )
                updated_metadata = json.loads(updated.metadata_json)
                updated_metadata["phase2_split"] = split
                updated = replace(
                    updated,
                    metadata_json=json.dumps(updated_metadata, sort_keys=True),
                )
                destination = clip_dir / source.name
                temporary = destination.with_name(destination.stem + ".tmp.npz")
                save_cache_clip(temporary, updated)
                os.replace(temporary, destination)
                entries.append(f"../clips/{split}/{destination.name}")
                frames += len(updated.frame_names)
                supervised += int(updated.target_rotation_valid.all(axis=1).sum())
            output_manifest = output_root / "splits" / f"{split}.json"
            output_manifest.write_text(
                json.dumps({"clips": entries}, indent=2) + "\n", encoding="utf-8"
            )
            report["splits"][split] = {
                "clips": len(entries),
                "frames": frames,
                "complete_body_and_hands_frames": supervised,
                "manifest": str(output_manifest.resolve()),
                "manifest_sha256": sha256_file(output_manifest),
            }
        (output_root / "materialization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output_root)
        raise
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--license-record", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path)
    parser.add_argument(
        "--audit-candidate",
        action="store_true",
        help=(
            "Build an explicitly formal-ineligible cache for overlay review; "
            "rematerialize without this flag after the audit passes."
        ),
    )
    parser.add_argument("--annotation-map", type=Path)
    parser.add_argument(
        "--signer-map",
        type=Path,
        required=True,
        help="JSON object mapping every source_clip to a stable signer identity.",
    )
    parser.add_argument(
        "--target-key",
        choices=("unsmooth_smplx", "smplx"),
        default="unsmooth_smplx",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(materialize(parse_args()), indent=2, sort_keys=True))
