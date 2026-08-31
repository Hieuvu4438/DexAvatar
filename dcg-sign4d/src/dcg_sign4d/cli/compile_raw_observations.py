from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dcg_sign4d.observations.sapiens_adapter import load_sapiens_clip
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse Sapiens JSON as uncalibrated raw cues")
    parser.add_argument("--initialization-root", required=True)
    parser.add_argument("--detector-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    if not args.development_only:
        raise PermissionError("unverified raw observation reuse requires --development-only")
    initialization = Path(args.initialization_root)
    if not (initialization / "CONVERSION_COMPLETE").is_file():
        raise ValueError("initialization root has no completion marker")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable raw observation artifact exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".compilation_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    rows = []
    clip_dirs = sorted(path for path in initialization.iterdir() if path.is_dir())
    for clip_index, clip_dir in enumerate(clip_dirs, start=1):
        init_metadata_path = clip_dir / "metadata.json"
        init_metadata = json.loads(init_metadata_path.read_text(encoding="utf-8"))
        frame_ids = [int(value) for value in init_metadata["frame_ids"]]
        raw, source_paths = load_sapiens_clip(
            clip_dir.name,
            frame_ids,
            fps=float(init_metadata["fps"]),
            detector_root=args.detector_root,
        )
        clip_output = output / clip_dir.name
        clip_output.mkdir()
        artifact_path = clip_output / "raw_keypoints.npz"
        np.savez_compressed(
            artifact_path,
            frame_ids=raw.frame_ids.numpy(),
            timestamps_sec=raw.timestamps_sec.numpy(),
            keypoints_2d=raw.keypoints_2d.numpy(),
            raw_score=raw.raw_score.numpy(),
            keypoint_available=raw.keypoint_available.numpy(),
            frame_available=raw.frame_available.numpy(),
        )
        image_paths = [
            Path(args.image_root) / clip_dir.name / f"low_{frame}.png" for frame in frame_ids
        ]
        if not all(path.is_file() for path in image_paths):
            missing = next(path for path in image_paths if not path.is_file())
            raise FileNotFoundError(missing)
        metadata = {
            "schema_version": "raw_keypoints_v1",
            "development_only": True,
            "scientific_status": "UNCALIBRATED_NOT_USABLE_AS_RELIABILITY",
            "clip_id": clip_dir.name,
            "frames": len(frame_ids),
            "joints": raw.keypoints_2d.shape[1],
            "fps": float(init_metadata["fps"]),
            "frame_ids": frame_ids,
            "timestamp_policy": "source_frame_id/fps",
            "extractor_name": "Sapiens-1B WholeBody (inferred from existing directory)",
            "extractor_checkpoint_sha256": None,
            "extractor_provenance_status": "INCOMPLETE_CHECKPOINT_HASH",
            "raw_score_status": "FINITE_UNBOUNDED_NOT_A_PROBABILITY",
            "selection_policy": "require_exactly_one_instance",
            "source_json_hashes": {path.name: file_sha256(path) for path in source_paths},
            "source_image_hashes": {path.name: file_sha256(path) for path in image_paths},
            "initialization_metadata_sha256": file_sha256(init_metadata_path),
            "artifact_sha256": file_sha256(artifact_path),
            "available_keypoints": int(raw.keypoint_available.sum()),
            "missing_frames": int((~raw.frame_available).sum()),
            "optional_cues": {"part_masks": False, "tracks": False, "depth": False},
        }
        metadata["source_identity_sha256"] = canonical_hash(
            {
                "source_json_hashes": metadata["source_json_hashes"],
                "source_image_hashes": metadata["source_image_hashes"],
                "frame_ids": frame_ids,
                "fps": metadata["fps"],
            }
        )
        (clip_output / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(metadata)
        print(f"[{clip_index:02d}/{len(clip_dirs):02d}] {clip_dir.name}", flush=True)
    report = {
        "schema_version": "raw_observation_compilation_v1",
        "development_only": True,
        "scientific_status": "BLOCKED_BEFORE_CALIBRATION_AND_PROVENANCE_GATES",
        "clips": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "joints": 133,
        "available_keypoints": sum(int(row["available_keypoints"]) for row in rows),
        "missing_frames": sum(int(row["missing_frames"]) for row in rows),
        "extractor_checkpoint_sha256": None,
        "calibration_model_sha256": None,
        "raw_score_status": "FINITE_UNBOUNDED_NOT_A_PROBABILITY",
        "part_masks_available": False,
        "tracks_available": False,
        "depth_available": False,
        "per_clip": rows,
    }
    (output / "compilation_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(incomplete, output / "COMPILATION_COMPLETE")
    print(json.dumps({key: value for key, value in report.items() if key != "per_clip"}))


if __name__ == "__main__":
    main()
