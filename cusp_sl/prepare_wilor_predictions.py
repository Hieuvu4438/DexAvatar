"""Materialize tested SMPLer-X + WiLoR A1 rotations for the shared renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from cusp_sl.evaluate_wilor_direct_development import select_detection
from cusp_sl.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from cusp_sl.retargeting import fuse_wilor_hand, mano_fingers_to_smplx
from cusp_sl.wilor_artifact import validate_wilor_raw_v3
from phase2_refiner.data.cache_schema import load_cache_clip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilor_candidate(
    base_axis_angle: np.ndarray,
    global_axis_angle: np.ndarray,
    image_records: list[dict],
    *,
    geometric_wrist_alignment: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply selected per-side WiLoR detections with explicit base fallback."""

    if len(base_axis_angle) != len(image_records):
        raise ValueError("Pose/image record lengths differ")
    candidate = axis_angle_to_matrix(torch.from_numpy(base_axis_angle).float())
    global_orient = axis_angle_to_matrix(
        torch.from_numpy(global_axis_angle).float()
    )
    detected = {"left": 0, "right": 0}
    for frame, image_record in enumerate(image_records):
        for side, is_right, start in (
            ("left", False, 21),
            ("right", True, 36),
        ):
            hand = select_detection(image_record.get("hands", []), is_right=is_right)
            if hand is None:
                continue
            fingers = torch.from_numpy(
                np.asarray(hand["pred_mano_pose_rotmat"], dtype=np.float32)
            )
            if geometric_wrist_alignment:
                wrist = torch.from_numpy(
                    np.asarray(
                        hand["pred_mano_global_orient_rotmat"], dtype=np.float32
                    )
                ).reshape(3, 3)
                candidate[frame] = fuse_wilor_hand(
                    candidate[frame],
                    global_orient[frame],
                    fingers,
                    wrist,
                    is_right=is_right,
                )
            else:
                candidate[frame, start : start + 15] = mano_fingers_to_smplx(
                    fingers, is_right=is_right
                )
            detected[side] += 1
    result = matrix_to_axis_angle(candidate).cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("A1 candidate contains non-finite rotations")
    return result, detected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wilor-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometric-wrist-alignment", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    (args.output / "clips").mkdir(parents=True)

    # This pickle must be the locally generated, hash-bound frozen-front-end
    # artifact, never an untrusted downloaded object.
    with args.wilor_pickle.open("rb") as handle:
        wilor = pickle.load(handle, encoding="latin1")
    images, wilor_meta = validate_wilor_raw_v3(wilor)
    source = json.loads(args.manifest.read_text(encoding="utf-8"))

    summaries = []
    frames = 0
    for entry in source["clips"]:
        relative = entry["cache"] if isinstance(entry, dict) else entry
        cache_path = Path(relative)
        if not cache_path.is_absolute():
            cache_path = args.manifest.parent / cache_path
        clip = load_cache_clip(cache_path)
        image_records = []
        for name in clip.frame_names.astype(str):
            keys = (f"{clip.clip_id}__{name}.png", f"{name}.png")
            record = next((images[key] for key in keys if key in images), None)
            if record is None:
                raise ValueError(
                    f"WiLoR artifact lacks explicit frame record for {clip.clip_id}/{name}"
                )
            image_records.append(record)
        selected, detected = wilor_candidate(
            clip.init_axis_angle,
            clip.global_orient,
            image_records,
            geometric_wrist_alignment=args.geometric_wrist_alignment,
        )
        output = args.output / "clips" / f"{clip.clip_id}.npz"
        np.savez_compressed(
            output,
            clip_id=np.asarray(clip.clip_id),
            frame_names=clip.frame_names,
            selected_axis_angle=selected,
            selected_index=np.asarray(1, dtype=np.int64),
            candidate_valid=np.asarray([True, True]),
        )
        frames += len(clip.frame_names)
        summaries.append(
            {
                "clip_id": clip.clip_id,
                "frames": len(clip.frame_names),
                "left_detected_frames": detected["left"],
                "right_detected_frames": detected["right"],
                "cache_sha256": sha256(cache_path),
                "prediction_sha256": sha256(output),
            }
        )
        print(
            f"[a1] {clip.clip_id}: {len(clip.frame_names)} frames, "
            f"left={detected['left']}, right={detected['right']}"
        )

    expected = source.get("expected_frames")
    if expected is not None and frames != int(expected):
        raise ValueError(f"Materialized {frames} != expected {expected}")
    if frames != len(images):
        raise ValueError(
            f"WiLoR artifact has {len(images)} records for {frames} source frames"
        )
    report = {
        "role": "frozen_a1_frontend_predictions",
        "geometric_wrist_alignment": args.geometric_wrist_alignment,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "wilor_pickle": str(args.wilor_pickle.resolve()),
        "wilor_pickle_sha256": sha256(args.wilor_pickle),
        "wilor_frame_manifest_sha256": wilor_meta["frame_manifest_sha256"],
        "wilor_checkpoint_sha256": wilor_meta["wilor_checkpoint_sha256"],
        "wilor_detector_checkpoint_sha256": wilor_meta[
            "detector_checkpoint_sha256"
        ],
        "frames": frames,
        "clips": summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"clips": len(summaries), "frames": frames}, indent=2))


if __name__ == "__main__":
    main()
