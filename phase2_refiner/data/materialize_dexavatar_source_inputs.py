"""Materialize frame-exact, target-free inputs for the DexAvatar fitter.

The legacy fitter expects one directory per sign containing decoded RGB frames,
Sapiens-style whole-body observations, SMPLer-X parameter pickles, and a HaMeR
pickle.  This adapter supplies that interface from the locked PHOENIX/WLASL
component selection.  It never reads released SOKE/SignAvatar target fields.

DexAvatar's released two-hand branch drops frames when HaMeR misses a side.  To
preserve the locked 16-frame windows, a missing side is represented explicitly
by the corresponding SMPLer-X H32 hand pose and the existing 133-point 2D hand
track.  Real-vs-fallback availability is recorded per frame for downstream
fusion and is never inferred from the training target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import cv2
import numpy as np

from phase2_refiner.data.build_sign_domain_cache import _load_hamer_outputs
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-dexavatar-source-input-v1"


def _numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value, dtype=dtype)


def _confidence(logits: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-(logits - 3.0)))).astype(np.float32)


def _camera_from_bbox(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match SMPLer-X inference.py for the H32 256x192 body crop."""

    x, y, width, height = np.asarray(bbox, dtype=np.float64)
    focal = np.asarray([5000.0 / 192.0 * width, 5000.0 / 256.0 * height])
    principal = np.asarray([x + 0.5 * width, y + 0.5 * height])
    return focal, principal


def _rotation_matrices(axis_angle: np.ndarray) -> np.ndarray:
    return np.stack([cv2.Rodrigues(row)[0] for row in axis_angle]).astype(np.float32)


def _hand_crop_coordinates(points: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    usable = points[finite]
    if len(usable) == 0:
        usable = np.zeros((1, 2), dtype=np.float32)
    low = usable.min(axis=0)
    high = usable.max(axis=0)
    center = (low + high) * 0.5
    size = float(max((high - low).max() * 1.5, 32.0))
    normalized = (points - center) / size
    if side == "left":
        normalized[:, 0] *= -1.0
    return normalized.astype(np.float32), center.astype(np.float32), size


def _normalize_hamer_entry(
    entry: Any,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> tuple[list[Any], dict[str, bool]]:
    if not isinstance(entry, (list, tuple)) or len(entry) < 5:
        entry = [
            {
                "pred_keypoints_2d": np.zeros((0, 21, 2), dtype=np.float32),
                "pred_keypoints_3d": np.zeros((0, 21, 3), dtype=np.float32),
                "pred_mano_params": {
                    "global_orient": np.zeros((0, 1, 3, 3), dtype=np.float32),
                    "hand_pose": np.zeros((0, 15, 3, 3), dtype=np.float32),
                    "betas": np.zeros((0, 10), dtype=np.float32),
                },
            },
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        ]
    prediction = entry[0]
    keypoints_2d = _numpy(prediction["pred_keypoints_2d"], np.float32)
    keypoints_3d = _numpy(prediction["pred_keypoints_3d"], np.float32)
    mano = prediction["pred_mano_params"]
    global_orient = _numpy(mano["global_orient"], np.float32)
    hand_pose = _numpy(mano["hand_pose"], np.float32)
    betas = _numpy(mano["betas"], np.float32)
    centers = _numpy(entry[1], np.float32).reshape(-1, 2)
    sizes = _numpy(entry[2], np.float32).reshape(-1)
    flags = _numpy(entry[3], np.float32).reshape(-1)
    camera = _numpy(entry[4], np.float32).reshape(-1, 3)
    lengths = {
        len(keypoints_2d), len(keypoints_3d), len(global_orient), len(hand_pose),
        len(betas), len(centers), len(sizes), len(flags), len(camera)
    }
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent HaMeR entry lengths: {sorted(lengths)}")

    available = {
        "left": bool(np.any(flags.astype(int) == 0)),
        "right": bool(np.any(flags.astype(int) == 1)),
    }
    identity = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3)
    for side, flag, pose, points in (
        ("left", 0.0, left_pose, left_points),
        ("right", 1.0, right_pose, right_points),
    ):
        if available[side]:
            continue
        crop_points, center, size = _hand_crop_coordinates(points, side)
        keypoints_2d = np.concatenate((keypoints_2d, crop_points[None]), axis=0)
        keypoints_3d = np.concatenate(
            (keypoints_3d, np.zeros((1, 21, 3), dtype=np.float32)), axis=0
        )
        global_orient = np.concatenate((global_orient, identity), axis=0)
        hand_pose = np.concatenate(
            (hand_pose, _rotation_matrices(pose.reshape(15, 3))[None]), axis=0
        )
        betas = np.concatenate((betas, np.zeros((1, 10), dtype=np.float32)), axis=0)
        centers = np.concatenate((centers, center[None]), axis=0)
        sizes = np.concatenate((sizes, np.asarray([size], dtype=np.float32)))
        flags = np.concatenate((flags, np.asarray([flag], dtype=np.float32)))
        camera = np.concatenate((camera, np.zeros((1, 3), dtype=np.float32)), axis=0)
    normalized = [
        {
            "pred_keypoints_2d": keypoints_2d,
            "pred_keypoints_3d": keypoints_3d,
            "pred_mano_params": {
                "global_orient": global_orient,
                "hand_pose": hand_pose,
                "betas": betas,
            },
        },
        centers,
        sizes,
        flags,
        camera,
    ]
    return normalized, available


def _decode_exact_frames(video: Path, indices: np.ndarray, destinations: list[Path]) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"Cannot open source video: {video}")
    try:
        for index, destination in zip(indices, destinations):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise IOError(f"Cannot decode frame {index} from {video}")
            if not cv2.imwrite(str(destination), frame):
                raise IOError(f"Cannot write decoded frame: {destination}")
    finally:
        capture.release()


def _write_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=2)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    smplerx_root = args.smplerx_root.resolve()
    wilor_root = args.wilor_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only DexAvatar input root exists: {output}")
    selection_path = smplerx_root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    entries = selection["clips"]
    if args.max_clips > 0:
        entries = entries[: args.max_clips]
    hamer, hamer_paths = _load_hamer_outputs(wilor_root)

    output.mkdir(parents=True)
    clips_root = output / "clips"
    clips_root.mkdir()
    sign_lines = []
    segments: dict[str, list[int]] = {}
    reports = []
    for clip_number, entry in enumerate(entries, start=1):
        clip_id = str(entry["clip_id"])
        teacher_path = smplerx_root / "clips" / f"{clip_id}.npz"
        with np.load(teacher_path, allow_pickle=False) as teacher:
            indices = teacher["sample_indices"].astype(np.int64)
            expected = np.asarray(entry["frame_indices"], dtype=np.int64)
            if not np.array_equal(indices, expected):
                raise ValueError(f"SMPLer-X/selection frame mismatch: {clip_id}")
            width, height = teacher["image_size"].astype(int)
            points = teacher["keypoints_2d"].astype(np.float32).copy()
            points[..., 0] *= width
            points[..., 1] *= height
            confidence = _confidence(teacher["keypoint_scores"].astype(np.float32))

            clip_root = clips_root / clip_id
            frames_root = clip_root / clip_id
            smplx_dir = clip_root / "smplerx" / "smplx"
            hamer_dir = clip_root / "hamer"
            frames_root.mkdir(parents=True)
            smplx_dir.mkdir(parents=True)
            hamer_dir.mkdir()
            frame_names = [f"{clip_id}_{int(index):06d}.png" for index in indices]
            destinations = [frames_root / name for name in frame_names]
            _decode_exact_frames(Path(entry["video"]), indices, destinations)

            sapiens: dict[str, Any] = {}
            clip_hamer: dict[str, Any] = {}
            availability = []
            for frame_offset, (index, frame_name) in enumerate(zip(indices, frame_names)):
                sapiens[f"{clip_id}/{frame_name}"] = [
                    points[frame_offset][None], confidence[frame_offset][None]
                ]
                normalized, real = _normalize_hamer_entry(
                    hamer.get(frame_name),
                    teacher["left_hand_pose"][frame_offset],
                    teacher["right_hand_pose"][frame_offset],
                    points[frame_offset, 91:112],
                    points[frame_offset, 112:133],
                )
                clip_hamer[frame_name] = normalized
                availability.append(
                    {
                        "frame_number": int(index),
                        "real_hamer_left": real["left"],
                        "real_hamer_right": real["right"],
                    }
                )
                focal, principal = _camera_from_bbox(teacher["bboxes"][frame_offset])
                parameters = {
                    "global_orient": teacher["global_orient"][frame_offset],
                    "body_pose": teacher["body_pose"][frame_offset],
                    "left_hand_pose": teacher["left_hand_pose"][frame_offset],
                    "right_hand_pose": teacher["right_hand_pose"][frame_offset],
                    "jaw_pose": teacher["jaw_pose"][frame_offset],
                    "leye_pose": np.zeros(3, dtype=np.float32),
                    "reye_pose": np.zeros(3, dtype=np.float32),
                    "betas": teacher["betas"][frame_offset],
                    "expression": teacher["expression"][frame_offset],
                    "transl": teacher["transl"][frame_offset],
                    "focal": focal,
                    "princpt": principal,
                }
                _write_pickle(smplx_dir / f"{Path(frame_name).stem}.pkl", parameters)

            _write_pickle(clip_root / "sapiens.pkl", sapiens)
            _write_pickle(hamer_dir / "hamer.pkl", clip_hamer)
            np.save(clip_root / "mean_shape_smplx.npy", np.median(teacher["betas"], axis=0))
            (clip_root / "gender.txt").write_text("neutral\n", encoding="utf-8")
            availability_path = clip_root / "expert_availability.json"
            availability_path.write_text(
                json.dumps(availability, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            real_left = sum(item["real_hamer_left"] for item in availability)
            real_right = sum(item["real_hamer_right"] for item in availability)
            reports.append(
                {
                    "clip_id": clip_id,
                    "dataset": entry["dataset"],
                    "frames": len(indices),
                    "real_hamer_left_frames": real_left,
                    "real_hamer_right_frames": real_right,
                    "frame_sha256": [sha256_file(path) for path in destinations],
                    "availability_sha256": sha256_file(availability_path),
                }
            )
        # "~0" is the legacy non-zero token for the two-hand fitter branch;
        # it carries no gloss identity and is not fed to either poser network.
        sign_lines.append(f"{clip_id} ~0")
        segments[clip_id] = [int(indices[0]), int(indices[-1])]
        print(f"[dexavatar-input] {clip_number}/{len(entries)} {clip_id}", flush=True)

    signs_path = output / "signs.txt"
    signs_path.write_text("\n".join(sign_lines) + "\n", encoding="utf-8")
    segments_path = output / "segment.json"
    segments_path.write_text(
        json.dumps(segments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256()
    for path in hamer_paths:
        digest.update(bytes.fromhex(sha256_file(path)))
    report = {
        "schema": SCHEMA,
        "split": selection["split"],
        "clips": len(reports),
        "frames": sum(item["frames"] for item in reports),
        "target_fields_read": False,
        "sgnify_labels_read": False,
        "routing": "two-hand fitting with explicit H32 fallback for missing HaMeR side",
        "smplerx_selection": str(selection_path),
        "smplerx_selection_sha256": sha256_file(selection_path),
        "hamer_outputs": [str(path) for path in hamer_paths],
        "hamer_outputs_combined_sha256": digest.hexdigest(),
        "signs_sha256": sha256_file(signs_path),
        "segments_sha256": sha256_file(segments_path),
        "clip_reports": reports,
    }
    report_path = output / "materialization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplerx-root", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=0)
    print(json.dumps(materialize(parser.parse_args()), indent=2, sort_keys=True))
