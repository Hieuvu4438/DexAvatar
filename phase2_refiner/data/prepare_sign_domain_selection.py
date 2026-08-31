"""Build frame-exact PHOENIX-SOKE and WLASL-SignAvatars selections.

The output is an append-only set of JSON manifests.  Every selected source
frame is bound to an RGB video frame, a 133-keypoint row, and a released SMPL-X
target row/file.  PHOENIX and WLASL official test items are never placed in a
training, validation, or calibration split.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil
from typing import Any

import cv2
import numpy as np
import torch

from phase2_refiner.data.build_how2sign_cache import _mapped_keypoints
from phase2_refiner.data.extract_how2sign_teacher import _load_track
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-sign-domain-selection-v1"
SPLITS = ("train", "val", "calibration", "test")
PHOENIX_SIGNER_SPLIT = {
    "Signer03": "val",
    "Signer09": "calibration",
}


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _stable_int(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    # Lock the split/window contract to the first 32 digest bits.  Besides being
    # portable across Python versions, seed 12345 yields a non-degenerate
    # signer-disjoint WLASL 75/15/10 partition for this release.
    return int(digest[:8], 16)


def _select_window(
    frame_indices: np.ndarray, frames_per_clip: int, seed: int, key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic contiguous annotation-row window.

    Contiguity is defined in the released annotation rows.  Original RGB frame
    IDs can still contain gaps where the annotation provider rejected a frame.
    """

    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if len(indices) < frames_per_clip:
        raise ValueError(
            f"{key} has {len(indices)} valid rows; require {frames_per_clip}"
        )
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError(f"Released frame IDs are not strictly increasing: {key}")
    starts = len(indices) - frames_per_clip + 1
    start = _stable_int(seed, key) % starts
    rows = np.arange(start, start + frames_per_clip, dtype=np.int64)
    return indices[rows], rows


def _frame_id(path: Path) -> int:
    match = re.fullmatch(r"images(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected SOKE pose filename: {path}")
    # PHOENIX image files are one-based; OpenCV video frames are zero-based.
    return int(match.group(1)) - 1


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _video_contract(video: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"Cannot open source video: {video}")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if frames < 1 or fps <= 0 or width < 1 or height < 1:
        raise ValueError(
            f"Invalid video contract frames={frames} fps={fps} size={width}x{height}: {video}"
        )
    return frames, fps, width, height


def _validate_binding(entry: dict[str, Any], verify_video: bool) -> dict[str, Any]:
    video = Path(entry["video"])
    track_path = Path(entry["keypoint_path"])
    if not video.is_file() or not track_path.is_file():
        raise FileNotFoundError(f"Missing RGB/track for {entry['clip_id']}")
    keypoints, scores = _load_track(track_path)
    indices = np.asarray(entry["frame_indices"], dtype=np.int64)
    if indices.max() >= len(keypoints):
        raise ValueError(
            f"Keypoint track is shorter than selected frame for {entry['clip_id']}: "
            f"track={len(keypoints)} max_frame={indices.max()}"
        )
    mapped, _, valid = _mapped_keypoints(keypoints[indices], scores[indices])
    if not np.isfinite(mapped).all():
        raise ValueError(f"Non-finite mapped keypoints: {entry['clip_id']}")
    contract: dict[str, Any] = {
        "keypoint_rows": len(keypoints),
        "selected_keypoint_valid_fraction": float(valid.mean()),
    }
    if verify_video:
        frames, fps, width, height = _video_contract(video)
        if indices.max() >= frames:
            raise ValueError(
                f"Video is shorter than selected frame for {entry['clip_id']}: "
                f"video={frames} max_frame={indices.max()}"
            )
        contract.update(
            {
                "video_frames": frames,
                "fps": fps,
                "width": width,
                "height": height,
            }
        )
    return contract


def _phoenix_tokens(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / f"phoenix14t.{split}"
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected SOKE token payload: {path}")
    return payload


def _phoenix_candidates(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    result = {split: [] for split in SPLITS}
    for official_split in ("train", "test"):
        for token in _phoenix_tokens(args.soke_tokens_root, official_split):
            source_clip = str(token["name"]).split("/", 1)[-1]
            signer = str(token["signer"])
            split = (
                "test"
                if official_split == "test"
                else PHOENIX_SIGNER_SPLIT.get(signer, "train")
            )
            pose_dir = args.soke_pose_root / official_split / source_clip
            pose_paths = sorted(pose_dir.glob("images*.pkl"), key=_frame_id)
            if len(pose_paths) < args.frames_per_clip:
                continue
            all_indices = np.asarray([_frame_id(path) for path in pose_paths])
            frame_indices, rows = _select_window(
                all_indices, args.frames_per_clip, args.seed, f"phoenix:{source_clip}"
            )
            selected_targets = [str(pose_paths[int(row)].resolve()) for row in rows]
            video = args.phoenix_video_root / official_split / f"{source_clip}.mp4"
            track = args.phoenix_keypoint_root / f"{source_clip}.pkl"
            result[split].append(
                {
                    "clip_id": f"phoenix_{source_clip}",
                    "source_clip": source_clip,
                    "dataset": "SOKE",
                    "target_provider": "SOKE PHOENIX released SMPL-X",
                    "official_split": official_split,
                    "phase2_split": split,
                    "signer_id": signer,
                    "source_group": f"phoenix_signer:{signer}",
                    "gloss": str(token.get("gloss", "")),
                    "video": str(video.resolve()),
                    "keypoint_path": str(track.resolve()),
                    "frame_indices": frame_indices.tolist(),
                    "target_paths": selected_targets,
                    "target_rows": rows.tolist(),
                }
            )
    return result


def _wlasl_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for gloss_record in payload:
        for instance in gloss_record["instances"]:
            video_id = str(instance["video_id"])
            if video_id in records:
                raise ValueError(f"Duplicate WLASL video ID: {video_id}")
            records[video_id] = {**instance, "gloss": str(gloss_record["gloss"])}
    return records


def _wlasl_split(signer: str, official_split: str, seed: int) -> str | None:
    if official_split == "test":
        return "test"
    if official_split != "train":
        return None
    bucket = _stable_int(seed, f"wlasl-signer:{signer}") % 100
    if bucket < 75:
        return "train"
    if bucket < 90:
        return "val"
    return "calibration"


def _wlasl_candidates(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    metadata = _wlasl_metadata(args.wlasl_metadata)
    result = {split: [] for split in SPLITS}
    for annotation in sorted(args.wlasl_annotations_root.glob("*.pkl")):
        video_id = annotation.stem
        record = metadata.get(video_id)
        if record is None:
            raise ValueError(f"SignAvatars WLASL ID is absent from metadata: {video_id}")
        signer = str(record["signer_id"])
        split = _wlasl_split(signer, str(record["split"]), args.wlasl_split_seed)
        if split is None:
            continue
        payload = _load_pickle(annotation)
        if not isinstance(payload, dict) or "smplx" not in payload:
            raise ValueError(f"Invalid SignAvatars annotation: {annotation}")
        original_indices = _numpy(payload["total_valid_index"]).astype(np.int64).reshape(-1)
        parameters = _numpy(payload["smplx"])
        if len(original_indices) != len(parameters):
            raise ValueError(
                f"SignAvatars index/target mismatch for {video_id}: "
                f"{len(original_indices)} != {len(parameters)}"
            )
        if len(original_indices) < args.frames_per_clip:
            continue
        frame_indices, rows = _select_window(
            original_indices, args.frames_per_clip, args.seed, f"wlasl:{video_id}"
        )
        result[split].append(
            {
                "clip_id": f"wlasl_{video_id}",
                "source_clip": video_id,
                "dataset": "SignAvatars",
                "target_provider": "SignAvatars WLASL released smoothed SMPL-X",
                "target_key": "smplx",
                "official_split": str(record["split"]),
                "phase2_split": split,
                "signer_id": signer,
                "source_group": f"wlasl_signer:{signer}",
                "gloss": str(record["gloss"]),
                "video": str((args.wlasl_video_root / f"{video_id}.mp4").resolve()),
                "keypoint_path": str(
                    (args.wlasl_keypoint_root / f"{video_id}.pkl").resolve()
                ),
                "annotation_path": str(annotation.resolve()),
                "frame_indices": frame_indices.tolist(),
                "target_rows": rows.tolist(),
            }
        )
    return result


def _take(entries: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=lambda item: _stable_int(seed, item["clip_id"]))
    return ordered[:maximum] if maximum > 0 else ordered


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only selection output exists: {output}")
    for name in (
        "soke_tokens_root",
        "soke_pose_root",
        "phoenix_video_root",
        "phoenix_keypoint_root",
        "wlasl_annotations_root",
        "wlasl_video_root",
        "wlasl_keypoint_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.wlasl_metadata = args.wlasl_metadata.resolve()

    phoenix = _phoenix_candidates(args)
    wlasl = _wlasl_candidates(args)
    maxima = {
        "train": args.max_phoenix_train,
        "val": args.max_phoenix_val,
        "calibration": args.max_phoenix_calibration,
        "test": args.max_phoenix_test,
    }
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "seed": args.seed,
        "wlasl_split_seed": args.wlasl_split_seed,
        "frames_per_clip": args.frames_per_clip,
        "splits": {},
    }
    groups_by_split: dict[str, set[str]] = {}
    try:
        split_dir = output / "splits"
        split_dir.mkdir()
        for split in SPLITS:
            selected_phoenix = _take(phoenix[split], maxima[split], args.seed + 1)
            selected_wlasl = _take(wlasl[split], args.max_wlasl_per_split, args.seed + 2)
            entries = sorted(
                selected_phoenix + selected_wlasl,
                key=lambda item: (item["dataset"], item["clip_id"]),
            )
            if not entries:
                raise ValueError(f"Selection produced an empty split: {split}")
            valid_fractions = []
            for index, entry in enumerate(entries, start=1):
                contract = _validate_binding(entry, args.verify_videos)
                entry["source_contract"] = contract
                valid_fractions.append(contract["selected_keypoint_valid_fraction"])
                if index % 100 == 0 or index == len(entries):
                    print(f"[selection] split={split} verified={index}/{len(entries)}", flush=True)
            groups = {entry["source_group"] for entry in entries}
            groups_by_split[split] = groups
            manifest = {
                "schema": SCHEMA,
                "dataset": "SOKE+SignAvatars",
                "split": split,
                "frames_per_clip": args.frames_per_clip,
                "clips": entries,
                "source_groups": sorted(groups),
                "sgnify_excluded": True,
            }
            path = split_dir / f"{split}.json"
            path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            counts = Counter(entry["dataset"] for entry in entries)
            report["splits"][split] = {
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "clips": len(entries),
                "frames": len(entries) * args.frames_per_clip,
                "datasets": dict(sorted(counts.items())),
                "source_groups": len(groups),
                "mean_selected_keypoint_valid_fraction": float(np.mean(valid_fractions)),
            }
        overlaps = {}
        for left_index, left in enumerate(("train", "val", "calibration")):
            for right in ("train", "val", "calibration")[left_index + 1 :]:
                overlap = sorted(groups_by_split[left] & groups_by_split[right])
                overlaps[f"{left}__{right}"] = overlap
                if overlap:
                    raise ValueError(f"Signer/source-group leakage: {overlap[:3]}")
        report["source_group_overlaps"] = overlaps
        report["passed"] = True
        report_path = output / "selection_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--wlasl-split-seed", type=int, default=12345)
    parser.add_argument("--max-phoenix-train", type=int, default=1200)
    parser.add_argument("--max-phoenix-val", type=int, default=200)
    parser.add_argument("--max-phoenix-calibration", type=int, default=100)
    parser.add_argument("--max-phoenix-test", type=int, default=200)
    parser.add_argument(
        "--max-wlasl-per-split",
        type=int,
        default=0,
        help="Zero keeps every eligible WLASL item in the assigned split.",
    )
    parser.add_argument("--verify-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--soke-tokens-root",
        type=Path,
        default=Path("data/SignAvatar_SOKE/extracted/soke_phoenix_tokens"),
    )
    parser.add_argument(
        "--soke-pose-root",
        type=Path,
        default=Path("data/SignAvatar_SOKE/extracted/soke_phoenix_frame_poses"),
    )
    parser.add_argument(
        "--phoenix-video-root",
        type=Path,
        default=Path("/home/dongvk/datasets/phoenix14T/videos_phoenix/videos"),
    )
    parser.add_argument(
        "--phoenix-keypoint-root",
        type=Path,
        default=Path(
            "/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/"
            "PHOENIX-2014-T/output/keypoint"
        ),
    )
    parser.add_argument(
        "--wlasl-annotations-root",
        type=Path,
        default=Path(
            "data/SignAvatar_SOKE/extracted/signavatars_wlasl_smplx/"
            "wlasl_pkls_cropFalse_defult_shape"
        ),
    )
    parser.add_argument(
        "--wlasl-metadata",
        type=Path,
        default=Path("data/SignAvatars/datasets/word2motion/WLASL_v0.3.json"),
    )
    parser.add_argument(
        "--wlasl-video-root",
        type=Path,
        default=Path("/home/dongvk/datasets/WLASL2000/WLASL2000"),
    )
    parser.add_argument(
        "--wlasl-keypoint-root",
        type=Path,
        default=Path("/home/dongvk/datasets/WLASL2000/keypoint"),
    )
    args = parser.parse_args()
    if args.frames_per_clip < 2:
        parser.error("--frames-per-clip must be at least 2")
    for name in (
        "max_phoenix_train",
        "max_phoenix_val",
        "max_phoenix_calibration",
        "max_phoenix_test",
        "max_wlasl_per_split",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
