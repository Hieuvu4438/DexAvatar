"""Build leakage-safe partial-target caches from official InterHand2.6M annotations.

The adapter uses only the requested official train/validation annotations.  It
does not read SGNify, DexAvatar evaluation ground truth, or Phase 1 outputs.
InterHand supervises the 15 articulated joints of each available hand; all
body rotations remain explicitly unsupervised.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np

from phase2_refiner.data.cache_schema import (
    NUM_JOINTS,
    NUM_OBSERVATION_FEATURES,
    CacheClip,
    save_cache_clip,
)


# InterHand skeleton joints corresponding to SMPL-X/MANO articulation order:
# index, middle, pinky, ring, thumb; MCP -> PIP -> DIP.
INTERHAND_TO_MANO = {
    "right": np.asarray([7, 6, 5, 11, 10, 9, 19, 18, 17, 15, 14, 13, 3, 2, 1]),
    "left": np.asarray([28, 27, 26, 32, 31, 30, 40, 39, 38, 36, 35, 34, 24, 23, 22]),
}
INTERHAND_WRIST = {"right": 20, "left": 41}
TOKEN_OFFSET = {"left": 21, "right": 36}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def _annotation_path(root: Path, split: str, suffix: str) -> Path:
    return root / split / f"InterHand2.6M_{split}_{suffix}.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _frame_metadata(path: Path) -> dict[tuple[int, int], dict]:
    """Stream the near-GB COCO file and retain one deterministic camera view."""
    output: dict[tuple[int, int], dict] = {}
    with path.open("rb") as handle:
        for item in ijson.items(handle, "images.item"):
            key = (int(item["capture"]), int(item["frame_idx"]))
            candidate = {
                "seq_name": str(item["seq_name"]),
                "camera": str(item["camera"]),
                "file_name": str(item["file_name"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "subject": str(item["subject"]),
            }
            current = output.get(key)
            if current is None or candidate["camera"] < current["camera"]:
                output[key] = candidate
    return output


def _palm_normal(points: np.ndarray, side: str) -> np.ndarray:
    mcp = points[[0, 3, 6, 9, 12]]
    center = mcp.mean(axis=0)
    normal = np.cross(points[0] - center, points[6] - center)
    if side == "right":
        normal = -normal
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)


def _make_clip(
    split: str,
    capture: int,
    seq_name: str,
    records: list[tuple[int, dict, dict, dict]],
    chunk_index: int,
    fps: float,
    annotation_root: Path,
) -> CacheClip:
    frames = len(records)
    target = np.zeros((frames, NUM_JOINTS, 3), dtype=np.float32)
    rotation_valid = np.zeros((frames, NUM_JOINTS), dtype=bool)
    positions = np.zeros_like(target)
    position_valid = np.zeros((frames, NUM_JOINTS), dtype=bool)
    observations = np.zeros(
        (frames, NUM_JOINTS, NUM_OBSERVATION_FEATURES), dtype=np.float32
    )
    palms = np.zeros((frames, 2, 3), dtype=np.float32)
    palm_valid = np.zeros((frames, 2), dtype=bool)
    shapes: list[np.ndarray] = []

    for frame_index, (_, _, mano, joints) in enumerate(records):
        world = np.asarray(joints["world_coord"], dtype=np.float32) / 1000.0
        for palm_index, side in enumerate(("left", "right")):
            hand = mano.get(side)
            if hand is None:
                continue
            offset = TOKEN_OFFSET[side]
            target[frame_index, offset : offset + 15] = np.asarray(
                hand["pose"], dtype=np.float32
            )[3:].reshape(15, 3)
            rotation_valid[frame_index, offset : offset + 15] = True
            mapped = world[INTERHAND_TO_MANO[side]] - world[INTERHAND_WRIST[side]]
            positions[frame_index, offset : offset + 15] = mapped
            position_valid[frame_index, offset : offset + 15] = True
            observations[frame_index, offset : offset + 15, 0] = 1.0
            observations[frame_index, offset : offset + 15, 1] = 1.0
            palms[frame_index, palm_index] = _palm_normal(mapped, side)
            palm_valid[frame_index, palm_index] = bool(
                np.linalg.norm(palms[frame_index, palm_index]) > 0
            )
            shapes.append(np.asarray(hand["shape"], dtype=np.float32))
    observations[..., 2] = 1.0 - observations[..., 1]
    if frames > 1:
        innovation = np.linalg.norm(target[1:] - target[:-1], axis=-1) / np.pi
        observations[1:, :, 5] = np.clip(innovation, 0.0, 1.0)

    frame_numbers = np.asarray([item[0] for item in records], dtype=np.int64)
    metadata = [item[1] for item in records]
    clip_id = (
        f"interhand_{split}_c{capture:02d}_{_safe_name(seq_name)}_{chunk_index:03d}"
    )
    refine_mask = np.zeros(NUM_JOINTS, dtype=bool)
    refine_mask[21:] = True
    return CacheClip(
        clip_id=clip_id,
        frame_names=np.asarray([f"frame_{value:06d}" for value in frame_numbers]),
        frame_numbers=frame_numbers,
        timestamps=frame_numbers.astype(np.float64) / fps,
        fps=fps,
        init_axis_angle=target.copy(),
        target_axis_angle=target,
        target_rotation_valid=rotation_valid,
        observation_features=observations,
        keypoints_2d=np.zeros((frames, NUM_JOINTS, 2), dtype=np.float32),
        keypoint_valid=np.zeros((frames, NUM_JOINTS), dtype=bool),
        refine_mask=refine_mask,
        betas=np.zeros(10, dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        jaw_pose=np.zeros((frames, 3), dtype=np.float32),
        leye_pose=np.zeros((frames, 3), dtype=np.float32),
        reye_pose=np.zeros((frames, 3), dtype=np.float32),
        expression=np.zeros((frames, 10), dtype=np.float32),
        source_paths=np.asarray([item["file_name"] for item in metadata]),
        image_size=np.asarray([[item["width"], item["height"]] for item in metadata]),
        keypoints_3d=positions.copy(),
        keypoint_3d_valid=position_valid.copy(),
        wrist_local_positions=positions.copy(),
        wrist_local_valid=position_valid.copy(),
        palm_normals=palms,
        palm_valid=palm_valid,
        target_joint_positions=positions,
        target_joint_valid=position_valid,
        metadata_json=json.dumps(
            {
                "dataset": "InterHand2.6M",
                "official_split": split,
                "capture": capture,
                "sequence": seq_name,
                "subject": metadata[0]["subject"],
                "camera_policy": "lexicographically-smallest-camera-metadata-only",
                "coordinate": "wrist-local metres",
                "target_scope": "MANO articulation only; body/wrists unsupervised",
                "mean_mano_shape": (
                    np.mean(shapes, axis=0).tolist() if shapes else [0.0] * 10
                ),
                "annotation_root": str(annotation_root),
                "forbidden_training_sources": [
                    "data/smplx_gt",
                    "evaluation_from_author",
                ],
            },
            sort_keys=True,
        ),
    )


def build_split(
    annotation_root: Path,
    output_root: Path,
    split: str,
    min_frames: int,
    max_frames: int,
    fps: float,
    max_clips: int | None,
) -> tuple[list[str], dict]:
    data_path = _annotation_path(annotation_root, split, "data")
    mano_path = _annotation_path(annotation_root, split, "MANO_NeuralAnnot")
    joint_path = _annotation_path(annotation_root, split, "joint_3d")
    for path in (data_path, mano_path, joint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frame_meta = _frame_metadata(data_path)
    mano = _load_json(mano_path)
    joints = _load_json(joint_path)
    sequences: dict[tuple[int, str], list[tuple[int, dict, dict, dict]]] = defaultdict(
        list
    )
    missing_metadata = missing_joints = 0
    for capture_text, capture_frames in mano.items():
        capture = int(capture_text)
        for frame_text, hand_annotation in capture_frames.items():
            frame = int(frame_text)
            metadata = frame_meta.get((capture, frame))
            if metadata is None:
                missing_metadata += 1
                continue
            joint_annotation = joints.get(capture_text, {}).get(frame_text)
            if joint_annotation is None:
                missing_joints += 1
                continue
            sequences[(capture, metadata["seq_name"])].append(
                (frame, metadata, hand_annotation, joint_annotation)
            )

    clip_dir = output_root / "clips" / split
    clip_dir.mkdir(parents=True, exist_ok=False)
    manifest_entries: list[str] = []
    source_sequences = kept_frames = dropped_short_frames = 0
    hand_frames = {"left": 0, "right": 0, "both": 0}
    stop = False
    for (capture, seq_name), records in sorted(sequences.items()):
        records.sort(key=lambda item: item[0])
        if len(records) < min_frames:
            dropped_short_frames += len(records)
            continue
        source_sequences += 1
        for chunk_index, start in enumerate(range(0, len(records), max_frames)):
            chunk = records[start : start + max_frames]
            if len(chunk) < min_frames:
                dropped_short_frames += len(chunk)
                continue
            clip = _make_clip(
                split,
                capture,
                seq_name,
                chunk,
                chunk_index,
                fps,
                annotation_root,
            )
            destination = clip_dir / f"{clip.clip_id}.npz"
            save_cache_clip(destination, clip)
            manifest_entries.append(
                str(Path("..") / "clips" / split / destination.name)
            )
            kept_frames += len(chunk)
            for _, _, annotation, _ in chunk:
                left = annotation.get("left") is not None
                right = annotation.get("right") is not None
                hand_frames["left"] += int(left)
                hand_frames["right"] += int(right)
                hand_frames["both"] += int(left and right)
            if max_clips is not None and len(manifest_entries) >= max_clips:
                stop = True
                break
        if stop:
            break
    audit = {
        "dataset": "InterHand2.6M",
        "official_split": split,
        "clips": len(manifest_entries),
        "source_sequences": source_sequences,
        "frames": kept_frames,
        "hand_frames": hand_frames,
        "minimum_frames": min_frames,
        "maximum_frames": max_frames,
        "missing_metadata": missing_metadata,
        "missing_joint_annotations": missing_joints,
        "dropped_short_frames": dropped_short_frames,
        "sgnify_training_reads": 0,
        "annotation_files": {
            path.name: {
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in (data_path, mano_path, joint_path)
        },
    }
    return manifest_entries, audit


def build(args: argparse.Namespace) -> dict:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"Append-only cache output already exists: {root}")
    usage_parent = root.parent
    while not usage_parent.exists():
        usage_parent = usage_parent.parent
    free = shutil.disk_usage(usage_parent).free
    if free < args.minimum_free_gb * 1024**3:
        raise OSError(
            f"Only {free / 1024**3:.2f} GiB free; require {args.minimum_free_gb:.2f} GiB"
        )
    (root / "splits").mkdir(parents=True, exist_ok=False)
    audits = {}
    try:
        for split in args.splits:
            entries, audit = build_split(
                args.annotations.resolve(),
                root,
                split,
                args.min_frames,
                args.max_frames,
                args.fps,
                args.max_clips,
            )
            payload = {
                "dataset": "InterHand2.6M",
                "official_split": split,
                "clips": entries,
                "training_target_scope": "hands-only",
                "sgnify_excluded": True,
            }
            with (root / "splits" / f"{split}.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            audits[split] = audit
        with (root / "audit.json").open("x", encoding="utf-8") as handle:
            json.dump(audits, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        # Do not leave an apparently valid partial cache after a failed build.
        shutil.rmtree(root)
        raise
    return audits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val"), default=("train", "val")
    )
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--minimum-free-gb", type=float, default=5.0)
    args = parser.parse_args()
    if not 2 <= args.min_frames <= args.max_frames:
        parser.error("require 2 <= --min-frames <= --max-frames")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
