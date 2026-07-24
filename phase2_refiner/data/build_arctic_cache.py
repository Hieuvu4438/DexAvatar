"""Build subject-disjoint full-pose caches from an ARCTIC raw-sequence ZIP."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import (
    NUM_JOINTS,
    NUM_OBSERVATION_FEATURES,
    CacheClip,
    save_cache_clip,
)


REFINED_BODY = (2, 5, 8, 11, 12, 13, 15, 16, 17, 18, 19, 20)
DEFAULT_SPLITS = {
    "train": ("s01", "s02", "s04", "s05", "s06", "s07"),
    "val": ("s08",),
    # s09/s10 remain untouched for later external testing.
}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _pose(payload: dict) -> np.ndarray:
    frames = len(payload["body_pose"])
    pose = np.zeros((frames, NUM_JOINTS, 3), dtype=np.float32)
    pose[:, :21] = np.asarray(payload["body_pose"], np.float32).reshape(frames, 21, 3)
    pose[:, 21:36] = np.asarray(payload["left_hand_pose"], np.float32).reshape(
        frames, 15, 3
    )
    pose[:, 36:51] = np.asarray(payload["right_hand_pose"], np.float32).reshape(
        frames, 15, 3
    )
    return pose


def _slice_or_zero(payload: dict, key: str, window: slice, width: int) -> np.ndarray:
    if key not in payload:
        length = window.stop - window.start
        return np.zeros((length, width), dtype=np.float32)
    return np.asarray(payload[key][window], dtype=np.float32).reshape(-1, width)


def _make_clip(
    archive: Path,
    member: str,
    payload: dict,
    split: str,
    subject: str,
    start: int,
    end: int,
    chunk_index: int,
    fps: float,
) -> CacheClip:
    window = slice(start, end)
    pose = _pose(payload)[window]
    frames = len(pose)
    refine = np.zeros(NUM_JOINTS, dtype=bool)
    refine[list(REFINED_BODY)] = True
    refine[21:] = True
    observations = np.zeros(
        (frames, NUM_JOINTS, NUM_OBSERVATION_FEATURES), dtype=np.float32
    )
    observations[:, refine, :2] = 1.0
    observations[..., 2] = 1.0 - observations[..., 1]
    if frames > 1:
        observations[1:, :, 5] = np.clip(
            np.linalg.norm(pose[1:] - pose[:-1], axis=-1) / np.pi, 0.0, 1.0
        )
    sequence = Path(member).name.removesuffix(".smplx.npy")
    clip_id = f"arctic_{split}_{subject}_{_safe(sequence)}_{chunk_index:03d}"
    frame_numbers = np.arange(start, end, dtype=np.int64)
    return CacheClip(
        clip_id=clip_id,
        frame_names=np.asarray(
            [f"{_safe(sequence)}_{index:06d}" for index in frame_numbers]
        ),
        frame_numbers=frame_numbers,
        timestamps=frame_numbers.astype(np.float64) / fps,
        fps=fps,
        init_axis_angle=pose.copy(),
        target_axis_angle=pose,
        target_rotation_valid=np.ones((frames, NUM_JOINTS), dtype=bool),
        observation_features=observations,
        keypoints_2d=np.zeros((frames, NUM_JOINTS, 2), dtype=np.float32),
        keypoint_valid=np.zeros((frames, NUM_JOINTS), dtype=bool),
        refine_mask=refine,
        betas=np.zeros(10, dtype=np.float32),
        global_orient=_slice_or_zero(payload, "global_orient", window, 3),
        transl=_slice_or_zero(payload, "transl", window, 3),
        jaw_pose=_slice_or_zero(payload, "jaw_pose", window, 3),
        leye_pose=_slice_or_zero(payload, "leye_pose", window, 3),
        reye_pose=_slice_or_zero(payload, "reye_pose", window, 3),
        expression=_slice_or_zero(payload, "expression", window, 10),
        source_paths=np.asarray(
            [f"{archive}!{member}#frame={index}" for index in frame_numbers]
        ),
        metadata_json=json.dumps(
            {
                "dataset": "ARCTIC",
                "official_split": split,
                "subject": subject,
                "source_sequence": sequence,
                "archive_member": member,
                "motion_domain": "generic_hand_object",
                "target_scope": "complete SMPL-X body and both hands",
                "betas_policy": "zero: raw sequence payload has no subject betas",
                "sgnify_training_reads": 0,
            },
            sort_keys=True,
        ),
    )


def build(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    archive = args.archive.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only cache output already exists: {output}")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < args.minimum_free_gb * 1024**3:
        raise OSError(
            f"Only {free / 1024**3:.2f} GiB free; require {args.minimum_free_gb:.2f} GiB"
        )
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=False)
    audits = {}
    try:
        with zipfile.ZipFile(archive) as zf:
            members = sorted(
                name for name in zf.namelist() if name.endswith(".smplx.npy")
            )
            for split, subjects in DEFAULT_SPLITS.items():
                clip_dir = output / "clips" / split
                clip_dir.mkdir(parents=True, exist_ok=False)
                entries = []
                source_sequences = source_frames = kept_frames = dropped_frames = 0
                for member in members:
                    parts = Path(member).parts
                    if len(parts) < 3 or parts[1] not in subjects:
                        continue
                    subject = parts[1]
                    payload = np.load(
                        io.BytesIO(zf.read(member)), allow_pickle=True
                    ).item()
                    lengths = {len(np.asarray(value)) for value in payload.values()}
                    if len(lengths) != 1:
                        raise ValueError(f"Inconsistent sequence fields: {member}")
                    frames = lengths.pop()
                    source_sequences += 1
                    source_frames += frames
                    for chunk_index, start in enumerate(
                        range(0, frames, args.max_frames)
                    ):
                        end = min(start + args.max_frames, frames)
                        if end - start < args.min_frames:
                            dropped_frames += end - start
                            continue
                        clip = _make_clip(
                            archive,
                            member,
                            payload,
                            split,
                            subject,
                            start,
                            end,
                            chunk_index,
                            args.fps,
                        )
                        destination = clip_dir / f"{clip.clip_id}.npz"
                        save_cache_clip(destination, clip)
                        entries.append(
                            str(Path("..") / "clips" / split / destination.name)
                        )
                        kept_frames += len(clip.frame_names)
                manifest = {
                    "dataset": "ARCTIC",
                    "official_split": split,
                    "subjects": list(subjects),
                    "clips": entries,
                    "training_target_scope": "complete-body-and-hands",
                    "motion_domain": "generic_hand_object",
                    "sgnify_excluded": True,
                }
                with (split_dir / f"{split}.json").open(
                    "x", encoding="utf-8"
                ) as handle:
                    json.dump(manifest, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                audits[split] = {
                    "subjects": list(subjects),
                    "source_sequences": source_sequences,
                    "source_frames": source_frames,
                    "clips": len(entries),
                    "kept_frames": kept_frames,
                    "dropped_short_tail_frames": dropped_frames,
                }
        with (output / "audit.json").open("x", encoding="utf-8") as handle:
            json.dump(audits, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        shutil.rmtree(output)
        raise
    return audits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--minimum-free-gb", type=float, default=10.0)
    args = parser.parse_args()
    if not 2 <= args.min_frames <= args.max_frames:
        parser.error("require 2 <= --min-frames <= --max-frames")
    return args


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
