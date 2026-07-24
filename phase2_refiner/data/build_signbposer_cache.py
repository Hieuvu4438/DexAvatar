"""Build body-only spatial warm-up caches from the SignBPoser pose bank.

Each pose is deliberately stored as a one-frame sample because the supplied
metadata does not contain sequence identity or temporal order.  These caches
must not be described as whole-sequence sign supervision.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import (
    NUM_JOINTS,
    NUM_OBSERVATION_FEATURES,
    CacheClip,
    save_cache_clip,
)


REFINED_BODY = np.asarray((2, 5, 8, 11, 12, 13, 15, 16, 17, 18, 19, 20))


def build(input_root: Path, output: Path, max_samples: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Append-only cache output already exists: {output}")
    poses = np.load(input_root / "body_poses.npy", mmap_mode="r")
    weights = np.load(input_root / "sample_weights.npy", mmap_mode="r")
    with (input_root / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    if poses.ndim != 2 or poses.shape[1] != 63:
        raise ValueError(f"Expected body_poses [N,63], got {poses.shape}")
    if len(weights) != len(poses) or len(metadata) != len(poses):
        raise ValueError("SignBPoser arrays and metadata have different lengths")
    count = min(len(poses), max_samples) if max_samples is not None else len(poses)
    clip_dir = output / "clips" / "train"
    split_dir = output / "splits"
    clip_dir.mkdir(parents=True, exist_ok=False)
    split_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for index in range(count):
        pose = np.zeros((1, NUM_JOINTS, 3), dtype=np.float32)
        pose[0, :21] = np.asarray(poses[index], dtype=np.float32).reshape(21, 3)
        valid = np.zeros((1, NUM_JOINTS), dtype=bool)
        valid[:, REFINED_BODY] = True
        observations = np.zeros((1, NUM_JOINTS, NUM_OBSERVATION_FEATURES), np.float32)
        observations[:, REFINED_BODY, 0:2] = 1.0
        observations[..., 2] = 1.0 - observations[..., 1]
        refine = np.zeros(NUM_JOINTS, dtype=bool)
        refine[REFINED_BODY] = True
        clip_id = f"signbposer_pose_{index:06d}"
        clip = CacheClip(
            clip_id=clip_id,
            frame_names=np.asarray([clip_id]),
            init_axis_angle=pose.copy(),
            target_axis_angle=pose,
            target_rotation_valid=valid,
            observation_features=observations,
            keypoints_2d=np.zeros((1, NUM_JOINTS, 2), np.float32),
            keypoint_valid=np.zeros((1, NUM_JOINTS), bool),
            refine_mask=refine,
            betas=np.zeros(10, np.float32),
            global_orient=np.zeros((1, 3), np.float32),
            transl=np.zeros((1, 3), np.float32),
            jaw_pose=np.zeros((1, 3), np.float32),
            leye_pose=np.zeros((1, 3), np.float32),
            reye_pose=np.zeros((1, 3), np.float32),
            expression=np.zeros((1, 10), np.float32),
            source_paths=np.asarray([str(input_root / "body_poses.npy")]),
            metadata_json=json.dumps(
                {
                    "dataset": "DexAvatar SignBPoser pose bank",
                    "temporal_supervision": False,
                    "source_metadata": metadata[index],
                    "sample_weight": float(weights[index]),
                    "target_scope": "body-only spatial warm-up",
                    "sgnify_training_reads": 0,
                },
                sort_keys=True,
            ),
        )
        destination = clip_dir / f"{clip_id}.npz"
        save_cache_clip(destination, clip)
        entries.append(str(Path("..") / "clips" / "train" / destination.name))
    with (split_dir / "train.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "DexAvatar SignBPoser pose bank",
                "clips": entries,
                "temporal_supervision": False,
                "training_target_scope": "body-only",
                "sgnify_excluded": True,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    audit = {
        "samples": count,
        "pose_shape": list(poses.shape),
        "temporal_supervision": False,
        "sgnify_training_reads": 0,
    }
    with (output / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build(args.input.resolve(), args.output.resolve(), args.max_samples),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
