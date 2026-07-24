"""Build Phase 2 sign-sequence caches from frozen How2Sign SMPL-X targets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import (
    NUM_JOINTS,
    NUM_OBSERVATION_FEATURES,
    CacheClip,
    save_cache_clip,
)


REFINED_BODY = (2, 5, 8, 11, 12, 13, 15, 16, 17, 18, 19, 20)
BODY_MAP = (
    (11,),
    (12,),
    (11, 12),
    (13,),
    (14,),
    (5, 6, 11, 12),
    (15,),
    (16,),
    (5, 6, 11, 12),
    (17, 19, 21),
    (18, 20, 22),
    (5, 6),
    (5,),
    (6,),
    (0, 1, 2, 3, 4),
    (5,),
    (6,),
    (7,),
    (8,),
    (9,),
    (10,),
)
HAND_MAP = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)


def _pose(payload: np.lib.npyio.NpzFile) -> np.ndarray:
    frames = len(payload["body_pose"])
    pose = np.zeros((frames, NUM_JOINTS, 3), dtype=np.float32)
    pose[:, :21] = payload["body_pose"].reshape(frames, 21, 3)
    pose[:, 21:36] = payload["left_hand_pose"].reshape(frames, 15, 3)
    pose[:, 36:51] = payload["right_hand_pose"].reshape(frames, 15, 3)
    return pose


def _mapped_keypoints(
    keypoints: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = len(keypoints)
    mapped = np.zeros((frames, NUM_JOINTS, 2), dtype=np.float32)
    logits = np.zeros((frames, NUM_JOINTS), dtype=np.float32)
    for destination, source in enumerate(BODY_MAP):
        mapped[:, destination] = keypoints[:, source].mean(axis=1)
        logits[:, destination] = scores[:, source].mean(axis=1)
    for offset, source in enumerate(HAND_MAP):
        mapped[:, 21 + offset] = keypoints[:, 91 + source]
        logits[:, 21 + offset] = scores[:, 91 + source]
        mapped[:, 36 + offset] = keypoints[:, 112 + source]
        logits[:, 36 + offset] = scores[:, 112 + source]
    confidence = 1.0 / (1.0 + np.exp(-(logits - 3.0)))
    valid = np.isfinite(mapped).all(axis=-1)
    valid &= (mapped[..., 0] >= -0.05) & (mapped[..., 0] <= 1.05)
    valid &= (mapped[..., 1] >= -0.05) & (mapped[..., 1] <= 1.05)
    valid &= confidence >= 0.05
    return mapped, confidence.astype(np.float32), valid


def _observations(
    keypoints: np.ndarray, confidence: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    frames = len(keypoints)
    observations = np.zeros(
        (frames, NUM_JOINTS, NUM_OBSERVATION_FEATURES), dtype=np.float32
    )
    observations[..., 0] = confidence
    observations[..., 1] = valid
    observations[..., 2] = ~valid
    observations[..., 4] = (
        (keypoints[..., 0] < 0.0)
        | (keypoints[..., 0] > 1.0)
        | (keypoints[..., 1] < 0.0)
        | (keypoints[..., 1] > 1.0)
    )
    if frames > 1:
        displacement = np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=-1)
        observations[1:, :, 5] = np.clip(displacement / 0.15, 0.0, 1.0)
    return observations


def _source_group(clip_id: str) -> str:
    # How2Sign clip names begin with the 11-character YouTube video ID.  The
    # ID alphabet itself includes underscores, so splitting on '_' is unsafe.
    if len(clip_id) < 12:
        raise ValueError(f"Invalid How2Sign clip id: {clip_id}")
    return clip_id[:11]


def _quality(pose: np.ndarray) -> dict[str, float | bool]:
    joint_angle = np.linalg.norm(pose, axis=-1)
    frame_outlier = (joint_angle > math.pi * 1.25).any(axis=-1)
    finite = np.isfinite(pose).all(axis=(1, 2))
    catastrophic = (~finite) | frame_outlier
    fraction = float(catastrophic.mean())
    return {
        "catastrophic_frame_fraction": fraction,
        "max_joint_axis_angle_radians": float(joint_angle.max()),
        "passed": fraction < 0.10,
    }


def _make_clip(
    teacher_path: Path,
    selection: dict,
    split: str,
    quality: dict,
) -> CacheClip:
    with np.load(teacher_path, allow_pickle=False) as payload:
        pose = _pose(payload)
        indices = payload["sample_indices"].astype(np.int64)
        keypoints, confidence, keypoint_valid = _mapped_keypoints(
            payload["keypoints_2d"].astype(np.float32),
            payload["keypoint_scores"].astype(np.float32),
        )
        observations = _observations(keypoints, confidence, keypoint_valid)
        refine = np.zeros(NUM_JOINTS, dtype=bool)
        refine[list(REFINED_BODY)] = True
        refine[21:] = True
        fps = float(payload["fps"])
        clip_id = selection["clip_id"]
        betas = np.median(payload["betas"], axis=0).astype(np.float32)
        source_paths = np.asarray(
            [f"{selection['video']}#frame={int(index)}" for index in indices]
        )
        return CacheClip(
            clip_id=f"how2sign_{split}_{clip_id}",
            frame_names=np.asarray(
                [f"{clip_id}_{int(index):06d}" for index in indices]
            ),
            frame_numbers=indices,
            timestamps=indices.astype(np.float64) / fps,
            fps=fps,
            image_size=np.repeat(payload["image_size"][None], len(pose), axis=0),
            init_axis_angle=pose.copy(),
            target_axis_angle=pose,
            target_rotation_valid=np.ones((len(pose), NUM_JOINTS), dtype=bool),
            observation_features=observations,
            keypoints_2d=keypoints,
            keypoint_valid=keypoint_valid,
            refine_mask=refine,
            betas=betas,
            global_orient=payload["global_orient"].reshape(-1, 3),
            transl=payload["transl"].reshape(-1, 3),
            jaw_pose=payload["jaw_pose"].reshape(-1, 3),
            leye_pose=np.zeros((len(pose), 3), dtype=np.float32),
            reye_pose=np.zeros((len(pose), 3), dtype=np.float32),
            expression=payload["expression"].reshape(-1, 10),
            source_paths=source_paths,
            u0_reliability=(confidence * keypoint_valid).astype(np.float32),
            metadata_json=json.dumps(
                {
                    "dataset": "How2Sign",
                    "official_split": split,
                    "source_group": _source_group(clip_id),
                    "source_clip": clip_id,
                    "motion_domain": "sign_language_asl",
                    "target_scope": "complete SMPL-X body and both hands",
                    "target_type": "SMPLer-X H32 pseudo-3D",
                    "teacher_path": str(teacher_path.resolve()),
                    "quality": quality,
                    "sgnify_training_reads": 0,
                },
                sort_keys=True,
            ),
        )


def _load_selection(root: Path, split: str) -> tuple[dict, list[dict]]:
    path = root / split / "selection.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("official_split") != split:
        raise ValueError(f"Official split mismatch: {path}")
    if payload.get("motion_domain") != "sign_language_asl":
        raise ValueError(f"Unexpected motion domain: {path}")
    return payload, payload["clips"]


def build(args: argparse.Namespace) -> dict:
    teacher_root = args.teacher_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only cache output already exists: {output}")
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=False)
    audit = {}
    split_groups = {}
    try:
        for split in ("train", "val"):
            selection_payload, selected = _load_selection(teacher_root, split)
            clip_dir = output / "clips" / split
            clip_dir.mkdir(parents=True, exist_ok=False)
            entries = []
            groups = set()
            rejected = []
            for item in selected:
                teacher_path = teacher_root / split / "clips" / f"{item['clip_id']}.npz"
                if not teacher_path.is_file():
                    continue
                with np.load(teacher_path, allow_pickle=False) as payload:
                    pose = _pose(payload)
                quality = _quality(pose)
                if not quality["passed"]:
                    rejected.append({"clip_id": item["clip_id"], **quality})
                    continue
                clip = _make_clip(teacher_path, item, split, quality)
                destination = clip_dir / f"{clip.clip_id}.npz"
                save_cache_clip(destination, clip)
                entries.append(str(Path("..") / "clips" / split / destination.name))
                groups.add(_source_group(item["clip_id"]))
            minimum = args.minimum_train_clips if split == "train" else 1
            if len(entries) < minimum:
                raise ValueError(
                    f"{split} has only {len(entries)} accepted clips; require {minimum}"
                )
            manifest = {
                "dataset": "How2Sign",
                "official_split": split,
                "clips": entries,
                "source_groups": sorted(groups),
                "motion_domain": "sign_language_asl",
                "training_target_scope": "complete-body-and-hands",
                "target_type": "SMPLer-X H32 pseudo-3D",
                "teacher_selection": str(
                    (teacher_root / split / "selection.json").resolve()
                ),
                "sgnify_excluded": True,
            }
            with (split_dir / f"{split}.json").open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            audit[split] = {
                "selected": len(selected),
                "accepted_clips": len(entries),
                "accepted_frames": len(entries)
                * int(selection_payload["frames_per_clip"]),
                "source_groups": len(groups),
                "rejected_quality": rejected,
            }
            split_groups[split] = groups
        overlap = sorted(split_groups["train"] & split_groups["val"])
        if overlap:
            raise ValueError(f"Train/validation source-group overlap: {overlap[:3]}")
        audit["train_validation_source_group_overlap"] = overlap
        if args.generic_train_manifest is not None:
            generic_manifest = args.generic_train_manifest.resolve()
            with generic_manifest.open("r", encoding="utf-8") as handle:
                generic_payload = json.load(handle)
            generic_entries = generic_payload.get("clips", generic_payload)
            if not isinstance(generic_entries, list):
                raise ValueError(f"Invalid generic manifest: {generic_manifest}")
            resolved_generic = [
                str(
                    (
                        generic_manifest.parent / entry
                        if not Path(entry).is_absolute()
                        else Path(entry)
                    ).resolve()
                )
                for entry in generic_entries
            ]
            with (split_dir / "train.json").open("r", encoding="utf-8") as handle:
                sign_payload = json.load(handle)
            resolved_sign = [
                str((split_dir / entry).resolve())
                if not Path(entry).is_absolute()
                else str(Path(entry).resolve())
                for entry in sign_payload["clips"]
            ]
            mixed = {
                "dataset": "How2Sign + generic retention",
                "official_split": "train",
                "clips": resolved_sign + resolved_generic,
                "sign_clips": len(resolved_sign),
                "generic_clips": len(resolved_generic),
                "generic_fraction": len(resolved_generic)
                / (len(resolved_sign) + len(resolved_generic)),
                "generic_manifest": str(generic_manifest),
                "sgnify_excluded": True,
            }
            with (split_dir / "train_mixed_generic.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(mixed, handle, indent=2, sort_keys=True)
                handle.write("\n")
            audit["mixed_training"] = {
                key: mixed[key]
                for key in ("sign_clips", "generic_clips", "generic_fraction")
            }
        with (output / "audit.json").open("x", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        shutil.rmtree(output)
        raise
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-train-clips", type=int, default=10000)
    parser.add_argument("--generic-train-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
