"""Build an immutable Phase 2 cache without modifying existing method outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from phase2_refiner.data.cache_schema import (
    NUM_JOINTS,
    NUM_OBSERVATION_FEATURES,
    SCHEMA_VERSION,
    CacheClip,
    save_cache_clip,
)
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance


BODY_NAMES = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
HAND_NAMES = [
    "index_1",
    "index_2",
    "index_3",
    "middle_1",
    "middle_2",
    "middle_3",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "ring_1",
    "ring_2",
    "ring_3",
    "thumb_1",
    "thumb_2",
    "thumb_3",
]
COCO_BODY_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
COCO_HAND_ORDER = [
    "root",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_tip",
    "index_1",
    "index_2",
    "index_3",
    "index_tip",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_tip",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_tip",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_tip",
]
REFINED_BODY_NAMES = {
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "left_collar",
    "right_collar",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
}


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"No frame number in {path.name}")
    return int(match.group(1))


def _array(value: Any, size: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < size:
        array = np.pad(array, (0, size - array.size), constant_values=default)
    return array[:size]


def _numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_from_params(params: dict[str, Any]) -> np.ndarray:
    body = _array(params.get("body_pose"), 63).reshape(21, 3)
    left = _array(params.get("left_hand_pose"), 45).reshape(15, 3)
    right = _array(params.get("right_hand_pose"), 45).reshape(15, 3)
    pose = np.concatenate((body, left, right), axis=0)
    if pose.shape != (NUM_JOINTS, 3):
        raise AssertionError(pose.shape)
    return pose


def _sapiens_frame(
    sapiens: dict[str, Any], sign: str, frame_name: str
) -> tuple[np.ndarray, np.ndarray]:
    keys = (f"{sign}/{frame_name}.png", f"{sign}/{frame_name}.jpg", f"{frame_name}.png")
    entry = next((sapiens[key] for key in keys if key in sapiens), None)
    if entry is None:
        return np.zeros((133, 2), np.float32), np.zeros(133, np.float32)
    points = np.asarray(entry[0], dtype=np.float32).reshape(-1, 2)
    confidence = np.asarray(entry[1], dtype=np.float32).reshape(-1)
    if points.shape[0] < 133 or confidence.shape[0] < 133:
        raise ValueError(f"Malformed Sapiens entry for {sign}/{frame_name}")
    return points[:133], confidence[:133]


def _token_2d_and_conf(
    points: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    conf = np.zeros(NUM_JOINTS, dtype=np.float32)
    body_map = {name: idx for idx, name in enumerate(COCO_BODY_NAMES)}
    torso_fallback = float(np.mean(confidence[[5, 6, 11, 12]]))
    torso_point = np.mean(points[[5, 6, 11, 12]], axis=0)
    for idx, name in enumerate(BODY_NAMES):
        if name in body_map:
            source = body_map[name]
            output[idx] = points[source]
            conf[idx] = confidence[source]
        else:
            output[idx] = torso_point
            conf[idx] = torso_fallback
    for side_idx, side in enumerate(("left", "right")):
        hand_offset = 91 if side == "left" else 112
        hand_map = {name: hand_offset + idx for idx, name in enumerate(COCO_HAND_ORDER)}
        for local_idx, name in enumerate(HAND_NAMES):
            target_idx = 21 + side_idx * 15 + local_idx
            source = hand_map[name]
            output[target_idx] = points[source]
            conf[target_idx] = confidence[source]
    return output, conf


def _hand_metadata(
    hamer: dict[str, Any], frame_name: str, width: int, height: int
) -> dict[str, tuple[bool, float, float, bool]]:
    entry = hamer.get(f"{frame_name}.png") or hamer.get(f"{frame_name}.jpg")
    result: dict[str, tuple[bool, float, float, bool]] = {
        "left": (False, 0.0, 1.0, False),
        "right": (False, 0.0, 1.0, False),
    }
    if entry is None or len(entry) < 4:
        return result
    centers = _numpy(entry[1], np.float32).reshape(-1, 2)
    sizes = _numpy(entry[2], np.float32).reshape(-1)
    sides = _numpy(entry[3]).reshape(-1)
    for side_value in (0, 1):
        indices = np.flatnonzero(sides.astype(int) == side_value)
        side = "right" if side_value else "left"
        if len(indices) == 0:
            continue
        idx = int(indices[0])
        size = float(sizes[idx])
        scale = size / max(float(max(width, height)), 1.0)
        cx, cy = centers[idx]
        half = size * 0.5
        outside = max(0.0, half - cx) + max(0.0, cx + half - width)
        outside += max(0.0, half - cy) + max(0.0, cy + half - height)
        truncation = min(1.0, outside / max(2.0 * size, 1.0))
        result[side] = (True, scale, truncation, len(indices) > 1)
    return result


def _refine_mask() -> np.ndarray:
    mask = np.zeros(NUM_JOINTS, dtype=bool)
    for idx, name in enumerate(BODY_NAMES):
        mask[idx] = name in REFINED_BODY_NAMES
    mask[21:] = True
    return mask


def build_clip(
    sign: str,
    result_files: list[Path],
    frames_root: Path,
    sapiens: dict[str, Any],
    hamer: dict[str, Any],
    target_dir: Path | None,
) -> CacheClip:
    poses, targets, frame_names, source_paths = [], [], [], []
    globals_, translations, jaws, leyes, reyes, expressions, betas = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    observations, points_2d, point_valid = [], [], []

    for result_path in sorted(result_files, key=_frame_number):
        name = result_path.stem
        params = _load_pickle(result_path)
        pose = _pose_from_params(params)
        image_path = frames_root / sign / f"{name}.png"
        if not image_path.exists():
            image_path = frames_root / sign / f"{name}.jpg"
        image = cv2.imread(str(image_path)) if image_path.exists() else None
        height, width = image.shape[:2] if image is not None else (1, 1)
        sapiens_points, sapiens_conf = _sapiens_frame(sapiens, sign, name)
        token_points, token_conf = _token_2d_and_conf(sapiens_points, sapiens_conf)
        token_points[:, 0] = token_points[:, 0] / max(width, 1) * 2.0 - 1.0
        token_points[:, 1] = token_points[:, 1] / max(height, 1) * 2.0 - 1.0
        valid = token_conf > 0.0
        metadata = _hand_metadata(hamer, name, width, height)
        features = np.zeros((NUM_JOINTS, NUM_OBSERVATION_FEATURES), dtype=np.float32)
        features[:, 0] = np.clip(token_conf, 0.0, 1.0)
        features[:21, 1] = valid[:21]
        for side_idx, side in enumerate(("left", "right")):
            start = 21 + side_idx * 15
            end = start + 15
            present, crop_scale, truncation, duplicate = metadata[side]
            features[start:end, 1] = float(present)
            features[start:end, 3] = crop_scale
            features[start:end, 4] = truncation
            features[start:end, 6] = float(duplicate)
            valid[start:end] &= present
        features[:, 2] = 1.0 - features[:, 1]
        features[:, 7] = 0.0

        poses.append(pose)
        observations.append(features)
        points_2d.append(token_points)
        point_valid.append(valid)
        frame_names.append(name)
        source_paths.append(str(result_path.resolve()))
        betas.append(_array(params.get("betas"), 10))
        globals_.append(_array(params.get("global_orient"), 3))
        translations.append(_array(params.get("transl"), 3))
        jaws.append(_array(params.get("jaw_pose"), 3))
        leyes.append(_array(params.get("leye_pose"), 3))
        reyes.append(_array(params.get("reye_pose"), 3))
        expressions.append(_array(params.get("expression"), 10))
        if target_dir is not None:
            target_path = target_dir / result_path.name
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Missing target for {sign}/{result_path.name}: {target_path}"
                )
            targets.append(_pose_from_params(_load_pickle(target_path)))

    pose_array = np.stack(poses).astype(np.float32)
    matrices = axis_angle_to_matrix(torch.from_numpy(pose_array))
    if len(pose_array) > 1:
        innovation = geodesic_distance(matrices[1:], matrices[:-1]).numpy() / np.pi
        innovation = np.concatenate((innovation[:1], innovation), axis=0)
    else:
        innovation = np.zeros((1, NUM_JOINTS), dtype=np.float32)
    observation_array = np.stack(observations)
    observation_array[:, :, 5] = innovation

    return CacheClip(
        clip_id=sign,
        frame_names=np.asarray(frame_names),
        init_axis_angle=pose_array,
        observation_features=observation_array,
        keypoints_2d=np.stack(points_2d).astype(np.float32),
        keypoint_valid=np.stack(point_valid),
        refine_mask=_refine_mask(),
        betas=np.median(np.stack(betas), axis=0).astype(np.float32),
        global_orient=np.stack(globals_).astype(np.float32),
        transl=np.stack(translations).astype(np.float32),
        jaw_pose=np.stack(jaws).astype(np.float32),
        leye_pose=np.stack(leyes).astype(np.float32),
        reye_pose=np.stack(reyes).astype(np.float32),
        expression=np.stack(expressions).astype(np.float32),
        source_paths=np.asarray(source_paths),
        target_axis_angle=np.stack(targets).astype(np.float32) if targets else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--initializer-root", type=Path, required=True)
    parser.add_argument("--initializer-subdir", default="smplifyx/results")
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--target-subdir", default="smplifyx/results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sign", action="append", help="Limit to one or more signs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initializer_root = args.initializer_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    signs = sorted(path.name for path in initializer_root.iterdir() if path.is_dir())
    if args.sign:
        requested = set(args.sign)
        signs = [sign for sign in signs if sign in requested]
    if not signs:
        raise ValueError("No matching sign directories found")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "initializer_root": str(initializer_root),
        "initializer_subdir": args.initializer_subdir,
        "target_root": str(args.target_root.resolve()) if args.target_root else None,
        "clips": [],
    }
    for sign in signs:
        result_dir = initializer_root / sign / args.initializer_subdir
        result_files = list(result_dir.glob("*.pkl"))
        if not result_files:
            continue
        sapiens_path = initializer_root / sign / "sapiens.pkl"
        hamer_path = initializer_root / sign / "hamer" / "hamer.pkl"
        sapiens = _load_pickle(sapiens_path) if sapiens_path.exists() else {}
        hamer = _load_pickle(hamer_path) if hamer_path.exists() else {}
        target_dir = (
            args.target_root / sign / args.target_subdir if args.target_root else None
        )
        clip = build_clip(
            sign, result_files, args.frames_root, sapiens, hamer, target_dir
        )
        cache_path = output_root / "clips" / f"{sign}.npz"
        if cache_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing cache: {cache_path}")
        save_cache_clip(cache_path, clip)
        manifest["clips"].append(
            {
                "clip_id": sign,
                "frames": len(clip.frame_names),
                "cache": str(cache_path.relative_to(output_root)),
                "sha256": _sha256(cache_path),
                "has_target": clip.target_axis_angle is not None,
            }
        )
        print(f"[cache] {sign}: {len(clip.frame_names)} frames -> {cache_path}")
    manifest_path = output_root / "manifest.json"
    with manifest_path.open(
        "x" if not args.overwrite else "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[cache] wrote {len(manifest['clips'])} clips -> {manifest_path}")


if __name__ == "__main__":
    main()
