"""Build full-PHOENIX caches from target-independent H32 + WiLoR experts.

The initializer is SMPLer-X H32 for all 51 local rotations, with each detected
WiLoR hand replacing its 15 finger rotations.  A side-specific WiLoR detector
dropout falls back to the corresponding H32 hand.  Released SOKE pose files
are opened only for supervision and are bound to RGB by their original
one-based ``imagesNNNN.pkl`` frame number.

WiLoR artifacts are consumed shard-by-shard so the full 810k-frame training
set never has to be loaded into memory at once.  Every raw-v3 artifact must
prove exact coverage of its immutable frame manifest before a cache is written.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase2_refiner.data.build_how2sign_cache import REFINED_BODY, _observations
from phase2_refiner.data.build_sign_domain_cache import _wilor_hands
from phase2_refiner.data.cache_schema import (
    NUM_JOINTS,
    PHASE2R_SEMANTIC_CONTRACT,
    CacheClip,
    save_cache_clip,
)
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-phoenix-soke-full-expert-cache-v1"
BODY_PROJECTED_MAP = (
    (1,),
    (2,),
    (0, 1, 2, 7),
    (3,),
    (4,),
    (0, 1, 2, 7),
    (5,),
    (6,),
    (0, 7, 8, 9),
    (5, 14, 15, 16),
    (6, 17, 18, 19),
    (7,),
    (7, 8),
    (7, 9),
    (20, 21, 22, 23, 24),
    (8,),
    (9,),
    (10,),
    (11,),
    (12,),
    (13,),
)
HAND_PROJECTED_MAP = (4, 5, 6, 8, 9, 10, 16, 17, 18, 12, 13, 14, 0, 1, 2)


def _load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _projected_to_image(payload: dict[str, Any]) -> np.ndarray:
    """Map H32 heatmap-space projections through its saved crop affine."""

    points = _numpy(payload["pred_2d"], np.float32).copy()
    if points.ndim != 3 or points.shape[1:] != (106, 2):
        raise ValueError(f"Unexpected H32 pred_2d shape: {points.shape}")
    # H32 projects into a 12x16 heatmap over a 384x512 model input crop.
    points[..., 0] *= 384.0 / 12.0
    points[..., 1] *= 512.0 / 16.0
    homogeneous = np.concatenate(
        (points, np.ones((*points.shape[:-1], 1), dtype=np.float32)), axis=-1
    )
    affine = _numpy(payload["bb2img_trans"], np.float32)
    image = np.einsum("tij,tkj->tki", affine, homogeneous)
    image[..., 0] /= float(payload["width"])
    image[..., 1] /= float(payload["height"])
    return image.astype(np.float32)


def _map_projected(points: np.ndarray) -> np.ndarray:
    mapped = np.zeros((len(points), NUM_JOINTS, 2), dtype=np.float32)
    for destination, source in enumerate(BODY_PROJECTED_MAP):
        mapped[:, destination] = points[:, source].mean(axis=1)
    for destination, source in enumerate(HAND_PROJECTED_MAP):
        mapped[:, 21 + destination] = points[:, 25 + source]
        mapped[:, 36 + destination] = points[:, 45 + source]
    return mapped


def _select_hand(hands: list[dict[str, Any]], is_right: bool) -> tuple[dict | None, bool]:
    matching = [
        hand
        for hand in hands
        if bool(round(float(hand.get("is_right", -1)))) is is_right
    ]
    if not matching:
        return None, False
    selected = max(matching, key=lambda hand: float(hand.get("detector_confidence", 0.0)))
    return selected, len(matching) > 1


def _target_pose(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    poses = []
    valid = []
    for path in paths:
        payload = _load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"SOKE target must be a mapping: {path}")

        def exact_field(name: str, width: int) -> np.ndarray:
            if name not in payload:
                raise ValueError(f"SOKE target lacks {name}: {path}")
            field = _numpy(payload[name], np.float32).reshape(-1)
            if field.size != width:
                raise ValueError(
                    f"SOKE target {name} has {field.size} values, expected "
                    f"{width}: {path}"
                )
            return field

        pose = np.concatenate(
            (
                exact_field("smplx_body_pose", 63),
                exact_field("smplx_lhand_pose", 45),
                exact_field("smplx_rhand_pose", 45),
            )
        ).reshape(NUM_JOINTS, 3)
        finite = np.isfinite(pose).all(axis=-1)
        poses.append(np.where(finite[:, None], pose, 0.0))
        valid.append(finite)
    return np.asarray(poses, dtype=np.float32), np.asarray(valid, dtype=bool)


def _h32_pose(parameters: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (parameters[:, 3:66], parameters[:, 66:111], parameters[:, 111:156]),
        axis=1,
    ).reshape(-1, NUM_JOINTS, 3).astype(np.float32)


def _artifact_paths(root: Path, shard_index: int) -> tuple[Path, Path]:
    shard = root / f"shard_{shard_index:04d}"
    return shard / "hamer" / "hamer.pkl", shard / "wilor" / "wilor.pkl"


def _verified_artifacts(
    root: Path, shard_manifest: Path, shard_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    hamer_path, wilor_path = _artifact_paths(root, shard_index)
    if not hamer_path.is_file():
        raise FileNotFoundError(hamer_path)
    if not wilor_path.is_file():
        raise FileNotFoundError(wilor_path)
    hamer = _load(hamer_path)
    wilor = _load(wilor_path)
    images = wilor.get("images", {})
    metadata = wilor.get("meta", {})
    expected_hash = sha256_file(shard_manifest)
    if metadata.get("frame_manifest_sha256") != expected_hash:
        raise ValueError(f"WiLoR/shard manifest hash mismatch: {wilor_path}")
    if metadata.get("frame_manifest_sources_verified") is not True:
        raise ValueError(f"WiLoR source verification is absent: {wilor_path}")
    manifest = json.loads(shard_manifest.read_text(encoding="utf-8"))
    expected_keys = {str(record["image_key"]) for record in manifest["records"]}
    if set(images) != expected_keys:
        raise ValueError(
            f"WiLoR raw coverage mismatch shard={shard_index}: "
            f"actual={len(images)} expected={len(expected_keys)}"
        )
    if not isinstance(hamer, dict):
        raise ValueError(f"Invalid HaMeR compatibility artifact: {hamer_path}")
    unexpected_hamer = set(hamer) - expected_keys
    if unexpected_hamer:
        sample = ", ".join(sorted(map(str, unexpected_hamer))[:5])
        raise ValueError(
            f"HaMeR keys outside shard manifest shard={shard_index}: {sample}"
        )
    return hamer, wilor


def _make_clip(
    entry: dict[str, Any],
    h32_path: Path,
    hamer: dict[str, Any],
    raw_images: dict[str, Any],
) -> tuple[CacheClip, dict[str, Any]]:
    h32 = _load(h32_path)
    all_indices = _numpy(h32["total_valid_index"], np.int64).reshape(-1)
    if len(all_indices) != len(set(all_indices.tolist())):
        raise ValueError(f"Duplicate H32 valid frame index: {h32_path}")
    rows = {int(frame): row for row, frame in enumerate(all_indices)}
    requested = [int(frame) for frame in entry["frame_indices"]]
    selected_indices = np.asarray(requested, dtype=np.int64)
    if not len(all_indices):
        raise ValueError(f"H32 retained no frames: {entry['clip_id']}")
    exact_h32 = np.asarray([int(frame) in rows for frame in selected_indices], dtype=bool)
    # Preserve full SOKE evaluation coverage. For a person-detector dropout,
    # use the temporally nearest target-independent H32 result as pose fallback
    # and mark all body observations absent. Never substitute a target pose.
    selected_rows = np.asarray(
        [
            rows[int(frame)]
            if int(frame) in rows
            else int(np.argmin(np.abs(all_indices - int(frame))))
            for frame in selected_indices
        ],
        dtype=np.int64,
    )
    parameters = _numpy(h32["smplx"], np.float32)[selected_rows]
    raw_pose = _h32_pose(parameters)
    initial = raw_pose.copy()
    projected = _map_projected(_projected_to_image(h32)[selected_rows])
    finite = np.isfinite(projected).all(axis=-1)
    in_frame = (
        finite
        & (projected[..., 0] >= -0.05)
        & (projected[..., 0] <= 1.05)
        & (projected[..., 1] >= -0.05)
        & (projected[..., 1] <= 1.05)
    )

    observations = np.zeros((len(selected_indices), NUM_JOINTS, 8), dtype=np.float32)
    observations[..., 0] = 1.0
    observations[..., 1] = in_frame
    observations[..., 2] = ~in_frame
    observations[..., 4] = ~in_frame
    detector_present = in_frame.copy()
    detector_present[~exact_h32, :21] = False
    in_frame[~exact_h32, :21] = False
    duplicated = np.zeros_like(in_frame)
    left_fused = np.zeros(len(selected_indices), dtype=bool)
    right_fused = np.zeros(len(selected_indices), dtype=bool)
    components = []
    fallback = []
    frame_names = []
    for offset, frame in enumerate(selected_indices):
        frame_name = f"{entry['clip_id']}_{int(frame):06d}"
        image_key = f"{frame_name}.png"
        frame_names.append(frame_name)
        if image_key not in raw_images:
            raise ValueError(f"Verified WiLoR artifact lacks {image_key}")
        hands = _wilor_hands(hamer.get(image_key))
        raw_hands = raw_images[image_key].get("hands", [])
        detected = []
        missing = []
        for side, is_right, start in (
            ("left", False, 21),
            ("right", True, 36),
        ):
            stop = start + 15
            hand_record, duplicate = _select_hand(raw_hands, is_right)
            hand_pose = hands.get(side)
            if hand_record is None or hand_pose is None:
                observations[offset, start:stop, 0] = 0.0
                observations[offset, start:stop, 1] = 0.0
                observations[offset, start:stop, 2] = 1.0
                observations[offset, start:stop, 4] = 1.0
                detector_present[offset, start:stop] = False
                in_frame[offset, start:stop] = False
                missing.append(side)
                continue
            initial[offset, start:stop] = hand_pose
            confidence = float(np.clip(hand_record.get("detector_confidence", 0.0), 0.0, 1.0))
            width = float(entry["source_contract"]["width"])
            height = float(entry["source_contract"]["height"])
            size = float(hand_record.get("box_size", 0.0))
            center = _numpy(hand_record.get("box_center", [0.0, 0.0]), np.float32)
            half = size * 0.5
            outside = max(0.0, half - center[0]) + max(0.0, center[0] + half - width)
            outside += max(0.0, half - center[1]) + max(0.0, center[1] + half - height)
            truncation = min(1.0, outside / max(2.0 * size, 1.0))
            observations[offset, start:stop, 0] = confidence
            observations[offset, start:stop, 1] = 1.0
            observations[offset, start:stop, 2] = 0.0
            observations[offset, start:stop, 3] = size / max(width, height, 1.0)
            observations[offset, start:stop, 4] = truncation
            observations[offset, start:stop, 6] = float(duplicate)
            detector_present[offset, start:stop] = True
            in_frame[offset, start:stop] = truncation < 1.0
            duplicated[offset, start:stop] = duplicate
            if side == "left":
                left_fused[offset] = True
            else:
                right_fused[offset] = True
            detected.append(side)
        components.append("body=smplerx;hands=" + ("wilor_" + "_".join(detected) if detected else "smplerx"))
        fallback.append("" if not missing else "wilor_dropout_" + "_".join(missing))

    raw_matrix = axis_angle_to_matrix(torch.from_numpy(raw_pose).float())
    initial_matrix = axis_angle_to_matrix(torch.from_numpy(initial).float())
    disagreement = geodesic_distance(raw_matrix, initial_matrix).numpy() / np.pi
    observations[..., 7] = np.clip(disagreement, 0.0, 1.0)
    if len(initial_matrix) > 1:
        motion = geodesic_distance(initial_matrix[1:], initial_matrix[:-1]).numpy() / np.pi
        observations[1:, :, 5] = np.clip(motion, 0.0, 1.0)
        observations[0, :, 5] = observations[1, :, 5]

    target_paths = [
        Path(entry["target_dir"]) / f"images{int(frame) + 1:04d}.pkl"
        for frame in selected_indices
    ]
    missing_targets = [path for path in target_paths if not path.is_file()]
    if missing_targets:
        raise FileNotFoundError(missing_targets[0])
    target, target_valid = _target_pose(target_paths)
    refine = np.zeros(NUM_JOINTS, dtype=bool)
    refine[list(REFINED_BODY)] = True
    refine[21:] = True
    keypoint_valid = detector_present & in_frame
    observations[..., 1] = keypoint_valid
    observations[..., 2] = ~keypoint_valid
    confidence = np.clip(observations[..., 0], 0.0, 1.0)
    reliability = (
        confidence
        * keypoint_valid
        * (1.0 - np.clip(observations[..., 4], 0.0, 1.0))
        * (1.0 - 0.5 * duplicated)
        * np.exp(-2.0 * np.clip(observations[..., 5], 0.0, 1.0))
    ).astype(np.float32)
    fps = float(entry["source_contract"]["fps"])
    width = int(h32["width"])
    height = int(h32["height"])
    metadata = {
        "dataset": "SOKE-PHOENIX-full",
        "official_split": entry["official_split"],
        "phase2_split": entry["phase2_split"],
        "source_clip": entry["source_clip"],
        "source_group": entry["source_group"],
        "signer_id": entry["signer_id"],
        "initializer_expert": "SMPLer-X H32 body/hands with per-side WiLoR hand replacement and H32 dropout fallback",
        "initializer_target_independent": True,
        "target_provider": entry["target_provider"],
        "target_dir": entry["target_dir"],
        "target_frame_binding": "images{video_frame_zero_based+1:04d}.pkl",
        "target_fields_used_as_input": False,
        "coordinate_policy": {
            "keypoints_2d": "normalized_image_0_to_1",
            "rotations": "smplx_local_axis_angle",
            "source_frame_binding": "zero_based_original_video_frame_index",
        },
        "h32_source": str(h32_path.resolve()),
        "h32_source_sha256": sha256_file(h32_path),
        "h32_requested_frames": len(requested),
        "h32_direct_frames": int(exact_h32.sum()),
        "h32_temporal_fallback_frames": int((~exact_h32).sum()),
        "h32_coverage": float(exact_h32.mean()),
        "wilor_left_coverage": float(left_fused.mean()),
        "wilor_right_coverage": float(right_fused.mean()),
        "requires_reprojection_enrichment": False,
        "reprojection_residual_policy": "zero; no independent full-body detector track in this extraction",
    }
    clip = CacheClip(
        clip_id=entry["clip_id"],
        frame_names=np.asarray(frame_names),
        frame_numbers=selected_indices,
        timestamps=selected_indices.astype(np.float64) / fps,
        fps=fps,
        image_size=np.repeat(np.asarray([[width, height]], dtype=np.int32), len(selected_indices), axis=0),
        init_axis_angle=initial,
        target_axis_angle=target,
        target_rotation_valid=target_valid,
        alternate_axis_angle=raw_pose,
        alternate_rotation_valid=np.ones((len(selected_indices), NUM_JOINTS), dtype=bool),
        observation_features=observations,
        keypoints_2d=projected,
        keypoint_valid=keypoint_valid,
        refine_mask=refine,
        betas=np.median(parameters[:, 159:169], axis=0).astype(np.float32),
        global_orient=parameters[:, :3].astype(np.float32),
        transl=parameters[:, 179:182].astype(np.float32),
        jaw_pose=parameters[:, 156:159].astype(np.float32),
        leye_pose=np.zeros((len(selected_indices), 3), dtype=np.float32),
        reye_pose=np.zeros((len(selected_indices), 3), dtype=np.float32),
        expression=parameters[:, 169:179].astype(np.float32),
        source_paths=np.asarray(
            [f"{entry['video']}#frame={int(frame)}" for frame in selected_indices]
        ),
        u0_reliability=reliability,
        reprojection_residual_2d=np.zeros((len(selected_indices), NUM_JOINTS, 2), dtype=np.float32),
        raw_confidence=confidence.astype(np.float32),
        calibrated_confidence=confidence.astype(np.float32),
        detector_present=detector_present,
        track_valid=keypoint_valid,
        in_frame=in_frame,
        copied_observation=np.zeros_like(keypoint_valid),
        interpolated_observation=np.zeros_like(keypoint_valid),
        target_quality=target_valid.astype(np.float32),
        initializer_component=np.asarray(components),
        fallback_reason=np.asarray(fallback),
        semantic_contract_version=PHASE2R_SEMANTIC_CONTRACT,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    report = {
        "clip_id": entry["clip_id"],
        "requested_frames": len(requested),
        "retained_frames": len(selected_indices),
        "target_joint_frames": int(target_valid.size),
        "target_valid_joint_frames": int(target_valid.sum()),
        "target_invalid_joint_frames": int((~target_valid).sum()),
        "target_complete_frames": int(target_valid.all(axis=1).sum()),
        "h32_direct_frames": int(exact_h32.sum()),
        "h32_temporal_fallback_frames": int((~exact_h32).sum()),
        "h32_coverage": float(exact_h32.mean()),
        "wilor_left_coverage": float(left_fused.mean()),
        "wilor_right_coverage": float(right_fused.mean()),
    }
    return clip, report


def build(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = args.selection.resolve()
    shard_root = args.wilor_shard_manifest_root.resolve()
    wilor_root = args.wilor_root.resolve()
    h32_root = args.smplerx_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only PHOENIX cache exists: {output}")
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    entries = {str(item["clip_id"]): item for item in selection["clips"]}
    split = str(selection["split"])
    shard_report = json.loads((shard_root / "shard_report.json").read_text(encoding="utf-8"))
    staging.mkdir(parents=True)
    clip_dir = staging / "clips" / split
    split_dir = staging / "splits"
    clip_dir.mkdir(parents=True)
    split_dir.mkdir()
    cache_entries = []
    cache_sha256: dict[str, str] = {}
    clip_reports = []
    seen = set()
    try:
        for shard_index, shard in enumerate(shard_report["shards"]):
            manifest_path = Path(shard["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in manifest["records"]:
                stem = Path(str(record["image_key"])).stem
                clip_id, separator, frame = stem.rpartition("_")
                if not separator or not frame.isdigit():
                    raise ValueError(f"Invalid WiLoR image key: {stem}")
                by_clip[clip_id].append(record)
            hamer, wilor = _verified_artifacts(
                wilor_root, manifest_path, shard_index
            )
            for clip_id in by_clip:
                if clip_id not in entries:
                    raise ValueError(f"WiLoR shard contains undeclared clip: {clip_id}")
                if clip_id in seen:
                    raise ValueError(f"Clip appears in multiple WiLoR shards: {clip_id}")
                entry = entries[clip_id]
                h32_path = h32_root / f"{entry['source_clip']}.pkl"
                if not h32_path.is_file():
                    raise FileNotFoundError(h32_path)
                clip, report = _make_clip(
                    entry, h32_path, hamer, wilor["images"]
                )
                destination = clip_dir / f"{clip_id}.npz"
                save_cache_clip(destination, clip)
                relative_cache = str(Path("..") / "clips" / split / destination.name)
                cache_entries.append(relative_cache)
                cache_sha256[relative_cache] = sha256_file(destination)
                clip_reports.append(report)
                seen.add(clip_id)
            print(
                f"[phoenix-cache] split={split} shard={shard_index + 1}/"
                f"{len(shard_report['shards'])} clips={len(seen)}/{len(entries)}",
                flush=True,
            )
        if seen != set(entries):
            missing = sorted(set(entries) - seen)
            raise ValueError(f"PHOENIX cache lacks selected clips: {missing[:3]}")
        manifest = {
            "schema": SCHEMA,
            "dataset": "SOKE-PHOENIX-full",
            "split": split,
            "official_split": selection["official_split"],
            "clips": cache_entries,
            "clip_sha256": cache_sha256,
            "initializer": "SMPLer-X H32 + WiLoR hands with per-side H32 fallback",
            "target": "released SOKE PHOENIX frame poses",
            "selection": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
        }
        manifest_path = split_dir / f"{split}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = {
            "schema": SCHEMA,
            "split": split,
            "official_split": selection["official_split"],
            "clips": len(clip_reports),
            "requested_frames": sum(item["requested_frames"] for item in clip_reports),
            "retained_frames": sum(item["retained_frames"] for item in clip_reports),
            "target_joint_frames": sum(
                item["target_joint_frames"] for item in clip_reports
            ),
            "target_valid_joint_frames": sum(
                item["target_valid_joint_frames"] for item in clip_reports
            ),
            "target_invalid_joint_frames": sum(
                item["target_invalid_joint_frames"] for item in clip_reports
            ),
            "target_complete_frames": sum(
                item["target_complete_frames"] for item in clip_reports
            ),
            "h32_direct_frames": sum(item["h32_direct_frames"] for item in clip_reports),
            "h32_temporal_fallback_frames": sum(
                item["h32_temporal_fallback_frames"] for item in clip_reports
            ),
            "mean_h32_coverage": float(
                np.mean([item["h32_coverage"] for item in clip_reports])
            ) if clip_reports else 0.0,
            "mean_wilor_left_coverage": float(
                np.mean([item["wilor_left_coverage"] for item in clip_reports])
            ) if clip_reports else 0.0,
            "mean_wilor_right_coverage": float(
                np.mean([item["wilor_right_coverage"] for item in clip_reports])
            ) if clip_reports else 0.0,
            "manifest": str(output / "splits" / f"{split}.json"),
            "manifest_sha256": sha256_file(manifest_path),
            "content_hashed_cache_files": len(cache_sha256),
            "clip_reports": clip_reports,
        }
        report["target_valid_joint_fraction"] = (
            report["target_valid_joint_frames"]
            / max(report["target_joint_frames"], 1)
        )
        (staging / "materialization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--smplerx-root", type=Path, required=True)
    parser.add_argument("--wilor-shard-manifest-root", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
