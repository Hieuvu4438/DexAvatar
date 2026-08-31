"""Fuse frozen SMPLer-X/WiLoR experts and attach released sign-domain GT."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np
import torch

from phase2_refiner.data.build_how2sign_cache import (
    REFINED_BODY,
    _mapped_keypoints,
    _observations,
    _pose,
    _quality,
)
from phase2_refiner.data.cache_schema import (
    NUM_JOINTS,
    PHASE2R_SEMANTIC_CONTRACT,
    CacheClip,
    save_cache_clip,
)
from phase2_refiner.geometry.rotations import matrix_to_axis_angle
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-sign-domain-expert-fusion-cache-v2"


def _numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _load_hamer_outputs(wilor_root: Path) -> tuple[dict[str, Any], list[Path]]:
    """Load one WiLoR output or merge disjoint append-only shard outputs."""

    single = wilor_root / "hamer" / "hamer.pkl"
    paths = [single] if single.is_file() else sorted(
        wilor_root.glob("shard_*/hamer/hamer.pkl")
    )
    if not paths:
        raise FileNotFoundError(
            f"No hamer/hamer.pkl or shard_*/hamer/hamer.pkl under {wilor_root}"
        )
    merged: dict[str, Any] = {}
    for path in paths:
        payload = _load_pickle(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid WiLoR/HaMeR output: {path}")
        duplicates = sorted(set(merged).intersection(payload))
        if duplicates:
            raise ValueError(f"Duplicate WiLoR frame keys across shards: {duplicates[:3]}")
        merged.update(payload)
    return merged, paths


def _array(value: Any, width: int) -> np.ndarray:
    array = _numpy(value, np.float32).reshape(-1)
    if len(array) < width:
        array = np.pad(array, (0, width - len(array)))
    return array[:width]


def _target_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _dexavatar_pose(
    dexavatar_root: Path,
    clip_id: str,
    frame_names: list[str],
) -> tuple[np.ndarray, list[Path]]:
    paths = [dexavatar_root / clip_id / "results" / f"{name}.pkl" for name in frame_names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DexAvatar fitted frames: {missing[:3]}")
    poses = []
    for path in paths:
        payload = _load_pickle(path)
        body = _array(payload.get("body_pose"), 63)
        left = _array(payload.get("left_hand_pose"), 45)
        right = _array(payload.get("right_hand_pose"), 45)
        poses.append(np.concatenate((body, left, right)).reshape(NUM_JOINTS, 3))
    pose = np.stack(poses).astype(np.float32)
    if not np.isfinite(pose).all():
        raise ValueError(f"Non-finite DexAvatar fitted pose: {clip_id}")
    return pose, paths


def _expert_disagreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_matrix = axis_angle_to_matrix(torch.from_numpy(first).float())
    second_matrix = axis_angle_to_matrix(torch.from_numpy(second).float())
    distance = geodesic_distance(first_matrix, second_matrix).cpu().numpy()
    return np.clip(distance / np.pi, 0.0, 1.0).astype(np.float32)


def _wilor_hands(entry: Any) -> dict[str, np.ndarray]:
    """Decode WiLoR rotations using DexAvatar's handedness convention."""

    if not isinstance(entry, (list, tuple)) or len(entry) < 4:
        return {}
    predictions = entry[0]
    flags = _numpy(entry[3]).reshape(-1).astype(int)
    rotations = predictions.get("pred_mano_params", {}).get("hand_pose")
    if rotations is None:
        return {}
    matrices = torch.as_tensor(_numpy(rotations), dtype=torch.float32)
    if matrices.ndim != 4 or matrices.shape[1:] != (15, 3, 3):
        raise ValueError(f"Unexpected WiLoR hand rotation shape: {tuple(matrices.shape)}")
    axis_angle = matrix_to_axis_angle(matrices).cpu().numpy().astype(np.float32)
    if len(axis_angle) != len(flags):
        raise ValueError("WiLoR side/rotation count mismatch")
    result: dict[str, np.ndarray] = {}
    for index, flag in enumerate(flags):
        side = "right" if flag == 1 else "left"
        pose = axis_angle[index].copy()
        if side == "left":
            # WiLoR predicts the left hand in canonical-right coordinates.
            pose[:, 1] *= -1.0
            pose[:, 2] *= -1.0
        result.setdefault(side, pose)
    return result


def _soke_target(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    paths = [Path(path) for path in entry["target_paths"]]
    poses = []
    for path in paths:
        payload = _load_pickle(path)
        body = _array(payload.get("smplx_body_pose"), 63)
        left = _array(payload.get("smplx_lhand_pose"), 45)
        right = _array(payload.get("smplx_rhand_pose"), 45)
        poses.append(np.concatenate((body, left, right)).reshape(NUM_JOINTS, 3))
    target = np.stack(poses).astype(np.float32)
    valid = np.isfinite(target).all(axis=-1)
    if not valid.all():
        target = np.where(valid[..., None], target, 0.0)
    return target, valid, _target_digest(paths)


def _signavatars_target(
    entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    path = Path(entry["annotation_path"])
    payload = _load_pickle(path)
    rows = np.asarray(entry["target_rows"], dtype=np.int64)
    parameters = _numpy(payload["smplx"], np.float32)
    if rows.max() >= len(parameters):
        raise ValueError(f"Target rows exceed SignAvatars annotation: {path}")
    selected = parameters[rows]
    target = np.concatenate(
        (selected[:, 3:66], selected[:, 66:111], selected[:, 111:156]), axis=1
    ).reshape(-1, NUM_JOINTS, 3)
    valid = np.ones((len(rows), NUM_JOINTS), dtype=bool)
    left = _numpy(payload["left_valid"]).reshape(-1)[rows].astype(bool)
    right = _numpy(payload["right_valid"]).reshape(-1)[rows].astype(bool)
    valid[:, 21:36] = left[:, None]
    valid[:, 36:51] = right[:, None]
    valid &= np.isfinite(target).all(axis=-1)
    target = np.where(valid[..., None], target, 0.0).astype(np.float32)
    return target, valid, sha256_file(path)


def _make_clip(
    teacher_path: Path,
    entry: dict[str, Any],
    hamer: dict[str, Any],
    dexavatar_root: Path | None = None,
    body_source: str = "smplerx",
    hand_source: str = "wilor",
) -> CacheClip:
    with np.load(teacher_path, allow_pickle=False) as teacher:
        initial = _pose(teacher)
        indices = teacher["sample_indices"].astype(np.int64)
        expected = np.asarray(entry["frame_indices"], dtype=np.int64)
        if not np.array_equal(indices, expected):
            raise ValueError(f"Teacher/source frame mismatch: {entry['clip_id']}")
        keypoints, confidence, keypoint_valid = _mapped_keypoints(
            teacher["keypoints_2d"].astype(np.float32),
            teacher["keypoint_scores"].astype(np.float32),
        )
        observations = _observations(keypoints, confidence, keypoint_valid)
        left_fused = np.zeros(len(indices), dtype=bool)
        right_fused = np.zeros(len(indices), dtype=bool)
        frame_names = []
        for frame_index, source_frame in enumerate(indices):
            frame_name = f"{entry['clip_id']}_{int(source_frame):06d}"
            frame_names.append(frame_name)
            hands = _wilor_hands(hamer.get(f"{frame_name}.png"))
            if "left" in hands:
                initial[frame_index, 21:36] = hands["left"]
                left_fused[frame_index] = True
            if "right" in hands:
                initial[frame_index, 36:51] = hands["right"]
                right_fused[frame_index] = True

        raw_fusion = initial.copy()
        alternate = raw_fusion.copy()
        alternate_valid = np.zeros((len(indices), NUM_JOINTS), dtype=bool)
        dexavatar_paths: list[Path] = []
        if dexavatar_root is not None:
            dexavatar, dexavatar_paths = _dexavatar_pose(
                dexavatar_root, entry["clip_id"], frame_names
            )
            alternate = dexavatar
            alternate_valid[:] = True
            observations[..., 7] = _expert_disagreement(raw_fusion, dexavatar)
            if body_source == "dexavatar":
                initial[:, :21] = dexavatar[:, :21]
            if hand_source == "dexavatar":
                initial[:, 21:] = dexavatar[:, 21:]
        elif body_source == "dexavatar" or hand_source == "dexavatar":
            raise ValueError("DexAvatar initializer source requires --dexavatar-root")

        if entry["dataset"] == "SOKE":
            target, target_valid, target_digest = _soke_target(entry)
        elif entry["dataset"] == "SignAvatars":
            target, target_valid, target_digest = _signavatars_target(entry)
        else:
            raise ValueError(f"Unsupported target dataset: {entry['dataset']}")
        if target.shape != initial.shape:
            raise ValueError(f"Initializer/target shape mismatch: {entry['clip_id']}")
        quality = _quality(target)
        if not quality["passed"]:
            raise ValueError(f"Released target quality rejection: {entry['clip_id']}")
        refine = np.zeros(NUM_JOINTS, dtype=bool)
        refine[list(REFINED_BODY)] = True
        refine[21:] = True
        fps = float(teacher["fps"])
        source_paths = np.asarray(
            [f"{entry['video']}#frame={int(index)}" for index in indices]
        )
        metadata = {
            "dataset": entry["dataset"],
            "motion_domain": (
                "sign_language_dgs_continuous"
                if entry["dataset"] == "SOKE"
                else "sign_language_asl_isolated"
            ),
            "official_split": entry["official_split"],
            "phase2_split": entry["phase2_split"],
            "source_clip": entry["source_clip"],
            "source_group": entry["source_group"],
            "signer_id": entry["signer_id"],
            "gloss": entry.get("gloss", ""),
            "sgnify_training_reads": 0,
            "teacher_path": str(teacher_path.resolve()),
            "initializer_expert": (
                f"body={body_source}; hands={hand_source}; "
                "frozen SMPLer-X H32 body/root/camera + WiLoR/HaMeR hands + "
                "optional DexAvatar SignBPoser/SignHPoser fitted proposal"
            ),
            "initializer_target_independent": True,
            "wilor_fused_left_frames": int(left_fused.sum()),
            "wilor_fused_right_frames": int(right_fused.sum()),
            "wilor_left_coverage": float(left_fused.mean()),
            "wilor_right_coverage": float(right_fused.mean()),
            "target_provider": entry["target_provider"],
            "target_key": entry.get("target_key"),
            "target_binding_sha256": target_digest,
            "target_quality": quality,
            "dexavatar_fitted": dexavatar_root is not None,
            "alternate_expert": (
                "DexAvatar SignBPoser/SignHPoser source fit"
                if dexavatar_root is not None
                else "absent"
            ),
            "alternate_expert_contract": (
                "alternate_axis_angle; alternate_rotation_valid; target-free"
                if dexavatar_root is not None
                else "absent"
            ),
            "dexavatar_result_sha256": [
                sha256_file(path) for path in dexavatar_paths
            ],
            "expert_disagreement_feature": (
                "observation_features[...,7]=geodesic(raw_fusion,dexavatar)/pi"
                if dexavatar_root is not None
                else "absent"
            ),
            "coordinate_policy": {
                "keypoints_2d": "normalized_image_0_to_1",
                "rotations": "smplx_local_axis_angle",
                "source_frame_binding": "zero_based_video_frame_index",
            },
            "requires_reprojection_enrichment": True,
        }
        image_size = teacher["image_size"].astype(np.int32)
        betas = np.median(teacher["betas"], axis=0).astype(np.float32)
        return CacheClip(
            clip_id=entry["clip_id"],
            frame_names=np.asarray(frame_names),
            frame_numbers=indices,
            timestamps=indices.astype(np.float64) / fps,
            fps=fps,
            image_size=np.repeat(image_size[None], len(indices), axis=0),
            init_axis_angle=initial.astype(np.float32),
            target_axis_angle=target,
            target_rotation_valid=target_valid,
            alternate_axis_angle=alternate,
            alternate_rotation_valid=alternate_valid,
            target_quality=target_valid.astype(np.float32),
            observation_features=observations,
            keypoints_2d=keypoints,
            keypoint_valid=keypoint_valid,
            refine_mask=refine,
            betas=betas,
            global_orient=teacher["global_orient"].reshape(-1, 3).astype(np.float32),
            transl=teacher["transl"].reshape(-1, 3).astype(np.float32),
            jaw_pose=teacher["jaw_pose"].reshape(-1, 3).astype(np.float32),
            leye_pose=np.zeros((len(indices), 3), dtype=np.float32),
            reye_pose=np.zeros((len(indices), 3), dtype=np.float32),
            expression=teacher["expression"].reshape(-1, 10).astype(np.float32),
            source_paths=source_paths,
            u0_reliability=(confidence * keypoint_valid).astype(np.float32),
            initializer_component=np.asarray(
                [f"body={body_source};hands={hand_source}"] * len(indices)
            ),
            semantic_contract_version=PHASE2R_SEMANTIC_CONTRACT,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    smplerx_root = args.smplerx_root.resolve()
    wilor_root = args.wilor_root.resolve()
    output = args.output.resolve()
    dexavatar_root = (
        args.dexavatar_root.resolve() if args.dexavatar_root is not None else None
    )
    if output.exists():
        raise FileExistsError(f"Append-only cache output exists: {output}")
    selection_path = smplerx_root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    split = str(selection["split"])
    hamer, hamer_paths = _load_hamer_outputs(wilor_root)
    dexavatar_report_path = None
    if dexavatar_root is not None:
        dexavatar_report_path = dexavatar_root / "fit_report.json"
        dexavatar_report = json.loads(
            dexavatar_report_path.read_text(encoding="utf-8")
        )
        if dexavatar_report.get("target_fields_read") is not False:
            raise ValueError("DexAvatar fit report does not prove target independence")
        fitted_clips = {
            item["clip_id"]: int(item["frames"])
            for item in dexavatar_report.get("clip_reports", [])
        }
        expected_clips = {
            item["clip_id"]: len(item["frame_indices"])
            for item in selection["clips"]
        }
        if fitted_clips != expected_clips:
            raise ValueError(
                "DexAvatar fit/SMPLer-X selection mismatch: "
                f"fitted={len(fitted_clips)} expected={len(expected_clips)}"
            )
    output.mkdir(parents=True)
    clip_dir = output / "clips" / split
    split_dir = output / "splits"
    clip_dir.mkdir(parents=True)
    split_dir.mkdir()
    entries = []
    dataset_counts: dict[str, int] = {}
    left_coverage = []
    right_coverage = []
    groups = set()
    try:
        for index, entry in enumerate(selection["clips"], start=1):
            teacher = smplerx_root / "clips" / f"{entry['clip_id']}.npz"
            if not teacher.is_file():
                raise FileNotFoundError(teacher)
            clip = _make_clip(
                teacher,
                entry,
                hamer,
                dexavatar_root=dexavatar_root,
                body_source=args.body_source,
                hand_source=args.hand_source,
            )
            destination = clip_dir / f"{clip.clip_id}.npz"
            save_cache_clip(destination, clip)
            entries.append(str(Path("..") / "clips" / split / destination.name))
            metadata = json.loads(clip.metadata_json)
            left_coverage.append(metadata["wilor_left_coverage"])
            right_coverage.append(metadata["wilor_right_coverage"])
            groups.add(entry["source_group"])
            dataset_counts[entry["dataset"]] = dataset_counts.get(entry["dataset"], 0) + 1
            if index % 25 == 0 or index == len(selection["clips"]):
                print(f"[expert-fusion-cache] split={split} {index}/{len(selection['clips'])}", flush=True)
        manifest = {
            "schema": SCHEMA,
            "dataset": "SOKE+SignAvatars",
            "split": split,
            "clips": entries,
            "source_groups": sorted(groups),
            "sgnify_excluded": True,
            "initializer": {
                "body_source": args.body_source,
                "hand_source": args.hand_source,
                "raw_experts": "SMPLer-X H32 + WiLoR/HaMeR",
                "dexavatar_fitted": dexavatar_root is not None,
                "observations": "133-point tracks",
            },
            "target": "released SOKE/SignAvatars dataset ground truth",
        }
        manifest_path = split_dir / f"{split}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = {
            "schema": SCHEMA,
            "split": split,
            "clips": len(entries),
            "frames": sum(len(item["frame_indices"]) for item in selection["clips"]),
            "datasets": dict(sorted(dataset_counts.items())),
            "mean_wilor_left_coverage": float(np.mean(left_coverage)),
            "mean_wilor_right_coverage": float(np.mean(right_coverage)),
            "smplerx_selection": str(selection_path),
            "smplerx_selection_sha256": sha256_file(selection_path),
            "hamer_outputs": [str(path) for path in hamer_paths],
            "hamer_output_sha256": [sha256_file(path) for path in hamer_paths],
            "dexavatar_fit_report": (
                str(dexavatar_report_path) if dexavatar_report_path else None
            ),
            "dexavatar_fit_report_sha256": (
                sha256_file(dexavatar_report_path) if dexavatar_report_path else None
            ),
            "body_source": args.body_source,
            "hand_source": args.hand_source,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
        (output / "materialization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    except Exception:
        shutil.rmtree(output)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplerx-root", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, required=True)
    parser.add_argument("--dexavatar-root", type=Path)
    parser.add_argument(
        "--body-source", choices=("smplerx", "dexavatar"), default="smplerx"
    )
    parser.add_argument(
        "--hand-source", choices=("wilor", "dexavatar"), default="wilor"
    )
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))
