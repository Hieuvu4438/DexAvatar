"""Build non-identity How2Sign targets with 2D-guided temporal bundle adjustment.

The frozen H32 pose is retained as the initializer.  A separate optimization
uses the ordered How2Sign whole-body tracks as an independent refinement signal
and constrains corrections temporally and by bounded pose anchors.  Outputs are
new cache files; the input cache and all legacy methods remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.build_how2sign_cache import (
    _mapped_keypoints,
    _observations,
)
from phase2_refiner.data.cache_schema import CacheClip, load_cache_clip, save_cache_clip
from phase2_refiner.render import create_smplx_model


# Cache joints are body_pose[0:21], left_hand_pose[0:15], right_hand_pose[0:15].
# The corresponding native SMPL-X output joints are 1:22, 25:40, and 40:55.
MODEL_JOINTS = tuple(range(1, 22)) + tuple(range(25, 55))
REGIONS = {
    "body": slice(0, 21),
    "left_hand": slice(21, 36),
    "right_hand": slice(36, 51),
}
TARGET_PROVIDER = "How2Sign 2D-track temporal bundle adjustment v1"
INITIALIZER_PROVIDER = "SMPLer-X H32 frozen per-frame"


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _select_paths(
    train_manifest: Path,
    val_manifest: Path,
    train_clips: int,
    validation_clips: int,
    calibration_clips: int,
    seed: int,
) -> dict[str, list[Path]]:
    train = _manifest_paths(train_manifest)
    val = _manifest_paths(val_manifest)
    train.sort(key=lambda path: _stable_rank(path.stem, seed))
    if train_clips > len(train):
        raise ValueError(f"Requested {train_clips} train clips, only {len(train)} exist")

    by_group: dict[str, list[Path]] = {}
    for path in val:
        clip = load_cache_clip(path)
        group = str(json.loads(clip.metadata_json).get("source_group", ""))
        if not group:
            raise ValueError(f"Missing source_group: {path}")
        by_group.setdefault(group, []).append(path)
    groups = sorted(by_group, key=lambda value: _stable_rank(value, seed))
    midpoint = len(groups) // 2
    validation_groups = set(groups[:midpoint])
    calibration_groups = set(groups[midpoint:])

    def take(groups_for_split: set[str], count: int) -> list[Path]:
        ordered_groups = sorted(
            groups_for_split, key=lambda value: _stable_rank(value, seed + 1)
        )
        per_group = {
            group: sorted(
                by_group[group], key=lambda path: _stable_rank(path.stem, seed + 2)
            )
            for group in ordered_groups
        }
        candidates = []
        for local_index in range(max(map(len, per_group.values()))):
            candidates.extend(
                per_group[group][local_index]
                for group in ordered_groups
                if local_index < len(per_group[group])
            )
        if count > len(candidates):
            raise ValueError(f"Requested {count} clips from only {len(candidates)} candidates")
        return candidates[:count]

    return {
        "train": train[:train_clips],
        "val": take(validation_groups, validation_clips),
        "calibration": take(calibration_groups, calibration_clips),
    }


def _teacher_observations(clip: CacheClip) -> tuple[np.ndarray, ...]:
    metadata = json.loads(clip.metadata_json)
    teacher_path = Path(metadata["teacher_path"])
    with np.load(teacher_path, allow_pickle=False) as teacher:
        keypoints, confidence, valid = _mapped_keypoints(
            teacher["keypoints_2d"].astype(np.float32),
            teacher["keypoint_scores"].astype(np.float32),
        )
        observations = _observations(keypoints, confidence, valid)
        bboxes = teacher["bboxes"].astype(np.float32)
        image_size = teacher["image_size"].astype(np.float32)
    if len(keypoints) != len(clip.frame_names):
        raise ValueError(f"Teacher/cache frame mismatch for {clip.clip_id}")
    return keypoints, confidence, valid, observations, bboxes, image_size


def _decode_joints(
    model,
    pose: torch.Tensor,
    clips: list[CacheClip],
    device: torch.device,
) -> torch.Tensor:
    batch, frames = pose.shape[:2]

    def stack(name: str) -> torch.Tensor:
        values = np.stack([getattr(clip, name) for clip in clips])
        return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(
            batch * frames, -1
        )

    betas = torch.as_tensor(
        np.stack([clip.betas for clip in clips]), dtype=torch.float32, device=device
    )
    betas = betas[:, None].expand(-1, frames, -1).reshape(batch * frames, -1)
    flattened = pose.reshape(batch * frames, 51, 3)
    output = model(
        betas=betas,
        global_orient=stack("global_orient"),
        body_pose=flattened[:, :21].flatten(1),
        left_hand_pose=flattened[:, 21:36].flatten(1),
        right_hand_pose=flattened[:, 36:51].flatten(1),
        transl=stack("transl"),
        jaw_pose=stack("jaw_pose"),
        leye_pose=stack("leye_pose"),
        reye_pose=stack("reye_pose"),
        expression=stack("expression"),
        return_verts=False,
    )
    return output.joints[:, MODEL_JOINTS].reshape(batch, frames, 51, 3)


def _project(
    joints: torch.Tensor, bboxes: torch.Tensor, image_sizes: torch.Tensor
) -> torch.Tensor:
    x, y, width, height = bboxes.unbind(-1)
    focal_x = 5000.0 / 192.0 * width
    focal_y = 5000.0 / 256.0 * height
    principal_x = x + width * 0.5
    principal_y = y + height * 0.5
    z = joints[..., 2].clamp_min(1e-5)
    pixel_x = joints[..., 0] / z * focal_x[..., None] + principal_x[..., None]
    pixel_y = joints[..., 1] / z * focal_y[..., None] + principal_y[..., None]
    normalizer = image_sizes[:, None, None, :]
    return torch.stack((pixel_x, pixel_y), dim=-1) / normalizer


def _weighted_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    return (distance * weight).sum() / weight.sum().clamp_min(1.0)


def _bounded_delta(
    raw_delta: torch.Tensor, limits: torch.Tensor, refine: torch.Tensor
) -> torch.Tensor:
    # Bound the 3-D correction vector, rather than each component independently.
    # sqrt(epsilon) keeps the derivative well-defined at the zero initialization.
    norm = torch.sqrt(raw_delta.square().sum(dim=-1, keepdim=True) + 1e-8)
    return (
        limits
        * torch.tanh(norm)
        * raw_delta
        / norm
        * refine[:, None, :, None]
    )


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    result = {"all": float(_weighted_error(prediction, target, weight))}
    for name, region in REGIONS.items():
        result[name] = float(
            _weighted_error(
                prediction[..., region, :], target[..., region, :], weight[..., region]
            )
        )
    return result


def _fit_batch(
    model,
    clips: list[CacheClip],
    teacher: list[tuple[np.ndarray, ...]],
    device: torch.device,
    iterations: int,
    learning_rate: float,
    body_max_degrees: float,
    hand_max_degrees: float,
) -> tuple[np.ndarray, list[dict]]:
    initial = torch.as_tensor(
        np.stack([clip.init_axis_angle for clip in clips]),
        dtype=torch.float32,
        device=device,
    )
    keypoints = torch.as_tensor(
        np.stack([item[0] for item in teacher]), dtype=torch.float32, device=device
    )
    confidence = torch.as_tensor(
        np.stack([item[1] for item in teacher]), dtype=torch.float32, device=device
    )
    valid = torch.as_tensor(
        np.stack([item[2] for item in teacher]), dtype=torch.bool, device=device
    )
    refine = torch.as_tensor(
        np.stack([clip.refine_mask for clip in clips]), dtype=torch.bool, device=device
    )
    weight = confidence * valid * refine[:, None, :]
    bboxes = torch.as_tensor(
        np.stack([item[4] for item in teacher]), dtype=torch.float32, device=device
    )
    image_sizes = torch.as_tensor(
        np.stack([item[5] for item in teacher]), dtype=torch.float32, device=device
    )
    limits = torch.full((1, 1, 51, 1), math.radians(hand_max_degrees), device=device)
    limits[..., :21, :] = math.radians(body_max_degrees)
    raw_delta = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam((raw_delta,), lr=learning_rate)

    with torch.no_grad():
        initial_projection = _project(
            _decode_joints(model, initial, clips, device), bboxes, image_sizes
        )
    for _ in range(iterations):
        delta = _bounded_delta(raw_delta, limits, refine)
        pose = initial + delta
        projection = _project(_decode_joints(model, pose, clips, device), bboxes, image_sizes)
        coordinate_loss = F.smooth_l1_loss(
            projection, keypoints, reduction="none", beta=0.01
        ).sum(dim=-1)
        reprojection = (coordinate_loss * weight).sum() / weight.sum().clamp_min(1.0)
        anchor = (delta.square() * refine[:, None, :, None]).sum() / (
            refine.sum().clamp_min(1) * initial.shape[1] * 3
        )
        velocity = (delta[:, 1:] - delta[:, :-1]).square().mean()
        acceleration = (
            delta[:, 2:] - 2.0 * delta[:, 1:-1] + delta[:, :-2]
        ).square().mean()
        loss = reprojection + 0.02 * anchor + 0.10 * velocity + 0.05 * acceleration
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_delta,), 1.0)
        optimizer.step()

    with torch.no_grad():
        delta = _bounded_delta(raw_delta, limits, refine)
        target_pose = initial + delta
        final_projection = _project(
            _decode_joints(model, target_pose, clips, device), bboxes, image_sizes
        )
        reports = []
        for index in range(len(clips)):
            initial_metrics = _metrics(
                initial_projection[index], keypoints[index], weight[index]
            )
            final_metrics = _metrics(final_projection[index], keypoints[index], weight[index])
            delta_degrees = torch.rad2deg(torch.linalg.vector_norm(delta[index], dim=-1))
            gain = {
                region: (
                    (initial_metrics[region] - final_metrics[region])
                    / max(initial_metrics[region], 1e-8)
                )
                for region in initial_metrics
            }
            accepted = gain["all"] >= 0.005 and all(
                gain[region] >= -0.02 for region in REGIONS
            ) and float(delta_degrees.max()) > 1e-4
            reports.append(
                {
                    "initial_reprojection": initial_metrics,
                    "final_reprojection": final_metrics,
                    "relative_gain": gain,
                    "mean_correction_degrees": float(delta_degrees.mean()),
                    "max_correction_degrees": float(delta_degrees.max()),
                    "accepted": accepted,
                }
            )
    return target_pose.detach().cpu().numpy(), reports


def _write_manifest(output: Path, split: str, entries: list[str], groups: set[str]) -> None:
    payload = {
        "dataset": "How2Sign",
        "split": split,
        "clips": entries,
        "source_groups": sorted(groups),
        "motion_domain": "sign_language_asl",
        "target_type": "independent_pseudo_target",
        "initializer_expert": INITIALIZER_PROVIDER,
        "target_provider": TARGET_PROVIDER,
        "independent_refinement_signal": "ordered How2Sign 133-point 2D tracks",
        "sgnify_excluded": True,
    }
    with (output / "splits" / f"{split}.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only output already exists: {output}")
    selected = _select_paths(
        args.train_manifest.resolve(),
        args.val_manifest.resolve(),
        args.train_clips,
        args.validation_clips,
        args.calibration_clips,
        args.seed,
    )
    output.mkdir(parents=True)
    (output / "splits").mkdir()
    device = torch.device(args.device)
    model = create_smplx_model(args.model_folder.resolve(), device)
    model.requires_grad_(False)
    summary: dict = {
        "schema_version": 1,
        "method": TARGET_PROVIDER,
        "initializer": INITIALIZER_PROVIDER,
        "locked_lane_a1_match": False,
        "scientific_scope": (
            "Tier-C proxy/pretraining target; does not by itself satisfy exact locked-A1 G4"
        ),
        "parameters": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "body_max_degrees": args.body_max_degrees,
            "hand_max_degrees": args.hand_max_degrees,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "splits": {},
    }
    try:
        for split, paths in selected.items():
            clip_dir = output / "clips" / split
            clip_dir.mkdir(parents=True)
            entries: list[str] = []
            groups: set[str] = set()
            reports: list[dict] = []
            for start in range(0, len(paths), args.batch_size):
                batch_paths = paths[start : start + args.batch_size]
                clips = [load_cache_clip(path) for path in batch_paths]
                teacher = [_teacher_observations(clip) for clip in clips]
                targets, batch_reports = _fit_batch(
                    model,
                    clips,
                    teacher,
                    device,
                    args.iterations,
                    args.learning_rate,
                    args.body_max_degrees,
                    args.hand_max_degrees,
                )
                for clip, observations, target, report in zip(
                    clips, teacher, targets, batch_reports, strict=True
                ):
                    report["clip_id"] = clip.clip_id
                    reports.append(report)
                    if not report["accepted"]:
                        continue
                    metadata = json.loads(clip.metadata_json)
                    group = str(metadata["source_group"])
                    metadata.update(
                        {
                            "target_type": "independent_pseudo_target",
                            "initializer_expert": INITIALIZER_PROVIDER,
                            "target_provider": TARGET_PROVIDER,
                            "independent_refinement_signal": (
                                "ordered How2Sign 133-point 2D tracks plus temporal constraints"
                            ),
                            "initializer_matches_locked_lane_a1": False,
                            "target_quality": report,
                        }
                    )
                    refined = replace(
                        clip,
                        target_axis_angle=target.astype(np.float32),
                        target_rotation_valid=np.ones_like(
                            clip.target_rotation_valid, dtype=bool
                        ),
                        keypoints_2d=observations[0],
                        keypoint_valid=observations[2],
                        observation_features=observations[3],
                        u0_reliability=(observations[1] * observations[2]).astype(
                            np.float32
                        ),
                        metadata_json=json.dumps(metadata, sort_keys=True),
                    )
                    destination = clip_dir / f"{clip.clip_id}.npz"
                    save_cache_clip(destination, refined)
                    entries.append(str(Path("..") / "clips" / split / destination.name))
                    groups.add(group)
                print(
                    f"[2d-temporal] {split} {min(start + len(clips), len(paths))}/"
                    f"{len(paths)} accepted={len(entries)}",
                    flush=True,
                )
            if not entries:
                raise RuntimeError(f"No {split} clips passed target quality filters")
            _write_manifest(output, split, entries, groups)
            accepted_reports = [report for report in reports if report["accepted"]]
            summary["splits"][split] = {
                "requested": len(paths),
                "accepted": len(accepted_reports),
                "rejected": len(reports) - len(accepted_reports),
                "frames": len(accepted_reports) * 32,
                "source_groups": len(groups),
                "mean_initial_reprojection": {
                    region: float(
                        np.mean([r["initial_reprojection"][region] for r in accepted_reports])
                    )
                    for region in ("all", *REGIONS)
                },
                "mean_final_reprojection": {
                    region: float(
                        np.mean([r["final_reprojection"][region] for r in accepted_reports])
                    )
                    for region in ("all", *REGIONS)
                },
                "mean_relative_gain": {
                    region: float(np.mean([r["relative_gain"][region] for r in accepted_reports]))
                    for region in ("all", *REGIONS)
                },
                "mean_correction_degrees": float(
                    np.mean([r["mean_correction_degrees"] for r in accepted_reports])
                ),
                "max_correction_degrees": float(
                    max(r["max_correction_degrees"] for r in accepted_reports)
                ),
            }
            with (output / f"{split}_per_clip.jsonl").open("x", encoding="utf-8") as handle:
                for report in reports:
                    handle.write(json.dumps(report, sort_keys=True) + "\n")
        split_groups = {
            split: set(payload["source_groups"])
            for split in selected
            for payload in [json.load(open(output / "splits" / f"{split}.json"))]
        }
        overlaps = {
            "train_validation": sorted(split_groups["train"] & split_groups["val"]),
            "train_calibration": sorted(
                split_groups["train"] & split_groups["calibration"]
            ),
            "validation_calibration": sorted(
                split_groups["val"] & split_groups["calibration"]
            ),
        }
        summary["source_group_overlap"] = overlaps
        summary["source_disjoint_verified"] = not any(overlaps.values())
        with (output / "refinement_report.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        shutil.rmtree(output)
        raise
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument("--train-clips", type=int, default=2000)
    parser.add_argument("--validation-clips", type=int, default=300)
    parser.add_argument("--calibration-clips", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--body-max-degrees", type=float, default=12.0)
    parser.add_argument("--hand-max-degrees", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
