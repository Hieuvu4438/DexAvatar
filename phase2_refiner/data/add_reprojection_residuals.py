"""Add observed-minus-projected 2D residuals to immutable Phase-2 caches."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip, save_cache_clip
from phase2_refiner.data.refine_how2sign_targets import (
    _decode_joints,
    _project,
    _teacher_observations,
)
from phase2_refiner.provenance import sha256_file
from phase2_refiner.render import create_smplx_model


PROVIDER = "frozen initializer SMPL-X joint reprojection v1"


def _camera_matrix(params: dict, source: str | Path) -> np.ndarray:
    """Read either a fitted 3x3 K or raw SMPLer-X focal/principal fields."""

    matrix = np.asarray(params.get("K"), dtype=np.float32)
    if matrix.shape == (3, 3):
        return matrix
    focal = np.asarray(params.get("focal"), dtype=np.float32).reshape(-1)
    principal = np.asarray(params.get("princpt"), dtype=np.float32).reshape(-1)
    if focal.size != 2 or principal.size != 2:
        raise ValueError(f"Missing 3x3 K or focal/princpt camera fields in {source}")
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0], matrix[1, 1] = focal
    matrix[0, 2], matrix[1, 2] = principal
    return matrix


def _masked_residual(
    observed: np.ndarray, projected: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, float]:
    residual = observed - projected
    finite = np.isfinite(residual).all(axis=-1)
    mask = valid & finite
    clipped = np.clip(residual, -2.0, 2.0)
    clipping_fraction = (
        float((np.abs(residual[mask]) > 2.0).mean()) if mask.any() else 0.0
    )
    return np.where(mask[..., None], clipped, 0.0).astype(np.float32), clipping_fraction


def _how2sign_batch(model, clips, device) -> list[tuple[np.ndarray, float]]:
    teacher = [_teacher_observations(clip) for clip in clips]
    pose = torch.as_tensor(
        np.stack([clip.init_axis_angle for clip in clips]),
        dtype=torch.float32,
        device=device,
    )
    bboxes = torch.as_tensor(
        np.stack([item[4] for item in teacher]), dtype=torch.float32, device=device
    )
    image_sizes = torch.as_tensor(
        np.stack([item[5] for item in teacher]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        projected = (
            _project(_decode_joints(model, pose, clips, device), bboxes, image_sizes)
            .cpu()
            .numpy()
        )
    return [
        _masked_residual(
            clip.keypoints_2d * 2.0 - 1.0,
            pred * 2.0 - 1.0,
            clip.keypoint_valid & clip.refine_mask[None],
        )
        for clip, pred in zip(clips, projected, strict=True)
    ]


def _lane_clip(model, clip, device) -> tuple[np.ndarray, float]:
    pose = torch.as_tensor(
        clip.init_axis_angle[None], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        joints = _decode_joints(model, pose, [clip], device)[0].cpu().numpy()
    matrices = []
    for source in clip.source_paths:
        with Path(source).open("rb") as handle:
            params = pickle.load(handle, encoding="latin1")
        matrices.append(_camera_matrix(params, source))
    camera = np.stack(matrices)
    z = np.maximum(joints[..., 2], 1e-5)
    pixel_x = joints[..., 0] / z * camera[:, None, 0, 0] + camera[:, None, 0, 2]
    pixel_y = joints[..., 1] / z * camera[:, None, 1, 1] + camera[:, None, 1, 2]
    height = clip.image_size[:, None, 0]
    width = clip.image_size[:, None, 1]
    projected = np.stack(
        (pixel_x / width * 2.0 - 1.0, pixel_y / height * 2.0 - 1.0), axis=-1
    )
    return _masked_residual(
        clip.keypoints_2d,
        projected,
        clip.keypoint_valid & clip.refine_mask[None],
    )


def _enriched(clip, residual: np.ndarray, clipping_fraction: float):
    metadata = json.loads(clip.metadata_json)
    coordinate_policy = metadata.setdefault("coordinate_policy", {})
    coordinate_policy["reprojection_residual_2d"] = (
        "observed_minus_projected_normalized_image_-1_to_1"
    )
    metadata["reprojection_residual_provider"] = PROVIDER
    metadata["reprojection_residual_clipping_fraction"] = clipping_fraction
    metadata.pop("requires_reprojection_enrichment", None)
    return replace(
        clip,
        reprojection_residual_2d=residual,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _write_how2sign(args, model, device, output: Path) -> dict:
    summary = {}
    (output / "splits").mkdir(parents=True)
    for split in args.splits:
        source_manifest = args.input_root / "splits" / f"{split}.json"
        with source_manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        paths = _manifest_paths(source_manifest)
        clip_dir = output / "clips" / split
        clip_dir.mkdir(parents=True)
        clipping = []
        entries = []
        for start in range(0, len(paths), args.batch_size):
            clips = [
                load_cache_clip(path) for path in paths[start : start + args.batch_size]
            ]
            results = _how2sign_batch(model, clips, device)
            for clip, (residual, fraction) in zip(clips, results, strict=True):
                destination = clip_dir / f"{clip.clip_id}.npz"
                save_cache_clip(destination, _enriched(clip, residual, fraction))
                entries.append(str(Path("..") / "clips" / split / destination.name))
                clipping.append(fraction)
            print(
                f"[reprojection] {split} {min(start + len(clips), len(paths))}/{len(paths)}",
                flush=True,
            )
        manifest["clips"] = entries
        manifest["reprojection_residual_provider"] = PROVIDER
        with (output / "splits" / f"{split}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        summary[split] = {
            "clips": len(entries),
            "mean_clipping_fraction": float(np.mean(clipping)),
            "max_clipping_fraction": float(np.max(clipping)),
        }
    return summary


def _write_lane(args, model, device, output: Path) -> dict:
    clip_dir = output / "clips"
    clip_dir.mkdir(parents=True)
    source_paths = sorted((args.input_root / "clips").glob("*.npz"))
    entries = []
    clipping = []
    for index, path in enumerate(source_paths, start=1):
        clip = load_cache_clip(path)
        residual, fraction = _lane_clip(model, clip, device)
        destination = clip_dir / path.name
        save_cache_clip(destination, _enriched(clip, residual, fraction))
        entries.append(
            {
                "cache": str(Path("clips") / path.name),
                "clip_id": clip.clip_id,
                "frames": len(clip.frame_names),
                "has_target": clip.target_axis_angle is not None,
                "sha256": sha256_file(destination),
            }
        )
        clipping.append(fraction)
        print(
            f"[reprojection] lane {index}/{len(source_paths)} {clip.clip_id}",
            flush=True,
        )
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {"clips": entries, "reprojection_residual_provider": PROVIDER},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return {
        "lane": {
            "clips": len(entries),
            "mean_clipping_fraction": float(np.mean(clipping)),
            "max_clipping_fraction": float(np.max(clipping)),
        }
    }


def build(args: argparse.Namespace) -> dict:
    args.input_root = args.input_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only output exists: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    model = create_smplx_model(args.model_folder.resolve(), device)
    model.requires_grad_(False)
    try:
        splits = (
            _write_how2sign(args, model, device, output)
            if args.mode == "how2sign"
            else _write_lane(args, model, device, output)
        )
        report = {
            "schema_version": 1,
            "input_root": str(args.input_root),
            "mode": args.mode,
            "provider": PROVIDER,
            "splits": splits,
        }
        with (output / "reprojection_report.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        shutil.rmtree(output)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("how2sign", "lane"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=("train", "val", "calibration", "test"),
        help=(
            "How2Sign split to enrich; repeat as needed. Defaults to the legacy "
            "train/val/calibration set."
        ),
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.splits is None:
        args.splits = ["train", "val", "calibration"]
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
