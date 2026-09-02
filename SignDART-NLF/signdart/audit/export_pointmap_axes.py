from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml

from signdart.geometry.arm_ik import BOUNDARY_X180, internal_intrinsics
from signdart.geometry.ray_sphere import project
from signdart.io.h1_state import H1State, read_manifest, sha256_file, state_path
from signdart.model import create_model, forward_state_batch
from signdart.pointmap import (
    PART_ENDPOINT_IDS,
    PART_JOINT_IDS,
    block_bootstrap_axes,
    face_part_labels,
    mask_bone_endpoints,
    render_visible_part_masks,
    robust_axis,
)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sapiens-root", type=Path, required=True)
    parser.add_argument("--sapiens-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--debug-first", type=int, default=0)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(args.sapiens_root))
    from sapiens.dense.models import init_model

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    forbidden = {"gt_root", "protocol_lock", "author_assets"}.intersection(
        config["paths"]
    )
    if forbidden:
        raise ValueError("pointmap inference config exposes evaluation-only paths")
    paths = {key: Path(value) for key, value in config["paths"].items()}
    records = read_manifest(paths["manifest"])
    if args.limit is not None:
        records = records[: args.limit]

    smplx_model = create_model(paths["model_root"], str(config["runtime"]["device"]))
    faces = np.asarray(smplx_model.faces, dtype=np.int64)
    labels = face_part_labels(
        smplx_model.lbs_weights.detach().cpu().numpy(), faces
    )
    pointmap_model = init_model(
        str(args.sapiens_config), str(args.checkpoint),
        device=str(config["runtime"]["device"]),
    )
    rows = []
    for ordinal, record in enumerate(records, start=1):
        state = H1State.load(state_path(paths["h1_state_root"], record))
        vertices_batch, joints_batch = forward_state_batch(
            smplx_model, state, state.arrays["body_pose"],
            str(config["runtime"]["device"]),
        )
        vertices = vertices_batch[0]
        joints_internal = joints_batch[0]
        if args.seg_root is None:
            masks = render_visible_part_masks(
                vertices, faces, labels, state.K_evaluator,
                int(record["height"]), int(record["width"]), erode_px=2,
            )
            endpoints = {
                name: (
                    project(internal_intrinsics(state.K_evaluator), joints_internal[parent]),
                    project(internal_intrinsics(state.K_evaluator), joints_internal[child]),
                )
                for name, (parent, child) in PART_ENDPOINT_IDS.items()
            }
            mask_source = "incumbent_smplx_renderer"
        else:
            seg_path = (
                args.seg_root / record["sign_id"]
                / f"{int(record['source_frame_id']):06d}.npz"
            )
            with np.load(seg_path, allow_pickle=False) as segmentation:
                names = segmentation["class_names"].astype(str).tolist()
                probabilities = (
                    segmentation["prob_q"].astype(np.float32)
                    * segmentation["prob_scale"][:, None, None]
                )
            mapping = {
                "left_upper": "l_upper_arm",
                "left_forearm": "l_lower_arm",
                "right_upper": "r_upper_arm",
                "right_forearm": "r_lower_arm",
            }
            kernel = np.ones((3, 3), dtype=np.uint8)
            masks = {}
            for name, channel in mapping.items():
                probability = cv2.resize(
                    probabilities[names.index(channel)],
                    (int(record["width"]), int(record["height"])),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = (probability >= 0.5).astype(np.uint8)
                masks[name] = cv2.erode(mask, kernel, iterations=1).astype(bool)
            endpoints = {}
            endpoint_errors = {}
            for side in ("left", "right"):
                try:
                    directed = mask_bone_endpoints(
                        masks[f"{side}_upper"], masks[f"{side}_forearm"]
                    )
                    endpoints[f"{side}_upper"] = directed["upper"]
                    endpoints[f"{side}_forearm"] = directed["forearm"]
                except ValueError as error:
                    endpoint_errors[side] = str(error)
            mask_source = "frozen_sapiens2_semantic_arm_masks"
        image = cv2.imread(str(record["rgb_path"]))
        if image is None:
            raise FileNotFoundError(record["rgb_path"])
        data = pointmap_model.pipeline(dict(img=image))
        data = pointmap_model.data_preprocessor(data)
        inputs, samples = data["inputs"], data["data_samples"]
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            pointmap, scale = pointmap_model(inputs)
        pointmap = (pointmap / scale).float()
        left, right, top, bottom = samples["meta"]["padding_size"]
        pointmap = pointmap[
            :, :, top : inputs.shape[2] - bottom, left : inputs.shape[3] - right
        ]
        pointmap = torch_functional.interpolate(
            pointmap, size=image.shape[:2], mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()

        arrays = {"record_id": np.asarray(record["record_id"])}
        frame_row = {"record_id": record["record_id"], "parts": {}}
        for name in PART_JOINT_IDS:
            side = name.split("_", maxsplit=1)[0]
            if args.seg_root is not None and side in endpoint_errors:
                arrays[f"{name}_valid"] = np.asarray(False)
                frame_row["parts"][name] = {
                    "pixels": int(np.count_nonzero(masks[name])),
                    "valid": False,
                    "reason": endpoint_errors[side],
                }
                continue
            yy, xx = np.nonzero(masks[name])
            points = pointmap[yy, xx]
            valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
            points = points[valid]
            pixels = np.stack((xx[valid], yy[valid]), axis=1).astype(np.float64)
            uv_parent, uv_child = endpoints[name]
            part_row = {"pixels": int(len(points)), "valid": False}
            arrays[f"{name}_valid"] = np.asarray(False)
            if len(points) >= 64:
                try:
                    axis, quality = robust_axis(
                        points, pixels, uv_parent, uv_child, iterations=5
                    )
                    bootstrap = block_bootstrap_axes(
                        points, pixels, uv_parent, uv_child, axis,
                        seed_key=f"{record['record_id']}:{name}", repetitions=256,
                    )
                    angles = np.rad2deg(np.arccos(np.clip(bootstrap @ axis, -1.0, 1.0)))
                    ci95 = float(np.quantile(angles, 0.95))
                    arrays[f"{name}_valid"] = np.asarray(True)
                    arrays[f"{name}_axis"] = axis.astype(np.float32)
                    arrays[f"{name}_bootstrap_axes"] = bootstrap
                    arrays[f"{name}_n"] = np.asarray(quality["n"])
                    arrays[f"{name}_eigen_gap"] = np.asarray(quality["eigen_gap"])
                    arrays[f"{name}_residual_mad"] = np.asarray(
                        quality["residual_mad"]
                    )
                    arrays[f"{name}_ci95_deg"] = np.asarray(ci95)
                    part_row = {
                        **quality, "ci95_deg": ci95, "valid": True,
                    }
                except ValueError as error:
                    part_row["reason"] = str(error)
            else:
                part_row["reason"] = "too few eroded visible pixels"
            frame_row["parts"][name] = part_row

        destination = (
            args.output_root / "frames" / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        atomic_npz(destination, **arrays)
        if ordinal <= args.debug_first:
            overlay = image.copy()
            colors = ((255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80))
            for color, name in zip(colors, PART_JOINT_IDS):
                overlay[masks[name]] = (
                    0.45 * overlay[masks[name]] + 0.55 * np.asarray(color)
                ).astype(np.uint8)
            debug_path = args.output_root / "debug" / f"{record['sign_id']}_{int(record['source_frame_id']):06d}.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), overlay)
        rows.append(frame_row)
        if ordinal % 10 == 0 or ordinal == len(records):
            print(f"[pointmap-axes] {ordinal}/{len(records)}", flush=True)

    valid_parts = sum(
        part["valid"] for row in rows for part in row["parts"].values()
    )
    report = {
        "schema_version": "signray.pointmap_axes.v1",
        "status": "complete",
        "frames": len(records),
        "parts": len(records) * 4,
        "valid_parts": int(valid_parts),
        "valid_part_fraction": valid_parts / max(len(records) * 4, 1),
        "uses_gt": False,
        "trained_on_sgnify": False,
        "mask_source": mask_source if rows else None,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "items": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: report[key] for key in (
            "status", "frames", "parts", "valid_parts", "valid_part_fraction",
            "uses_gt", "trained_on_sgnify",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
