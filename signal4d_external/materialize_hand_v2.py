"""Materialize the frozen external-only V2 hand policy on target-free caches."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip
from phase2_refiner.geometry.rotations import geodesic_distance, matrix_to_axis_angle
from phase2_refiner.infer import _predict_sequence
from phase2_refiner.provenance import run_provenance, sha256_file
from phase2_refiner.render import render_source_anchored_directory
from signal4d_external.hand_v2_core import (
    HAND_REGIONS,
    exact_rank_selection,
    geodesic_blend,
    hand_eligibility,
    smooth_clips,
)
from signal4d_external.infer import _load_model


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed baseline result: {path}")
    return payload


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    calibration = _load_json(args.calibration.resolve())
    if calibration.get("decision") != "PASS":
        raise ValueError("External V2H calibration did not pass")
    if sha256_file(args.config) != calibration.get("config_sha256"):
        raise ValueError("Config does not match external V2H calibration")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != calibration.get("checkpoint_sha256"):
        raise ValueError("Checkpoint does not match external V2H calibration")

    cache_root = args.cache_root.resolve()
    cache_manifest = cache_root / "manifest.json"
    cache_payload = _load_json(cache_manifest)
    if any(bool(row.get("has_target")) for row in cache_payload.get("clips", [])):
        raise ValueError("Target inference manifest contains supervision")
    cache_paths = sorted((cache_root / "clips").glob("*.npz"))
    if len(cache_paths) != 57:
        raise ValueError(f"Expected 57 target-free clips, got {len(cache_paths)}")
    baseline_root = args.baseline_root.resolve()
    baseline_manifest = _load_json(baseline_root / "run_manifest.json")
    if baseline_manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Baseline V1 output does not use the calibrated checkpoint")
    if int(baseline_manifest.get("sgnify_target_reads_before_evaluation", -1)) != 0:
        raise ValueError("Baseline V1 output does not prove target-free inference")

    device = torch.device(args.device)
    model = _load_model(config, args.checkpoint.resolve(), device)
    reprojection_scale = float(
        config["data"].get("reprojection_residual_scale", 10.0)
    )
    records = []
    for index, cache_path in enumerate(cache_paths, start=1):
        clip = load_cache_clip(cache_path)
        if clip.target_axis_angle is not None or clip.target_joint_positions is not None:
            raise ValueError(f"Inference cache contains target fields: {cache_path}")
        metadata = json.loads(clip.metadata_json)
        if int(metadata.get("sgnify_target_reads", 0)) != 0:
            raise ValueError(f"Inference cache reports target reads: {cache_path}")
        features, initial = features_from_clip(
            clip,
            input_dim=45,
            reprojection_residual_scale=reprojection_scale,
        )
        prediction = _predict_sequence(
            model,
            features,
            initial,
            torch.from_numpy(clip.refine_mask),
            device,
        )
        records.append(
            {
                "clip": clip,
                "cache_path": cache_path,
                "initial": initial,
                "candidate": prediction["matrix"].cpu(),
                "probability": prediction["benefit_logit"].sigmoid().cpu().numpy(),
            }
        )
        print(f"[hand-v2-target-predict] {index}/{len(cache_paths)} {clip.clip_id}")

    selections: dict[str, list[np.ndarray]] = {}
    for region, (_, _, group_index) in HAND_REGIONS.items():
        policy = calibration["regions"][region]["selected_on_validation"]
        scores = smooth_clips(
            [record["probability"][:, group_index] for record in records],
            [record["clip"].timestamps for record in records],
            float(policy["smoothing_half_window_seconds"]),
            [hand_eligibility(record["clip"], region) for record in records],
        )
        selections[region] = exact_rank_selection(
            scores,
            float(policy["coverage"]),
            [hand_eligibility(record["clip"], region) for record in records],
        )

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    summaries = []
    for clip_index, record in enumerate(records):
        clip = record["clip"]
        initial = record["initial"]
        output_hands = {}
        fallback_masks = {}
        for region, (start, end, _) in HAND_REGIONS.items():
            policy = calibration["regions"][region]["selected_on_validation"]
            chosen = selections[region][clip_index].copy()
            blended = geodesic_blend(
                initial[:, start:end],
                record["candidate"][:, start:end],
                float(policy["alpha"]),
            )
            angular = torch.rad2deg(
                geodesic_distance(blended, initial[:, start:end])
            )
            fallback = (~torch.isfinite(blended).all(dim=(-1, -2))).any(dim=-1)
            fallback |= (angular > 25.0 + 1e-3).any(dim=-1)
            chosen[fallback.numpy()] = False
            axis_angle = matrix_to_axis_angle(blended).numpy()
            identity = geodesic_distance(blended, initial[:, start:end]).numpy() < 1e-12
            axis_angle[identity] = clip.init_axis_angle[:, start:end][identity]
            output_hands[region] = axis_angle
            fallback_masks[region] = fallback.numpy()
            selections[region][clip_index] = chosen

        result_dir = args.output / clip.clip_id / "smplifyx" / "results"
        diagnostics_dir = args.output / clip.clip_id / "external_hand_v2_diagnostics"
        result_dir.mkdir(parents=True)
        diagnostics_dir.mkdir(parents=True)
        for frame, frame_name in enumerate(clip.frame_names.astype(str)):
            baseline_path = (
                baseline_root / clip.clip_id / "smplifyx" / "results" / f"{frame_name}.pkl"
            )
            payload = copy.deepcopy(_load_baseline(baseline_path))
            if selections["lhand"][clip_index][frame]:
                payload["left_hand_pose"] = output_hands["lhand"][frame].reshape(
                    1, 45
                ).astype(np.float32)
            if selections["rhand"][clip_index][frame]:
                payload["right_hand_pose"] = output_hands["rhand"][frame].reshape(
                    1, 45
                ).astype(np.float32)
            with (result_dir / f"{frame_name}.pkl").open("xb") as handle:
                pickle.dump(payload, handle, protocol=2)
        np.savez_compressed(
            diagnostics_dir / "sequence.npz",
            frame_names=clip.frame_names,
            benefit_probability=record["probability"],
            selected_lhand=selections["lhand"][clip_index],
            selected_rhand=selections["rhand"][clip_index],
            fallback_lhand=fallback_masks["lhand"],
            fallback_rhand=fallback_masks["rhand"],
        )
        summaries.append(
            {
                "clip_id": clip.clip_id,
                "frames": len(clip.frame_names),
                "cache": str(record["cache_path"].resolve()),
                "cache_sha256": sha256_file(record["cache_path"]),
                "selected_lhand_frames": int(selections["lhand"][clip_index].sum()),
                "selected_rhand_frames": int(selections["rhand"][clip_index].sum()),
                "fallback_lhand_frames": int(fallback_masks["lhand"].sum()),
                "fallback_rhand_frames": int(fallback_masks["rhand"].sum()),
            }
        )

    frames = sum(row["frames"] for row in summaries)
    if frames != 1493:
        raise ValueError(f"Expected 1493 frames, got {frames}")
    if args.render:
        for index, record in enumerate(records, start=1):
            clip = record["clip"]
            sign_root = args.output / clip.clip_id / "smplifyx"
            render_source_anchored_directory(
                sign_root / "results",
                sign_root / "meshes",
                clip.source_paths,
                args.model_folder,
                device,
            )
            print(f"[hand-v2-render] {index}/{len(records)} {clip.clip_id}")

    run_manifest = {
        "schema_version": 1,
        "method": "SIGNAL4D_EXTERNAL_HAND_V2_RANK_SOFT_RESIDUAL",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "baseline_v1_manifest": str((baseline_root / "run_manifest.json").resolve()),
        "baseline_v1_manifest_sha256": sha256_file(
            baseline_root / "run_manifest.json"
        ),
        "policies": {
            region: calibration["regions"][region]["selected_on_validation"]
            for region in HAND_REGIONS
        },
        "target_covariate_use": (
            "global ranks of unlabeled benefit probabilities only; no target labels"
        ),
        "sgnify_target_reads_before_evaluation": 0,
        "clips": summaries,
        "frames": frames,
        "provenance": run_provenance(args.config, int(config.get("seed", 42))),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
