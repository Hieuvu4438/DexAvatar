"""Run the frozen external-only checkpoint on target-free inference caches."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip
from phase2_refiner.geometry.rotations import geodesic_distance, matrix_to_axis_angle
from phase2_refiner.infer import (
    _apply_safety_fallback,
    _predict_sequence,
    _sha256,
    _standard_result,
)
from phase2_refiner.provenance import run_provenance, sha256_file
from phase2_refiner.render import render_source_anchored_directory

from .model import model_from_config


REGIONS = {"ubody": (0, 21), "lhand": (21, 36), "rhand": (36, 51)}


def _load_model(config: dict, checkpoint: Path, device: torch.device):
    model = model_from_config(config, initialize=False).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("ema_model") or payload["model"], strict=True)
    return model.eval()


def _apply_region_thresholds(
    prediction: dict[str, torch.Tensor],
    initial: torch.Tensor,
    thresholds: dict[str, float],
) -> torch.Tensor:
    if "benefit_logit" not in prediction:
        raise ValueError("External checkpoint has no benefit head")
    probability = prediction["benefit_logit"].sigmoid()
    abstained = torch.zeros_like(probability, dtype=torch.bool)
    for group_index, (name, (start, end)) in enumerate(REGIONS.items()):
        threshold = float(thresholds[name])
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Invalid {name} threshold: {threshold}")
        reject = probability[..., group_index] < threshold
        abstained[..., group_index] = reject
        prediction["matrix"][..., start:end, :, :] = torch.where(
            reject[..., None, None, None],
            initial[..., start:end, :, :].to(prediction["matrix"].device),
            prediction["matrix"][..., start:end, :, :],
        )
        for key in ("raw_delta", "gate"):
            prediction[key][..., start:end, :] = torch.where(
                reject[..., None, None],
                torch.zeros_like(prediction[key][..., start:end, :]),
                prediction[key][..., start:end, :],
            )
    return abstained


@torch.no_grad()
def infer_clip(
    cache_path: Path,
    output_root: Path,
    model,
    device: torch.device,
    thresholds: dict[str, float],
) -> dict:
    clip = load_cache_clip(cache_path)
    if clip.target_axis_angle is not None or clip.target_joint_positions is not None:
        raise ValueError(f"Inference cache contains target fields: {cache_path}")
    metadata = json.loads(clip.metadata_json)
    if int(metadata.get("sgnify_target_reads", 0)) != 0:
        raise ValueError(f"Inference cache reports target reads: {cache_path}")
    features, initial_matrix = features_from_clip(
        clip,
        input_dim=45,
        reprojection_residual_scale=10.0,
    )
    prediction = _predict_sequence(
        model,
        features,
        initial_matrix,
        torch.from_numpy(clip.refine_mask),
        device,
    )
    abstained = _apply_region_thresholds(
        prediction, initial_matrix.to(device), thresholds
    )
    body_limit = float(torch.rad2deg(model.max_angles[:21].max()).cpu())
    hand_limit = float(torch.rad2deg(model.max_angles[21:].max()).cpu())
    output_matrix, fallback = _apply_safety_fallback(
        prediction["matrix"], initial_matrix, body_limit, hand_limit
    )
    delta = prediction["raw_delta"]
    gate = prediction["gate"]
    for group_index, (_, (start, end)) in enumerate(REGIONS.items()):
        invalid = fallback[:, group_index]
        delta[invalid, start:end] = 0.0
        gate[invalid, start:end] = 0.0
    output_axis_angle = matrix_to_axis_angle(output_matrix).cpu().numpy()
    identity = (
        geodesic_distance(output_matrix, initial_matrix.to(device)).cpu().numpy()
        < 1e-12
    )
    output_axis_angle[identity] = clip.init_axis_angle[identity]

    result_dir = output_root / clip.clip_id / "smplifyx" / "results"
    diagnostics_dir = output_root / clip.clip_id / "external_diagnostics"
    planned = [result_dir / f"{name}.pkl" for name in clip.frame_names]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    source_resolved = {Path(path).resolve() for path in clip.source_paths}
    if any(path.resolve() in source_resolved for path in planned):
        raise ValueError("Output would overwrite the frozen initializer")
    result_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_name in enumerate(clip.frame_names):
        with (result_dir / f"{frame_name}.pkl").open("xb") as handle:
            pickle.dump(
                _standard_result(clip, index, output_axis_angle[index]),
                handle,
                protocol=2,
            )
    diagnostics_path = diagnostics_dir / "sequence.npz"
    np.savez_compressed(
        diagnostics_path,
        frame_names=clip.frame_names,
        delta_rotvec=delta.cpu().numpy(),
        gate=gate.cpu().numpy(),
        reliability=prediction["reliability"].cpu().numpy(),
        benefit_probability=prediction["benefit_logit"].sigmoid().cpu().numpy(),
        abstained_group_frames=abstained.cpu().numpy(),
        fallback_group_frames=fallback.cpu().numpy(),
    )
    summary = {
        "clip_id": clip.clip_id,
        "frames": len(clip.frame_names),
        "cache": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
        "max_delta_degrees": float(
            torch.rad2deg(
                geodesic_distance(output_matrix, initial_matrix.to(device))
            ).max()
        ),
        "mean_gate": float(gate.mean()),
        "abstained_group_frames": int(abstained.sum()),
        "fallback_group_frames": int(fallback.sum()),
    }
    with (diagnostics_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    with args.calibration.resolve().open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("decision") != "PASS":
        raise ValueError("External calibration did not pass")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != calibration.get("checkpoint_sha256"):
        raise ValueError("Checkpoint does not match external calibration")
    cache_manifest = args.cache_root / "manifest.json"
    with cache_manifest.open("r", encoding="utf-8") as handle:
        cache_payload = json.load(handle)
    entries = cache_payload.get("clips", [])
    if entries and isinstance(entries[0], dict):
        if any(bool(item.get("has_target")) for item in entries):
            raise ValueError("Inference cache manifest contains targets")
    cache_paths = sorted((args.cache_root / "clips").glob("*.npz"))
    if len(cache_paths) != 57:
        raise ValueError(f"Expected 57 inference clips, got {len(cache_paths)}")
    device = torch.device(args.device)
    model = _load_model(config, args.checkpoint.resolve(), device)
    args.output.mkdir(parents=True)
    summaries = []
    for index, cache_path in enumerate(cache_paths, start=1):
        summary = infer_clip(
            cache_path,
            args.output,
            model,
            device,
            calibration["thresholds"],
        )
        if args.render:
            clip = load_cache_clip(cache_path)
            sign_root = args.output / summary["clip_id"] / "smplifyx"
            summary["meshes"] = render_source_anchored_directory(
                sign_root / "results",
                sign_root / "meshes",
                clip.source_paths,
                args.model_folder,
                device,
            )
        summaries.append(summary)
        print(f"[external-infer] {index}/{len(cache_paths)} {summary['clip_id']}")
    frame_count = sum(item["frames"] for item in summaries)
    if frame_count != 1493:
        raise ValueError(f"Expected 1493 frames, got {frame_count}")
    run_manifest = {
        "schema_version": 1,
        "method": "SIGNAL4D_EXTERNAL_HOW2SIGN_CLIPNORM_V1",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "cache_manifest": str(cache_manifest.resolve()),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "thresholds": calibration["thresholds"],
        "sgnify_target_reads_before_evaluation": 0,
        "clips": summaries,
        "frames": frame_count,
        "provenance": run_provenance(args.config, int(config.get("seed", 42))),
    }
    with (args.output / "run_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
