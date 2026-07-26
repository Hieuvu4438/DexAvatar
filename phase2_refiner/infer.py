"""Run Phase 2 inference and export non-destructive DexAvatar-compatible results."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from phase2_refiner.config import load_config, validate_config
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip
from phase2_refiner.geometry.rotations import (
    geodesic_distance,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from phase2_refiner.models import WholeSequenceRefiner
from phase2_refiner.provenance import run_provenance, sha256_file
from phase2_refiner.render import render_source_anchored_directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(
    config: dict,
    checkpoint: Path | None,
    device: torch.device,
    use_ema: bool = True,
) -> WholeSequenceRefiner:
    model_config = config.get("model", {})
    checkpoint_data = None
    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_config = checkpoint_data.get("model_config", model_config)
    model = WholeSequenceRefiner(**model_config).to(device)
    if checkpoint_data is not None:
        state = checkpoint_data.get("ema_model") if use_ema else None
        model.load_state_dict(state or checkpoint_data["model"], strict=True)
    return model.eval()


def _pad_sequence(
    features: torch.Tensor, matrix: torch.Tensor, max_frames: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = len(features)
    if length > max_frames:
        raise ValueError(f"Window has {length} frames but max_frames={max_frames}")
    padded_features = torch.zeros(max_frames, *features.shape[1:], dtype=features.dtype)
    padded_matrix = torch.zeros(max_frames, *matrix.shape[1:], dtype=matrix.dtype)
    padded_features[:length] = features
    padded_matrix[:length] = matrix
    frame_valid = torch.zeros(max_frames, dtype=torch.bool)
    frame_valid[:length] = True
    return padded_features, padded_matrix, frame_valid


@torch.no_grad()
def _predict_sequence(
    model: WholeSequenceRefiner,
    features: torch.Tensor,
    initial_matrix: torch.Tensor,
    refine_mask: torch.Tensor,
    device: torch.device,
    uncertainty_offset: float = 0.0,
) -> dict[str, torch.Tensor]:
    length = len(features)
    window = model.max_frames
    starts = (
        [0]
        if length <= window
        else list(range(0, length - window + 1, max(1, window // 2)))
    )
    if starts[-1] != max(0, length - window):
        starts.append(length - window)
    quaternion_sum = torch.zeros(length, 51, 4, device=device)
    quaternion_reference = torch.zeros_like(quaternion_sum)
    weight_sum = torch.zeros(length, 1, 1, device=device)
    delta_sum = torch.zeros(length, 51, 3, device=device)
    gate_sum = torch.zeros(length, 51, 1, device=device)
    variance_sum = torch.zeros(length, 51, 1, device=device)
    position_sum = torch.zeros(length, 51, 3, device=device)
    reliability_sum = torch.zeros(length, 51, device=device)
    has_variance = False
    for start in starts:
        end = min(start + window, length)
        padded_features, padded_matrix, frame_valid = _pad_sequence(
            features[start:end], initial_matrix[start:end], window
        )
        prediction = model(
            padded_features[None].to(device),
            padded_matrix[None].to(device),
            frame_valid[None].to(device),
            refine_mask[None].to(device),
            uncertainty_offset=uncertainty_offset,
        )
        current_length = end - start
        weights = torch.hann_window(
            current_length + 2, periodic=False, device=device, dtype=features.dtype
        )[1:-1].clamp_min(1e-3)[:, None, None]
        quaternion = matrix_to_quaternion(prediction["matrix"][0, :current_length])
        existing = weight_sum[start:end] > 0
        reference = quaternion_reference[start:end]
        sign = torch.where(
            existing & ((reference * quaternion).sum(dim=-1, keepdim=True) < 0),
            -1.0,
            1.0,
        )
        quaternion = quaternion * sign
        quaternion_reference[start:end] = torch.where(existing, reference, quaternion)
        quaternion_sum[start:end] += quaternion * weights
        delta_sum[start:end] += prediction["raw_delta"][0, :current_length] * weights
        gate_sum[start:end] += prediction["gate"][0, :current_length] * weights
        position_sum[start:end] += (
            prediction["joint_position"][0, :current_length] * weights
        )
        reliability_sum[start:end] += prediction["reliability"][
            0, :current_length
        ] * weights.squeeze(-1)
        if "log_variance" in prediction:
            variance_sum[start:end] += (
                prediction["log_variance"][0, :current_length].exp() * weights
            )
            has_variance = True
        weight_sum[start:end] += weights
    result = {
        "matrix": quaternion_to_matrix(quaternion_sum / weight_sum),
        "raw_delta": delta_sum / weight_sum,
        "gate": gate_sum / weight_sum,
        "joint_position": position_sum / weight_sum,
        "reliability": reliability_sum / weight_sum.squeeze(-1),
    }
    if has_variance:
        result["log_variance"] = (variance_sum / weight_sum).clamp_min(1e-8).log()
    return result


def _standard_result(clip, index: int, pose: np.ndarray) -> dict[str, np.ndarray]:
    body = pose[:21].reshape(1, 63).astype(np.float32)
    return {
        "betas": clip.betas.reshape(1, 10).astype(np.float32),
        "global_orient": clip.global_orient[index].reshape(1, 3).astype(np.float32),
        "body_pose": body,
        "body_pose_fore": body[:, :45].copy(),
        "body_pose_op": body[:, 45:].copy(),
        "transl": clip.transl[index].reshape(1, 3).astype(np.float32),
        "left_hand_pose": pose[21:36].reshape(1, 45).astype(np.float32),
        "right_hand_pose": pose[36:51].reshape(1, 45).astype(np.float32),
        "jaw_pose": clip.jaw_pose[index].reshape(1, 3).astype(np.float32),
        "leye_pose": clip.leye_pose[index].reshape(1, 3).astype(np.float32),
        "reye_pose": clip.reye_pose[index].reshape(1, 3).astype(np.float32),
        "expression": clip.expression[index].reshape(1, 10).astype(np.float32),
    }


def _apply_safety_fallback(
    output: torch.Tensor,
    initial: torch.Tensor,
    body_limit_degrees: float = 25.0,
    hand_limit_degrees: float = 35.0,
    log_variance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace invalid body/hand group frames with their frozen initializer."""
    output = output.clone()
    fallback = torch.zeros(output.shape[0], 3, dtype=torch.bool, device=output.device)
    groups = (
        (0, 21, body_limit_degrees),
        (21, 36, hand_limit_degrees),
        (36, 51, hand_limit_degrees),
    )
    finite = torch.isfinite(output).all(dim=-1).all(dim=-1)
    safe_output = torch.where(
        finite[..., None, None], output, initial.to(output.device)
    )
    angular = torch.rad2deg(geodesic_distance(safe_output, initial.to(output.device)))
    for group_idx, (start, end, limit) in enumerate(groups):
        invalid = (~finite[:, start:end]).any(dim=-1) | (
            angular[:, start:end] > limit + 1e-3
        ).any(dim=-1)
        if log_variance is not None:
            uncertainty_invalid = (~torch.isfinite(log_variance[:, start:end])).any(
                dim=(-1, -2)
            ) | (
                (log_variance[:, start:end] < -8.0) | (log_variance[:, start:end] > 6.0)
            ).any(dim=(-1, -2))
            invalid = invalid | uncertainty_invalid
        fallback[:, group_idx] = invalid
        output[:, start:end] = torch.where(
            invalid[:, None, None, None],
            initial[:, start:end].to(output.device),
            safe_output[:, start:end],
        )
    return output, fallback


@torch.no_grad()
def infer_clip(
    cache_path: Path,
    output_root: Path,
    model: WholeSequenceRefiner,
    device: torch.device,
    overwrite: bool,
    uncertainty_offset: float = 0.0,
    reprojection_residual_scale: float = 10.0,
) -> dict:
    clip = load_cache_clip(cache_path)
    features, initial_matrix = features_from_clip(
        clip,
        input_dim=model.token_embedding.input_projection.in_features,
        reprojection_residual_scale=reprojection_residual_scale,
    )
    prediction = _predict_sequence(
        model,
        features,
        initial_matrix,
        torch.from_numpy(clip.refine_mask),
        device,
        uncertainty_offset,
    )
    length = len(clip.frame_names)
    body_limit = float(torch.rad2deg(model.max_angles[:21].max()).cpu())
    hand_limit = float(torch.rad2deg(model.max_angles[21:].max()).cpu())
    output_matrix, fallback = _apply_safety_fallback(
        prediction["matrix"],
        initial_matrix,
        body_limit,
        hand_limit,
        prediction.get("log_variance"),
    )
    delta = prediction["raw_delta"]
    gate = prediction["gate"]
    if fallback.any():
        for group_index, (start, end) in enumerate(((0, 21), (21, 36), (36, 51))):
            invalid = fallback[:, group_index]
            delta[invalid, start:end] = 0.0
            gate[invalid, start:end] = 0.0
    output_axis_angle = matrix_to_axis_angle(output_matrix).cpu().numpy()
    identity = torch.linalg.vector_norm(delta, dim=-1).cpu().numpy() < 1e-12
    output_axis_angle[identity] = clip.init_axis_angle[identity]

    result_dir = output_root / clip.clip_id / "smplifyx" / "results"
    diagnostics_dir = output_root / clip.clip_id / "phase2_diagnostics"
    planned_results = [result_dir / f"{name}.pkl" for name in clip.frame_names]
    stale_results = set(result_dir.glob("*.pkl")) - set(planned_results)
    if stale_results:
        raise RuntimeError(
            f"Output contains stale result PKLs; first: {sorted(stale_results)[0]}"
        )
    source_resolved = {Path(path).resolve() for path in clip.source_paths}
    collision = [path for path in planned_results if path.resolve() in source_resolved]
    if collision:
        raise ValueError(f"Output would overwrite frozen initializer: {collision[0]}")
    diagnostics_path = diagnostics_dir / "sequence.npz"
    summary_path = diagnostics_dir / "summary.json"
    existing = [
        path
        for path in (*planned_results, diagnostics_path, summary_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} outputs; first: {existing[0]}"
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_name in enumerate(clip.frame_names):
        result_path = result_dir / f"{frame_name}.pkl"
        with result_path.open("wb") as handle:
            pickle.dump(
                _standard_result(clip, index, output_axis_angle[index]),
                handle,
                protocol=2,
            )
    written_names = {path.stem for path in result_dir.glob("*.pkl")}
    expected_names = set(clip.frame_names.astype(str))
    if written_names != expected_names:
        missing = sorted(expected_names - written_names)
        extra = sorted(written_names - expected_names)
        raise RuntimeError(
            f"Output coverage mismatch for {clip.clip_id}: missing={missing[:1]}, extra={extra[:1]}"
        )
    np.savez_compressed(
        diagnostics_path,
        frame_names=clip.frame_names,
        delta_rotvec=delta.cpu().numpy(),
        gate=gate.cpu().numpy(),
        reliability=prediction["reliability"].cpu().numpy(),
        fallback_mask=fallback.cpu().numpy(),
        joint_position=prediction["joint_position"].cpu().numpy(),
        log_variance=(
            prediction["log_variance"].cpu().numpy()
            if "log_variance" in prediction
            else np.zeros((length, 51, 1), np.float32)
        ),
    )
    summary = {
        "clip_id": clip.clip_id,
        "frames": length,
        "cache": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
        "max_delta_degrees": float(
            torch.rad2deg(
                geodesic_distance(output_matrix, initial_matrix.to(device))
            ).max()
        ),
        "mean_gate": float(gate.mean()),
        "fallback_group_frames": int(fallback.sum()),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--cache-root", "--cache", dest="cache_root", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Held-out calibration JSON produced by phase2_refiner.calibrate",
    )
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Use raw rather than EMA checkpoint weights",
    )
    parser.add_argument(
        "--identity", action="store_true", help="Allow zero-head identity model"
    )
    parser.add_argument("--sign", action="append")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is None and not args.identity:
        raise ValueError(
            "Provide --checkpoint, or explicitly use --identity for a smoke test"
        )
    config = load_config(args.config)
    validate_config(config)
    device = torch.device(args.device)
    model = _load_model(config, args.checkpoint, device, use_ema=not args.raw_weights)
    if model.predict_uncertainty and args.calibration is None and not args.identity:
        raise ValueError("U1 inference requires a held-out --calibration report")
    uncertainty_offset = 0.0
    if args.calibration is not None:
        with args.calibration.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        if not calibration.get("gate", {}).get("passed", False):
            raise ValueError("Calibration report did not pass the U1 gate")
        uncertainty_offset = float(calibration["all"]["log_variance_offset"])
    cache_paths = sorted((args.cache_root / "clips").glob("*.npz"))
    if args.sign:
        requested = set(args.sign)
        cache_paths = [path for path in cache_paths if path.stem in requested]
    if not cache_paths:
        raise ValueError("No matching cache clips")
    output_root = args.output.resolve()
    run_manifest_path = output_root / "run_manifest.json"
    if run_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {run_manifest_path}")
    summaries = []
    reprojection_residual_scale = float(
        config.get("data", {}).get("reprojection_residual_scale", 10.0)
    )
    for cache_path in cache_paths:
        summary = infer_clip(
            cache_path,
            output_root,
            model,
            device,
            args.overwrite,
            uncertainty_offset,
            reprojection_residual_scale,
        )
        if args.render:
            sign_root = output_root / summary["clip_id"] / "smplifyx"
            clip = load_cache_clip(cache_path)
            summary["meshes"] = render_source_anchored_directory(
                sign_root / "results",
                sign_root / "meshes",
                clip.source_paths,
                args.model_folder,
                device,
                overwrite=args.overwrite,
            )
        summaries.append(summary)
        print(f"[infer] {summary['clip_id']}: {summary['frames']} frames")
    run_manifest = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        "calibration": str(args.calibration.resolve()) if args.calibration else None,
        "calibration_sha256": (
            sha256_file(args.calibration) if args.calibration else None
        ),
        "uncertainty_offset": uncertainty_offset,
        "reprojection_residual_scale": reprojection_residual_scale,
        "cache_manifest": (
            str((args.cache_root / "manifest.json").resolve())
            if (args.cache_root / "manifest.json").exists()
            else None
        ),
        "cache_manifest_sha256": (
            sha256_file(args.cache_root / "manifest.json")
            if (args.cache_root / "manifest.json").exists()
            else None
        ),
        "weights": "raw" if args.raw_weights else "ema",
        "identity": bool(args.identity),
        "provenance": run_provenance(args.config, int(config.get("seed", 42))),
        "clips": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with run_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
