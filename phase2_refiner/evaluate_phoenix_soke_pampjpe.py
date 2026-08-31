"""Calibrate and evaluate full-PHOENIX reconstruction with SOKE PA-MPJPE.

``calibrate`` selects one benefit threshold per region on official PHOENIX dev
using full-sequence rotation error. ``evaluate`` opens official test only after
the checkpoint and thresholds are frozen, then reports the exact SOKE Table-3
body/hand PA-MPJPE construction: per-frame similarity alignment of J14 body
joints and independent 21-joint alignment of each hand, averaged in mm.

For decoder comparability, both target and prediction use SOKE's fixed shape,
zero root/lower-body convention and the same neutral SMPL-X layer.  The current
Transformer does not reconstruct jaw/expression, which do not enter the body or
hand joint regressors used by these two metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase2_refiner.config import load_config
from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip
from phase2_refiner.geometry.rotations import (
    geodesic_distance,
    matrix_to_axis_angle,
)
from phase2_refiner.infer import _load_model, _predict_sequence
from phase2_refiner.provenance import run_provenance, sha256_file


SCHEMA = "signal4d-phoenix-soke-pampjpe-v1"
REGIONS = {"body": (0, 21), "left_hand": (21, 36), "right_hand": (36, 51)}
SOKE_FIXED_SHAPE = np.asarray(
    [
        -0.07284723,
        0.1795129,
        -0.27608207,
        0.135155,
        0.10748172,
        0.16037364,
        -0.01616933,
        -0.03450319,
        0.01369138,
        0.01108842,
    ],
    dtype=np.float32,
)
SOKE_TABLE3 = {
    "phoenix_body_pa_mpjpe_mm": 25.79,
    "phoenix_hand_pa_mpjpe_mm": 6.78,
    "training": "joint How2Sign + CSL-Daily + PHOENIX-2014T tokenizer",
    "source": "Signs as Tokens (SOKE), ICCV 2025, Table 3",
}


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _predict(model, clip, config: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    data = config.get("data", {})
    features, initial = features_from_clip(
        clip,
        input_dim=int(config.get("model", {}).get("input_dim", 43)),
        reprojection_residual_scale=float(data.get("reprojection_residual_scale", 10.0)),
        physical_time_motion=bool(data.get("physical_time_motion", False)),
        motion_reference_seconds=float(data.get("motion_reference_seconds", 0.04)),
    )
    prediction = _predict_sequence(
        model,
        features,
        initial,
        torch.from_numpy(clip.refine_mask),
        device,
    )
    return initial.to(device), prediction


@torch.no_grad()
def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device)
    model = _load_model(config, checkpoint, device, use_ema=True)
    candidates = [0.0] + np.linspace(0.05, 0.95, 19).tolist() + [1.0]
    totals = {
        region: {
            float(threshold): {"baseline": 0.0, "candidate": 0.0, "frames": 0}
            for threshold in candidates
        }
        for region in REGIONS
    }
    paths = _manifest_paths(manifest)
    for index, path in enumerate(paths, start=1):
        clip = load_cache_clip(path)
        metadata = json.loads(clip.metadata_json)
        if metadata.get("official_split") != "dev":
            raise ValueError(f"Calibration is not official dev: {path}")
        initial, prediction = _predict(model, clip, config, device)
        target = torch.from_numpy(clip.target_axis_angle).float().to(device)
        from phase2_refiner.geometry.rotations import axis_angle_to_matrix

        target_matrix = axis_angle_to_matrix(target)
        baseline_error = geodesic_distance(initial, target_matrix)
        candidate_error = geodesic_distance(prediction["matrix"].float(), target_matrix)
        probability = prediction["benefit_logit"].sigmoid().float()
        target_valid = torch.from_numpy(clip.target_rotation_valid).bool().to(device)
        refine = torch.from_numpy(clip.refine_mask).bool().to(device)
        for group, (region, (start, stop)) in enumerate(REGIONS.items()):
            valid = target_valid[:, start:stop] & refine[start:stop]
            denominator = valid.sum(dim=-1)
            valid_frame = denominator > 0
            baseline_frame = (
                baseline_error[:, start:stop] * valid
            ).sum(dim=-1) / denominator.clamp_min(1)
            candidate_frame = (
                candidate_error[:, start:stop] * valid
            ).sum(dim=-1) / denominator.clamp_min(1)
            for threshold in candidates:
                selected = probability[:, group] >= float(threshold)
                error = torch.where(selected, candidate_frame, baseline_frame)
                cell = totals[region][float(threshold)]
                cell["baseline"] += float(baseline_frame[valid_frame].sum())
                cell["candidate"] += float(error[valid_frame].sum())
                cell["frames"] += int(valid_frame.sum())
        if index % 25 == 0 or index == len(paths):
            print(f"[phoenix-calibrate] {index}/{len(paths)}", flush=True)

    regions = {}
    thresholds = {}
    for region, grid in totals.items():
        rows = []
        for threshold, cell in grid.items():
            baseline = cell["baseline"] / max(cell["frames"], 1)
            candidate = cell["candidate"] / max(cell["frames"], 1)
            rows.append(
                {
                    "threshold": threshold,
                    "baseline_radians": baseline,
                    "candidate_radians": candidate,
                    "candidate_over_baseline": candidate / max(baseline, 1e-12),
                    "frames": cell["frames"],
                }
            )
        # Prefer greater abstention on exact numerical ties.
        selected = min(rows, key=lambda row: (row["candidate_radians"], -row["threshold"]))
        thresholds[region] = selected["threshold"]
        regions[region] = {"selected": selected, "grid": rows}
    return {
        "schema": SCHEMA,
        "mode": "calibration",
        "selection_split": "official PHOENIX dev",
        "metric": "frame-micro mean geodesic rotation proxy",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "thresholds": thresholds,
        "regions": regions,
        "clips": len(paths),
    }


def _rigid_align_soke(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SOKE's batched scale+rotation+translation Procrustes implementation."""

    n = prediction.shape[1]
    centroid_p = prediction.mean(dim=1, keepdim=True)
    centroid_q = target.mean(dim=1, keepdim=True)
    p = prediction - centroid_p
    q = target - centroid_q
    covariance = p.transpose(1, 2) @ q / n
    u, singular, vt = torch.linalg.svd(covariance)
    determinant = torch.det(vt.transpose(1, 2) @ u.transpose(1, 2))
    flip = determinant < 0
    if flip.any():
        vt = vt.clone()
        singular = singular.clone()
        vt[flip, -1] *= -1.0
        singular[flip, -1] *= -1.0
    rotation = vt.transpose(1, 2) @ u.transpose(1, 2)
    variance = torch.var(prediction, dim=1, correction=0).sum(dim=-1)
    scale = (singular.sum(dim=-1) / variance).view(-1, 1, 1)
    translation = -(scale * rotation) @ centroid_p.transpose(1, 2)
    translation = translation + centroid_q.transpose(1, 2)
    return (
        (scale * rotation) @ prediction.transpose(1, 2) + translation
    ).transpose(1, 2)


def _hand_regressors(model, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    regressor = model.J_regressor
    if getattr(regressor, "is_sparse", False):
        regressor = regressor.to_dense()
    regressor = regressor.float().to(device)
    vertices = regressor.shape[1]

    def vertex_row(index: int) -> torch.Tensor:
        row = torch.zeros(vertices, device=device)
        row[index] = 1.0
        return row

    left = torch.stack(
        [
            regressor[20], regressor[37], regressor[38], regressor[39], vertex_row(5361),
            regressor[25], regressor[26], regressor[27], vertex_row(4933),
            regressor[28], regressor[29], regressor[30], vertex_row(5058),
            regressor[34], regressor[35], regressor[36], vertex_row(5169),
            regressor[31], regressor[32], regressor[33], vertex_row(5286),
        ]
    )
    right = torch.stack(
        [
            regressor[21], regressor[52], regressor[53], regressor[54], vertex_row(8079),
            regressor[40], regressor[41], regressor[42], vertex_row(7669),
            regressor[43], regressor[44], regressor[45], vertex_row(7794),
            regressor[49], regressor[50], regressor[51], vertex_row(7905),
            regressor[46], regressor[47], regressor[48], vertex_row(8022),
        ]
    )
    # SOKE's orig_hand_regressor includes wrist plus 4 joints per finger = 21.
    return left, right


def _create_soke_decoder(model_folder: Path, device: torch.device):
    import smplx

    folder = model_folder.resolve()
    model = smplx.create(
        str(folder),
        "smplx",
        gender="NEUTRAL",
        use_pca=False,
        use_face_contour=True,
        num_betas=10,
        num_expression_coeffs=10,
        create_global_orient=False,
        create_body_pose=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_betas=False,
        create_expression=False,
        create_transl=False,
    ).to(device).eval()
    return model


@torch.no_grad()
def _decode_vertices(model, pose: torch.Tensor, device: torch.device) -> torch.Tensor:
    frames = len(pose)
    body = torch.zeros(frames, 21, 3, device=device)
    # SOKE removes root + first 11 body joints before tokenization, then fills
    # those 36 axis-angle dimensions with zeros during reconstruction.
    body[:, 11:] = pose[:, 11:21]
    zeros3 = torch.zeros(frames, 3, device=device)
    output = model(
        betas=torch.as_tensor(SOKE_FIXED_SHAPE, device=device).repeat(frames, 1),
        global_orient=zeros3,
        body_pose=body.reshape(frames, 63),
        left_hand_pose=pose[:, 21:36].reshape(frames, 45),
        right_hand_pose=pose[:, 36:51].reshape(frames, 45),
        jaw_pose=zeros3,
        leye_pose=zeros3,
        reye_pose=zeros3,
        expression=torch.zeros(frames, 10, device=device),
        transl=zeros3,
    )
    return output.vertices


def _region_errors(
    predicted_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    j14: torch.Tensor,
    left_regressor: torch.Tensor,
    right_regressor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_body = torch.einsum("jv,tvc->tjc", j14, target_vertices)
    predicted_body = torch.einsum("jv,tvc->tjc", j14, predicted_vertices)
    aligned_body = _rigid_align_soke(predicted_body, target_body)
    body = torch.linalg.vector_norm(aligned_body - target_body, dim=-1).mean(dim=-1)
    hand_errors = []
    for regressor in (left_regressor, right_regressor):
        target_hand = torch.einsum("jv,tvc->tjc", regressor, target_vertices)
        predicted_hand = torch.einsum("jv,tvc->tjc", regressor, predicted_vertices)
        aligned_hand = _rigid_align_soke(predicted_hand, target_hand)
        hand_errors.append(
            torch.linalg.vector_norm(aligned_hand - target_hand, dim=-1).mean(dim=-1)
        )
    hand = 0.5 * (hand_errors[0] + hand_errors[1])
    return body, hand


def _gated_matrix(
    prediction: dict[str, torch.Tensor],
    initial: torch.Tensor,
    thresholds: dict[str, float],
) -> torch.Tensor:
    result = prediction["matrix"].clone()
    probability = prediction["benefit_logit"].sigmoid()
    for group, (region, (start, stop)) in enumerate(REGIONS.items()):
        threshold = float(thresholds[region])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid {region} threshold: {threshold}")
        reject = probability[:, group] < threshold
        result[:, start:stop] = torch.where(
            reject[:, None, None, None], initial[:, start:stop], result[:, start:stop]
        )
    return result


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    calibration_path = args.calibration.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("mode") != "calibration":
        raise ValueError("Invalid PHOENIX calibration artifact")
    if calibration.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("Checkpoint/calibration hash mismatch")
    device = torch.device(args.device)
    model = _load_model(config, checkpoint, device, use_ema=True)
    decoder = _create_soke_decoder(args.model_folder, device)
    assets = args.model_folder.resolve() / "smplx"
    with (assets / "SMPLX_to_J14.pkl").open("rb") as handle:
        j14 = torch.as_tensor(pickle.load(handle, encoding="latin1"), device=device).float()
    left_regressor, right_regressor = _hand_regressors(decoder, device)
    paths = _manifest_paths(manifest)
    totals = {
        variant: {"body": 0.0, "hand": 0.0, "frames": 0}
        for variant in ("initializer", "transformer_always", "transformer_gated")
    }
    clip_rows = []
    for clip_index, path in enumerate(paths, start=1):
        clip = load_cache_clip(path)
        metadata = json.loads(clip.metadata_json)
        if metadata.get("official_split") != "test":
            raise ValueError(f"Final evaluation is not official test: {path}")
        if clip.target_axis_angle is None or clip.target_rotation_valid is None:
            raise ValueError(f"Official-test cache lacks reconstruction target: {path}")
        if not np.asarray(clip.target_rotation_valid, dtype=bool).all():
            invalid = int((~np.asarray(clip.target_rotation_valid, dtype=bool)).sum())
            raise ValueError(
                f"Official-test target contains {invalid} invalid joint-frames: {path}"
            )
        initial, prediction = _predict(model, clip, config, device)
        variants = {
            "initializer": initial,
            "transformer_always": prediction["matrix"].float(),
            "transformer_gated": _gated_matrix(
                prediction, initial, calibration["thresholds"]
            ),
        }
        target_pose = torch.from_numpy(clip.target_axis_angle).float().to(device)
        clip_sum = {
            variant: {"body": 0.0, "hand": 0.0, "frames": 0}
            for variant in variants
        }
        for start in range(0, len(target_pose), args.decode_batch_size):
            stop = min(start + args.decode_batch_size, len(target_pose))
            target_vertices = _decode_vertices(decoder, target_pose[start:stop], device)
            for variant, matrix in variants.items():
                pose = matrix_to_axis_angle(matrix[start:stop])
                predicted_vertices = _decode_vertices(decoder, pose, device)
                body, hand = _region_errors(
                    predicted_vertices,
                    target_vertices,
                    j14,
                    left_regressor,
                    right_regressor,
                )
                count = stop - start
                body_sum = float(body.sum())
                hand_sum = float(hand.sum())
                totals[variant]["body"] += body_sum
                totals[variant]["hand"] += hand_sum
                totals[variant]["frames"] += count
                clip_sum[variant]["body"] += body_sum
                clip_sum[variant]["hand"] += hand_sum
                clip_sum[variant]["frames"] += count
        clip_rows.append(
            {
                "clip_id": clip.clip_id,
                "frames": len(target_pose),
                "metrics_mm": {
                    variant: {
                        "body_pa_mpjpe": values["body"] / values["frames"] * 1000.0,
                        "hand_pa_mpjpe": values["hand"] / values["frames"] * 1000.0,
                    }
                    for variant, values in clip_sum.items()
                },
            }
        )
        if clip_index % 10 == 0 or clip_index == len(paths):
            print(f"[phoenix-pampjpe] {clip_index}/{len(paths)}", flush=True)

    metrics = {
        variant: {
            "body_pa_mpjpe_mm": values["body"] / values["frames"] * 1000.0,
            "hand_pa_mpjpe_mm": values["hand"] / values["frames"] * 1000.0,
            "frames": values["frames"],
        }
        for variant, values in totals.items()
    }
    gated = metrics["transformer_gated"]
    comparison = {
        "body_delta_vs_soke_mm": gated["body_pa_mpjpe_mm"]
        - SOKE_TABLE3["phoenix_body_pa_mpjpe_mm"],
        "hand_delta_vs_soke_mm": gated["hand_pa_mpjpe_mm"]
        - SOKE_TABLE3["phoenix_hand_pa_mpjpe_mm"],
        "beats_soke_body": gated["body_pa_mpjpe_mm"]
        < SOKE_TABLE3["phoenix_body_pa_mpjpe_mm"],
        "beats_soke_hand": gated["hand_pa_mpjpe_mm"]
        < SOKE_TABLE3["phoenix_hand_pa_mpjpe_mm"],
    }
    comparison["beats_soke_both"] = bool(
        comparison["beats_soke_body"] and comparison["beats_soke_hand"]
    )
    return {
        "schema": SCHEMA,
        "mode": "final_test_evaluation",
        "evaluation_split": "official PHOENIX test",
        "metric_definition": {
            "body": "per-frame Procrustes-aligned J14 MPJPE",
            "hand": "mean of independently Procrustes-aligned left/right 21-joint hand MPJPE",
            "aggregation": "frame-micro mean, millimeters",
            "decoder": "SOKE fixed shape; zero root and first 11 body joints",
        },
        "protocol_difference": (
            "Our Transformer is trained on PHOENIX only from an RGB-expert initializer; "
            "SOKE Table 3 is an encode/decode tokenizer trained jointly on H2S+CSL+PHOENIX."
        ),
        "soke_table3_reference": SOKE_TABLE3,
        "metrics": metrics,
        "comparison": comparison,
        "clips": len(paths),
        "frames": sum(row["frames"] for row in clip_rows),
        "per_clip": clip_rows,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "provenance": run_provenance(args.config, int(config.get("seed", 42))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.mode == "evaluate" and args.calibration is None:
        parser.error("--calibration is required for evaluate mode")
    report = calibrate(args) if args.mode == "calibrate" else evaluate(args)
    _write(args.output.resolve(), report)
    summary = {key: report[key] for key in report if key not in {"per_clip", "regions"}}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
