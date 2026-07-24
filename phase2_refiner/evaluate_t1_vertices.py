"""Decode SMPL-X and evaluate regional T1 vertex recovery on partial targets."""

from __future__ import annotations

import argparse
import json
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase2_refiner.config import load_config, validate_config
from phase2_refiner.data.corruptions import apply_burst_corruption
from phase2_refiner.data.dataset import SequenceCacheDataset, collate_sequences
from phase2_refiner.geometry.smplx_decode import decode_smplx_sequence
from phase2_refiner.infer import _load_model
from phase2_refiner.render import create_smplx_model


REGION_JOINTS = {
    "ubody": slice(0, 21),
    "left_hand": slice(21, 36),
    "right_hand": slice(36, 51),
}


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _face(batch: dict) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].float()
        for key in ("jaw_pose", "leye_pose", "reye_pose", "expression")
    }


def _decode(model, matrix: torch.Tensor, batch: dict) -> torch.Tensor:
    vertices, _ = decode_smplx_sequence(
        model,
        matrix.float(),
        batch["betas"].float(),
        batch["global_orient"].float(),
        batch["transl"].float(),
        **_face(batch),
    )
    return vertices


def _accumulate(
    totals: dict[str, dict[str, float]],
    name: str,
    error: torch.Tensor,
    frame_mask: torch.Tensor,
) -> None:
    selected = error[frame_mask]
    if selected.numel() == 0:
        return
    totals[name]["sum"] += float(selected.sum())
    totals[name]["count"] += int(selected.numel())
    totals[name]["frames"] += int(frame_mask.sum())


def _means(
    totals: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int | None]]:
    return {
        name: {
            "mean_mm": (values["sum"] / values["count"] if values["count"] else None),
            "frames": int(values["frames"]),
        }
        for name, values in totals.items()
    }


def _empty(regions: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {name: {"sum": 0.0, "count": 0.0, "frames": 0.0} for name in regions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--vertex-ids", type=Path, required=True)
    parser.add_argument("--upper-body-ids", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument(
        "--eval-precision",
        choices=("fp32", "training"),
        default="fp32",
        help=(
            "Use exact FP32 for the formal gate (default), or reproduce the "
            "training autocast precision."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite vertex report: {args.output}")
    config = load_config(args.config)
    validate_config(config, require_data=True, require_validation=True)
    device = torch.device(args.device)
    dataset = SequenceCacheDataset(
        config["data"]["val_glob"],
        max_frames=int(config["model"]["max_frames"]),
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        pin_memory=device.type == "cuda",
    )
    refiner = _load_model(
        config, args.checkpoint.resolve(), device, use_ema=not args.raw_weights
    )
    body_model = create_smplx_model(args.model_folder, device)
    with args.vertex_ids.open("rb") as handle:
        hand_ids = pickle.load(handle, encoding="latin1")
    vertex_ids = {
        "left_hand": torch.as_tensor(hand_ids["left_hand"], device=device),
        "right_hand": torch.as_tensor(hand_ids["right_hand"], device=device),
    }
    if args.upper_body_ids is not None:
        vertex_ids["ubody"] = torch.as_tensor(
            np.load(args.upper_body_ids), device=device
        )
    regions = tuple(vertex_ids)
    precision = str(config.get("training", {}).get("precision", "bf16")).lower()
    amp_enabled = (
        args.eval_precision == "training"
        and device.type == "cuda"
        and precision in {"bf16", "fp16"}
    )
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    def autocast_context():
        if amp_enabled:
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    clean_totals = _empty(regions)
    duration_totals = {
        duration: {"injected": _empty(regions), "residual": _empty(regions)}
        for duration in (4, 8, 16)
    }
    base_corruption = dict(config.get("validation_corruption", {}))
    cuda_devices = (
        [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        for batch_index, batch in enumerate(loader):
            batch = _to_device(batch, device)
            target_vertices = _decode(body_model, batch["target_matrix"], batch)
            with autocast_context():
                clean_prediction = refiner(
                    batch["features"],
                    batch["initial_matrix"],
                    batch["frame_valid"],
                    batch["refine_mask"],
                    batch["initial_joint_position"],
                )
            clean_vertices = _decode(body_model, clean_prediction["matrix"], batch)
            for region, ids in vertex_ids.items():
                valid = batch["target_rotation_valid"][..., REGION_JOINTS[region]].all(
                    -1
                )
                valid &= batch["frame_valid"]
                error = (
                    torch.linalg.vector_norm(
                        clean_vertices.index_select(-2, ids)
                        - target_vertices.index_select(-2, ids),
                        dim=-1,
                    )
                    * 1000.0
                )
                _accumulate(clean_totals, region, error, valid)

            for duration in (4, 8, 16):
                torch.manual_seed(args.seed + duration + batch_index * 1009)
                corruption = dict(base_corruption)
                corruption.update(
                    probability=1.0, min_duration=duration, max_duration=duration
                )
                features, initial_matrix, corruption_mask = apply_burst_corruption(
                    batch["features"],
                    batch["initial_matrix"],
                    batch["frame_valid"],
                    target_rotation_valid=batch["target_rotation_valid"],
                    **corruption,
                )
                with autocast_context():
                    prediction = refiner(
                        features,
                        initial_matrix,
                        batch["frame_valid"],
                        batch["refine_mask"],
                        batch["initial_joint_position"],
                    )
                initial_vertices = _decode(body_model, initial_matrix, batch)
                predicted_vertices = _decode(body_model, prediction["matrix"], batch)
                for region, ids in vertex_ids.items():
                    frame_mask = corruption_mask[..., REGION_JOINTS[region]].any(-1)
                    frame_mask &= batch["target_rotation_valid"][
                        ..., REGION_JOINTS[region]
                    ].all(-1)
                    injected = (
                        torch.linalg.vector_norm(
                            initial_vertices.index_select(-2, ids)
                            - target_vertices.index_select(-2, ids),
                            dim=-1,
                        )
                        * 1000.0
                    )
                    residual = (
                        torch.linalg.vector_norm(
                            predicted_vertices.index_select(-2, ids)
                            - target_vertices.index_select(-2, ids),
                            dim=-1,
                        )
                        * 1000.0
                    )
                    _accumulate(
                        duration_totals[duration]["injected"],
                        region,
                        injected,
                        frame_mask,
                    )
                    _accumulate(
                        duration_totals[duration]["residual"],
                        region,
                        residual,
                        frame_mask,
                    )

    clean = _means(clean_totals)
    durations = {}
    region_recovery_go = True
    clean_go = True
    for duration, totals in duration_totals.items():
        injected = _means(totals["injected"])
        residual = _means(totals["residual"])
        region_metrics = {}
        for region in regions:
            before = injected[region]["mean_mm"]
            after = residual[region]["mean_mm"]
            recovery = (
                None if before in (None, 0.0) or after is None else 1.0 - after / before
            )
            clean_error = clean[region]["mean_mm"]
            clean_ratio = (
                None
                if before in (None, 0.0) or clean_error is None
                else clean_error / before
            )
            if recovery is not None:
                region_recovery_go &= recovery >= 0.30
            if clean_ratio is not None:
                clean_go &= clean_ratio < 0.02
            region_metrics[region] = {
                "injected_mm": before,
                "residual_mm": after,
                "recovery_fraction": recovery,
                "corrupted_frames": injected[region]["frames"],
                "clean_mm": clean_error,
                "clean_to_injected_fraction": clean_ratio,
            }
        durations[str(duration)] = region_metrics
    available = {
        region: clean[region]["frames"] > 0
        and any(
            durations[str(duration)][region]["corrupted_frames"] > 0
            for duration in (4, 8, 16)
        )
        for region in regions
    }
    full_regions = all(available.get(name, False) for name in REGION_JOINTS)
    report = {
        "stage": "T1 decoded regional SMPL-X vertex recovery",
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": "raw" if args.raw_weights else "EMA",
        "evaluation_precision": (
            precision if args.eval_precision == "training" else "fp32"
        ),
        "units": "millimetres",
        "clean": clean,
        "durations": durations,
        "available_regions": available,
        "gates": {
            "regional_recovery_at_least_30_percent": region_recovery_go,
            "clean_to_injected_below_2_percent": clean_go,
            "all_upper_body_left_right_regions_available": full_regions,
            "G3": region_recovery_go and clean_go and full_regions,
        },
    }
    report["decision"] = (
        "GO: formal G3 passed"
        if report["gates"]["G3"]
        else (
            "GO: available-region vertex gates passed; full G3 awaits missing regions"
            if region_recovery_go and clean_go
            else "NO-GO: decoded regional vertex gate failed"
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
