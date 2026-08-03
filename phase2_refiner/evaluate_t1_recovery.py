"""Evaluate the predeclared T1 4/8/16-frame synthetic-recovery gate."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phase2_refiner.config import load_config, validate_config
from phase2_refiner.data.dataset import SequenceCacheDataset, collate_sequences
from phase2_refiner.infer import _load_model
from phase2_refiner.losses import RefinerLoss
from phase2_refiner.train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument(
        "--eval-precision",
        choices=("fp32", "training"),
        default="fp32",
        help=(
            "Use exact FP32 for the formal report (default), or reproduce the "
            "training autocast precision."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite T1 report: {args.output}")
    config = load_config(args.config)
    validate_config(config, require_data=True, require_validation=True)
    device = torch.device(args.device)
    max_frames = int(config.get("model", {}).get("max_frames", 64))
    data_config = config["data"]
    dataset = SequenceCacheDataset(
        data_config["val_glob"],
        max_frames=max_frames,
        training=False,
        input_dim=int(config.get("model", {}).get("input_dim", 43)),
        reprojection_residual_scale=float(
            data_config.get("reprojection_residual_scale", 10.0)
        ),
        physical_time_motion=bool(data_config.get("physical_time_motion", False)),
        motion_reference_seconds=float(
            data_config.get("motion_reference_seconds", 0.04)
        ),
        require_phase2r_semantics=bool(
            data_config.get("require_phase2r_semantics", False)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        pin_memory=device.type == "cuda",
    )
    model = _load_model(
        config, args.checkpoint.resolve(), device, use_ema=not args.raw_weights
    )
    loss_fn = RefinerLoss(**config.get("loss", {})).to(device)
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

    clean = evaluate(model, loss_fn, loader, device, autocast_context, None)
    durations = {}
    base_corruption = dict(config.get("validation_corruption", {}))
    for duration in (4, 8, 16):
        corruption = dict(base_corruption)
        corruption.update(probability=1.0, min_duration=duration, max_duration=duration)
        durations[str(duration)] = evaluate(
            model,
            loss_fn,
            loader,
            device,
            autocast_context,
            None,
            corruption=corruption,
            corruption_seed=args.seed + duration,
        )
    recovery_go = all(
        metrics["recovery_fraction"] >= 0.30 for metrics in durations.values()
    )
    report = {
        "stage": "T1 synthetic hand-corruption recovery",
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": "raw" if args.raw_weights else "EMA",
        "evaluation_precision": (
            precision if args.eval_precision == "training" else "fp32"
        ),
        "clean": clean,
        "durations": durations,
        "gates": {
            "rotation_recovery_at_least_30_percent_all_durations": recovery_go,
            "vertex_clean_preservation_below_2_percent": None,
            "G3": False,
        },
        "decision": (
            "NO-GO: rotation recovery below 30%"
            if not recovery_go
            else "PENDING: rotation proxy passed; run decoded vertex preservation gate"
        ),
        "note": (
            "G3 remains false until decoded regional vertex recovery and clean "
            "preservation are measured; rotation recovery alone is not sufficient."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
