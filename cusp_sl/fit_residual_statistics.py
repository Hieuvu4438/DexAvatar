"""Fit residual mean/std on the training split and declared data mixture only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config
from cusp_sl.train_flow import prepare_residual_targets
from cusp_sl.training import config_sha256, make_loader, resolve_device, seed_everything, to_device


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--minimum-std-degrees", type=float, default=0.25,
        help="Numerical floor applied after estimating each coordinate std",
    )
    parser.add_argument(
        "--max-batches", type=int,
        help="Smoke-test only; omit for the release statistic over one train epoch",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only statistics output exists: {args.output}")
    if args.minimum_std_degrees <= 0:
        raise ValueError("--minimum-std-degrees must be positive")

    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    # shuffle=True also activates the seeded random training-window policy. The
    # loader is consumed exactly once, so every included clip contributes once.
    loader = make_loader(
        config, "train", shuffle=True, drop_last=False,
        pin_memory=device.type == "cuda",
    )
    weighted_sum = torch.zeros(51, 3, dtype=torch.float64, device=device)
    weighted_square_sum = torch.zeros_like(weighted_sum)
    count = torch.zeros(51, dtype=torch.float64, device=device)
    mode_counts = torch.zeros(3, dtype=torch.int64, device=device)
    batches = 0
    for index, batch in enumerate(loader):
        if args.max_batches is not None and index >= args.max_batches:
            break
        batch = to_device(batch, device)
        _, _, target, weight, modes = prepare_residual_targets(batch, config)
        value = target.double()
        token_weight = weight.double()
        weighted_sum += (value * token_weight[..., None]).sum(dim=(0, 1))
        weighted_square_sum += (
            value.square() * token_weight[..., None]
        ).sum(dim=(0, 1))
        count += token_weight.sum(dim=(0, 1))
        mode_counts += torch.bincount(modes, minlength=3)
        batches += 1
        if batches % 100 == 0:
            print(f"[residual-statistics] batches={batches}, tokens={float(count.sum()):.0f}")

    if batches == 0 or not (count > 0).any():
        raise ValueError("No weighted training observations were found")
    supported = count > 0
    mean = torch.zeros_like(weighted_sum)
    mean[supported] = weighted_sum[supported] / count[supported, None]
    variance = torch.zeros_like(weighted_sum)
    variance[supported] = (
        weighted_square_sum[supported] / count[supported, None]
        - mean[supported].square()
    )
    floor = math.radians(args.minimum_std_degrees)
    std = torch.ones_like(variance)
    std[supported] = variance[supported].clamp_min(0).sqrt().clamp_min(floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        mean=mean.float().cpu().numpy(),
        std=std.float().cpu().numpy(),
        weighted_count=count.cpu().numpy(),
        supported_joint=supported.cpu().numpy(),
        mode_counts=mode_counts.cpu().numpy(),
        minimum_std_degrees=np.asarray(args.minimum_std_degrees),
        batches=np.asarray(batches),
        max_batches=np.asarray(-1 if args.max_batches is None else args.max_batches),
        seed=np.asarray(config.training.seed),
        config_sha256=np.asarray(config_sha256(args.config)),
        train_manifest_sha256=np.asarray(sha256(config.data.train_manifest)),
    )
    report = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "batches": batches,
        "mode_counts": mode_counts.cpu().tolist(),
        "weighted_tokens": float(count.sum()),
        "supported_joints": int(supported.sum()),
        "identity_statistics_joints": torch.nonzero(~supported).flatten().cpu().tolist(),
        "minimum_std_degrees": args.minimum_std_degrees,
        "mean_abs_degrees": float(torch.rad2deg(mean[supported].abs()).mean()),
        "std_degrees_median": float(torch.rad2deg(std[supported]).median()),
        "std_degrees_max": float(torch.rad2deg(std[supported]).max()),
        "release_complete_epoch": args.max_batches is None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
