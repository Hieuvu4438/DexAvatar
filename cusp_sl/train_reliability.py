"""Train Q and fit its scalar temperature on the source-disjoint validation set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from cusp_sl.config import load_config
from cusp_sl.geometry import geodesic_distance
from cusp_sl.models import ReliabilityCalibrator, TemperatureScaler
from cusp_sl.training import (
    append_jsonl, autocast_context, binary_calibration_metrics, config_sha256,
    make_loader, resolve_device, save_checkpoint, seed_everything, to_device,
)
from phase2_refiner.data.corruptions import apply_residual_mixture


def _tolerance(config, device) -> torch.Tensor:
    value = torch.full((51,), math.radians(config.reliability.hand_tolerance_degrees), device=device)
    value[:21] = math.radians(config.reliability.body_tolerance_degrees)
    return value


def corrupt_and_label(batch: dict, config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, base, _, _ = apply_residual_mixture(
        batch["features"], batch["initial_matrix"], batch["target_matrix"],
        batch["frame_valid"], batch["target_rotation_valid"],
        real_fraction=config.training.real_fraction,
        synthetic_fraction=config.training.synthetic_fraction,
        clean_fraction=config.training.clean_fraction,
        corruption={
            "min_duration": 2, "max_duration": config.data.window_size,
            "max_rotation_degrees": config.flow.hand_max_degrees,
        },
    )
    error = geodesic_distance(base, batch["target_matrix"])
    labels = error <= _tolerance(config, error.device)[None, None]
    supervised = batch["frame_valid"][:, :, None] & batch["target_rotation_valid"]
    supervised &= batch["refine_mask"][:, None]
    return features, labels, supervised


@torch.no_grad()
def collect_validation(model, loader, config, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    torch.manual_seed(config.training.seed + 1001)
    for batch in loader:
        batch = to_device(batch, device)
        features, target, mask = corrupt_and_label(batch, config)
        with autocast_context(config, device):
            prediction = model(features)
        logits.append(prediction[mask].float().cpu())
        labels.append(target[mask].float().cpu())
    return torch.cat(logits), torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path, help="Versioned run directory; defaults to config.output_dir/reliability")
    parser.add_argument("--resume", type=Path, help="Resume model/optimizer; --steps remains the target total step")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    train = make_loader(config, "train", shuffle=True)
    val = make_loader(config, "val", shuffle=False)
    model = ReliabilityCalibrator(
        config.data.input_dim, config.reliability.hidden_size,
        config.reliability.temporal_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    output = args.output or (Path(config.output_dir) / "reliability")
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    steps = args.steps or config.training.reliability_steps
    start_step = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("config_sha256") != config_sha256(args.config):
            raise ValueError("Resume checkpoint/config hash mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        if start_step > steps:
            raise ValueError(f"Resume step {start_step - 1} already reaches target {steps}")
    iterator = iter(train)
    best_brier = float("inf")
    best_step = None
    model.train()
    for step in range(start_step, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train)
            batch = next(iterator)
        batch = to_device(batch, device)
        features, labels, mask = corrupt_and_label(batch, config)
        positives = labels[mask].float().sum()
        negatives = mask.sum() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(0.25, 4.0)
        with autocast_context(config, device):
            logits = model(features)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float(), pos_weight=pos_weight
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()
        if step % 25 == 0 or step == 1:
            append_jsonl(output / "train.jsonl", {"step": step, "loss": float(loss.detach()), "tokens": int(mask.sum())})
        if step % config.training.validate_every == 0 or step == steps:
            val_logits, val_labels = collect_validation(model, val, config, device)
            scaler = TemperatureScaler().to(device)
            temperature = scaler.fit(val_logits.to(device), val_labels.to(device))
            probability = torch.sigmoid(val_logits / temperature).numpy()
            metrics = binary_calibration_metrics(probability, val_labels.numpy())
            metrics.update({"step": step, "temperature": temperature})
            append_jsonl(output / "validation.jsonl", metrics)
            if metrics["brier"] < best_brier:
                best_brier = metrics["brier"]
                best_step = step
                (output / "calibration.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                save_checkpoint(
                    output / "best.pt", model, optimizer, step, args.config,
                    temperature=temperature, validation=metrics,
                    selection_metric="validation_brier",
                )
            model.train()
    print(json.dumps({
        "checkpoint": str(output / "best.pt"), "steps": steps,
        "start_step": start_step, "best_step": best_step,
        "best_validation_brier": best_brier,
    }, indent=2))


if __name__ == "__main__":
    main()
