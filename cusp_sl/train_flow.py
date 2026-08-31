"""Train the conditional residual rectified flow after freezing Q."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from cusp_sl.config import load_config
from cusp_sl.geometry import residual_target
from cusp_sl.models import ReliabilityCalibrator, SelectiveResidualFlow
from cusp_sl.normalization import ResidualNormalizer
from cusp_sl.training import (
    append_jsonl, autocast_context, config_sha256, make_loader, resolve_device,
    save_checkpoint, seed_everything,
    to_device,
)
from phase2_refiner.data.corruptions import apply_residual_mixture


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_q(config, path: Path, device, config_path: Path | None = None):
    q = ReliabilityCalibrator(
        config.data.input_dim, config.reliability.hidden_size,
        config.reliability.temporal_layers,
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # Check against the active file, not the serialized path (which may move).
    if (
        config_path is not None
        and checkpoint.get("config_sha256") != config_sha256(config_path)
    ):
        raise ValueError("Reliability checkpoint/config hash mismatch")
    q.load_state_dict(checkpoint["model"])
    q.eval().requires_grad_(False)
    return q, float(checkpoint["temperature"])


def prepare_residual_targets(batch, config):
    features, base, corruption, modes = apply_residual_mixture(
        batch["features"], batch["initial_matrix"], batch["target_matrix"],
        batch["frame_valid"], batch["target_rotation_valid"],
        real_fraction=config.training.real_fraction,
        synthetic_fraction=config.training.synthetic_fraction,
        clean_fraction=config.training.clean_fraction,
        corruption={"min_duration": 2, "max_duration": config.data.window_size,
                    "max_rotation_degrees": config.flow.hand_max_degrees},
    )
    target = residual_target(base, batch["target_matrix"])
    real = modes == 0
    clean = modes == 2
    mask = corruption.clone()
    if real.any():
        real_mask = batch["frame_valid"][real, :, None] & batch["target_rotation_valid"][real]
        real_mask &= batch["refine_mask"][real, None]
        mask[real] = real_mask
    if clean.any():
        clean_mask = batch["frame_valid"][clean, :, None] & batch["target_rotation_valid"][clean]
        clean_mask &= batch["refine_mask"][clean, None]
        mask[clean] = clean_mask
    quality = batch["target_quality"]
    quality = torch.where(quality > 0, quality, torch.ones_like(quality))
    weight = mask.float() * quality * batch["frame_valid"][:, :, None]
    return features, base, target, weight, modes


def prepare_batch(batch, config, q, temperature, normalizer=None):
    features, base, target, weight, _ = prepare_residual_targets(batch, config)
    with torch.no_grad():
        probability = torch.sigmoid(q(features) / temperature)
    condition = torch.cat((features, probability[..., None]), dim=-1)
    if normalizer is not None:
        target = normalizer.normalize(target)
    return condition, base, target, weight


@torch.no_grad()
def validate(model, loader, config, q, temperature, normalizer, device, batches=20):
    model.eval()
    total, tokens, endpoint = 0.0, 0.0, 0.0
    torch.manual_seed(config.training.seed + 2001)
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = to_device(batch, device)
        condition, base, target, weight = prepare_batch(
            batch, config, q, temperature, normalizer
        )
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=device)
        state = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * target
        with autocast_context(config, device):
            velocity = model(state, time, condition, batch["frame_valid"])
        squared = (velocity - (target - noise)).square().sum(dim=-1)
        total += float((squared * weight).sum())
        tokens += float(weight.sum())
        one_step = noise + velocity
        endpoint += float((one_step - target).square().sum(dim=-1).mul(weight).sum())
    model.train()
    return {"flow_mse": total / max(tokens, 1.0), "one_step_endpoint_mse": endpoint / max(tokens, 1.0), "weighted_tokens": tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reliability-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path, help="Versioned run directory; defaults to config.output_dir/flow")
    parser.add_argument("--resume", type=Path, help="Resume model/optimizer; --steps remains the target total step")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    train = make_loader(config, "train", shuffle=True)
    val = make_loader(config, "val", shuffle=False)
    q, temperature = _load_q(
        config, args.reliability_checkpoint, device, args.config
    )
    normalizer = ResidualNormalizer.from_path(config.flow.normalization_statistics)
    flow = SelectiveResidualFlow(
        config.data.input_dim + 1, config.flow.hidden_size, config.flow.layers,
        config.flow.heads, config.flow.mlp_ratio, config.flow.dropout,
        config.flow.body_max_degrees, config.flow.hand_max_degrees,
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    output = args.output or (Path(config.output_dir) / "flow")
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    steps = args.steps or config.training.flow_steps
    start_step = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("config_sha256") != config_sha256(args.config):
            raise ValueError("Resume checkpoint/config hash mismatch")
        flow.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        best = float(checkpoint.get("best_validation_metric", float("inf")))
        if start_step > steps:
            raise ValueError(f"Resume step {start_step - 1} already reaches target {steps}")
    iterator = iter(train)
    if not args.resume:
        best = float("inf")
    flow.train()
    for step in range(start_step, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train)
            batch = next(iterator)
        batch = to_device(batch, device)
        condition, _, target, weight = prepare_batch(
            batch, config, q, temperature, normalizer
        )
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=device)
        state = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * target
        with autocast_context(config, device):
            velocity = flow(state, time, condition, batch["frame_valid"])
            squared = (velocity - (target - noise)).square().sum(dim=-1)
            loss = (squared * weight).sum() / weight.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), config.training.gradient_clip)
        optimizer.step()
        if step % 25 == 0 or step == 1:
            append_jsonl(output / "train.jsonl", {"step": step, "loss": float(loss.detach()), "weighted_tokens": float(weight.sum())})
        if step % config.training.validate_every == 0 or step == steps:
            metrics = validate(
                flow, val, config, q, temperature, normalizer, device
            )
            metrics["step"] = step
            append_jsonl(output / "validation.jsonl", metrics)
            if metrics["flow_mse"] < best:
                best = metrics["flow_mse"]
                save_checkpoint(
                    output / "best.pt", flow, optimizer, step, args.config,
                    model_kind="flow",
                    reliability_checkpoint=str(args.reliability_checkpoint.resolve()),
                    reliability_checkpoint_sha256=sha256(
                        args.reliability_checkpoint
                    ),
                    reliability_temperature=temperature, validation=metrics,
                    residual_statistics=(
                        str(normalizer.source) if normalizer.source is not None else None
                    ),
                    residual_statistics_sha256=normalizer.sha256,
                    best_validation_metric=best,
                )
            save_checkpoint(
                output / "last.pt", flow, optimizer, step, args.config,
                model_kind="flow",
                reliability_checkpoint=str(args.reliability_checkpoint.resolve()),
                reliability_checkpoint_sha256=sha256(args.reliability_checkpoint),
                reliability_temperature=temperature, validation=metrics,
                residual_statistics=(
                    str(normalizer.source) if normalizer.source is not None else None
                ),
                residual_statistics_sha256=normalizer.sha256,
                best_validation_metric=best,
            )
    print(json.dumps({"checkpoint": str(output / "best.pt"), "steps": steps, "start_step": start_step, "best_flow_mse": best}, indent=2))


if __name__ == "__main__":
    main()
