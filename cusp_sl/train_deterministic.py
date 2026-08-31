"""Train the parameter-matched deterministic K=1 residual control."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from cusp_sl.config import load_config
from cusp_sl.models import SelectiveResidualFlow
from cusp_sl.normalization import ResidualNormalizer
from cusp_sl.train_flow import _load_q, prepare_batch, sha256
from cusp_sl.training import (
    append_jsonl, autocast_context, config_sha256, make_loader, resolve_device,
    save_checkpoint, seed_everything, to_device,
)


def deterministic_residual(model, condition, frame_valid):
    """Use the exact flow backbone at fixed zero state/time, without sampling."""
    state = torch.zeros(
        condition.shape[:-1] + (3,), device=condition.device,
        dtype=condition.dtype,
    )
    time = torch.zeros(condition.shape[0], device=condition.device)
    return model(state, time, condition, frame_valid)


def with_training_seed(config, seed: int):
    """Override only runtime stochasticity while retaining the frozen file hash."""
    if seed < 0:
        raise ValueError("Training seed must be non-negative")
    return replace(config, training=replace(config.training, seed=seed))


@torch.no_grad()
def validate(model, loader, config, q, temperature, normalizer, device, batches=20):
    model.eval()
    total, tokens = 0.0, 0.0
    torch.manual_seed(config.training.seed + 3001)
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = to_device(batch, device)
        condition, _, target, weight = prepare_batch(
            batch, config, q, temperature, normalizer
        )
        with autocast_context(config, device):
            prediction = deterministic_residual(
                model, condition, batch["frame_valid"]
            )
            squared = (prediction - target).square().sum(dim=-1)
        total += float((squared * weight).sum())
        tokens += float(weight.sum())
    model.train()
    return {
        "deterministic_residual_mse": total / max(tokens, 1.0),
        "weighted_tokens": tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reliability-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--seed", type=int,
        help="Independent restart seed; does not alter the frozen config hash",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    configured_seed = config.training.seed
    run_seed = configured_seed if args.seed is None else args.seed
    config = with_training_seed(config, run_seed)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    train = make_loader(config, "train", shuffle=True)
    val = make_loader(config, "val", shuffle=False)
    q, temperature = _load_q(
        config, args.reliability_checkpoint, device, args.config
    )
    normalizer = ResidualNormalizer.from_path(config.flow.normalization_statistics)
    model = SelectiveResidualFlow(
        config.data.input_dim + 1, config.flow.hidden_size, config.flow.layers,
        config.flow.heads, config.flow.mlp_ratio, config.flow.dropout,
        config.flow.body_max_degrees, config.flow.hand_max_degrees,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    output = args.output or (Path(config.output_dir) / "deterministic")
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    steps = args.steps or config.training.flow_steps
    start_step = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("config_sha256") != config_sha256(args.config):
            raise ValueError("Resume checkpoint/config hash mismatch")
        if checkpoint.get("model_kind") != "deterministic_residual":
            raise ValueError("Resume checkpoint is not a deterministic residual model")
        checkpoint_seed = int(checkpoint.get("training_seed", configured_seed))
        if checkpoint_seed != run_seed:
            raise ValueError(
                f"Resume checkpoint seed {checkpoint_seed} != requested {run_seed}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        best = float(checkpoint.get("best_validation_metric", float("inf")))
        if start_step > steps:
            raise ValueError(
                f"Resume step {start_step - 1} already reaches target {steps}"
            )
    iterator = iter(train)
    if not args.resume:
        best = float("inf")
    model.train()
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
        with autocast_context(config, device):
            prediction = deterministic_residual(
                model, condition, batch["frame_valid"]
            )
            squared = (prediction - target).square().sum(dim=-1)
            loss = (squared * weight).sum() / weight.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()
        if step % 25 == 0 or step == 1:
            append_jsonl(output / "train.jsonl", {
                "step": step, "loss": float(loss.detach()),
                "weighted_tokens": float(weight.sum()),
            })
        if step % config.training.validate_every == 0 or step == steps:
            metrics = validate(
                model, val, config, q, temperature, normalizer, device
            )
            metrics["step"] = step
            append_jsonl(output / "validation.jsonl", metrics)
            if metrics["deterministic_residual_mse"] < best:
                best = metrics["deterministic_residual_mse"]
                save_checkpoint(
                    output / "best.pt", model, optimizer, step, args.config,
                    model_kind="deterministic_residual",
                    reliability_checkpoint=str(args.reliability_checkpoint.resolve()),
                    reliability_checkpoint_sha256=sha256(
                        args.reliability_checkpoint
                    ),
                    reliability_temperature=temperature,
                    training_seed=run_seed,
                    residual_statistics=(
                        str(normalizer.source) if normalizer.source is not None else None
                    ),
                    residual_statistics_sha256=normalizer.sha256,
                    best_validation_metric=best,
                    validation=metrics,
                )
            save_checkpoint(
                output / "last.pt", model, optimizer, step, args.config,
                model_kind="deterministic_residual",
                reliability_checkpoint=str(args.reliability_checkpoint.resolve()),
                reliability_checkpoint_sha256=sha256(args.reliability_checkpoint),
                reliability_temperature=temperature,
                training_seed=run_seed,
                residual_statistics=(
                    str(normalizer.source) if normalizer.source is not None else None
                ),
                residual_statistics_sha256=normalizer.sha256,
                best_validation_metric=best,
                validation=metrics,
            )
    print(json.dumps({
        "checkpoint": str(output / "best.pt"), "steps": steps,
        "start_step": start_step, "best_deterministic_residual_mse": best,
        "training_seed": run_seed,
    }, indent=2))


if __name__ == "__main__":
    main()
