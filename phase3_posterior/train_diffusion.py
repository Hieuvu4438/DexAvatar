"""Stages R3-R6 masked temporal-relational diffusion training."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from phase3_posterior.config import load_config
from phase3_posterior.data.corruptions import sample_conditioning_mask
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.losses.diffusion import SubVPSDE, region_balanced_score_loss
from phase3_posterior.losses.geometry import masked_geodesic_loss, target_motion_loss
from phase3_posterior.losses.relation import relation_losses
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.provenance import sha256_file
from phase3_posterior.training import (
    ExponentialMovingAverage,
    cosine_warmup_scheduler,
    load_weights,
    prepare_run,
    rng_state,
    save_checkpoint,
    seed_everything,
)


def _condition_masks(
    valid: torch.Tensor, seed: int, step: int, dropout: float
) -> torch.Tensor:
    masks = []
    for index, item in enumerate(valid):
        generator = torch.Generator().manual_seed(seed + step * 10_007 + index)
        mask = sample_conditioning_mask(item.cpu(), generator).conditioning
        if torch.rand((), generator=generator) < dropout:
            mask = torch.zeros_like(mask)
        masks.append(mask)
    return torch.stack(masks).to(valid.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--relation-init")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    output = prepare_run(config, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Phase3Dataset(
        config["data"]["train_index"],
        int(config["model"]["max_frames"]),
        training=True,
        seed=seed,
        input_dim=int(config["model"].get("observation_dim", 45)),
        identity_target=bool(config["data"].get("identity_target", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config["training"].get("workers", 0)),
        collate_fn=collate_phase3,
    )
    model = RelationalDiffusionPosterior(config["model"]).to(device)
    if args.init:
        load_weights(model, args.init)
    if args.relation_init:
        load_weights(model.relation_graph, args.relation_init, strict=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.05)),
    )
    max_steps = int(config["training"]["max_steps"])
    scheduler = cosine_warmup_scheduler(optimizer, max_steps)
    ema = ExponentialMovingAverage(model, float(config["training"].get("ema", 0.9999)))
    sde = SubVPSDE(
        **{key: config["diffusion"][key] for key in ("beta_min", "beta_max", "eps")}
    )
    iterator = iter(loader)
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(1, max_steps * accumulation + 1):
        step = (micro_step + accumulation - 1) // accumulation
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        valid = batch["target_rotation_valid"] & batch["frame_valid"][..., None]
        if not valid.any():
            raise RuntimeError("Training batch has no supervised rotation targets")
        time = (
            torch.rand(len(batch["target_state"]), device=device) * (1.0 - sde.eps)
            + sde.eps
        )
        noisy, noise, std = sde.perturb(batch["target_state"], time)
        condition = _condition_masks(
            batch["joint_valid"] & batch["frame_valid"][..., None],
            seed,
            step,
            float(config["training"].get("condition_dropout", 0.1)),
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            result = model(
                noisy,
                time,
                batch["features"],
                batch["frame_valid"],
                batch["edge_features"],
                batch["edge_index"],
                batch["edge_valid"],
                condition,
            )
        score_loss, regional = region_balanced_score_loss(
            result["score"], noise, std, valid, batch["target_weight"]
        )
        x0 = sde.x0_from_score(noisy, result["score"], time)
        rotation = masked_geodesic_loss(
            x0, batch["target_matrix"], valid, batch["target_weight"]
        )
        motion = target_motion_loss(
            x0, batch["target_state"], valid, batch["target_weight"]
        )
        relation = relation_losses(result, batch)
        weights = config.get("loss", {})
        loss = (
            score_loss
            + float(weights.get("rotation", 0.5)) * rotation
            + float(weights.get("motion", 0.25)) * motion
        )
        loss = (
            loss
            + float(weights.get("contact", 0.25)) * relation["contact"]
            + float(weights.get("persistence", 0.1)) * relation["persistence"]
            + float(weights.get("depth", 0.1)) * relation["depth"]
        )
        (loss / accumulation).backward()
        if micro_step % accumulation:
            continue
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"].get("gradient_clip", 1.0))
        )
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % int(config["training"].get("log_interval", 50)) == 0:
            metrics = {
                "step": step,
                "loss": float(loss.detach()),
                "score": float(score_loss.detach()),
                "rotation": float(rotation.detach()),
                "motion": float(motion.detach()),
            }
            metrics.update(
                {
                    f"score_{key}": float(value.detach())
                    for key, value in regional.items()
                }
            )
            print(json.dumps(metrics), flush=True)
    payload = {
        "model": model.state_dict(),
        "ema_model": ema.state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": max_steps,
        "config": config,
        "rng_state": rng_state(),
        "initialization": {
            "diffusion": {
                "path": args.init,
                "sha256": sha256_file(args.init) if args.init else None,
            },
            "relation": {
                "path": args.relation_init,
                "sha256": sha256_file(args.relation_init)
                if args.relation_init
                else None,
            },
        },
    }
    save_checkpoint(output / "last.pt", payload)
    save_checkpoint(output / "best.pt", payload)


if __name__ == "__main__":
    main()
