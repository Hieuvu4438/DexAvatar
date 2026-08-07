"""Stage R2 relation/contact pretraining."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from phase3_posterior.config import load_config
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.losses.relation import relation_losses
from phase3_posterior.models.relation_graph import RelationGraphEncoder
from phase3_posterior.training import (
    ExponentialMovingAverage,
    cosine_warmup_scheduler,
    prepare_run,
    rng_state,
    save_checkpoint,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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
        identity_target=bool(config["data"].get("identity_target", False)),
    )
    loader_kwargs = dict(
        batch_size=int(config["training"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config["training"].get("workers", 0)),
        collate_fn=collate_phase3,
    )
    loader = DataLoader(dataset, **loader_kwargs)
    generic_indices = [
        index
        for index, entry in enumerate(dataset.entries)
        if entry.source in {"arctic", "interhand26m"}
    ]
    generic_steps = int(config["training"].get("generic_warmup_steps", 0))
    if generic_steps and not generic_indices:
        raise RuntimeError("R2 generic warm-up requested but no generic clips exist")
    generic_loader = (
        DataLoader(Subset(dataset, generic_indices), **loader_kwargs)
        if generic_steps
        else loader
    )
    model = RelationGraphEncoder(
        int(config["model"].get("relation_width", 128)),
        int(config["model"].get("relation_layers", 3)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.05)),
    )
    max_steps = int(config["training"]["max_steps"])
    scheduler = cosine_warmup_scheduler(optimizer, max_steps)
    ema = ExponentialMovingAverage(model, float(config["training"].get("ema", 0.9999)))
    active_loader = generic_loader if generic_steps else loader
    iterator = iter(active_loader)
    curriculum_stage = "generic_warmup" if generic_steps else "joint_adaptation"
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))

    def checkpoint_payload(step: int) -> dict:
        return {
            "model": model.state_dict(),
            "ema_model": ema.state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "rng_state": rng_state(),
            "curriculum": curriculum_stage,
        }

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(1, max_steps * accumulation + 1):
        step = (micro_step + accumulation - 1) // accumulation
        requested_stage = (
            "generic_warmup" if step <= generic_steps else "joint_adaptation"
        )
        if requested_stage != curriculum_stage:
            curriculum_stage = requested_stage
            active_loader = loader
            iterator = iter(active_loader)
            print(
                json.dumps(
                    {
                        "event": "relation_curriculum",
                        "stage": curriculum_stage,
                        "step": step,
                    }
                ),
                flush=True,
            )
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(active_loader)
            batch = next(iterator)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if not batch["edge_valid"].any():
            raise RuntimeError(
                "No relation sidecar supervision is present; run build_relation_targets"
            )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            result = model(
                batch["edge_features"], batch["edge_index"], batch["edge_valid"]
            )
        losses = relation_losses(result, batch)
        loss = losses["contact"] + 0.4 * losses["depth"] + 0.4 * losses["persistence"]
        (loss / accumulation).backward()
        if micro_step % accumulation:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % int(config["training"].get("log_interval", 50)) == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "curriculum": curriculum_stage,
                        "loss": float(loss.detach()),
                        **{key: float(value.detach()) for key, value in losses.items()},
                    }
                ),
                flush=True,
            )
        if step % checkpoint_interval == 0:
            save_checkpoint(output / "last.pt", checkpoint_payload(step))
    payload = checkpoint_payload(max_steps)
    save_checkpoint(output / "last.pt", payload)
    save_checkpoint(output / "best.pt", payload)


if __name__ == "__main__":
    main()
