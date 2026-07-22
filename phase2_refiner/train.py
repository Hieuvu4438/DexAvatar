"""Train the deterministic Phase 2 whole-sequence refiner."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from phase2_refiner.config import load_config
from phase2_refiner.data.corruptions import apply_burst_corruption
from phase2_refiner.data.dataset import SequenceCacheDataset, collate_sequences
from phase2_refiner.losses import RefinerLoss
from phase2_refiner.models import WholeSequenceRefiner


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict) -> WholeSequenceRefiner:
    return WholeSequenceRefiner(**config.get("model", {}))


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loss_fn, loader, device) -> float:
    model.eval()
    totals = []
    for batch in loader:
        batch = _to_device(batch, device)
        prediction = model(
            batch["features"],
            batch["initial_matrix"],
            batch["frame_valid"],
            batch["refine_mask"],
        )
        losses = loss_fn(
            prediction,
            batch["initial_matrix"],
            batch["target_matrix"],
            batch["frame_valid"],
            batch["refine_mask"],
            batch["features"][..., 18],
        )
        totals.append(float(losses["total"]))
    return float(np.mean(totals)) if totals else float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--identity-target", action="store_true", help="Smoke tests only"
    )
    parser.add_argument("--train-glob", help="Override data.train_glob")
    parser.add_argument("--val-glob", help="Override data.val_glob")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device)
    data_config = config.get("data", {})
    if args.train_glob:
        data_config["train_glob"] = args.train_glob
    if args.val_glob:
        data_config["val_glob"] = args.val_glob
    max_frames = int(config.get("model", {}).get("max_frames", 64))
    train_dataset = SequenceCacheDataset(
        data_config["train_glob"],
        max_frames=max_frames,
        training=True,
        identity_target=args.identity_target,
        seed=seed,
    )
    val_glob = None if args.no_validation else data_config.get("val_glob")
    val_dataset = (
        SequenceCacheDataset(
            val_glob,
            max_frames=max_frames,
            training=False,
            identity_target=args.identity_target,
            seed=seed,
        )
        if val_glob
        else None
    )
    train_config = config.get("training", {})
    batch_size = int(args.batch_size or train_config.get("batch_size", 8))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_config.get("workers", 0)),
        collate_fn=collate_sequences,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sequences,
        )
        if val_dataset
        else None
    )
    model = make_model(config).to(device)
    loss_fn = RefinerLoss(**config.get("loss", {})).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.05)),
    )
    max_steps = int(args.max_steps or train_config.get("max_steps", 100000))
    warmup = max(1, int(max_steps * float(train_config.get("warmup_fraction", 0.05))))
    scheduler = SequentialLR(
        optimizer,
        schedulers=(
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup)),
        ),
        milestones=(warmup,),
    )
    output = Path(
        args.output_dir or config.get("output_dir", "outputs/phase2_training")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    accumulation = int(train_config.get("gradient_accumulation", 1))
    clip_norm = float(train_config.get("gradient_clip", 1.0))
    log_every = int(train_config.get("log_every", 20))
    validate_every = int(train_config.get("validate_every", 500))
    corruption = config.get("corruption", {})
    best = float("inf")
    step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch in train_loader:
            model.train()
            batch = _to_device(batch, device)
            features, initial_matrix, _ = apply_burst_corruption(
                batch["features"],
                batch["initial_matrix"],
                batch["frame_valid"],
                **corruption,
            )
            prediction = model(
                features, initial_matrix, batch["frame_valid"], batch["refine_mask"]
            )
            losses = loss_fn(
                prediction,
                initial_matrix,
                batch["target_matrix"],
                batch["frame_valid"],
                batch["refine_mask"],
                features[..., 18],
            )
            (losses["total"] / accumulation).backward()
            micro_step += 1
            if micro_step % accumulation != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % log_every == 0 or step == 1:
                fields = " ".join(
                    f"{key}={float(value.detach()):.6f}"
                    for key, value in losses.items()
                )
                print(f"step={step} {fields}")
            if val_loader is not None and (
                step % validate_every == 0 or step == max_steps
            ):
                score = evaluate(model, loss_fn, val_loader, device)
                print(f"step={step} val_total={score:.6f}")
                if score < best:
                    best = score
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_config": config.get("model", {}),
                            "step": step,
                        },
                        output / "best.pt",
                    )
            if step >= max_steps:
                break
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": config.get("model", {}),
            "step": step,
        },
        output / "last.pt",
    )
    print(f"Training complete at step {step}; checkpoints: {output}")


if __name__ == "__main__":
    main()
