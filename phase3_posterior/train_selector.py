"""Stage R7 listwise candidate evidence selector training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase3_posterior.config import load_config
from phase3_posterior.losses.selector import listwise_ranking_loss
from phase3_posterior.models.evidence_selector import EvidenceSelector
from phase3_posterior.provenance import sha256_file
from phase3_posterior.training import (
    prepare_run,
    rng_state,
    save_checkpoint,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--posterior", required=True, help="Frozen posterior checkpoint provenance"
    )
    args = parser.parse_args()
    if not Path(args.posterior).is_file():
        raise FileNotFoundError(args.posterior)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    output = prepare_run(config, args.config)
    feature_path = Path(config["selector"]["candidate_features"])
    with np.load(feature_path, allow_pickle=False) as data:
        features = torch.from_numpy(data["evidence_features"]).float()
        errors = torch.from_numpy(data["target_errors"]).float()
    if features.ndim != 3 or errors.shape != features.shape[:2]:
        raise ValueError(
            "Selector data must contain evidence_features (N,K,F) and target_errors (N,K)"
        )
    if features.shape[-1] != int(config["selector"]["feature_dim"]):
        raise ValueError("Selector feature dimension does not match config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvidenceSelector(
        features.shape[-1], int(config["selector"].get("width", 128))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.05)),
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["training"].get("batch_size", 32))
    max_steps = int(config["training"]["max_steps"])
    for step in range(1, max_steps + 1):
        indices = torch.randint(len(features), (batch_size,), generator=generator)
        feature = features[indices].to(device)
        error = errors[indices].to(device)
        scores = model(feature)
        loss = listwise_ranking_loss(scores, error)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % int(config["training"].get("log_interval", 50)) == 0:
            accuracy = (scores.argmax(-1) == error.argmin(-1)).float().mean()
            print(
                json.dumps(
                    {"step": step, "loss": float(loss), "top1": float(accuracy)}
                ),
                flush=True,
            )
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": max_steps,
        "posterior": str(Path(args.posterior).resolve()),
        "posterior_sha256": sha256_file(args.posterior),
        "candidate_features_sha256": sha256_file(feature_path),
        "config": config,
        "rng_state": rng_state(),
    }
    save_checkpoint(output / "last.pt", payload)
    save_checkpoint(output / "best.pt", payload)


if __name__ == "__main__":
    main()
