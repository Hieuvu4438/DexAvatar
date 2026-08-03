"""Shared deterministic training utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase3_posterior.provenance import atomic_json, require_new_output, run_provenance


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_run(config: dict[str, Any], config_path: str) -> Path:
    output = require_new_output(config["output_dir"])
    output.mkdir()
    atomic_json(output / "resolved_config.json", config)
    inputs = [config["data"]["train_index"]]
    if config["data"].get("val_index"):
        inputs.append(config["data"]["val_index"])
    atomic_json(
        output / "provenance.json",
        run_provenance(config_path, int(config.get("seed", 42)), inputs),
    )
    return output


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_weights(
    model: torch.nn.Module, path: str, strict: bool = False
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("ema_model", payload.get("model", payload))
    overlap = set(state) & set(model.state_dict())
    if not overlap:
        raise ValueError(
            f"Checkpoint has no tensors compatible with {type(model).__name__}"
        )
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return payload


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.state[key].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.state[key].copy_(value)


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    import math

    warmup = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
