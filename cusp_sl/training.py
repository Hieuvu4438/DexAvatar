"""Shared deterministic training utilities and checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cusp_sl.config import CUSPConfig
from phase2_refiner.data.dataset import SequenceCacheDataset, collate_sequences


def autocast_context(config: CUSPConfig, device: torch.device):
    if device.type != "cuda" or config.training.precision == "fp32":
        return nullcontext()
    if config.training.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported training precision without a GradScaler: {config.training.precision}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(
    config: CUSPConfig, split: str, *, shuffle: bool, batch_size: int | None = None,
    drop_last: bool | None = None, pin_memory: bool | None = None,
) -> DataLoader:
    manifest = config.data.train_manifest if split == "train" else config.data.val_manifest
    dataset = SequenceCacheDataset(
        manifest,
        max_frames=config.data.window_size,
        training=shuffle,
        seed=config.training.seed,
        input_dim=config.data.input_dim,
        physical_time_motion=True,
        require_phase2r_semantics=config.data.require_phase2r_semantics,
    )
    generator = torch.Generator().manual_seed(config.training.seed + (0 if split == "train" else 1))
    return DataLoader(
        dataset,
        batch_size=batch_size or config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.workers,
        collate_fn=collate_sequences,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        generator=generator,
        drop_last=shuffle if drop_last is None else drop_last,
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def config_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    step: int, config_path: str | Path, **extra,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256(config_path),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **extra,
    }
    torch.save(payload, temporary)
    os.replace(temporary, output)


def append_jsonl(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def binary_calibration_metrics(
    probability: np.ndarray, labels: np.ndarray, bins: int = 15
) -> dict:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    brier = float(np.mean((probability - labels) ** 2))
    if len(probability) != len(labels) or len(labels) == 0:
        raise ValueError("probability and labels must have equal non-zero length")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    predicted = probability >= 0.5
    true_positive_rate = float(predicted[labels == 1].mean()) if positives else float("nan")
    true_negative_rate = float((~predicted[labels == 0]).mean()) if negatives else float("nan")

    # Mann-Whitney AUROC with average ranks for tied probabilities.
    auroc = float("nan")
    if positives and negatives:
        ascending = np.argsort(probability, kind="mergesort")
        sorted_probability = probability[ascending]
        ranks = np.empty(len(labels), dtype=np.float64)
        start = 0
        while start < len(labels):
            stop = start + 1
            while stop < len(labels) and sorted_probability[stop] == sorted_probability[start]:
                stop += 1
            ranks[ascending[start:stop]] = 0.5 * (start + stop - 1) + 1.0
            start = stop
        rank_sum = ranks[labels == 1].sum()
        auroc = float(
            (rank_sum - positives * (positives + 1) / 2.0)
            / (positives * negatives)
        )

    average_precision = float("nan")
    if positives:
        descending = np.argsort(-probability, kind="mergesort")
        sorted_labels = labels[descending]
        precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
        average_precision = float((precision * sorted_labels).sum() / positives)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    log_loss = float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    diagram = []
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if not selected.any():
            continue
        confidence = float(probability[selected].mean())
        accuracy = float(labels[selected].mean())
        fraction = float(selected.mean())
        ece += fraction * abs(confidence - accuracy)
        diagram.append({"low": float(edges[index]), "high": float(edges[index + 1]), "n": int(selected.sum()), "confidence": confidence, "accuracy": accuracy})
    order = np.argsort(-probability)
    risk_coverage = {}
    error = 1.0 - labels
    for coverage in (1.0, 0.9, 0.75, 0.5, 0.25):
        count = max(1, int(round(len(order) * coverage)))
        risk_coverage[str(coverage)] = float(error[order[:count]].mean())
    return {
        "observations": int(len(labels)),
        "positive_prevalence": float(labels.mean()),
        "brier": brier,
        "log_loss": log_loss,
        "ece_15": float(ece),
        "auroc": auroc,
        "average_precision": average_precision,
        "accuracy_at_0.5": float((predicted == labels.astype(bool)).mean()),
        "balanced_accuracy_at_0.5": float(
            0.5 * (true_positive_rate + true_negative_rate)
        ),
        "reliability_diagram": diagram,
        "selective_risk": risk_coverage,
    }
