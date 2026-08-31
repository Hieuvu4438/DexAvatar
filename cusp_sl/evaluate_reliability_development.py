"""Audit frozen Q on natural frontend errors without training corruptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config
from cusp_sl.geometry import gate_from_reliability, geodesic_distance
from cusp_sl.models import ReliabilityCalibrator
from cusp_sl.training import (
    binary_calibration_metrics, config_sha256, make_loader, resolve_device,
    seed_everything, to_device,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reliability-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.reliability_checkpoint, map_location=device, weights_only=False
    )
    if checkpoint.get("config_sha256") != config_sha256(args.config):
        raise ValueError("Reliability checkpoint/config hash mismatch")
    model = ReliabilityCalibrator(
        config.data.input_dim, config.reliability.hidden_size,
        config.reliability.temporal_layers,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    temperature = float(checkpoint["temperature"])
    loader = make_loader(
        config, "val", shuffle=False, pin_memory=device.type == "cuda"
    )
    tolerances = torch.full(
        (51,), math.radians(config.reliability.hand_tolerance_degrees),
        device=device,
    )
    tolerances[:21] = math.radians(
        config.reliability.body_tolerance_degrees
    )
    probabilities, labels, errors, joint_groups, gates = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            probability = torch.sigmoid(model(batch["features"]) / temperature)
            error = geodesic_distance(
                batch["initial_matrix"], batch["target_matrix"]
            )
            label = error <= tolerances[None, None]
            valid = batch["frame_valid"][:, :, None].expand_as(
                batch["target_rotation_valid"]
            ).clone()
            valid &= batch["target_rotation_valid"]
            valid &= batch["refine_mask"][:, None]
            gate = gate_from_reliability(
                probability, config.reliability.tau_low,
                config.reliability.tau_high, config.reliability.dilation,
            )
            group = torch.arange(51, device=device) >= 21
            group = group[None, None].expand_as(valid)
            probabilities.append(probability[valid].float().cpu())
            labels.append(label[valid].float().cpu())
            errors.append(torch.rad2deg(error[valid]).float().cpu())
            joint_groups.append(group[valid].cpu())
            gates.append(gate[valid].float().cpu())
    probability = torch.cat(probabilities).numpy()
    label = torch.cat(labels).numpy()
    error = torch.cat(errors).numpy()
    hand = torch.cat(joint_groups).numpy().astype(bool)
    gate = torch.cat(gates).numpy()

    def summarize(selected: np.ndarray) -> dict:
        metrics = binary_calibration_metrics(probability[selected], label[selected])
        metrics.update({
            "mean_error_degrees": float(error[selected].mean()),
            "median_error_degrees": float(np.median(error[selected])),
            "p90_error_degrees": float(np.quantile(error[selected], 0.9)),
            "probability_mean": float(probability[selected].mean()),
            "gate_mean": float(gate[selected].mean()),
            "probability_error_pearson": float(
                np.corrcoef(probability[selected], error[selected])[0, 1]
            ),
        })
        return metrics

    report = {
        "role": "development_natural_frontend_reliability_audit",
        "corruption_applied": False,
        "groups": {
            "all_refinable": summarize(np.ones(len(label), dtype=bool)),
            "body_refinable": summarize(~hand),
            "hands": summarize(hand),
        },
        "body_tolerance_degrees": config.reliability.body_tolerance_degrees,
        "hand_tolerance_degrees": config.reliability.hand_tolerance_degrees,
        "tau_low": config.reliability.tau_low,
        "tau_high": config.reliability.tau_high,
        "config_sha256": sha256(args.config),
        "validation_manifest_sha256": sha256(Path(config.data.val_manifest)),
        "reliability_checkpoint_sha256": sha256(args.reliability_checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "temperature": temperature,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
