"""Select region abstention thresholds on held-out external calibration data."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from phase2_refiner.data.dataset import SequenceCacheDataset, collate_sequences
from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.provenance import sha256_file
from phase2_refiner.train import _to_device, set_seed

from .leakage import audit_protocol
from .model import model_from_config


REGIONS = {"ubody": (0, 21), "lhand": (21, 36), "rhand": (36, 51)}


def _load(config: dict, checkpoint: Path, device: torch.device):
    model = model_from_config(config, initialize=False).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("ema_model") or payload["model"], strict=True)
    return model.eval()


@torch.no_grad()
def calibrate(args: argparse.Namespace) -> dict:
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    data = config["data"]
    lineage = audit_protocol(
        data["train_glob"], data["val_glob"], data["calibration_glob"]
    )
    set_seed(int(config.get("seed", 42)))
    device = torch.device(args.device)
    model = _load(config, args.checkpoint.resolve(), device)
    dataset = SequenceCacheDataset(
        data["calibration_glob"],
        max_frames=int(config["model"].get("max_frames", 64)),
        training=False,
        input_dim=45,
        reprojection_residual_scale=float(data.get("reprojection_residual_scale", 10.0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_sequences,
    )
    candidates = np.linspace(0.05, 0.95, 19)
    totals = {
        name: {
            float(threshold): {"baseline": 0.0, "candidate": 0.0, "count": 0}
            for threshold in candidates
        }
        for name in REGIONS
    }
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if device.type == "cuda"
        else nullcontext
    )
    for batch in loader:
        batch = _to_device(batch, device)
        with autocast():
            prediction = model(
                batch["features"],
                batch["initial_matrix"],
                batch["frame_valid"],
                batch["refine_mask"],
                batch["initial_joint_position"],
            )
        initial_error = geodesic_distance(
            batch["initial_matrix"].float(), batch["target_matrix"].float()
        )
        candidate_error = geodesic_distance(
            prediction["matrix"].float(), batch["target_matrix"].float()
        )
        probability = prediction["benefit_logit"].sigmoid().float()
        for group_index, (name, (start, end)) in enumerate(REGIONS.items()):
            joint_valid = (
                batch["frame_valid"][:, :, None]
                & batch["target_rotation_valid"][:, :, start:end]
                & batch["refine_mask"][:, None, start:end]
            )
            frame_denominator = joint_valid.sum(dim=-1)
            valid_frame = frame_denominator > 0
            baseline_frame = (
                initial_error[:, :, start:end] * joint_valid
            ).sum(dim=-1) / frame_denominator.clamp_min(1)
            refined_frame = (
                candidate_error[:, :, start:end] * joint_valid
            ).sum(dim=-1) / frame_denominator.clamp_min(1)
            for threshold in candidates:
                selected = probability[..., group_index] >= float(threshold)
                error = torch.where(selected, refined_frame, baseline_frame)
                clip_count = valid_frame.sum(dim=1)
                eligible = clip_count > 0
                if not eligible.any():
                    continue
                baseline_clip = (baseline_frame * valid_frame).sum(dim=1) / clip_count.clamp_min(1)
                error_clip = (error * valid_frame).sum(dim=1) / clip_count.clamp_min(1)
                cell = totals[name][float(threshold)]
                cell["baseline"] += float(baseline_clip[eligible].sum())
                cell["candidate"] += float(error_clip[eligible].sum())
                cell["count"] += int(eligible.sum())

    selections = {}
    for name, by_threshold in totals.items():
        rows = []
        for threshold, cell in by_threshold.items():
            baseline = cell["baseline"] / cell["count"]
            candidate = cell["candidate"] / cell["count"]
            rows.append(
                {
                    "threshold": threshold,
                    "baseline_radians": baseline,
                    "candidate_radians": candidate,
                    "candidate_over_baseline": candidate / baseline,
                    "clips": cell["count"],
                }
            )
        # Conservative tie break: abstain more when scores are numerically equal.
        selected = min(rows, key=lambda row: (row["candidate_over_baseline"], -row["threshold"]))
        selections[name] = {"selected": selected, "grid": rows}
    non_inferior = {
        name: values["selected"]["candidate_over_baseline"] <= 1.0
        for name, values in selections.items()
    }
    return {
        "schema_version": 1,
        "decision": "PASS" if all(non_inferior.values()) else "FAIL",
        "decision_rule": "selected candidate_over_baseline <= 1.0 in every region",
        "non_inferior": non_inferior,
        "selection_data": "How2Sign calibration only",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "calibration_manifest": str(Path(data["calibration_glob"]).resolve()),
        "calibration_manifest_sha256": sha256_file(data["calibration_glob"]),
        "thresholds": {
            name: values["selected"]["threshold"] for name, values in selections.items()
        },
        "regions": selections,
        "lineage": lineage,
        "sgnify_training_or_selection_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = calibrate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
