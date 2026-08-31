"""Fit Q gate thresholds on one development fold and audit a disjoint fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config
from cusp_sl.geometry import (
    axis_angle_to_matrix, compose_right, gate_from_reliability,
    geodesic_distance, joint_max_angles,
)
from cusp_sl.inference import candidate_seed, load_models, sample_velocity_blend
from cusp_sl.train_deterministic import deterministic_residual
from cusp_sl.training import resolve_device, seed_everything
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold(source_group: str) -> str:
    """Assign a complete source video to one deterministic development fold."""
    value = hashlib.sha256(source_group.encode("utf-8")).digest()[0]
    return "fit" if value < 128 else "audit"


@torch.no_grad()
def calibration_residual(
    model, condition: torch.Tensor, base: torch.Tensor, config, clip_id: str,
    device: torch.device, normalizer, generator_kind: str,
) -> torch.Tensor:
    """Match the generator execution path used by locked inference."""
    if generator_kind == "deterministic":
        normalized = deterministic_residual(
            model,
            condition,
            torch.ones(
                (condition.shape[0], condition.shape[1]),
                device=device,
                dtype=torch.bool,
            ),
        )
        return normalizer.denormalize(normalized)
    if generator_kind != "flow":
        raise ValueError(f"Unsupported generator kind: {generator_kind}")
    generator = torch.Generator(device=device).manual_seed(
        candidate_seed(config.training.seed, clip_id, 0)
    )
    residual, _ = sample_velocity_blend(
        model,
        condition,
        base,
        torch.ones(condition.shape[:-1], device=device),
        steps=config.flow.ode_steps,
        window=config.data.window_size,
        overlap=config.flow.overlap,
        generator=generator,
        normalizer=normalizer,
    )
    return residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reliability-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--generator-kind", choices=("flow", "deterministic"), default="flow"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-step", type=float, default=0.1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if not 0 < args.grid_step <= 0.5:
        raise ValueError("--grid-step must be in (0,0.5]")
    args.output.mkdir(parents=True)
    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    q, temperature, flow_model, normalizer = load_models(
        config, args.reliability_checkpoint, args.flow_checkpoint, device,
        args.config, args.generator_kind,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for entry in manifest["clips"]:
        path = Path(entry) if Path(entry).is_absolute() else args.manifest.parent / entry
        clip = load_cache_clip(path)
        if clip.target_axis_angle is None:
            raise ValueError(f"Development clip lacks target: {path}")
        if clip.target_rotation_valid is None:
            raise ValueError(f"Development clip lacks target-valid mask: {path}")
        metadata = json.loads(clip.metadata_json)
        source_group = str(metadata.get("source_group", ""))
        if not source_group:
            raise ValueError(f"Development clip lacks source_group: {path}")
        features, base = features_from_clip(
            clip, input_dim=config.data.input_dim, physical_time_motion=True
        )
        features, base = features[None].to(device), base[None].to(device)
        with torch.no_grad():
            probability = torch.sigmoid(q(features) / temperature)
        condition = torch.cat((features, probability[..., None]), dim=-1)
        raw_residual = calibration_residual(
            flow_model, condition, base, config, clip.clip_id, device,
            normalizer, args.generator_kind,
        )
        target = axis_angle_to_matrix(torch.as_tensor(clip.target_axis_angle, device=device))
        valid = torch.as_tensor(clip.target_rotation_valid, device=device)
        valid &= torch.as_tensor(clip.refine_mask, device=device)[None]
        records.append({
            "clip_id": clip.clip_id,
            "source_group": source_group,
            "fold": fold(source_group),
            "base": base,
            "residual": raw_residual,
            "probability": probability,
            "target": target,
            "valid": valid,
            "refine": torch.as_tensor(
                clip.refine_mask, device=device, dtype=probability.dtype
            )[None, None],
        })
        print(f"[gate-calibration] cached {clip.clip_id} ({records[-1]['fold']})")

    if {record["fold"] for record in records} != {"fit", "audit"}:
        raise ValueError("Hash split did not produce both fit and audit folds")
    maximum = joint_max_angles(
        device, records[0]["base"].dtype, config.flow.body_max_degrees,
        config.flow.hand_max_degrees,
    )

    def score(low: float, high: float, selected_fold: str) -> dict[str, float | int]:
        names = ("overall", "body", "hands")
        error_sum = {name: 0.0 for name in names}
        tokens = {name: 0 for name in names}
        gate_sum = {name: 0.0 for name in names}
        for record in records:
            if record["fold"] != selected_fold:
                continue
            gate = gate_from_reliability(
                record["probability"], low, high, config.reliability.dilation
            )
            gate = gate * record["refine"]
            rotation = compose_right(
                record["base"], record["residual"], gate=gate, max_angle=maximum
            )[0]
            error = geodesic_distance(rotation, record["target"])
            valid = record["valid"]
            joint_index = torch.arange(error.shape[1], device=device)[None]
            masks = {
                "overall": valid,
                "body": valid & (joint_index < 21),
                "hands": valid & (joint_index >= 21),
            }
            for name, mask in masks.items():
                error_sum[name] += float(error[mask].sum())
                gate_sum[name] += float(gate[0][mask].sum())
                tokens[name] += int(mask.sum())
        result: dict[str, float | int] = {}
        for name in names:
            if tokens[name] == 0:
                raise ValueError(f"Gate calibration has no {name} tokens")
            result[f"{name}_degrees"] = float(
                np.degrees(error_sum[name] / tokens[name])
            )
            result[f"{name}_gate_mean"] = gate_sum[name] / tokens[name]
            result[f"{name}_tokens"] = tokens[name]
        return result

    grid = np.arange(0.0, 1.0 + args.grid_step / 2, args.grid_step)
    candidates = []
    for low in grid:
        for high in grid:
            if low >= high:
                continue
            fit = score(float(low), float(high), "fit")
            candidates.append({
                "tau_low": float(low), "tau_high": float(high),
                "fit_degrees": fit["overall_degrees"],
                "fit_gate_mean": fit["overall_gate_mean"],
                "fit_tokens": fit["overall_tokens"],
                "fit_body_degrees": fit["body_degrees"],
                "fit_hands_degrees": fit["hands_degrees"],
            })
    candidates.sort(key=lambda row: (row["fit_degrees"], row["tau_high"] - row["tau_low"]))
    best = candidates[0]
    audit = score(
        best["tau_low"], best["tau_high"], "audit"
    )
    default_fit = score(
        config.reliability.tau_low, config.reliability.tau_high, "fit"
    )
    default_audit = score(
        config.reliability.tau_low, config.reliability.tau_high, "audit"
    )
    report = {
        "role": "development_gate_calibration_with_hash_disjoint_audit",
        "selection_control": (
            "deterministic_point_estimate"
            if args.generator_kind == "deterministic"
            else "fixed_seed_k1"
        ),
        "generator_kind": args.generator_kind,
        "selection_metric": "fit_overall_geodesic_degrees",
        "candidate_seed_policy": (
            "sha256(global_seed:clip_id:candidate_index)_mod_2^63-1"
        ),
        "split_unit": "source_group",
        "best_fit": best,
        "heldout_audit": audit,
        "config_default": {
            "tau_low": config.reliability.tau_low,
            "tau_high": config.reliability.tau_high,
            "fit": default_fit,
            "audit": default_audit,
        },
        "fold_counts": {
            name: sum(record["fold"] == name for record in records)
            for name in ("fit", "audit")
        },
        "source_group_counts": {
            name: len({
                record["source_group"]
                for record in records
                if record["fold"] == name
            })
            for name in ("fit", "audit")
        },
        "grid_step": args.grid_step,
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "reliability_checkpoint_sha256": sha256(args.reliability_checkpoint),
        "flow_checkpoint_sha256": sha256(args.flow_checkpoint),
        "top_grid_results": candidates[:10],
    }
    (args.output / "gate_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
