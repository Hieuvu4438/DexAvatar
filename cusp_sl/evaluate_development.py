"""Measure residual-candidate headroom on target-bearing development caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config
from cusp_sl.gate_artifact import load_gate_thresholds
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


def clustered_delta_interval(
    records: list[dict], method_key: str, *, replicates: int, seed: int,
    weight_key: str = "tokens",
    base_key: str = "base_degrees",
) -> dict[str, float | int]:
    """Paired cluster bootstrap of token-weighted method-minus-base error."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["source_group"]), []).append(record)
    if len(grouped) < 2:
        raise ValueError("cluster bootstrap requires at least two source groups")
    numerator, denominator = [], []
    for group_records in grouped.values():
        weights = np.asarray(
            [record[weight_key] for record in group_records], dtype=np.float64
        )
        deltas = np.asarray(
            [record[method_key] - record[base_key] for record in group_records],
            dtype=np.float64,
        )
        numerator.append(float(np.sum(weights * deltas)))
        denominator.append(float(np.sum(weights)))
    numerator_array = np.asarray(numerator)
    denominator_array = np.asarray(denominator)
    observed = float(numerator_array.sum() / denominator_array.sum())
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(grouped), size=(replicates, len(grouped)), endpoint=False
    )
    sampled = numerator_array[indices].sum(axis=1) / denominator_array[indices].sum(axis=1)
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return {
        "clusters": len(grouped),
        "replicates": replicates,
        "delta_degrees": observed,
        "ci95_low_degrees": float(lower),
        "ci95_high_degrees": float(upper),
        "bootstrap_probability_improvement": float(np.mean(sampled < 0.0)),
    }


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
    parser.add_argument("--gate-calibration", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    args.output.mkdir(parents=True)
    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    q, temperature, flow, normalizer = load_models(
        config, args.reliability_checkpoint, args.flow_checkpoint, device,
        args.config, args.generator_kind,
    )
    tau_low = config.reliability.tau_low
    tau_high = config.reliability.tau_high
    gate_calibration_sha256 = None
    if args.gate_calibration is not None:
        tau_low, tau_high, _ = load_gate_thresholds(
            args.gate_calibration,
            config_path=args.config,
            reliability_checkpoint=args.reliability_checkpoint,
            generator_checkpoint=args.flow_checkpoint,
        )
        gate_calibration_sha256 = sha256(args.gate_calibration)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for entry in manifest["clips"]:
        path = Path(entry) if Path(entry).is_absolute() else args.manifest.parent / entry
        clip = load_cache_clip(path)
        metadata = json.loads(clip.metadata_json)
        source_group = str(metadata.get("source_group", ""))
        if not source_group:
            raise ValueError(f"Development clip lacks source_group: {path}")
        if clip.target_axis_angle is None:
            raise ValueError(f"Development clip lacks target: {path}")
        features, base = features_from_clip(
            clip, input_dim=config.data.input_dim, physical_time_motion=True
        )
        features, base = features[None].to(device), base[None].to(device)
        with torch.no_grad():
            probability = torch.sigmoid(q(features) / temperature)
        gate = gate_from_reliability(
            probability, tau_low, tau_high, config.reliability.dilation,
        )
        refine = torch.as_tensor(clip.refine_mask, device=device)[None, None]
        refine = refine.expand(1, base.shape[1], -1).float()
        gate = gate * refine
        condition = torch.cat((features, probability[..., None]), dim=-1)
        maximum = joint_max_angles(
            device, base.dtype, config.flow.body_max_degrees,
            config.flow.hand_max_degrees,
        )
        candidates = [base[0]]
        ungated_candidates = [base[0]]
        candidate_count = (
            1 if args.generator_kind == "deterministic"
            else config.flow.candidates
        )
        with torch.no_grad():
            for index in range(candidate_count):
                if args.generator_kind == "deterministic":
                    normalized = deterministic_residual(
                        flow, condition, torch.ones(
                            (1, base.shape[1]), device=device, dtype=torch.bool
                        )
                    )
                    residual = normalizer.denormalize(normalized)
                    rotation = compose_right(
                        base, residual, gate=gate, max_angle=maximum
                    )
                else:
                    generator = torch.Generator(device=device).manual_seed(
                        candidate_seed(config.training.seed, clip.clip_id, index)
                    )
                    residual, rotation = sample_velocity_blend(
                        flow, condition, base, gate,
                        steps=config.flow.ode_steps,
                        window=config.data.window_size,
                        overlap=config.flow.overlap,
                        generator=generator, normalizer=normalizer,
                    )
                candidates.append(rotation[0])
                ungated_candidates.append(
                    compose_right(
                        base, residual, gate=refine, max_angle=maximum
                    )[0]
                )
        rotations = torch.stack(candidates)
        ungated_rotations = torch.stack(ungated_candidates)
        target = axis_angle_to_matrix(
            torch.as_tensor(clip.target_axis_angle, device=device)
        )
        valid = torch.as_tensor(clip.target_rotation_valid, device=device)
        valid &= torch.as_tensor(clip.refine_mask, device=device)[None]
        errors = geodesic_distance(rotations, target[None])
        means = (errors * valid[None]).sum(dim=(1, 2)) / valid.sum().clamp_min(1)
        ungated_errors = geodesic_distance(ungated_rotations, target[None])
        ungated_means = (
            (ungated_errors * valid[None]).sum(dim=(1, 2))
            / valid.sum().clamp_min(1)
        )
        generated_best = means[1:].min()
        all_best = means.min()
        ungated_generated_best = ungated_means[1:].min()
        ungated_all_best = ungated_means.min()
        joint_index = torch.arange(51, device=device)[None]
        body_valid = valid & (joint_index < 21)
        hand_valid = valid & (joint_index >= 21)

        def group_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return (value * mask[None]).sum(dim=(1, 2)) / mask.sum().clamp_min(1)

        body_means = group_mean(errors, body_valid)
        hand_means = group_mean(errors, hand_valid)
        ungated_body_means = group_mean(ungated_errors, body_valid)
        ungated_hand_means = group_mean(ungated_errors, hand_valid)
        selected_edit = geodesic_distance(rotations[1], base[0])
        records.append({
            "clip_id": clip.clip_id, "source_group": source_group,
            "tokens": int(valid.sum()),
            "body_tokens": int(body_valid.sum()),
            "hand_tokens": int(hand_valid.sum()),
            "base_degrees": float(torch.rad2deg(means[0])),
            "k1_degrees": float(torch.rad2deg(means[1])),
            "generated_oracle_degrees": float(torch.rad2deg(generated_best)),
            "base_plus_generated_oracle_degrees": float(torch.rad2deg(all_best)),
            "generated_beats_base": bool(generated_best < means[0]),
            "ungated_k1_degrees": float(torch.rad2deg(ungated_means[1])),
            "ungated_generated_oracle_degrees": float(torch.rad2deg(ungated_generated_best)),
            "ungated_base_plus_generated_oracle_degrees": float(torch.rad2deg(ungated_all_best)),
            "ungated_generated_beats_base": bool(ungated_generated_best < ungated_means[0]),
            "body_base_degrees": float(torch.rad2deg(body_means[0])),
            "body_k1_degrees": float(torch.rad2deg(body_means[1])),
            "hand_base_degrees": float(torch.rad2deg(hand_means[0])),
            "hand_k1_degrees": float(torch.rad2deg(hand_means[1])),
            "ungated_body_k1_degrees": float(
                torch.rad2deg(ungated_body_means[1])
            ),
            "ungated_hand_k1_degrees": float(
                torch.rad2deg(ungated_hand_means[1])
            ),
            "k1_edit_degrees": float(
                torch.rad2deg(selected_edit[valid]).mean()
            ),
            "gate_mean": float(gate.mean()),
        })
        print(f"[development] {clip.clip_id}: base={records[-1]['base_degrees']:.3f}, oracle={records[-1]['generated_oracle_degrees']:.3f}")
    with (args.output / "per_clip.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    weights = np.asarray([record["tokens"] for record in records], dtype=np.float64)
    def weighted(name: str) -> float:
        return float(np.average([record[name] for record in records], weights=weights))
    body_weights = np.asarray(
        [record["body_tokens"] for record in records], dtype=np.float64
    )
    hand_weights = np.asarray(
        [record["hand_tokens"] for record in records], dtype=np.float64
    )

    def group_weighted(name: str, group_weights: np.ndarray) -> float:
        return float(np.average(
            [record[name] for record in records], weights=group_weights
        ))
    summary = {
        "role": "development_only",
        "generator_kind": args.generator_kind,
        "candidate_seed_policy": (
            "sha256(global_seed:clip_id:candidate_index)_mod_2^63-1"
        ),
        "generated_candidates": candidate_count,
        "clips": len(records), "tokens": int(weights.sum()),
        "base_degrees": weighted("base_degrees"),
        "k1_degrees": weighted("k1_degrees"),
        "generated_oracle_degrees": weighted("generated_oracle_degrees"),
        "base_plus_generated_oracle_degrees": weighted("base_plus_generated_oracle_degrees"),
        "generated_beats_base_clip_fraction": float(np.mean([record["generated_beats_base"] for record in records])),
        "ungated_k1_degrees": weighted("ungated_k1_degrees"),
        "ungated_generated_oracle_degrees": weighted("ungated_generated_oracle_degrees"),
        "ungated_base_plus_generated_oracle_degrees": weighted("ungated_base_plus_generated_oracle_degrees"),
        "ungated_generated_beats_base_clip_fraction": float(np.mean([record["ungated_generated_beats_base"] for record in records])),
        "body_base_degrees": group_weighted("body_base_degrees", body_weights),
        "body_k1_degrees": group_weighted("body_k1_degrees", body_weights),
        "hand_base_degrees": group_weighted("hand_base_degrees", hand_weights),
        "hand_k1_degrees": group_weighted("hand_k1_degrees", hand_weights),
        "ungated_body_k1_degrees": group_weighted(
            "ungated_body_k1_degrees", body_weights
        ),
        "ungated_hand_k1_degrees": group_weighted(
            "ungated_hand_k1_degrees", hand_weights
        ),
        "k1_edit_degrees": weighted("k1_edit_degrees"),
        "gate_mean": weighted("gate_mean"),
        "manifest_sha256": sha256(args.manifest),
        "reliability_checkpoint_sha256": sha256(args.reliability_checkpoint),
        "flow_checkpoint_sha256": sha256(args.flow_checkpoint),
        "gate_threshold_source": (
            "development_artifact" if args.gate_calibration else "config_default"
        ),
        "gate_calibration_sha256": gate_calibration_sha256,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "clustered_paired_deltas": {
            "k1_minus_base": clustered_delta_interval(
                records, "k1_degrees",
                replicates=config.protocol.bootstrap_replicates,
                seed=config.training.seed + 4101,
            ),
            "generated_oracle_minus_base": clustered_delta_interval(
                records, "generated_oracle_degrees",
                replicates=config.protocol.bootstrap_replicates,
                seed=config.training.seed + 4102,
            ),
            "ungated_k1_minus_base": clustered_delta_interval(
                records, "ungated_k1_degrees",
                replicates=config.protocol.bootstrap_replicates,
                seed=config.training.seed + 4103,
            ),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
