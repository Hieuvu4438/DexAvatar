"""Locked CUSP-SL inference with one-time gating and velocity-blended windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config
from cusp_sl.evidence import candidate_evidence_terms
from cusp_sl.gate_artifact import load_gate_thresholds
from cusp_sl.geometry import (
    compose_right, gate_from_reliability, joint_max_angles,
    matrix_to_axis_angle,
)
from cusp_sl.models import ReliabilityCalibrator, SelectiveResidualFlow
from cusp_sl.normalization import ResidualNormalizer
from cusp_sl.train_deterministic import deterministic_residual
from cusp_sl.selection import EnergyStatistics, candidate_energy, select_candidates
from cusp_sl.training import config_sha256, resolve_device, seed_everything
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip
from phase2_refiner.render import create_smplx_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def starts(length: int, window: int, overlap: int) -> list[int]:
    if length < 1 or window < 1:
        raise ValueError("length and window must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must satisfy 0 <= overlap < window")
    if length <= window:
        return [0]
    stride = window - overlap
    # Match HandFlow's stride-grid contract. The final window is allowed to be
    # shorter (equivalent to a padded tail); anchoring it at length-window can
    # create an unintended near-full-window overlap.
    count = math.ceil((length - window) / stride) + 1
    return [index * stride for index in range(count)]


def center_weights(length: int, device, dtype) -> torch.Tensor:
    if length == 1:
        return torch.ones(1, device=device, dtype=dtype)
    coordinate = torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype)
    # HandFlow's released velocity-blending contract retains a non-zero 0.01
    # contribution at window boundaries. Keep this explicit so a future
    # refactor cannot silently turn per-step blending into endpoint averaging.
    return (1.0 - coordinate.abs()).clamp_min(0.01)


def generated_candidate_count(
    variant: str, generator_kind: str, configured_candidates: int
) -> int:
    if generator_kind == "deterministic" or variant == "a4_k1":
        return 1
    return configured_candidates


def variant_uses_energy(variant: str) -> bool:
    return variant in {
        "a7_geometry",
        "a7_hands_only",
        "a9_combined",
        "a10_always_on",
    }


def random_candidate_index(clip_id: str, seed: int, candidate_count: int) -> int:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    digest = hashlib.sha256(f"{seed}:{clip_id}".encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % candidate_count


def candidate_seed(seed: int, clip_id: str, candidate_index: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{clip_id}:{candidate_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@torch.no_grad()
def sample_velocity_blend(
    model, condition, base, gate, *, steps: int, window: int, overlap: int,
    generator: torch.Generator, normalizer: ResidualNormalizer | None = None,
):
    """Blend velocities at every Euler step, never post-hoc endpoints."""
    _, length, joints, _ = condition.shape
    state = torch.randn((1, length, joints, 3), device=condition.device, dtype=condition.dtype, generator=generator)
    locations = starts(length, window, overlap)
    dt = 1.0 / steps
    for step in range(steps):
        velocity_sum = torch.zeros_like(state)
        weight_sum = torch.zeros((1, length, 1, 1), device=state.device, dtype=state.dtype)
        for start in locations:
            stop = min(start + window, length)
            valid = torch.ones((1, stop - start), device=state.device, dtype=torch.bool)
            time = torch.full((1,), step * dt, device=state.device)
            velocity = model(state[:, start:stop], time, condition[:, start:stop], valid)
            # A short final segment represents the valid prefix of a padded
            # full window, so retain the full-window center-distance weights.
            weight = center_weights(window, state.device, state.dtype)[: stop - start]
            weight = weight[None, :, None, None]
            velocity_sum[:, start:stop] += velocity * weight
            weight_sum[:, start:stop] += weight
        state = state + dt * velocity_sum / weight_sum.clamp_min(1e-6)
    residual = (
        state if normalizer is None else normalizer.denormalize(state)
    )
    maximum = joint_max_angles(
        residual.device, residual.dtype,
        model.body_max_degrees, model.hand_max_degrees,
    )
    return residual, compose_right(base, residual, gate=gate, max_angle=maximum)


def load_models(
    config, q_path: Path, flow_path: Path, device,
    config_path: Path | None = None, generator_kind: str = "flow",
):
    q_checkpoint = torch.load(q_path, map_location=device, weights_only=False)
    active_config_hash = config_sha256(config_path) if config_path is not None else None
    if active_config_hash is not None and q_checkpoint.get("config_sha256") != active_config_hash:
        raise ValueError("Reliability checkpoint/config hash mismatch")
    q = ReliabilityCalibrator(config.data.input_dim, config.reliability.hidden_size, config.reliability.temporal_layers).to(device)
    q.load_state_dict(q_checkpoint["model"])
    q.eval()
    flow_checkpoint = torch.load(flow_path, map_location=device, weights_only=False)
    if active_config_hash is not None and flow_checkpoint.get("config_sha256") != active_config_hash:
        raise ValueError("Flow checkpoint/config hash mismatch")
    if flow_checkpoint.get("reliability_checkpoint_sha256") != sha256(q_path):
        raise ValueError("Generator checkpoint/reliability-checkpoint hash mismatch")
    model_kind = flow_checkpoint.get("model_kind", "flow")
    expected_kind = "deterministic_residual" if generator_kind == "deterministic" else "flow"
    if model_kind != expected_kind:
        raise ValueError(
            f"Generator checkpoint kind mismatch: checkpoint={model_kind}, "
            f"requested={expected_kind}"
        )
    flow = SelectiveResidualFlow(
        config.data.input_dim + 1, config.flow.hidden_size, config.flow.layers,
        config.flow.heads, config.flow.mlp_ratio, config.flow.dropout,
        config.flow.body_max_degrees, config.flow.hand_max_degrees,
    ).to(device)
    flow.load_state_dict(flow_checkpoint["model"])
    flow.eval()
    normalizer = ResidualNormalizer.from_path(config.flow.normalization_statistics)
    checkpoint_statistics = flow_checkpoint.get("residual_statistics_sha256")
    if checkpoint_statistics != normalizer.sha256:
        raise ValueError(
            "Flow checkpoint/residual-normalization hash mismatch: "
            f"checkpoint={checkpoint_statistics}, config={normalizer.sha256}"
        )
    return q, float(q_checkpoint["temperature"]), flow, normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--reliability-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--flow-checkpoint", "--generator-checkpoint", dest="flow_checkpoint",
        type=Path, required=True,
    )
    parser.add_argument(
        "--generator-kind", choices=("flow", "deterministic"), default="flow"
    )
    parser.add_argument("--energy-statistics", type=Path)
    parser.add_argument(
        "--gate-calibration",
        type=Path,
        help="Source-disjoint development artifact that freezes tau_low/tau_high",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--protocol-role", choices=("locked_evaluation", "development_validation"),
        default="locked_evaluation",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "a3_deterministic", "a4_k1", "a5_random", "a7_geometry",
            "a7_hands_only", "a9_combined", "a10_always_on",
        ),
        default="a7_geometry",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = resolve_device(args.device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
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
            generator_kind=args.generator_kind,
        )
        gate_calibration_sha256 = sha256(args.gate_calibration)
    if (args.variant == "a3_deterministic") != (args.generator_kind == "deterministic"):
        raise ValueError(
            "a3_deterministic requires --generator-kind deterministic, and "
            "deterministic checkpoints require --variant a3_deterministic"
        )
    uses_energy = variant_uses_energy(args.variant)
    smplx = None
    if uses_energy:
        smplx = create_smplx_model(config.protocol.smplx_model_folder, device)
        smplx.requires_grad_(False)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    entries = manifest["clips"]
    if args.energy_statistics:
        payload = np.load(args.energy_statistics)
        if (
            "schema_version" not in payload.files
            or str(payload["schema_version"].item())
            != "cusp_sl_energy_statistics_v2"
            or str(payload["role"].item())
            != "development_candidate_energy_normalization"
        ):
            raise ValueError("Unsupported energy-statistics artifact")
        expected_energy_binding = {
            "config_sha256": sha256(args.config),
            "reliability_checkpoint_sha256": sha256(
                args.reliability_checkpoint
            ),
            "flow_checkpoint_sha256": sha256(args.flow_checkpoint),
            "gate_calibration_sha256": gate_calibration_sha256 or "",
        }
        for field, expected in expected_energy_binding.items():
            if str(payload[field].item()) != expected:
                raise ValueError(f"Energy-statistics {field} mismatch")
        stats = EnergyStatistics(torch.from_numpy(payload["median"]).to(device), torch.from_numpy(payload["mad"]).to(device))
    else:
        if uses_energy and args.protocol_role == "locked_evaluation":
            raise ValueError(
                "Locked energy-based inference requires development-fitted "
                "--energy-statistics"
            )
        stats = EnergyStatistics(torch.zeros(4, device=device), torch.ones(4, device=device))
    weights = torch.tensor([
        config.selection.observation_weight, config.selection.motion_weight,
        config.selection.physical_weight,
        config.selection.form_weight if args.variant == "a9_combined" else 0.0,
    ], device=device)
    total_frames = 0
    summaries = []
    for entry in entries:
        clip_started = time.perf_counter()
        cache_path = args.input_manifest.parent / (entry["cache"] if isinstance(entry, dict) else entry)
        clip = load_cache_clip(cache_path)
        if (
            clip.target_axis_angle is not None
            or clip.target_joint_positions is not None
            or bool(np.any(clip.target_quality))
        ):
            raise ValueError(
                f"Inference refuses target-bearing cache: {cache_path}"
            )
        features, base = features_from_clip(clip, input_dim=config.data.input_dim, physical_time_motion=True)
        features, base = features[None].to(device), base[None].to(device)
        with torch.no_grad():
            probability = torch.sigmoid(q(features) / temperature)
        gate = gate_from_reliability(
            probability, tau_low, tau_high, config.reliability.dilation,
        )
        refine = torch.as_tensor(
            clip.refine_mask, device=device, dtype=gate.dtype
        )[None, None]
        gate = gate * refine
        if args.variant == "a10_always_on":
            gate = torch.ones_like(gate) * refine
        if args.variant == "a7_hands_only":
            # Explicit exploratory control motivated by How2Sign development:
            # its body pseudo-target has little 2D headroom relative to hands.
            gate = gate.clone()
            gate[..., :21] = 0.0
        condition = torch.cat((features, probability[..., None]), dim=-1)
        no_active_tokens = not bool(torch.any(gate > 0.0))
        candidate_count = generated_candidate_count(
            args.variant, args.generator_kind, config.flow.candidates
        )
        if no_active_tokens:
            rotations = base[0][None]
            residual_stack = torch.zeros(
                (1, *base.shape[1:3], 3), device=device, dtype=base.dtype
            )
            generated_count = 0
        else:
            generated = []
            residuals = []
            for candidate_index in range(candidate_count):
                if args.generator_kind == "deterministic":
                    with torch.no_grad():
                        normalized = deterministic_residual(
                            flow, condition, torch.ones(
                                (1, base.shape[1]),
                                device=device,
                                dtype=torch.bool,
                            )
                        )
                    residual = normalizer.denormalize(normalized)
                    maximum = joint_max_angles(
                        device, residual.dtype, config.flow.body_max_degrees,
                        config.flow.hand_max_degrees,
                    )
                    rotation = compose_right(
                        base, residual, gate=gate, max_angle=maximum
                    )
                else:
                    generator = torch.Generator(device=device).manual_seed(
                        candidate_seed(
                            config.training.seed, clip.clip_id, candidate_index
                        )
                    )
                    residual, rotation = sample_velocity_blend(
                        flow, condition, base, gate,
                        steps=config.flow.ode_steps,
                        window=config.data.window_size,
                        overlap=config.flow.overlap,
                        generator=generator,
                        normalizer=normalizer,
                    )
                residuals.append(residual[0])
                generated.append(rotation[0])
            rotations = torch.stack((base[0], *generated), dim=0)
            residual_stack = torch.stack(
                (torch.zeros_like(residuals[0]), *residuals), dim=0
            )
            generated_count = candidate_count
        valid = torch.isfinite(rotations).all(dim=(-1, -2, -3, -4))
        if uses_energy:
            terms = candidate_evidence_terms(
                smplx, rotations, clip, device,
                huber_delta=config.selection.huber_delta,
                rom_threshold_degrees=config.selection.rom_threshold_degrees,
            )
            energy = candidate_energy(terms, stats, weights)
        else:
            terms = torch.zeros(
                (rotations.shape[0], 4), device=device, dtype=rotations.dtype
            )
            energy = torch.full(
                (rotations.shape[0],), float("nan"),
                device=device, dtype=rotations.dtype,
            )
        if no_active_tokens:
            selected_index = torch.tensor(0, device=device)
            result = {"index": selected_index, "rotation": rotations[0], "weights": torch.ones(1, device=device), "disagreement": torch.zeros_like(gate[0]), "energy_margin": torch.tensor(float("nan"), device=device)}
        elif args.variant in {"a3_deterministic", "a4_k1"}:
            selected_index = torch.tensor(
                1 if bool(valid[1]) else 0, device=device
            )
            result = {"index": selected_index, "rotation": rotations[selected_index], "weights": torch.nn.functional.one_hot(selected_index, rotations.shape[0]).float(), "disagreement": torch.zeros_like(gate[0]), "energy_margin": torch.tensor(float("nan"), device=device)}
        elif args.variant == "a5_random":
            proposed_index = random_candidate_index(
                clip.clip_id, config.training.seed, candidate_count
            )
            selected_index = torch.tensor(
                proposed_index if bool(valid[proposed_index]) else 0,
                device=device,
            )
            result = {"index": selected_index, "rotation": rotations[selected_index], "weights": torch.nn.functional.one_hot(selected_index, rotations.shape[0]).float(), "disagreement": torch.zeros_like(gate[0]), "energy_margin": torch.tensor(float("nan"), device=device)}
        else:
            batched = select_candidates(rotations[None], energy[None], valid[None], config.selection.energy_temperature)
            result = {key: value[0] for key, value in batched.items()}
        output = args.output / "clips" / f"{clip.clip_id}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            clip_id=np.asarray(clip.clip_id), frame_names=clip.frame_names,
            selected_rotation=result["rotation"].cpu().numpy(),
            selected_axis_angle=matrix_to_axis_angle(result["rotation"]).cpu().numpy(),
            candidate_rotation=rotations.cpu().numpy(), candidate_residual=residual_stack.cpu().numpy(),
            reliability=probability[0].cpu().numpy(), gate=gate[0].cpu().numpy(),
            energy_terms=terms.cpu().numpy(), energy=energy.cpu().numpy(),
            candidate_valid=valid.cpu().numpy(), selected_index=result["index"].cpu().numpy(),
            energy_weights=result["weights"].cpu().numpy(), disagreement=result["disagreement"].cpu().numpy(),
            energy_margin=result["energy_margin"].cpu().numpy(),
        )
        total_frames += len(clip.frame_names)
        summaries.append({
            "clip_id": clip.clip_id,
            "frames": len(clip.frame_names),
            "selected_index": int(result["index"]),
            "edited_fraction": float((gate[0] > 0).float().mean()),
            "runtime_seconds": time.perf_counter() - clip_started,
            "prediction_sha256": sha256(output),
            "generated_candidates": generated_count,
            "no_active_tokens": no_active_tokens,
            "energy_computed": uses_energy,
        })
        print(f"[infer] {clip.clip_id}: {len(clip.frame_names)} frames, candidate={int(result['index'])}")
    if args.protocol_role == "locked_evaluation" and total_frames != config.protocol.expected_frames:
        raise ValueError(f"Inference coverage {total_frames} != {config.protocol.expected_frames}")
    runtime_seconds = time.perf_counter() - started
    report = {
        "variant": args.variant, "protocol_role": args.protocol_role,
        "input_manifest_role": manifest.get("role", "legacy_unspecified"),
        "generator_kind": args.generator_kind,
        "candidate_seed_policy": (
            "sha256(global_seed:clip_id:candidate_index)_mod_2^63-1"
        ),
        "frames": total_frames, "clips": len(summaries),
        "config_sha256": sha256(args.config), "input_manifest_sha256": sha256(args.input_manifest),
        "reliability_checkpoint_sha256": sha256(args.reliability_checkpoint),
        "flow_checkpoint_sha256": sha256(args.flow_checkpoint),
        "gate_threshold_source": (
            "development_artifact" if args.gate_calibration else "config_default"
        ),
        "gate_calibration_sha256": gate_calibration_sha256,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "runtime_seconds": runtime_seconds,
        "frames_per_second": total_frames / runtime_seconds,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "summaries": summaries,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "summaries"}, indent=2))


if __name__ == "__main__":
    main()
