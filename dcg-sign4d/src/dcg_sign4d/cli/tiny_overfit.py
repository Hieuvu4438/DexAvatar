"""Deterministic development-only Stage 2/3 tiny-overfit verification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.diffusion.trajectory_denoiser import PartAwareTrajectoryDenoiser
from dcg_sign4d.synthetic import make_observations, make_state
from dcg_sign4d.training.checkpoint import CheckpointMetadata, save_model_checkpoint
from dcg_sign4d.training.steps import contact_objective, diffusion_objective
from dcg_sign4d.utils.hashing import file_sha256


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--manifest", default="manifests/smoke.jsonl")
    parser.add_argument("--third-party-manifest", default="third_party/manifest.yaml")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-only", action="store_true")
    return parser.parse_args()


def _dependencies(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {item["name"]: item["commit"] for item in payload["repositories"]}


def _event_fixture(time_steps: int) -> torch.Tensor:
    if time_steps != 8:
        raise ValueError("tiny-overfit event fixture requires exactly eight frames")
    return torch.tensor(
        [
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [2, 1],
                [3, 2],
                [0, 3],
                [0, 0],
                [0, 0],
            ]
        ],
        dtype=torch.long,
    )


def _graph(labels: torch.Tensor, fps: float) -> ContactGraphBatch:
    probability = torch.nn.functional.one_hot(labels, 4).float()
    batch, time_steps, edges = labels.shape
    return ContactGraphBatch(
        event_state=labels,
        event_probability=probability,
        edge_valid=torch.ones(batch, edges, dtype=torch.bool, device=labels.device),
        uncertain_mask=torch.zeros_like(labels, dtype=torch.bool),
        segment_id=torch.zeros_like(labels),
        segment_duration=torch.full_like(labels, 1 / fps, dtype=torch.float),
    ).validate()


def _model_class(model: nn.Module) -> str:
    return f"{type(model).__module__}.{type(model).__qualname__}"


def main() -> None:
    args = _arguments()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not args.development_only
        or not config["experiment"].get("development_only")
        or config_path.name != "smoke.yaml"
    ):
        raise PermissionError("tiny overfit requires configs/smoke.yaml and --development-only")
    if not 0 < args.learning_rate <= 0.1:
        raise ValueError("development learning rate must lie in (0,0.1]")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable tiny-overfit output exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".tiny_overfit_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    start = time.time()
    seed = int(config["experiment"]["seed"])
    steps = int(config["diffusion"]["train_steps"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    state = make_state(time=8)
    observations = make_observations(time=8)
    state = state.__class__(
        **{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name in state.__dataclass_fields__
            if (value := getattr(state, name)) is not None
        }
    )
    observations = observations.__class__(
        **{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name in observations.__dataclass_fields__
            if (value := getattr(observations, name)) is not None
        }
    )
    encoded, context = StateCodec().encode(state)
    labels = _event_fixture(8).to(device)
    durations = torch.ones_like(labels)
    uncertain = torch.zeros_like(labels, dtype=torch.bool)
    edge_valid = torch.ones(1, 2, dtype=torch.bool, device=device)
    geometry_features = torch.randn(1, 8, 2, 5, device=device)
    class_counts = torch.bincount(labels.flatten(), minlength=4)

    contact = ContactProposal(
        encoded.shape[-1],
        edge_count=2,
        max_duration=8,
        hidden_dim=16,
        heads=4,
        layers=1,
    ).to(device)
    contact_optimizer = torch.optim.AdamW(contact.parameters(), lr=args.learning_rate)

    def contact_loss() -> torch.Tensor:
        prediction = contact(observations, state, geometry_features)
        return contact_objective(
            prediction,
            event_state=labels,
            duration_frames=durations,
            edge_valid=edge_valid,
            frame_valid=state.valid_mask,
            uncertain=uncertain,
            class_counts=class_counts,
        ).total

    contact_losses = [float(contact_loss().detach())]
    for _ in range(steps):
        contact_optimizer.zero_grad(set_to_none=True)
        loss = contact_loss()
        loss.backward()
        contact_optimizer.step()
        contact_losses.append(float(contact_loss().detach()))

    part_dims = (
        context.widths[0] + context.widths[1] + context.widths[2],
        context.widths[3],
        context.widths[4],
        context.widths[5],
    )
    denoiser = PartAwareTrajectoryDenoiser(part_dims, hidden_dim=16, heads=4, layers=1).to(device)
    token_encoder = ContactTokenEncoder(2, 16).to(device)
    diffusion_module = nn.ModuleDict({"denoiser": denoiser, "contact_token_encoder": token_encoder})
    diffusion_optimizer = torch.optim.AdamW(diffusion_module.parameters(), lr=args.learning_rate)
    schedule = DiffusionSchedule(steps)
    graph = _graph(labels, fps=30)
    fixed_noise = torch.randn(encoded.shape, device=device)
    fixed_timestep = torch.full((1,), steps - 1, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)

    def diffusion_loss() -> torch.Tensor:
        return diffusion_objective(
            denoiser,
            schedule,
            StateCodec(),
            token_encoder,
            state,
            graph,
            observations,
            conditioning_mode="dynamic",
            channel_weights=torch.ones(encoded.shape[-1], device=device),
            generator=generator,
            timesteps=fixed_timestep,
            noise=fixed_noise,
        ).total

    diffusion_losses = [float(diffusion_loss().detach())]
    for _ in range(steps):
        diffusion_optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss()
        loss.backward()
        diffusion_optimizer.step()
        diffusion_losses.append(float(diffusion_loss().detach()))

    config_hash = file_sha256(config_path)
    manifest_hash = file_sha256(args.manifest)
    dependencies = _dependencies(Path(args.third_party_manifest))
    common = {
        "step": steps,
        "epoch": 0,
        "seed": seed,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "dependency_commits": dependencies,
        "development_only": True,
    }
    save_model_checkpoint(
        output / "contact_checkpoint",
        contact,
        CheckpointMetadata(
            stage="contact_proposal",
            model_class=_model_class(contact),
            metrics={"tiny_overfit_loss": contact_losses[-1]},
            **common,
        ),
    )
    save_model_checkpoint(
        output / "diffusion_checkpoint",
        diffusion_module,
        CheckpointMetadata(
            stage="trajectory_diffusion",
            model_class=_model_class(diffusion_module),
            metrics={"tiny_overfit_loss": diffusion_losses[-1]},
            **common,
        ),
    )
    contact_pass = contact_losses[-1] < contact_losses[0]
    diffusion_pass = diffusion_losses[-1] < diffusion_losses[0]
    report = {
        "schema_version": "dcg_tiny_overfit_v1",
        "scientific_status": "DEVELOPMENT_WIRING_ONLY_NOT_A_MODEL_RESULT",
        "development_only": True,
        "seed": seed,
        "steps": steps,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "cpu"
        ),
        "elapsed_seconds": time.time() - start,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "contact": {
            "initial_loss": contact_losses[0],
            "final_loss": contact_losses[-1],
            "losses": contact_losses,
            "pass": contact_pass,
        },
        "diffusion": {
            "initial_loss": diffusion_losses[0],
            "final_loss": diffusion_losses[-1],
            "losses": diffusion_losses,
            "pass": diffusion_pass,
        },
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "dependency_commits": dependencies,
    }
    (output / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(incomplete, output / "TINY_OVERFIT_COMPLETE")
    print(json.dumps(report, sort_keys=True, indent=2))
    if not contact_pass or not diffusion_pass:
        raise RuntimeError("tiny-overfit loss did not decrease")


if __name__ == "__main__":
    main()
