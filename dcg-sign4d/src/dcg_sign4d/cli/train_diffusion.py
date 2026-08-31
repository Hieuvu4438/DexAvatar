"""Proposal Stage 3 holistic official-DPoser-X trajectory diffusion trainer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from dcg_sign4d.diffusion.contact_encoder import ConditioningMode, ContactTokenEncoder
from dcg_sign4d.diffusion.dposer_bridge import OfficialDPoserXBridge
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.diffusion.trajectory_denoiser import DPoserXConditionedTrajectoryDenoiser
from dcg_sign4d.training.batch import SupervisedWindowBatch, load_supervised_windows
from dcg_sign4d.training.checkpoint import CheckpointMetadata, save_model_checkpoint
from dcg_sign4d.training.steps import diffusion_objective
from dcg_sign4d.utils.hashing import file_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(_StrictModel):
    seed: int
    development_only: bool = False


class DataConfig(_StrictModel):
    train_bundle: Path
    validation_bundle: Path


class DPoserConfig(_StrictModel):
    source_root: Path
    runtime_root: Path
    registry: Path
    expected_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class ModelConfig(_StrictModel):
    hidden_dim: int = Field(gt=0)
    heads: int = Field(gt=0)
    layers: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def compatible_heads(self) -> ModelConfig:
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        return self


class DiffusionConfig(_StrictModel):
    steps: int = Field(ge=2)
    beta_start: float = Field(gt=0, lt=1)
    beta_end: float = Field(gt=0, lt=1)
    conditioning_mode: ConditioningMode
    graph_dropout_probability: float = Field(ge=0, le=1)
    edge_dropout_probability: float = Field(ge=0, le=1)
    reliability_dropout_probability: float = Field(ge=0, le=1)
    root_weight: float = Field(gt=0)
    body_weight: float = Field(gt=0)
    left_hand_weight: float = Field(gt=0)
    right_hand_weight: float = Field(gt=0)
    face_weight: float = Field(gt=0)

    @model_validator(mode="after")
    def increasing_beta(self) -> DiffusionConfig:
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be smaller than beta_end")
        return self


class OptimizationConfig(_StrictModel):
    train_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0, le=0.1)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    validation_interval: int = Field(gt=0)


class DiffusionTrainingConfig(_StrictModel):
    experiment: ExperimentConfig
    data: DataConfig
    dposer_x: DPoserConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    optimization: OptimizationConfig
    third_party_manifest: Path


def _dependencies(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {item["name"]: item["commit"] for item in payload["repositories"]}


def _channel_weights(config: DiffusionConfig, reference: torch.Tensor) -> torch.Tensor:
    parts = DPoserXConditionedTrajectoryDenoiser.PRODUCTION_PART_DIMS
    values = (
        config.root_weight,
        config.body_weight,
        config.left_hand_weight,
        config.right_hand_weight,
        config.face_weight,
    )
    return torch.cat(
        [
            torch.full((width,), value, device=reference.device)
            for width, value in zip(parts, values, strict=True)
        ]
    )


def _without_reliability(batch: SupervisedWindowBatch) -> SupervisedWindowBatch:
    observations = replace(
        batch.observations,
        keypoint_reliability=torch.zeros_like(batch.observations.keypoint_reliability),
        mask_reliability=(
            torch.zeros_like(batch.observations.mask_reliability)
            if batch.observations.mask_reliability is not None
            else None
        ),
        track_reliability=(
            torch.zeros_like(batch.observations.track_reliability)
            if batch.observations.track_reliability is not None
            else None
        ),
        depth_reliability=(
            torch.zeros_like(batch.observations.depth_reliability)
            if batch.observations.depth_reliability is not None
            else None
        ),
    )
    return replace(batch, observations=observations).validate()


def _with_edge_dropout(
    batch: SupervisedWindowBatch,
    probability: float,
    generator: torch.Generator,
) -> SupervisedWindowBatch:
    if probability <= 0:
        return batch
    keep = (
        torch.rand(
            batch.graph.edge_valid.shape,
            generator=generator,
            device=batch.graph.edge_valid.device,
        )
        >= probability
    )
    graph = replace(batch.graph, edge_valid=batch.graph.edge_valid & keep)
    return replace(batch, graph=graph).validate()


def _objective(
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    token_encoder: nn.Module,
    batch: SupervisedWindowBatch,
    codec: StateCodec,
    config: DiffusionConfig,
    generator: torch.Generator,
    *,
    training: bool,
) -> torch.Tensor:
    mode = config.conditioning_mode
    working = batch
    random_device = batch.trajectory.root_rot6d.device
    if (
        training
        and torch.rand((), generator=generator, device=random_device).item()
        < config.graph_dropout_probability
    ):
        mode = "null"
    if (
        training
        and torch.rand((), generator=generator, device=random_device).item()
        < config.reliability_dropout_probability
    ):
        working = _without_reliability(batch)
    if training:
        working = _with_edge_dropout(working, config.edge_dropout_probability, generator)
    encoded, _ = codec.encode(working.trajectory)
    return diffusion_objective(
        denoiser,
        schedule,
        codec,
        token_encoder,
        working.trajectory,
        working.graph,
        working.observations,
        conditioning_mode=mode,
        channel_weights=_channel_weights(config, encoded),
        generator=generator,
        supervision_mask=working.supervision_mask,
    ).total


def train(
    config_path: str | Path,
    output: str | Path,
    *,
    device: str,
    bridge_factory: Callable[[DPoserConfig, torch.device], nn.Module] | None = None,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = DiffusionTrainingConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    if config.experiment.development_only and config_path.name != "smoke.yaml":
        raise ValueError("development training config must be named smoke.yaml")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable diffusion training output exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".training_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    started = time.time()
    torch.manual_seed(config.experiment.seed)
    execution_device = torch.device(device)
    if execution_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device)
    train_batch = load_supervised_windows(
        config.data.train_bundle,
        expected_split="train",
        allow_development=config.experiment.development_only,
    )
    validation_batch = load_supervised_windows(
        config.data.validation_bundle,
        expected_split="validation",
        allow_development=config.experiment.development_only,
    )
    for name in ("patch_map_sha256", "observation_cache_sha256"):
        if getattr(train_batch.metadata, name) != getattr(validation_batch.metadata, name):
            raise ValueError(f"train/validation {name} mismatch")
    codec = StateCodec.fit(train_batch.trajectory, train_batch.supervision_mask)
    normalizer_path = output / "trajectory_normalizer.json"
    normalizer_path.write_text(
        json.dumps(codec.to_payload(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    train_width = codec.encode(train_batch.trajectory)[0].shape[-1]
    validation_width = codec.encode(validation_batch.trajectory)[0].shape[-1]
    expected_width = sum(DPoserXConditionedTrajectoryDenoiser.PRODUCTION_PART_DIMS)
    if train_width != expected_width or validation_width != expected_width:
        raise ValueError("official DCG diffusion requires the exact 337-D trajectory topology")
    edge_count = train_batch.graph.event_state.shape[2]
    if validation_batch.graph.event_state.shape[2] != edge_count:
        raise ValueError("train/validation edge topology mismatch")
    if train_batch.metadata.edge_names != validation_batch.metadata.edge_names:
        raise ValueError("train/validation edge-name topology mismatch")
    if bridge_factory is None:
        bridge = OfficialDPoserXBridge(
            source_root=config.dposer_x.source_root,
            runtime_root=config.dposer_x.runtime_root,
            registry_path=config.dposer_x.registry,
            expected_commit=config.dposer_x.expected_commit,
            device=execution_device,
        )
    else:
        bridge = bridge_factory(config.dposer_x, execution_device)
    if any(parameter.requires_grad for parameter in bridge.parameters()):
        raise ValueError("official DPoser-X bridge must remain frozen")
    denoiser = DPoserXConditionedTrajectoryDenoiser(
        bridge,
        trajectory_steps=config.diffusion.steps,
        hidden_dim=config.model.hidden_dim,
        heads=config.model.heads,
        layers=config.model.layers,
        dropout=config.model.dropout,
    ).to(execution_device)
    token_encoder = ContactTokenEncoder(
        edge_count, config.model.hidden_dim, train_batch.metadata.edge_names
    ).to(execution_device)
    model = nn.ModuleDict({"denoiser": denoiser, "contact_token_encoder": token_encoder})
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    schedule = DiffusionSchedule(
        config.diffusion.steps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )
    order_generator = torch.Generator().manual_seed(config.experiment.seed)
    train_generator = torch.Generator(device=execution_device).manual_seed(
        config.experiment.seed + 1
    )
    order = torch.randperm(train_batch.trajectory.valid_mask.shape[0], generator=order_generator)
    cursor = 0
    history = []
    best_loss = float("inf")
    best_step = -1
    best_state = None
    for step in range(1, config.optimization.train_steps + 1):
        if cursor >= len(order):
            order = torch.randperm(len(order), generator=order_generator)
            cursor = 0
        indices = order[cursor : cursor + config.optimization.batch_size]
        cursor += len(indices)
        batch = train_batch.select(indices).to(execution_device)
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(
            denoiser,
            schedule,
            token_encoder,
            batch,
            codec,
            config.diffusion,
            train_generator,
            training=True,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("diffusion training produced NaN/Inf loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config.optimization.gradient_clip_norm
        )
        optimizer.step()
        row: dict[str, float | int] = {
            "step": step,
            "train_loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
        }
        should_validate = step % config.optimization.validation_interval == 0
        should_validate |= step == config.optimization.train_steps
        if should_validate:
            validation_generator = torch.Generator(device=execution_device).manual_seed(
                config.experiment.seed + 10_000
            )
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    _objective(
                        denoiser,
                        schedule,
                        token_encoder,
                        validation_batch.to(execution_device),
                        codec,
                        config.diffusion,
                        validation_generator,
                        training=False,
                    )
                )
            model.train()
            row["validation_loss"] = validation_loss
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
        history.append(row)
    if best_state is None:
        raise RuntimeError("diffusion trainer produced no validation checkpoint")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name].to(parameter))
    dependencies = _dependencies(config.third_party_manifest)
    save_model_checkpoint(
        output / "checkpoint",
        model,
        CheckpointMetadata(
            stage="trajectory_diffusion",
            model_class=f"{type(model).__module__}.{type(model).__qualname__}",
            step=best_step,
            epoch=0,
            seed=config.experiment.seed,
            config_sha256=file_sha256(config_path),
            manifest_sha256=train_batch.metadata.manifest_sha256,
            dependency_commits=dependencies,
            metrics={"best_validation_loss": best_loss},
            development_only=config.experiment.development_only,
            asset_sha256={
                "dposer_x_registry": file_sha256(config.dposer_x.registry),
                "trajectory_normalizer": file_sha256(normalizer_path),
            },
        ),
        state_scope="trainable",
    )
    report: dict[str, object] = {
        "schema_version": "dcg_diffusion_training_v1",
        "development_only": config.experiment.development_only,
        "selection_split": "validation",
        "selection_rule": "minimum_fixed_noise_epsilon_objective",
        "conditioning_mode": config.diffusion.conditioning_mode,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "history": history,
        "official_dposer_x_frozen": True,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "train_bundle_sha256": file_sha256(config.data.train_bundle / "windows.npz"),
        "validation_bundle_sha256": file_sha256(config.data.validation_bundle / "windows.npz"),
        "config_sha256": file_sha256(config_path),
        "dposer_x_registry_sha256": file_sha256(config.dposer_x.registry),
        "trajectory_normalizer_sha256": file_sha256(normalizer_path),
        "device": str(execution_device),
        "hardware": (
            torch.cuda.get_device_name(execution_device)
            if execution_device.type == "cuda"
            else platform.machine()
        ),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(execution_device))
            if execution_device.type == "cuda"
            else 0
        ),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(incomplete, output / "TRAINING_COMPLETE")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.output, device=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
