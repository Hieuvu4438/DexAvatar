"""Proposal Stage 2 contact-proposal trainer with validation-only selection."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dcg_sign4d.contact.balanced_sampler import balanced_window_sample_weights
from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.training.batch import SupervisedWindowBatch, load_supervised_windows
from dcg_sign4d.training.checkpoint import CheckpointMetadata, save_model_checkpoint
from dcg_sign4d.training.steps import contact_objective
from dcg_sign4d.utils.hashing import file_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(_StrictModel):
    seed: int
    development_only: bool = False


class DataConfig(_StrictModel):
    train_bundle: Path
    validation_bundle: Path


class ModelConfig(_StrictModel):
    hidden_dim: int = Field(gt=0)
    heads: int = Field(gt=0)
    layers: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    max_duration: int = Field(gt=0)

    @model_validator(mode="after")
    def compatible_heads(self) -> ModelConfig:
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        return self


class OptimizationConfig(_StrictModel):
    steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0, le=0.1)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    validation_interval: int = Field(gt=0)


class SamplingConfig(_StrictModel):
    balanced: bool
    effective_number_beta: float = Field(ge=0, lt=1)


class ObjectiveConfig(_StrictModel):
    event_weight: float = Field(ge=0)
    duration_weight: float = Field(ge=0)
    transition_weight: float = Field(ge=0)
    calibration_weight: float = Field(ge=0)
    gold_label_weight: float = Field(gt=0)
    accepted_pseudo_label_weight: float = Field(gt=0)

    @model_validator(mode="after")
    def gold_is_not_downweighted(self) -> ObjectiveConfig:
        if self.gold_label_weight < self.accepted_pseudo_label_weight:
            raise ValueError("gold label weight must be at least accepted-pseudo weight")
        return self


class ContactTrainingConfig(_StrictModel):
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    sampling: SamplingConfig
    objective: ObjectiveConfig
    third_party_manifest: Path


def _dependencies(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {item["name"]: item["commit"] for item in payload["repositories"]}


def _class_counts(batch: SupervisedWindowBatch) -> torch.Tensor:
    valid = batch.graph.edge_valid[:, None, :].expand_as(batch.graph.event_state)
    valid = valid & batch.trajectory.valid_mask[:, :, None]
    valid = valid & ~batch.graph.uncertain_mask
    if not bool(valid.any()):
        raise ValueError("contact training bundle has no certain valid labels")
    return torch.bincount(batch.graph.event_state[valid], minlength=4)


def _loss(
    model: ContactProposal,
    batch: SupervisedWindowBatch,
    class_counts: torch.Tensor,
    objective: ObjectiveConfig,
) -> torch.Tensor:
    output = model(batch.observations, batch.trajectory, batch.geometry_features)
    weights = {
        "gold": objective.gold_label_weight,
        "accepted_pseudo": objective.accepted_pseudo_label_weight,
        "synthetic_fixture": 1.0,
    }
    sample_weight = torch.tensor(
        [weights[value] for value in batch.metadata.sample_label_status],
        device=output.event_logits.device,
        dtype=output.event_logits.dtype,
    )
    return contact_objective(
        output,
        event_state=batch.graph.event_state,
        duration_frames=batch.duration_frames,
        edge_valid=batch.graph.edge_valid,
        frame_valid=batch.trajectory.valid_mask,
        uncertain=batch.graph.uncertain_mask,
        class_counts=class_counts,
        event_weight=objective.event_weight,
        duration_weight=objective.duration_weight,
        transition_weight=objective.transition_weight,
        calibration_weight=objective.calibration_weight,
        sample_weight=sample_weight,
    ).total


def train(
    config_path: str | Path,
    output: str | Path,
    *,
    device: str,
) -> dict[str, object]:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ContactTrainingConfig.model_validate(raw)
    if config.experiment.development_only and config_path.name != "smoke.yaml":
        raise ValueError("development training config must be named smoke.yaml")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable contact training output exists: {output}")
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
    encoded, _ = StateCodec().encode(train_batch.trajectory)
    validation_encoded, _ = StateCodec().encode(validation_batch.trajectory)
    if encoded.shape[-1] != validation_encoded.shape[-1]:
        raise ValueError("train/validation trajectory topology mismatch")
    edge_count = train_batch.graph.event_state.shape[2]
    if validation_batch.graph.event_state.shape[2] != edge_count:
        raise ValueError("train/validation edge topology mismatch")
    if train_batch.metadata.edge_names != validation_batch.metadata.edge_names:
        raise ValueError("train/validation edge-name topology mismatch")
    if bool((train_batch.duration_frames > config.model.max_duration).any()) or bool(
        (validation_batch.duration_frames > config.model.max_duration).any()
    ):
        raise ValueError("duration target exceeds configured maximum")
    model = ContactProposal(
        encoded.shape[-1],
        edge_count,
        config.model.max_duration,
        hidden_dim=config.model.hidden_dim,
        heads=config.model.heads,
        layers=config.model.layers,
        dropout=config.model.dropout,
        edge_names=train_batch.metadata.edge_names,
    ).to(execution_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    counts = _class_counts(train_batch).to(execution_device)
    validation_counts = _class_counts(validation_batch).to(execution_device)
    generator = torch.Generator().manual_seed(config.experiment.seed)
    if config.sampling.balanced:
        window_weights = balanced_window_sample_weights(
            train_batch.graph.event_state,
            train_batch.graph.edge_valid,
            train_batch.trajectory.valid_mask,
            train_batch.graph.uncertain_mask,
            beta=config.sampling.effective_number_beta,
        ).cpu()
    else:
        window_weights = torch.ones(train_batch.trajectory.valid_mask.shape[0])
    history = []
    best_loss = float("inf")
    best_step = -1
    best_state = None
    for step in range(1, config.optimization.steps + 1):
        indices = torch.multinomial(
            window_weights,
            config.optimization.batch_size,
            replacement=True,
            generator=generator,
        )
        batch = train_batch.select(indices).to(execution_device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, counts, config.objective)
        if not torch.isfinite(loss):
            raise FloatingPointError("contact training produced NaN/Inf loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimization.gradient_clip_norm
        )
        optimizer.step()
        row: dict[str, float | int] = {
            "step": step,
            "train_loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
        }
        active = batch.graph.edge_valid[:, None, :] & batch.trajectory.valid_mask[:, :, None]
        active = active & ~batch.graph.uncertain_mask
        row["batch_event_counts"] = torch.bincount(
            batch.graph.event_state[active].detach().cpu(), minlength=4
        ).tolist()
        should_validate = step % config.optimization.validation_interval == 0
        should_validate |= step == config.optimization.steps
        if should_validate:
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    _loss(
                        model,
                        validation_batch.to(execution_device),
                        validation_counts,
                        config.objective,
                    )
                )
            model.train()
            row["validation_loss"] = validation_loss
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
        history.append(row)
    if best_state is None:
        raise RuntimeError("contact trainer produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    dependencies = _dependencies(config.third_party_manifest)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    save_model_checkpoint(
        output / "checkpoint",
        model,
        CheckpointMetadata(
            stage="contact_proposal",
            model_class=model_class,
            step=best_step,
            epoch=0,
            seed=config.experiment.seed,
            config_sha256=file_sha256(config_path),
            manifest_sha256=train_batch.metadata.manifest_sha256,
            dependency_commits=dependencies,
            metrics={"best_validation_loss": best_loss},
            development_only=config.experiment.development_only,
        ),
    )
    report: dict[str, object] = {
        "schema_version": "dcg_contact_training_v1",
        "development_only": config.experiment.development_only,
        "selection_split": "validation",
        "selection_rule": "minimum_total_contact_objective",
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "history": history,
        "train_bundle_sha256": file_sha256(config.data.train_bundle / "windows.npz"),
        "validation_bundle_sha256": file_sha256(config.data.validation_bundle / "windows.npz"),
        "config_sha256": file_sha256(config_path),
        "balanced_sampling": config.sampling.balanced,
        "window_sampling_weights": window_weights.tolist(),
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
