"""Train the deterministic Phase 2 whole-sequence refiner."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from phase2_refiner.config import load_config, validate_config
from phase2_refiner.data.corruptions import apply_burst_corruption
from phase2_refiner.data.dataset import (
    TORSO_POSITION,
    TORSO_POSITION_VALID,
    U0_RELIABILITY,
    WRIST_POSITION,
    WRIST_POSITION_VALID,
    LengthBucketBatchSampler,
    SequenceCacheDataset,
    collate_sequences,
)
from phase2_refiner.losses import RefinerLoss
from phase2_refiner.models import WholeSequenceRefiner
from phase2_refiner.models.pretrained import load_compatible_initialization
from phase2_refiner.provenance import run_provenance, sha256_file
from phase2_refiner.geometry.smplx_decode import decode_smplx_sequence
from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.render import create_smplx_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict) -> WholeSequenceRefiner:
    return WholeSequenceRefiner(**config.get("model", {}))


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loss_arguments(batch: dict, features: torch.Tensor) -> dict:
    torso = features[..., TORSO_POSITION]
    torso_valid = features[..., TORSO_POSITION_VALID].bool()
    wrist = features[..., WRIST_POSITION]
    wrist_valid = features[..., WRIST_POSITION_VALID].bool()
    observed_position = torch.where(wrist_valid[..., None], wrist, torso)
    observed_valid = torso_valid | wrist_valid
    return {
        "target_rotation_valid": batch["target_rotation_valid"],
        "target_joint_position": batch["target_joint_position"],
        "target_joint_valid": batch["target_joint_valid"],
        "target_palm_normal": batch["target_palm_normal"],
        "target_palm_valid": batch["target_palm_valid"],
        "observed_joint_position": observed_position,
        "observed_joint_valid": observed_valid,
    }


def _geometry_context(config: dict, device: torch.device) -> dict | None:
    geometry = config.get("geometry", {})
    if not geometry.get("enabled", False):
        return None
    model = create_smplx_model(geometry["model_folder"], device)
    model.requires_grad_(False)
    assets = Path(geometry["assets_root"])
    with (assets / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        hand_ids = pickle.load(handle, encoding="latin1")
    upper = np.load(
        assets / "sgnify_part_segm_above_pelvis_joint" / "upper_body_minus_face.npy"
    )
    return {
        "model": model,
        "region_masks": {
            "ubody": torch.as_tensor(upper, device=device, dtype=torch.long),
            "lhand": torch.as_tensor(
                hand_ids["left_hand"], device=device, dtype=torch.long
            ),
            "rhand": torch.as_tensor(
                hand_ids["right_hand"], device=device, dtype=torch.long
            ),
        },
    }


def _geometry_loss_arguments(
    context: dict | None, prediction: dict, batch: dict, device: torch.device
) -> dict:
    if context is None:
        return {}
    face = {
        "jaw_pose": batch["jaw_pose"].float(),
        "leye_pose": batch["leye_pose"].float(),
        "reye_pose": batch["reye_pose"].float(),
        "expression": batch["expression"].float(),
    }
    with torch.autocast(device_type=device.type, enabled=False):
        predicted_vertices, _ = decode_smplx_sequence(
            context["model"],
            prediction["matrix"].float(),
            batch["betas"].float(),
            batch["global_orient"].float(),
            batch["transl"].float(),
            **face,
        )
        with torch.no_grad():
            target_vertices, _ = decode_smplx_sequence(
                context["model"],
                batch["target_matrix"].float(),
                batch["betas"].float(),
                batch["global_orient"].float(),
                batch["transl"].float(),
                **face,
            )
    return {
        "predicted_vertices": predicted_vertices,
        "target_vertices": target_vertices.detach(),
        "vertex_region_masks": context["region_masks"],
    }


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[key].copy_(value)


def _optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias") or "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        (
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=learning_rate,
    )


@torch.no_grad()
def evaluate(
    model,
    loss_fn,
    loader,
    device,
    autocast_context,
    geometry_context,
    corruption: dict | None = None,
    corruption_seed: int = 1729,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    cuda_devices = (
        [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(corruption_seed)
        for batch in loader:
            batch = _to_device(batch, device)
            features = batch["features"]
            initial_matrix = batch["initial_matrix"]
            corruption_mask = None
            if corruption is not None:
                features, initial_matrix, corruption_mask = apply_burst_corruption(
                    features,
                    initial_matrix,
                    batch["frame_valid"],
                    target_rotation_valid=batch["target_rotation_valid"],
                    **corruption,
                )
            with autocast_context():
                prediction = model(
                    features,
                    initial_matrix,
                    batch["frame_valid"],
                    batch["refine_mask"],
                    batch["initial_joint_position"],
                )
                losses = loss_fn(
                    prediction,
                    initial_matrix,
                    batch["target_matrix"],
                    batch["frame_valid"],
                    batch["refine_mask"],
                    features[..., U0_RELIABILITY],
                    **_loss_arguments(batch, features),
                    **_geometry_loss_arguments(
                        geometry_context, prediction, batch, device
                    ),
                )
            for name, value in losses.items():
                totals.setdefault(name, []).append(float(value))
            if corruption_mask is not None:
                mask = corruption_mask & batch["target_rotation_valid"]
                count = mask.sum().clamp_min(1)
                injected = (
                    geodesic_distance(initial_matrix, batch["target_matrix"]) * mask
                ).sum() / count
                residual = (
                    geodesic_distance(prediction["matrix"], batch["target_matrix"])
                    * mask
                ).sum() / count
                totals.setdefault("injected_rotation_error", []).append(float(injected))
                totals.setdefault("residual_rotation_error", []).append(float(residual))
    if corruption is not None:
        injected = float(np.mean(totals.get("injected_rotation_error", [0.0])))
        residual = float(np.mean(totals.get("residual_rotation_error", [0.0])))
        totals["recovery_fraction"] = [
            0.0 if injected <= 1e-8 else 1.0 - residual / injected
        ]
    return {name: float(np.mean(values)) for name, values in totals.items()}


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _checkpoint(
    model,
    ema,
    optimizer,
    scheduler,
    config,
    provenance,
    step,
    micro_step,
    best,
    train_dataset,
    loader_generator,
    batch_sampler,
) -> dict:
    return {
        "format_version": 2,
        "model": model.state_dict(),
        "ema_model": ema.shadow,
        "model_config": config.get("model", {}),
        "resolved_config": config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _rng_state(),
        "dataset_rng": train_dataset.rng.bit_generator.state,
        "loader_rng": loader_generator.get_state(),
        "batch_sampler": (
            batch_sampler.state_dict() if batch_sampler is not None else None
        ),
        "step": step,
        "micro_step": micro_step,
        "best": best,
        "provenance": provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--identity-target", action="store_true", help="Smoke tests only"
    )
    parser.add_argument("--train-glob", help="Override data.train_glob/split manifest")
    parser.add_argument("--val-glob", help="Override data.val_glob/split manifest")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--spatial-init",
        type=Path,
        help="Optional adapter-produced compatible spatial-prior checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(
        config, require_data=True, require_validation=not args.no_validation
    )
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device)
    data_config = config.get("data", {})
    if args.train_glob:
        data_config["train_glob"] = args.train_glob
    if args.val_glob:
        data_config["val_glob"] = args.val_glob
    max_frames = int(config.get("model", {}).get("max_frames", 64))
    train_dataset = SequenceCacheDataset(
        data_config["train_glob"],
        max_frames=max_frames,
        training=True,
        identity_target=args.identity_target,
        seed=seed,
    )
    val_glob = None if args.no_validation else data_config.get("val_glob")
    val_dataset = (
        SequenceCacheDataset(
            val_glob,
            max_frames=max_frames,
            training=False,
            identity_target=args.identity_target,
            seed=seed,
        )
        if val_glob
        else None
    )
    train_config = config.get("training", {})
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.gradient_accumulation is not None:
        train_config["gradient_accumulation"] = args.gradient_accumulation
    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    batch_size = int(args.batch_size or train_config.get("batch_size", 8))
    loader_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "num_workers": int(train_config.get("workers", 0)),
        "collate_fn": collate_sequences,
        "generator": loader_generator,
        "pin_memory": device.type == "cuda",
    }
    if train_config.get("bucket_by_length", True):
        batch_sampler = LengthBucketBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    else:
        batch_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sequences,
            pin_memory=device.type == "cuda",
        )
        if val_dataset
        else None
    )
    model = make_model(config).to(device)
    initialization_report = None
    if args.spatial_init is not None:
        if args.resume is not None:
            raise ValueError("--spatial-init and --resume are mutually exclusive")
        initialization_report = load_compatible_initialization(model, args.spatial_init)
    loss_fn = RefinerLoss(**config.get("loss", {})).to(device)
    geometry_context = _geometry_context(config, device)
    optimizer = _optimizer(
        model,
        float(train_config.get("learning_rate", 2e-4)),
        float(train_config.get("weight_decay", 0.05)),
    )
    max_steps = int(train_config.get("max_steps", 100000))
    warmup = max(1, int(max_steps * float(train_config.get("warmup_fraction", 0.05))))
    scheduler = SequentialLR(
        optimizer,
        schedulers=(
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup)),
        ),
        milestones=(warmup,),
    )
    ema = ExponentialMovingAverage(model, float(train_config.get("ema_decay", 0.999)))
    precision = str(train_config.get("precision", "bf16")).lower()
    amp_enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    def autocast_context():
        if amp_enabled:
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    output = Path(
        args.output_dir or config.get("output_dir", "outputs/phase2_training")
    ).resolve()
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(
            f"Refusing to reuse non-empty training directory without --resume: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    provenance = run_provenance(args.config, seed)
    if initialization_report is not None:
        initialization_report["sha256"] = sha256_file(args.spatial_init)
        provenance["spatial_initialization"] = initialization_report
    for name in ("train_glob", "val_glob"):
        candidate = Path(str(data_config.get(name, "")))
        if candidate.is_file():
            provenance[f"{name}_sha256"] = sha256_file(candidate)
    with (output / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": config, "provenance": provenance},
            handle,
            indent=2,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")

    accumulation = int(train_config.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient accumulation must be at least 1")
    clip_norm = float(train_config.get("gradient_clip", 1.0))
    log_every = int(train_config.get("log_every", 20))
    validate_every = int(train_config.get("validate_every", 500))
    checkpoint_every = int(train_config.get("checkpoint_every", 2000))
    early_patience = int(train_config.get("early_stopping_patience", 20))
    uncertainty_only_steps = int(train_config.get("uncertainty_only_steps", 0))
    corruption = config.get("corruption", {})
    validation_corruption = config.get("validation_corruption")
    best = float("inf")
    validations_without_improvement = 0
    step = micro_step = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("resolved_config") not in (None, config):
            raise ValueError(
                "Resume configuration differs from checkpoint configuration"
            )
        model.load_state_dict(state["model"], strict=True)
        ema.shadow = state.get("ema_model", copy.deepcopy(state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        micro_step = int(state.get("micro_step", step * accumulation))
        best = float(state.get("best", best))
        _restore_rng_state(state["rng"])
        if "dataset_rng" in state:
            train_dataset.rng.bit_generator.state = state["dataset_rng"]
        if "loader_rng" in state:
            loader_generator.set_state(state["loader_rng"].cpu())
        if batch_sampler is not None and state.get("batch_sampler") is not None:
            batch_sampler.load_state_dict(state["batch_sampler"])

    optimizer.zero_grad(set_to_none=True)
    stop = False
    while step < max_steps and not stop:
        for batch in train_loader:
            model.train()
            batch = _to_device(batch, device)
            features, initial_matrix, _ = apply_burst_corruption(
                batch["features"],
                batch["initial_matrix"],
                batch["frame_valid"],
                target_rotation_valid=batch["target_rotation_valid"],
                **corruption,
            )
            with autocast_context():
                prediction = model(
                    features,
                    initial_matrix,
                    batch["frame_valid"],
                    batch["refine_mask"],
                    batch["initial_joint_position"],
                )
                losses = loss_fn(
                    prediction,
                    initial_matrix,
                    batch["target_matrix"],
                    batch["frame_valid"],
                    batch["refine_mask"],
                    features[..., U0_RELIABILITY],
                    **_loss_arguments(batch, features),
                    **_geometry_loss_arguments(
                        geometry_context, prediction, batch, device
                    ),
                )
                scaled_loss = losses["total"] / accumulation
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(
                    f"Non-finite loss before backward at step={step} "
                    f"clips={batch['clip_id']}"
                )
            scaled_loss.backward()
            if step < uncertainty_only_steps:
                for name, parameter in model.named_parameters():
                    if "reliability_head" not in name:
                        parameter.grad = None
            micro_step += 1
            if micro_step % accumulation != 0:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_norm, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient at step={step}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(model)
            step += 1
            if step % log_every == 0 or step == 1:
                fields = " ".join(
                    f"{key}={float(value.detach()):.6f}"
                    for key, value in losses.items()
                )
                print(f"step={step} {fields}")
            if val_loader is not None and (
                step % validate_every == 0 or step == max_steps
            ):
                metrics = evaluate(
                    model,
                    loss_fn,
                    val_loader,
                    device,
                    autocast_context,
                    geometry_context,
                    corruption=validation_corruption,
                    corruption_seed=seed + 10000,
                )
                score = metrics["total"]
                print(f"step={step} val={json.dumps(metrics, sort_keys=True)}")
                if validation_corruption is not None:
                    clean_metrics = evaluate(
                        model,
                        loss_fn,
                        val_loader,
                        device,
                        autocast_context,
                        geometry_context,
                    )
                    print(
                        f"step={step} val_clean="
                        f"{json.dumps(clean_metrics, sort_keys=True)}"
                    )
                if score < best:
                    best = score
                    validations_without_improvement = 0
                    torch.save(
                        _checkpoint(
                            model,
                            ema,
                            optimizer,
                            scheduler,
                            config,
                            provenance,
                            step,
                            micro_step,
                            best,
                            train_dataset,
                            loader_generator,
                            batch_sampler,
                        ),
                        output / "best.pt",
                    )
                else:
                    validations_without_improvement += 1
                    stop = validations_without_improvement >= early_patience
            if step % checkpoint_every == 0:
                torch.save(
                    _checkpoint(
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        config,
                        provenance,
                        step,
                        micro_step,
                        best,
                        train_dataset,
                        loader_generator,
                        batch_sampler,
                    ),
                    output / f"step_{step:07d}.pt",
                )
            if step >= max_steps or stop:
                break
    torch.save(
        _checkpoint(
            model,
            ema,
            optimizer,
            scheduler,
            config,
            provenance,
            step,
            micro_step,
            best,
            train_dataset,
            loader_generator,
            batch_sampler,
        ),
        output / "last.pt",
    )
    print(f"Training complete at step {step}; checkpoints: {output}")


if __name__ == "__main__":
    main()
