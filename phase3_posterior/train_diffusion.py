"""Stages R3-R6 masked temporal-relational diffusion training."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from phase2_refiner.data.corruptions import refresh_rotation_features
from phase3_posterior.config import load_config
from phase3_posterior.data.corruptions import sample_conditioning_mask
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.losses.diffusion import SubVPSDE, region_balanced_score_loss
from phase3_posterior.losses.geometry import masked_geodesic_loss, target_motion_loss
from phase3_posterior.losses.relation import relation_losses
from phase3_posterior.geometry.relation_anchors import mask_relation_inputs
from phase3_posterior.masked_spatial import (
    evaluate_rotation_proxy,
    inject_masked_rotation_corruption,
)
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.training import (
    ExponentialMovingAverage,
    cosine_warmup_scheduler,
    load_weights,
    prepare_run,
    rng_state,
    save_checkpoint,
    seed_everything,
)


def _condition_masks(
    valid: torch.Tensor, seed: int, step: int, dropout: float
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = []
    corruptions = []
    for index, item in enumerate(valid):
        generator = torch.Generator().manual_seed(seed + step * 10_007 + index)
        mask = sample_conditioning_mask(item.cpu(), generator).conditioning
        corruption = item.cpu() & ~mask
        if torch.rand((), generator=generator) < dropout:
            mask = torch.zeros_like(mask)
            corruption = torch.zeros_like(corruption)
        masks.append(mask)
        corruptions.append(corruption)
    return (
        torch.stack(masks).to(valid.device),
        torch.stack(corruptions).to(valid.device),
    )


def _conditioning_valid(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Validity of the cached pose observation used for synthetic masking."""
    return batch["frame_valid"][..., None].expand(-1, -1, 51)


def _reset_dormant_conditioning_projections(
    model: RelationalDiffusionPosterior,
) -> None:
    """Activate conditioning from an unconditional warm start without a score jump."""
    with torch.no_grad():
        model.residual.observation.weight.zero_()
        model.residual.relation.weight.zero_()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--relation-init")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    output = prepare_run(config, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Phase3Dataset(
        config["data"]["train_index"],
        int(config["model"]["max_frames"]),
        training=True,
        seed=seed,
        input_dim=int(config["model"].get("observation_dim", 45)),
        identity_target=bool(config["data"].get("identity_target", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config["training"].get("workers", 0)),
        collate_fn=collate_phase3,
    )
    validation_interval = int(config["training"].get("validation_interval", 0))
    validation_loader = None
    if validation_interval > 0:
        val_index = config["data"].get("val_index")
        if not val_index:
            raise ValueError("validation_interval requires data.val_index")
        validation_dataset = Phase3Dataset(
            val_index,
            int(config["model"]["max_frames"]),
            training=False,
            seed=seed + 1,
            input_dim=int(config["model"].get("observation_dim", 45)),
            identity_target=bool(config["data"].get("identity_target", False)),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(
                config["training"].get(
                    "validation_batch_size",
                    config["training"].get("batch_size", 4),
                )
            ),
            shuffle=False,
            num_workers=int(config["training"].get("workers", 0)),
            collate_fn=collate_phase3,
        )
    model = RelationalDiffusionPosterior(config["model"]).to(device)
    if args.init:
        load_weights(model, args.init)
    if args.relation_init:
        load_weights(model.relation_graph, args.relation_init, strict=True)
    if bool(config["model"].get("reset_conditioning_projections_on_init", False)):
        if not args.init:
            raise ValueError(
                "reset_conditioning_projections_on_init requires --init"
            )
        _reset_dormant_conditioning_projections(model)
    if bool(config["model"].get("freeze_relation_backbone", False)):
        model.relation_graph.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.05)),
    )
    max_steps = int(config["training"]["max_steps"])
    scheduler = cosine_warmup_scheduler(optimizer, max_steps)
    ema = ExponentialMovingAverage(model, float(config["training"].get("ema", 0.9999)))
    sde = SubVPSDE(
        **{key: config["diffusion"][key] for key in ("beta_min", "beta_max", "eps")}
    )
    iterator = iter(loader)
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    hint_only_steps = int(config["training"].get("hint_only_steps", 0))
    contact_energy_enabled = bool(
        config["model"].get("contact_energy_enabled", True)
    )

    best_score = float("inf")
    best_step: int | None = None
    best_validation: dict | None = None
    validations_without_improvement = 0
    early_stop_patience = int(config["training"].get("early_stop_patience", 0))
    completed_step = 0

    def checkpoint_payload(step: int, validation: dict | None = None) -> dict:
        return {
            "model": model.state_dict(),
            "ema_model": ema.state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "rng_state": rng_state(),
            "initialization": {
                "diffusion": {
                    "path": args.init,
                    "sha256": sha256_file(args.init) if args.init else None,
                },
                "relation": {
                    "path": args.relation_init,
                    "sha256": sha256_file(args.relation_init)
                    if args.relation_init
                    else None,
                    "frozen": bool(
                        config["model"].get("freeze_relation_backbone", False)
                    ),
                },
                "fallback": config.get("fallback", {}),
            },
            "validation": validation,
            "best_validation": best_validation,
            "best_step": best_step,
        }

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(1, max_steps * accumulation + 1):
        step = (micro_step + accumulation - 1) // accumulation
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        valid = batch["target_rotation_valid"] & batch["frame_valid"][..., None]
        if not valid.any():
            raise RuntimeError("Training batch has no supervised rotation targets")
        time = (
            torch.rand(len(batch["target_state"]), device=device) * (1.0 - sde.eps)
            + sde.eps
        )
        noisy, noise, std = sde.perturb(batch["target_state"], time)
        # Phase 3 synthetic masking conditions on the cached pose observation.
        # ``joint_valid`` denotes optional decoded 3D joint-position supervision
        # and is legitimately all-false for the clean ARCTIC/InterHand cache;
        # using it here silently turned R3 into unconditional diffusion. Target
        # validity cannot replace it either: hand-only labels still require the
        # full initializer body as attachment conditioning.
        condition, corruption_mask = _condition_masks(
            _conditioning_valid(batch),
            seed,
            step,
            float(config["training"].get("condition_dropout", 0.1)),
        )
        corruption_mask &= batch["target_rotation_valid"]
        condition |= (
            batch["frame_valid"][..., None]
            & ~batch["target_rotation_valid"]
            & ~corruption_mask
        )
        model_features = batch["features"]
        rotation_hint_mask = None
        if bool(config["model"].get("masked_rotation_hints", False)):
            corrupted_matrix, corruption_mask = inject_masked_rotation_corruption(
                batch["initial_matrix"].float(),
                corruption_mask,
                seed=seed + micro_step * 100_003,
                max_degrees=float(
                    config["training"].get(
                        "masked_rotation_corruption_degrees", 35.0
                    )
                ),
            )
            model_features = refresh_rotation_features(
                batch["features"], corrupted_matrix
            )
            rotation_hint_mask = corruption_mask
        conditioned_edges, conditioned_edge_valid = mask_relation_inputs(
            batch["edge_features"],
            batch["edge_valid"],
            batch["edge_index"],
            condition,
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            result = model(
                noisy,
                time,
                model_features,
                batch["frame_valid"],
                conditioned_edges,
                batch["edge_index"],
                conditioned_edge_valid,
                condition,
                rotation_hint_mask=rotation_hint_mask,
            )
        score_loss, regional = region_balanced_score_loss(
            result["score"], noise, std, valid, batch["target_weight"]
        )
        x0 = sde.x0_from_score(noisy, result["score"], time)
        auxiliary_weight = sde.clipped_auxiliary_weight(
            time, float(config.get("loss", {}).get("auxiliary_snr_gamma", 5.0))
        )
        auxiliary_sample_weight = batch["target_weight"] * auxiliary_weight
        rotation = masked_geodesic_loss(
            x0, batch["target_matrix"], valid, auxiliary_sample_weight
        )
        motion = target_motion_loss(
            x0, batch["target_state"], valid, auxiliary_sample_weight
        )
        # Do not supervise a relation target whose endpoint was deliberately
        # hidden from the conditioning graph.  Otherwise the contact head is
        # trained to guess labels from an all-zero edge token.
        weights = config.get("loss", {})
        loss = (
            score_loss
            + float(weights.get("rotation", 0.5)) * rotation
            + float(weights.get("motion", 0.25)) * motion
        )
        if contact_energy_enabled:
            relation_batch = dict(batch)
            relation_batch["edge_valid"] = conditioned_edge_valid
            relation_batch["contact_valid"] = (
                batch["contact_valid"] & conditioned_edge_valid
            )
            relation = relation_losses(result, relation_batch)
            loss = (
                loss
                + float(weights.get("contact", 0.25)) * relation["contact"]
                + float(weights.get("persistence", 0.1)) * relation["persistence"]
                + float(weights.get("depth", 0.1)) * relation["depth"]
            )
        (loss / accumulation).backward()
        if micro_step % accumulation:
            continue
        if hint_only_steps > 0 and step <= hint_only_steps:
            for name, parameter in model.named_parameters():
                if "residual.corruption_observation" not in name:
                    parameter.grad = None
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"].get("gradient_clip", 1.0))
        )
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        completed_step = step
        if step == 1 or step % int(config["training"].get("log_interval", 50)) == 0:
            metrics = {
                "step": step,
                "loss": float(loss.detach()),
                "score": float(score_loss.detach()),
                "rotation": float(rotation.detach()),
                "motion": float(motion.detach()),
                "contact_energy_enabled": contact_energy_enabled,
                "relation_backbone_frozen": not any(
                    parameter.requires_grad
                    for parameter in model.relation_graph.parameters()
                ),
                "auxiliary_snr_weight_mean": float(auxiliary_weight.mean()),
                "rotation_hints_enabled": rotation_hint_mask is not None,
                "rotation_hint_fraction": float(corruption_mask.float().mean()),
                "hint_only_optimization": step <= hint_only_steps,
            }
            metrics.update(
                {
                    f"score_{key}": float(value.detach())
                    for key, value in regional.items()
                }
            )
            print(json.dumps(metrics), flush=True)
        if step % checkpoint_interval == 0:
            save_checkpoint(output / "last.pt", checkpoint_payload(step))
        if validation_loader is not None and step % validation_interval == 0:
            with ema.average_parameters(model):
                validation = evaluate_rotation_proxy(
                    model,
                    validation_loader,
                    sde,
                    device,
                    steps=int(
                        config["training"].get("validation_sampling_steps", 10)
                    ),
                    seed=int(config["training"].get("validation_seed", seed + 2000)),
                    max_batches=int(
                        config["training"].get("validation_max_batches", 8)
                    ),
                )
            validation["step"] = step
            atomic_json(output / f"validation_{step:06d}.json", validation)
            print(json.dumps({"validation": validation}), flush=True)
            score = float(validation["selection_score"])
            if score < best_score:
                best_score = score
                best_step = step
                best_validation = validation
                validations_without_improvement = 0
                save_checkpoint(
                    output / "best.pt", checkpoint_payload(step, validation)
                )
            else:
                validations_without_improvement += 1
            if (
                early_stop_patience > 0
                and validations_without_improvement >= early_stop_patience
            ):
                print(
                    json.dumps(
                        {
                            "early_stop": True,
                            "step": step,
                            "best_step": best_step,
                            "validations_without_improvement": (
                                validations_without_improvement
                            ),
                        }
                    ),
                    flush=True,
                )
                break
    payload = checkpoint_payload(completed_step)
    save_checkpoint(output / "last.pt", payload)
    if not (output / "best.pt").exists():
        save_checkpoint(output / "best.pt", payload)


if __name__ == "__main__":
    main()
