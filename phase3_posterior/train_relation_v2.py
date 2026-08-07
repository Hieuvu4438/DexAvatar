"""Corrected R2 training with comparators and source-disjoint validation."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from phase3_posterior.config import load_config
from phase3_posterior.data.cache_schema import load_relation_sidecar
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.losses.relation import relation_losses
from phase3_posterior.models.relation_baseline import GeometryOnlyRelationMLP
from phase3_posterior.models.relation_graph import RelationGraphEncoder
from phase3_posterior.relation_evaluation import (
    RelationMetricAccumulator,
    g2_metrics,
    relation_edge_masks,
)
from phase3_posterior.training import (
    ExponentialMovingAverage,
    cosine_warmup_scheduler,
    prepare_run,
    rng_state,
    save_checkpoint,
    seed_everything,
)


def _mixture_loader(
    dataset: Phase3Dataset,
    mixture: dict[str, float],
    batch_size: int,
    workers: int,
    seed: int,
    sign_contact_clip_probability: float | None = None,
) -> DataLoader:
    indices = [
        index for index, entry in enumerate(dataset.entries) if entry.source in mixture
    ]
    counts = {
        source: sum(dataset.entries[index].source == source for index in indices)
        for source in mixture
    }
    missing = [source for source, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"R2 mixture sources have no clips: {missing}")
    total_probability = sum(float(value) for value in mixture.values())
    probabilities = {
        source: float(value) / total_probability for source, value in mixture.items()
    }
    sign_positive: set[int] = set()
    if sign_contact_clip_probability is not None:
        if not 0.0 < sign_contact_clip_probability < 1.0:
            raise ValueError("sign_contact_clip_probability must be in (0,1)")
        for index in indices:
            entry = dataset.entries[index]
            if entry.source != "how2sign":
                continue
            relation = load_relation_sidecar(entry.relation_path)
            source_node, target_node = relation.edge_index
            hand_body = (source_node >= 10) ^ (target_node >= 10)
            if (relation.contact_target[:, hand_body] & relation.contact_valid[:, hand_body]).any():
                sign_positive.add(index)
        sign_total = counts.get("how2sign", 0)
        if not sign_positive or len(sign_positive) == sign_total:
            raise RuntimeError("Sign contact clip stratification requires both classes")
    weights = []
    for index in indices:
        source = dataset.entries[index].source
        probability = probabilities[source]
        if source == "how2sign" and sign_contact_clip_probability is not None:
            if index in sign_positive:
                probability *= sign_contact_clip_probability / len(sign_positive)
            else:
                probability *= (1.0 - sign_contact_clip_probability) / (
                    counts[source] - len(sign_positive)
                )
        else:
            probability /= counts[source]
        weights.append(probability)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(len(indices), batch_size),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_phase3,
    )


def _relation_weights(batch: dict, sign_hand_body_weight: float) -> torch.Tensor:
    masks = relation_edge_masks(batch["edge_index"][0])
    sign = torch.tensor(
        [source == "how2sign" for source in batch["source"]],
        device=batch["edge_features"].device,
    )[:, None, None]
    hand_body = masks["hand_body"][None, None, :]
    weights = torch.ones_like(batch["contact_target"], dtype=torch.float32)
    weights = torch.where(sign & hand_body, sign_hand_body_weight, weights)
    return weights * batch["target_weight"][:, None, None]


@torch.inference_mode()
def _validate(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict:
    accumulators = {
        name: RelationMetricAccumulator(threshold) for name in models
    }
    for model in models.values():
        model.eval()
    for batch in loader:
        source = batch["source"]
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batch["source"] = source
        masks = relation_edge_masks(batch["edge_index"][0])
        for name, model in models.items():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(
                    batch["edge_features"],
                    batch["edge_index"],
                    batch["edge_valid"],
                )
            accumulators[name].update(outputs, batch, masks)
    for model in models.values():
        model.train()
    results = {name: value.result() for name, value in accumulators.items()}
    return {
        "results": results,
        "metrics_for_gate": g2_metrics(
            results["graph"], results["geometry_mlp"], results["no_persistence"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    output = prepare_run(config, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_frames = int(config["model"]["max_frames"])
    train_dataset = Phase3Dataset(
        config["data"]["train_index"],
        max_frames,
        training=True,
        seed=seed,
        require_relation_targets=True,
    )
    val_dataset = Phase3Dataset(
        config["data"]["val_index"],
        max_frames,
        training=False,
        seed=seed,
        require_relation_targets=True,
    )
    batch_size = int(config["training"].get("batch_size", 8))
    workers = int(config["training"].get("workers", 0))
    generic_loader = _mixture_loader(
        train_dataset,
        config["training"]["generic_mixture"],
        batch_size,
        workers,
        seed,
    )
    joint_loader = _mixture_loader(
        train_dataset,
        config["training"]["joint_mixture"],
        batch_size,
        workers,
        seed + 1,
        float(config["training"].get("sign_contact_clip_probability", 0.35)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("validation_batch_size", 8)),
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_phase3,
    )
    width = int(config["model"].get("relation_width", 128))
    layers = int(config["model"].get("relation_layers", 3))
    models = {
        "graph": RelationGraphEncoder(
            width, layers, predict_distance=True, edge_identity=True
        ).to(device),
        "geometry_mlp": GeometryOnlyRelationMLP(width, predict_distance=True).to(
            device
        ),
        "no_persistence": RelationGraphEncoder(
            width, layers, predict_distance=True, edge_identity=True
        ).to(device),
    }
    # The no-persistence comparator starts from the exact graph initialization;
    # its sole experimental difference is the zero persistence-loss weight.
    models["no_persistence"].load_state_dict(models["graph"].state_dict())
    max_steps = int(config["training"]["max_steps"])
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"].get("learning_rate", 2e-4)),
            weight_decay=float(config["training"].get("weight_decay", 0.05)),
        )
        for name, model in models.items()
    }
    schedulers = {
        name: cosine_warmup_scheduler(optimizer, max_steps)
        for name, optimizer in optimizers.items()
    }
    emas = {
        name: ExponentialMovingAverage(
            model, float(config["training"].get("ema", 0.9999))
        )
        for name, model in models.items()
    }
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    generic_steps = int(config["training"].get("generic_warmup_steps", 0))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    validation_interval = int(config["training"].get("validation_interval", 2000))
    positive_alpha = float(config["training"].get("contact_positive_alpha", 0.75))
    sign_weight = float(config["training"].get("sign_hand_body_weight", 4.0))
    distance_weight = float(config["training"].get("distance_loss_weight", 0.5))
    threshold = float(config["evaluation"].get("contact_threshold", 0.5))
    active_loader = generic_loader if generic_steps else joint_loader
    iterator = iter(active_loader)
    curriculum = "generic_warmup" if generic_steps else "joint_adaptation"
    best_score = float("-inf")
    last_validation: dict = {}
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    def checkpoint_payload(step: int) -> dict:
        return {
            "model": models["graph"].state_dict(),
            "ema_model": emas["graph"].state,
            "baseline_model": models["geometry_mlp"].state_dict(),
            "baseline_ema_model": emas["geometry_mlp"].state,
            "no_persistence_model": models["no_persistence"].state_dict(),
            "no_persistence_ema_model": emas["no_persistence"].state,
            "optimizers": {key: value.state_dict() for key, value in optimizers.items()},
            "schedulers": {key: value.state_dict() for key, value in schedulers.items()},
            "step": step,
            "config": config,
            "rng_state": rng_state(),
            "curriculum": curriculum,
            "validation": last_validation,
        }

    for micro_step in range(1, max_steps * accumulation + 1):
        step = (micro_step + accumulation - 1) // accumulation
        requested = "generic_warmup" if step <= generic_steps else "joint_adaptation"
        if requested != curriculum:
            curriculum = requested
            active_loader = joint_loader
            iterator = iter(active_loader)
            print(
                json.dumps(
                    {"event": "relation_v2_curriculum", "stage": curriculum, "step": step}
                ),
                flush=True,
            )
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(active_loader)
            batch = next(iterator)
        source = batch["source"]
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batch["source"] = source
        edge_masks = relation_edge_masks(batch["edge_index"][0])
        batch["distance_valid"] = batch["edge_valid"] & edge_masks["hand_hand"][
            None, None, :
        ]
        contact_weight = _relation_weights(batch, sign_weight)
        step_losses = {}
        for name, model in models.items():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(
                    batch["edge_features"], batch["edge_index"], batch["edge_valid"]
                )
            losses = relation_losses(
                outputs,
                batch,
                positive_alpha=positive_alpha,
                contact_weight=contact_weight,
            )
            persistence_weight = 0.0 if name == "no_persistence" else 0.4
            total = (
                losses["contact"]
                + 0.4 * losses["depth"]
                + persistence_weight * losses["persistence"]
                + distance_weight * losses["distance"]
            )
            (total / accumulation).backward()
            step_losses[name] = {
                "loss": float(total.detach()),
                **{key: float(value.detach()) for key, value in losses.items()},
            }
        if micro_step % accumulation:
            continue
        for name, model in models.items():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            schedulers[name].step()
            emas[name].update(model)
            optimizers[name].zero_grad(set_to_none=True)
        if step == 1 or step % int(config["training"].get("log_interval", 50)) == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "curriculum": curriculum,
                        "models": step_losses,
                    }
                ),
                flush=True,
            )
        if step % checkpoint_interval == 0:
            save_checkpoint(output / "last.pt", checkpoint_payload(step))
        if step % validation_interval == 0 or step == max_steps:
            original_states = {}
            for name, model in models.items():
                original_states[name] = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                model.load_state_dict(emas[name].state)
            validation = _validate(models, val_loader, device, threshold)
            for name, model in models.items():
                model.load_state_dict(original_states[name])
            metrics = validation["metrics_for_gate"]
            checks = (
                float(metrics["relation_mae_gain"]) >= 0.10,
                float(metrics["contact_f1"]) >= 0.65,
                float(metrics["sign_contact_f1"]) >= 0.60,
                float(metrics["depth_order_accuracy"]) >= 0.80,
                float(metrics["contact_slip_gain"]) >= 0.15,
                bool(metrics["contact_slip_comparison_available"]),
                float(metrics["max_region_regression"]) <= 0.01,
            )
            score = 10.0 * sum(checks) + sum(
                float(metrics[key])
                for key in (
                    "relation_mae_gain",
                    "contact_f1",
                    "sign_contact_f1",
                    "depth_order_accuracy",
                    "contact_slip_gain",
                )
            )
            last_validation = {"step": step, **validation, "selection_score": score}
            with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_validation, sort_keys=True) + "\n")
            print(json.dumps({"event": "relation_v2_validation", **last_validation}), flush=True)
            if score > best_score:
                best_score = score
                save_checkpoint(output / "best.pt", checkpoint_payload(step))
    save_checkpoint(output / "last.pt", checkpoint_payload(max_steps))


if __name__ == "__main__":
    main()
