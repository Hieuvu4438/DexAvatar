"""Contact-focused R2 recovery with temporal persistence and balanced sign loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phase3_posterior.config import load_config
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.losses.relation import (
    conditional_persistence_loss,
    focal_bce,
    stratified_sign_contact_loss,
)
from phase3_posterior.models.relation_baseline import GeometryOnlyRelationMLP
from phase3_posterior.models.relation_graph import RelationGraphEncoder
from phase3_posterior.models.temporal_contact import TemporalContactRefiner
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.relation_evaluation import (
    RelationMetricAccumulator,
    g2_metrics,
    relation_edge_masks,
)
from phase3_posterior.train_relation_v2 import _mixture_loader, _relation_weights
from phase3_posterior.training import (
    ExponentialMovingAverage,
    cosine_warmup_scheduler,
    prepare_run,
    rng_state,
    save_checkpoint,
    seed_everything,
)


def _base_model(config: dict, device: torch.device) -> RelationGraphEncoder:
    return RelationGraphEncoder(
        int(config["model"].get("relation_width", 128)),
        int(config["model"].get("relation_layers", 3)),
        predict_distance=True,
        edge_identity=True,
    ).to(device)


def build_models(
    config: dict,
    initialization: dict,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    width = int(config["model"].get("relation_width", 128))
    frozen_state = initialization.get("ema_model", initialization["model"])
    graph_base = _base_model(config, device)
    graph_base.load_state_dict(frozen_state)
    ablation_base = _base_model(config, device)
    # The temporal graph and its no-persistence comparator start identically.
    ablation_base.load_state_dict(frozen_state)
    temporal = config["model"]["temporal_contact"]
    graph = TemporalContactRefiner(
        graph_base,
        width=width,
        temporal_hidden=int(temporal.get("hidden", width)),
        persistence_fusion_weight=float(temporal["persistence_fusion_weight"]),
        train_contact_encoder=bool(temporal.get("train_contact_encoder", False)),
        observation_features=bool(temporal.get("observation_features", False)),
        observation_graph_layers=int(temporal.get("observation_graph_layers", 0)),
        observation_logit_residual=bool(
            temporal.get("observation_logit_residual", False)
        ),
        observation_only_training=bool(
            temporal.get("observation_only_training", False)
        ),
        observation_hand_body_only=bool(
            temporal.get("observation_hand_body_only", False)
        ),
    ).to(device)
    no_persistence = TemporalContactRefiner(
        ablation_base,
        width=width,
        temporal_hidden=int(temporal.get("hidden", width)),
        persistence_fusion_weight=0.0,
        train_contact_encoder=bool(temporal.get("train_contact_encoder", False)),
        observation_features=bool(temporal.get("observation_features", False)),
        observation_graph_layers=int(temporal.get("observation_graph_layers", 0)),
        observation_logit_residual=bool(
            temporal.get("observation_logit_residual", False)
        ),
        observation_only_training=bool(
            temporal.get("observation_only_training", False)
        ),
        observation_hand_body_only=bool(
            temporal.get("observation_hand_body_only", False)
        ),
    ).to(device)
    baseline = GeometryOnlyRelationMLP(width, predict_distance=True).to(device)
    baseline.load_state_dict(
        initialization.get("baseline_ema_model", initialization["baseline_model"])
    )
    baseline.requires_grad_(False)
    baseline.eval()
    return {
        "graph": graph,
        "geometry_mlp": baseline,
        "no_persistence": no_persistence,
    }


def _forward_model(model: torch.nn.Module, batch: dict) -> dict[str, torch.Tensor]:
    kwargs = {}
    if isinstance(model, TemporalContactRefiner) and (
        model.observation_projection is not None or model.observation_encoder is not None
    ):
        kwargs = {
            "observation_edge_features": batch["observation_edge_features"],
            "observation_edge_valid": batch["observation_edge_valid"],
        }
    return model(
        batch["edge_features"], batch["edge_index"], batch["edge_valid"], **kwargs
    )


@torch.inference_mode()
def validate(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict:
    accumulators = {
        "graph": RelationMetricAccumulator(
            threshold, contact_logits_key="guided_contact_logits"
        ),
        "geometry_mlp": RelationMetricAccumulator(threshold),
        "no_persistence": RelationMetricAccumulator(threshold),
    }
    modes = {name: model.training for name, model in models.items()}
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
                outputs = _forward_model(model, batch)
            accumulators[name].update(outputs, batch, masks)
    for name, model in models.items():
        model.train(modes[name])
    results = {name: item.result() for name, item in accumulators.items()}
    return {
        "results": results,
        "metrics_for_gate": g2_metrics(
            results["graph"],
            results["geometry_mlp"],
            results["no_persistence"],
        ),
    }


def _contact_losses(
    model: torch.nn.Module,
    batch: dict,
    sign_hand_body: torch.Tensor,
    contact_weight: torch.Tensor,
    config: dict,
    *,
    persistence_enabled: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.autocast(
        device_type=batch["edge_features"].device.type,
        dtype=torch.bfloat16,
        enabled=batch["edge_features"].is_cuda,
    ):
        outputs = _forward_model(model, batch)
        overall = focal_bce(
            outputs["contact_logits"],
            batch["contact_target"],
            batch["contact_valid"],
            positive_alpha=float(config["training"].get("contact_positive_alpha", 0.75)),
        )
        sign = stratified_sign_contact_loss(
            outputs["contact_logits"],
            batch["contact_target"],
            sign_hand_body,
            hard_negative_ratio=int(
                config["training"].get("sign_hard_negative_ratio", 8)
            ),
        )
        persistence = (
            conditional_persistence_loss(
                outputs["persistence_logits"],
                batch["persistence_target"],
                batch["contact_target"],
                batch["contact_valid"],
                weight=contact_weight,
            )
            if persistence_enabled
            else outputs["persistence_logits"].sum() * 0.0
        )
        total = (
            float(config["loss"].get("contact", 1.0)) * overall
            + float(config["loss"].get("sign_contact", 4.0)) * sign
            + float(config["loss"].get("persistence", 1.0)) * persistence
        )
    return total, {
        "loss": float(total.detach()),
        "contact": float(overall.detach()),
        "sign_contact": float(sign.detach()),
        "persistence": float(persistence.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    output = prepare_run(config, args.config)
    initialization = torch.load(args.init, map_location="cpu", weights_only=False)
    expected_hash = config["initialization"]["sha256"]
    if sha256_file(args.init) != expected_hash:
        raise RuntimeError("R2 recovery initialization hash mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Phase3Dataset(
        config["data"]["train_index"],
        int(config["model"]["max_frames"]),
        training=True,
        seed=seed,
        require_relation_targets=True,
    )
    val_dataset = Phase3Dataset(
        config["data"]["val_index"],
        int(config["model"]["max_frames"]),
        training=False,
        seed=seed,
        require_relation_targets=True,
    )
    workers = int(config["training"].get("workers", 0))
    batch_size = int(config["training"].get("batch_size", 8))
    train_loader = _mixture_loader(
        train_dataset,
        config["training"]["joint_mixture"],
        batch_size,
        workers,
        seed,
        float(config["training"]["sign_contact_clip_probability"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("validation_batch_size", 8)),
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_phase3,
    )
    models = build_models(config, initialization, device)
    adapter_initialization = config.get("adapter_initialization")
    if adapter_initialization:
        adapter_path = adapter_initialization["path"]
        if sha256_file(adapter_path) != adapter_initialization["sha256"]:
            raise RuntimeError("R2 adapter initialization hash mismatch")
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
        for name, key in (
            ("graph", "ema_model"),
            ("no_persistence", "no_persistence_ema_model"),
        ):
            incompatible = models[name].load_state_dict(adapter[key], strict=False)
            expected_missing = {
                item
                for item in models[name].state_dict()
                if item not in adapter[key]
                and item.startswith(
                    (
                        "contact_encoder.",
                        "observation_projection.",
                        "observation_encoder.",
                        "observation_graph_gate.",
                        "observation_contact_delta.",
                    )
                )
            }
            if set(incompatible.missing_keys) != expected_missing:
                raise RuntimeError(
                    f"Unexpected adapter missing keys for {name}: "
                    f"{incompatible.missing_keys}"
                )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected adapter keys for {name}: "
                    f"{incompatible.unexpected_keys}"
                )
    trainable = {key: models[key] for key in ("graph", "no_persistence")}
    max_steps = int(config["training"]["max_steps"])
    optimizers = {
        name: torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(config["training"].get("learning_rate", 1e-4)),
            weight_decay=float(config["training"].get("weight_decay", 0.01)),
        )
        for name, model in trainable.items()
    }
    schedulers = {
        name: cosine_warmup_scheduler(optimizer, max_steps)
        for name, optimizer in optimizers.items()
    }
    emas = {
        name: ExponentialMovingAverage(
            model, float(config["training"].get("ema", 0.999))
        )
        for name, model in trainable.items()
    }
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    validation_interval = int(config["training"].get("validation_interval", 1000))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    threshold = float(config["evaluation"]["contact_threshold"])
    patience = int(config["training"].get("early_stop_patience", 8))
    iterator = iter(train_loader)
    best_score = float("-inf")
    best_step = None
    stale = 0
    completed_step = 0
    last_validation: dict = {}
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    def payload(step: int) -> dict:
        return {
            "model": models["graph"].state_dict(),
            "ema_model": emas["graph"].state,
            "baseline_model": models["geometry_mlp"].state_dict(),
            "no_persistence_model": models["no_persistence"].state_dict(),
            "no_persistence_ema_model": emas["no_persistence"].state,
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
            "step": step,
            "best_step": best_step,
            "config": config,
            "rng_state": rng_state(),
            "validation": last_validation,
            "initialization": {
                "path": str(Path(args.init).resolve()),
                "sha256": expected_hash,
                "geometry_frozen": True,
                "adapter": adapter_initialization,
            },
        }

    for micro_step in range(1, max_steps * accumulation + 1):
        step = (micro_step + accumulation - 1) // accumulation
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        source = batch["source"]
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batch["source"] = source
        edge_masks = relation_edge_masks(batch["edge_index"][0])
        sign = torch.tensor(
            [item == "how2sign" for item in source], device=device
        )[:, None, None]
        sign_hand_body = (
            batch["contact_valid"]
            & sign
            & edge_masks["hand_body"][None, None, :]
        )
        contact_weight = _relation_weights(
            batch, float(config["training"].get("sign_hand_body_weight", 4.0))
        )
        losses = {}
        for name, model in trainable.items():
            total, values = _contact_losses(
                model,
                batch,
                sign_hand_body,
                contact_weight,
                config,
                persistence_enabled=name == "graph",
            )
            (total / accumulation).backward()
            losses[name] = values
        if micro_step % accumulation:
            continue
        for name, model in trainable.items():
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            optimizers[name].step()
            schedulers[name].step()
            emas[name].update(model)
            optimizers[name].zero_grad(set_to_none=True)
        completed_step = step
        if step == 1 or step % int(config["training"].get("log_interval", 50)) == 0:
            print(json.dumps({"step": step, "models": losses}), flush=True)
        if step % checkpoint_interval == 0:
            save_checkpoint(output / "last.pt", payload(step))
        if step % validation_interval == 0:
            states = {}
            for name, model in trainable.items():
                states[name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(emas[name].state)
            validation = validate(models, val_loader, device, threshold)
            for name, model in trainable.items():
                model.load_state_dict(states[name])
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
            print(json.dumps({"event": "relation_v3_validation", **last_validation}), flush=True)
            if score > best_score:
                best_score = score
                best_step = step
                stale = 0
                save_checkpoint(output / "best.pt", payload(step))
            else:
                stale += 1
            if stale >= patience:
                print(
                    json.dumps(
                        {"early_stop": True, "step": step, "best_step": best_step}
                    ),
                    flush=True,
                )
                break
    save_checkpoint(output / "last.pt", payload(completed_step))
    atomic_json(
        output / "completion.json",
        {"completed_step": completed_step, "best_step": best_step},
    )


if __name__ == "__main__":
    main()
