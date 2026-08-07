"""Formal evaluation for the temporal-contact P3-G2 recovery track."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from phase3_posterior.config import load_config
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.relation_evaluation import (
    RelationMetricAccumulator,
    g2_metrics,
    relation_edge_masks,
)
from phase3_posterior.train_relation_v3 import _forward_model, build_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--manifest",
        help="Optional sealed evaluation manifest; defaults to config data.val_index.",
    )
    parser.add_argument(
        "--weights",
        choices=("ema", "model"),
        default="ema",
        help="Evaluate EMA weights (formal default) or live weights (diagnostic only).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    init_path = config["initialization"]["path"]
    if sha256_file(init_path) != config["initialization"]["sha256"]:
        raise RuntimeError("Frozen v2b initialization hash mismatch")
    initialization = torch.load(init_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_models(config, initialization, device)
    graph_key = "ema_model" if args.weights == "ema" else "model"
    no_persistence_key = (
        "no_persistence_ema_model"
        if args.weights == "ema"
        else "no_persistence_model"
    )
    models["graph"].load_state_dict(checkpoint[graph_key], strict=True)
    models["geometry_mlp"].load_state_dict(checkpoint["baseline_model"], strict=True)
    models["no_persistence"].load_state_dict(
        checkpoint[no_persistence_key],
        strict=True,
    )
    for model in models.values():
        model.eval()
    manifest = args.manifest or config["data"]["val_index"]
    dataset = Phase3Dataset(
        manifest,
        int(config["model"]["max_frames"]),
        training=False,
        seed=int(config.get("seed", 42)),
        require_relation_targets=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("validation_batch_size", 8)),
        shuffle=False,
        num_workers=int(config["training"].get("workers", 0)),
        collate_fn=collate_phase3,
    )
    threshold = float(config["evaluation"]["contact_threshold"])
    accumulators = {
        "graph": RelationMetricAccumulator(
            threshold, contact_logits_key="guided_contact_logits"
        ),
        "geometry_mlp": RelationMetricAccumulator(threshold),
        "no_persistence": RelationMetricAccumulator(threshold),
    }
    with torch.inference_mode():
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
    results = {name: item.result() for name, item in accumulators.items()}
    metrics = g2_metrics(
        results["graph"], results["geometry_mlp"], results["no_persistence"]
    )
    result = {
        "schema_version": 1,
        "stage": "R2 temporal contact recovery v3",
        "checkpoint_step": int(checkpoint["step"]),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "threshold": threshold,
        "contact_score": (
            "contact_logits + "
            f"{config['model']['temporal_contact']['persistence_fusion_weight']} "
            "* persistence_logits"
        ),
        "evaluation_scope": "explicit_manifest" if args.manifest else "config_validation",
        "weight_source": args.weights,
        "results": results,
        "metrics_for_gate": metrics,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
