"""Evaluate a corrected R2 checkpoint and emit fail-closed P3-G2 metrics."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from phase3_posterior.config import load_config
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.models.relation_baseline import GeometryOnlyRelationMLP
from phase3_posterior.models.relation_graph import RelationGraphEncoder
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.relation_evaluation import (
    RelationMetricAccumulator,
    g2_metrics,
    relation_edge_masks,
)


def evaluate_models(
    config: dict,
    checkpoint: dict,
    manifest: str,
    device: torch.device,
) -> dict:
    width = int(config["model"].get("relation_width", 128))
    layers = int(config["model"].get("relation_layers", 3))
    graph = RelationGraphEncoder(
        width,
        layers,
        predict_distance=True,
        edge_identity=True,
    ).to(device)
    baseline = GeometryOnlyRelationMLP(width, predict_distance=True).to(device)
    no_persistence = RelationGraphEncoder(
        width,
        layers,
        predict_distance=True,
        edge_identity=True,
    ).to(device)
    graph.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    baseline.load_state_dict(
        checkpoint.get("baseline_ema_model", checkpoint["baseline_model"])
    )
    no_persistence.load_state_dict(
        checkpoint.get(
            "no_persistence_ema_model", checkpoint["no_persistence_model"]
        )
    )
    models = {
        "graph": graph.eval(),
        "geometry_mlp": baseline.eval(),
        "no_persistence": no_persistence.eval(),
    }
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
    threshold = float(config["evaluation"].get("contact_threshold", 0.5))
    accumulators = {
        name: RelationMetricAccumulator(threshold) for name in models
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
                    outputs = model(
                        batch["edge_features"],
                        batch["edge_index"],
                        batch["edge_valid"],
                    )
                accumulators[name].update(outputs, batch, masks)
    results = {name: accumulator.result() for name, accumulator in accumulators.items()}
    return {
        "schema_version": 1,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest),
        "checkpoint_step": int(checkpoint["step"]),
        "threshold": threshold,
        "results": results,
        "metrics_for_gate": g2_metrics(
            results["graph"], results["geometry_mlp"], results["no_persistence"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = args.manifest or config["data"]["val_index"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_models(config, checkpoint, manifest, device)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
