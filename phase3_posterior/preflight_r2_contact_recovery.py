"""Fail-closed preflight for the temporal-contact P3-G2 recovery track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase3_posterior.config import load_config
from phase3_posterior.data.cache_schema import load_index, load_relation_sidecar
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.relation_evaluation import relation_edge_masks
from phase3_posterior.train_relation_v3 import build_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--g2-decision", required=True)
    parser.add_argument("--smoke-completion", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    decision = json.loads(Path(args.g2_decision).read_text(encoding="utf-8"))
    failures = {
        key for key, value in decision["checks"].items() if not value["passed"]
    }
    initialization_path = config["initialization"]["path"]
    initialization_hash = sha256_file(initialization_path)
    initialization = torch.load(
        initialization_path, map_location="cpu", weights_only=False
    )
    models = build_models(config, initialization, torch.device("cpu"))
    graph = models["graph"]
    identities: dict[tuple[str, str, str], str] = {}
    split_counts = {}
    source_counts: dict[str, int] = {}
    positive_sign_edges = {"train": 0, "val": 0}
    leakage = []
    for split, key in (("train", "train_index"), ("val", "val_index")):
        entries = load_index(config["data"][key])
        split_counts[split] = len(entries)
        for entry in entries:
            source_counts[entry.source] = source_counts.get(entry.source, 0) + 1
            for kind, value in (
                ("signer", entry.signer),
                ("source_group", entry.source_group),
            ):
                identity = (entry.source, kind, value)
                previous = identities.setdefault(identity, split)
                if previous != split:
                    leakage.append(f"{entry.source}:{kind}:{value}")
            if entry.source == "how2sign":
                relation = load_relation_sidecar(entry.relation_path)
                masks = relation_edge_masks(torch.from_numpy(relation.edge_index))
                sign_contact = (
                    torch.from_numpy(relation.contact_valid)
                    & torch.from_numpy(relation.contact_target)
                    & masks["hand_body"][None, :]
                )
                positive_sign_edges[split] += int(sign_contact.sum())
    smoke = json.loads(Path(args.smoke_completion).read_text(encoding="utf-8"))
    temporal = config["model"]["temporal_contact"]
    adapter_initialization = config.get("adapter_initialization")
    adapter_hash_ok = True
    if adapter_initialization:
        adapter_hash_ok = (
            sha256_file(adapter_initialization["path"])
            == adapter_initialization["sha256"]
        )
    checks = {
        "formal_v2b_no_go_acknowledged": decision.get("gate") == "P3-G2"
        and decision.get("passed") is False,
        "exact_failure_set": failures
        == {"sign_contact_f1", "contact_slip_gain"},
        "initialization_hash_locked": initialization_hash
        == config["initialization"]["sha256"],
        "initialization_step_36000": int(initialization.get("step", -1)) == 36000,
        "geometry_backbone_frozen": not any(
            parameter.requires_grad for parameter in graph.backbone.parameters()
        ),
        "temporal_identity_initialization": int(
            torch.count_nonzero(graph.temporal_projection.weight)
        )
        == 0,
        "adapter_initialization_hash_locked": adapter_hash_ok,
        "contact_encoder_policy": (
            graph.contact_encoder is not None
            and (
                not any(
                    parameter.requires_grad
                    for parameter in graph.contact_encoder.parameters()
                )
                if temporal.get("observation_only_training", False)
                else any(
                    parameter.requires_grad
                    for parameter in graph.contact_encoder.parameters()
                )
            )
            if temporal.get("train_contact_encoder", False)
            else graph.contact_encoder is None
        ),
        "observation_recovery_branch_trainable": (
            not temporal.get("observation_features", False)
            or (
                (
                    graph.observation_encoder is not None
                    and any(
                        parameter.requires_grad
                        for parameter in graph.observation_encoder.parameters()
                    )
                    and graph.observation_contact_delta is not None
                    and any(
                        parameter.requires_grad
                        for parameter in graph.observation_contact_delta.parameters()
                    )
                )
                if temporal.get("observation_logit_residual", False)
                else graph.observation_projection is not None
                and any(
                    parameter.requires_grad
                    for parameter in graph.observation_projection.parameters()
                )
            )
        ),
        "contained_base_frozen_when_requested": (
            not temporal.get("observation_only_training", False)
            or all(
                not parameter.requires_grad
                for name, parameter in graph.named_parameters()
                if not name.startswith(
                    ("observation_encoder.", "observation_contact_delta.")
                )
            )
        ),
        "zero_initialized_branch_ema_adapts": (
            not temporal.get("observation_only_training", False)
            or float(config["training"].get("ema", 0.999)) <= 0.99
        ),
        "observation_delta_hand_body_contained": (
            not temporal.get("observation_logit_residual", False)
            or bool(temporal.get("observation_hand_body_only", False))
        ),
        "persistence_used_explicitly": float(
            temporal["persistence_fusion_weight"]
        )
        > 0,
        "threshold_frozen": float(config["evaluation"]["contact_threshold"])
        == 0.5
        and config["evaluation"]["threshold_policy"]
        == "frozen_before_v3_validation",
        "source_signer_group_disjoint": not leakage,
        "no_lane_l_data": all(
            "evaluation_from_author" not in config["data"][key]
            and "smplx_gt" not in config["data"][key]
            for key in ("train_index", "val_index")
        ),
        "sign_positive_support": positive_sign_edges["train"] >= 2000
        and positive_sign_edges["val"] >= 200,
        "cpu_worker_cap": int(config["training"].get("workers", 0)) <= 4,
        "smoke_completed": int(smoke.get("completed_step", 0)) == 2,
    }
    blockers = [key for key, passed in checks.items() if not passed]
    result = {
        "schema_version": 1,
        "pipeline_id": "R2_temporal_contact_recovery_v3",
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "checks": checks,
        "split_counts": split_counts,
        "source_counts": source_counts,
        "positive_sign_hand_body_edges": positive_sign_edges,
        "config_sha256": sha256_file(args.config),
        "initialization_sha256": initialization_hash,
        "source_g2_decision_sha256": sha256_file(args.g2_decision),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
