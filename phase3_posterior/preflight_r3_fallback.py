"""Fail-closed preflight for R2-geometry-only to R3 progression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_posterior.config import load_config
from phase3_posterior.data.cache_schema import load_index
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.training import load_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--relation-checkpoint", required=True)
    parser.add_argument("--g2-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    blockers = []
    fallback = config.get("fallback", {})
    decision = json.loads(Path(args.g2_decision).read_text(encoding="utf-8"))
    if decision.get("gate") != "P3-G2" or decision.get("passed") is not False:
        blockers.append("source P3-G2 decision is not the required formal NO-GO")
    expected_failures = {"sign_contact_f1", "contact_slip_gain"}
    actual_failures = {
        key
        for key, value in decision.get("checks", {}).items()
        if not value.get("passed", False)
    }
    if actual_failures != expected_failures:
        blockers.append(
            f"unexpected P3-G2 failure set: {sorted(actual_failures)}"
        )
    checkpoint_hash = sha256_file(args.relation_checkpoint)
    decision_hash = sha256_file(args.g2_decision)
    if checkpoint_hash != fallback.get("relation_checkpoint_sha256"):
        blockers.append("relation checkpoint hash does not match frozen fallback config")
    if decision_hash != fallback.get("source_gate_decision_sha256"):
        blockers.append("P3-G2 decision hash does not match frozen fallback config")
    model = RelationalDiffusionPosterior(config["model"])
    try:
        payload = load_weights(
            model.relation_graph, args.relation_checkpoint, strict=True
        )
    except (RuntimeError, ValueError) as error:
        blockers.append(f"strict relation initialization failed: {error}")
        payload = {}
    if int(payload.get("step", -1)) != int(fallback.get("relation_checkpoint_step", -2)):
        blockers.append("relation checkpoint step does not match frozen fallback config")
    if model.contact_energy_enabled:
        blockers.append("model contact energy is enabled")
    if float(config.get("loss", {}).get("contact", -1.0)) != 0.0:
        blockers.append("contact loss is nonzero")
    if float(config.get("loss", {}).get("persistence", -1.0)) != 0.0:
        blockers.append("persistence loss is nonzero")
    training = config["training"]
    if training.get("conditioning_validity") != "frame_valid_initializer":
        blockers.append("masked conditioning is not bound to the full initializer")
    if int(training.get("validation_interval", 0)) <= 0:
        blockers.append("validation-based checkpoint selection is disabled")
    if int(training.get("validation_max_batches", 0)) <= 0:
        blockers.append("deterministic validation has no batches")
    if int(training.get("validation_sampling_steps", 0)) <= 0:
        blockers.append("deterministic validation sampler is disabled")
    if int(training.get("early_stop_patience", 0)) <= 0:
        blockers.append("validation early stopping is disabled")
    warm_start = config["model"].get(
        "warm_start_conditioning", "unconditional_v3"
    )
    if warm_start == "unconditional_v3":
        projection_initialization_ok = (
            config["model"].get("reset_conditioning_projections_on_init") is True
        )
    elif warm_start == "conditional_v4b":
        projection_initialization_ok = (
            config["model"].get("reset_conditioning_projections_on_init") is False
            and config["model"].get("masked_rotation_hints") is True
            and float(
                config["training"].get(
                    "masked_rotation_corruption_degrees", 0.0
                )
            )
            > 0
        )
    else:
        projection_initialization_ok = False
    if not projection_initialization_ok:
        blockers.append("warm-start conditioning initialization is invalid")
    identities: dict[tuple[str, str, str], str] = {}
    split_counts = {}
    source_counts = {}
    for split, key in (("train", "train_index"), ("val", "val_index")):
        entries = load_index(config["data"][key])
        split_counts[split] = len(entries)
        for entry in entries:
            source_counts[entry.source] = source_counts.get(entry.source, 0) + 1
            if entry.source not in {"arctic", "interhand26m"}:
                blockers.append(f"non-Tier-A/B source in R3 {split}: {entry.source}")
            for kind, value in (
                ("signer", entry.signer),
                ("source_group", entry.source_group),
            ):
                identity = (entry.source, kind, value)
                previous = identities.setdefault(identity, split)
                if previous != split:
                    blockers.append(
                        f"{kind} leakage for {entry.source}:{value}: {previous}/{split}"
                    )
    checks = {
        "formal_g2_no_go_acknowledged": decision.get("passed") is False,
        "exact_expected_failure_set": actual_failures == expected_failures,
        "relation_checkpoint_hash_locked": checkpoint_hash
        == fallback.get("relation_checkpoint_sha256"),
        "relation_checkpoint_strict_load": bool(payload),
        "relation_checkpoint_step_36000": int(payload.get("step", -1)) == 36000,
        "contact_energy_disabled": not model.contact_energy_enabled,
        "contact_loss_zero": float(config.get("loss", {}).get("contact", -1.0))
        == 0.0,
        "persistence_loss_zero": float(
            config.get("loss", {}).get("persistence", -1.0)
        )
        == 0.0,
        "force_coupling_disabled": fallback.get("force_coupling_enabled") is False,
        "persistence_constraints_disabled": fallback.get(
            "persistence_constraints_enabled"
        )
        is False,
        "source_signer_group_disjoint": not any("leakage" in item for item in blockers),
        "tier_ab_only": set(source_counts) <= {"arctic", "interhand26m"},
        "frame_valid_initializer_conditioning": training.get(
            "conditioning_validity"
        )
        == "frame_valid_initializer",
        "validation_based_checkpoint_selection": int(
            training.get("validation_interval", 0)
        )
        > 0
        and int(training.get("validation_max_batches", 0)) > 0
        and int(training.get("validation_sampling_steps", 0)) > 0,
        "validation_early_stopping": int(
            training.get("early_stop_patience", 0)
        )
        > 0,
        "warm_start_conditioning_initialization": projection_initialization_ok,
    }
    result = {
        "schema_version": 1,
        "pipeline_id": "R2_geometry_only_R3_progression",
        "passed": not blockers and all(checks.values()),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "checks": checks,
        "split_counts": split_counts,
        "source_counts": source_counts,
        "relation_checkpoint_sha256": checkpoint_hash,
        "g2_decision_sha256": decision_hash,
        "config_sha256": sha256_file(args.config),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
