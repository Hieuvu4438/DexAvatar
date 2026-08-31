"""Validate and summarize matched-compute deterministic restart artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture_signature(state: dict[str, torch.Tensor]) -> list[list[object]]:
    return [[name, list(tensor.shape)] for name, tensor in sorted(state.items())]


def validate_common_checkpoint(
    checkpoint: dict[str, Any], expected_seed: int
) -> dict[str, object]:
    actual_seed = checkpoint.get("training_seed")
    # Seed 42 predates the append-only training_seed field; its frozen config
    # and declared control seed are retained instead of rewriting the checkpoint.
    if actual_seed is None and expected_seed == 42:
        actual_seed = 42
    if actual_seed != expected_seed:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {expected_seed}, got {actual_seed}"
        )
    if int(checkpoint.get("step", -1)) != 10000:
        raise ValueError("A11 requires exactly 10,000 training steps per restart")
    if checkpoint.get("model_kind") != "deterministic_residual":
        raise ValueError("A11 requires deterministic residual checkpoints")
    model = checkpoint.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("Checkpoint has no model state")
    validation = checkpoint.get("best_validation_metric")
    if validation is None:
        validation = checkpoint.get("validation", {}).get(
            "deterministic_residual_mse"
        )
    if validation is None:
        raise ValueError("Checkpoint has no validation metric")
    return {
        "seed": expected_seed,
        "steps": 10000,
        "parameter_count": int(sum(t.numel() for t in model.values())),
        "architecture_signature": architecture_signature(model),
        "config_sha256": checkpoint.get("config_sha256"),
        "reliability_checkpoint_sha256": checkpoint.get(
            "reliability_checkpoint_sha256"
        ),
        "residual_statistics_sha256": checkpoint.get(
            "residual_statistics_sha256"
        ),
        "validation_residual_mse": float(validation),
    }


def require_equal(rows: list[dict[str, object]], key: str) -> object:
    values = [row[key] for row in rows]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Matched restart invariant differs: {key}")
    if values[0] in (None, ""):
        raise ValueError(f"Matched restart invariant is absent: {key}")
    return values[0]


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        metavar=("SEED", "CHECKPOINT", "PREDICTIONS", "EVIDENCE", "PSEUDO_EVAL"),
        required=True,
        help="One A11 run and its frozen development artifacts",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if len(args.run) != 3:
        raise ValueError("A11 requires exactly three matched restarts")

    rows: list[dict[str, object]] = []
    for seed_text, checkpoint_text, predictions_text, evidence_text, pseudo_text in args.run:
        seed = int(seed_text)
        checkpoint_path = Path(checkpoint_text)
        predictions_path = Path(predictions_text)
        evidence_path = Path(evidence_text)
        pseudo_path = Path(pseudo_text)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        row = validate_common_checkpoint(checkpoint, seed)
        checkpoint_hash = sha256(checkpoint_path)
        prediction_manifest_path = predictions_path / "manifest.json"
        prediction = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
        prediction_hash = sha256(prediction_manifest_path)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
        if prediction.get("flow_checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"Prediction/checkpoint hash mismatch for seed {seed}")
        if prediction.get("generator_kind") != "deterministic" or prediction.get(
            "variant"
        ) != "a3_deterministic":
            raise ValueError(f"Prediction is not deterministic A3 for seed {seed}")
        if evidence.get("prediction_manifest_sha256") != prediction_hash:
            raise ValueError(f"Evidence/prediction hash mismatch for seed {seed}")
        if pseudo.get("inference_manifest_sha256") != prediction_hash:
            raise ValueError(f"Pseudo evaluation/prediction hash mismatch for seed {seed}")
        if evidence.get("target_reads") != 0:
            raise ValueError(f"Target-free evidence read targets for seed {seed}")
        if evidence.get("config_sha256") != row["config_sha256"]:
            raise ValueError(f"Evidence/config hash mismatch for seed {seed}")
        row.update(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "prediction_manifest_sha256": prediction_hash,
                "target_reads": 0,
                "selected_observation": float(evidence["selected_observation"]),
                "observation_delta": float(
                    evidence["clustered_selected_minus_base_observation"]["delta"]
                ),
                "selected_motion": float(evidence["selected_motion"]),
                "motion_delta": float(
                    evidence["clustered_selected_minus_base_motion"]["delta"]
                ),
                "pseudo_selected_overall_degrees": float(
                    pseudo["selected_overall_degrees"]
                ),
                "pseudo_delta_degrees": float(
                    pseudo["clustered_selected_minus_base"]["delta_degrees"]
                ),
            }
        )
        rows.append(row)

    rows.sort(key=lambda row: int(row["seed"]))
    if [row["seed"] for row in rows] != [42, 43, 44]:
        raise ValueError("A11 seed set must be exactly 42, 43 and 44")
    invariants = {
        key: require_equal(rows, key)
        for key in (
            "steps",
            "parameter_count",
            "architecture_signature",
            "config_sha256",
            "reliability_checkpoint_sha256",
            "residual_statistics_sha256",
        )
    }
    summary: dict[str, object] = {
        "role": "a11_matched_compute_deterministic_restart_audit",
        "development_only": True,
        "protocol_evaluation_reused": False,
        "seeds": [42, 43, 44],
        "runs": rows,
        "matched_invariants": invariants,
    }
    for key in (
        "validation_residual_mse",
        "selected_observation",
        "observation_delta",
        "selected_motion",
        "motion_delta",
        "pseudo_selected_overall_degrees",
        "pseudo_delta_degrees",
    ):
        summary[f"{key}_distribution"] = distribution(
            [float(row[key]) for row in rows]
        )
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
