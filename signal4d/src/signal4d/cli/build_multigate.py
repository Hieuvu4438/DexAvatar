from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..models.gating import ExtraTreesArtifact, decode_multigate_sequence
from ..utils.hashing import sha256_file


def _load_oof(root: Path) -> tuple[list[tuple[str, str, int]], np.ndarray, np.ndarray]:
    with (root / "oof_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["dataset"], row["clip_id"], int(row["frame_id"])) for row in rows]
    prediction = np.asarray(
        [float(row["predicted_candidate_minus_baseline_mm"]) for row in rows]
    )
    target = np.asarray(
        [float(row["observed_candidate_minus_baseline_mm"]) for row in rows]
    )
    return keys, prediction, target


def run(config_path: str, output: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("hypotheses"), list):
        raise ValueError("multi-gate config requires a hypotheses list")
    switch_penalty = float(config["switch_penalty_mm"])
    seed = int(config.get("seed", 12345))
    hypothesis_metadata: list[dict[str, Any]] = []
    reference_keys: list[tuple[str, str, int]] | None = None
    predictions = []
    targets = []
    for hypothesis in config["hypotheses"]:
        root = Path(hypothesis["artifact"])
        artifact = ExtraTreesArtifact.load(root)
        keys, prediction, target = _load_oof(root)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("multi-gate OOF rows are not exactly paired")
        predictions.append(prediction - artifact.decision_threshold_mm)
        targets.append(target)
        hypothesis_metadata.append(
            {
                "label": str(hypothesis["label"]),
                "artifact": str(root),
                "forest_sha256": artifact.metadata["forest_sha256"],
                "metadata_sha256": sha256_file(root / "metadata.json"),
                "oof_predictions_sha256": sha256_file(root / "oof_predictions.csv"),
                "decision_threshold_mm": artifact.decision_threshold_mm,
            }
        )
    if reference_keys is None:
        raise ValueError("multi-gate requires at least one non-baseline hypothesis")
    groups = np.asarray([f"{dataset}:{clip}" for dataset, clip, _ in reference_keys])
    emission = np.column_stack((np.zeros(len(groups)), *predictions))
    target_matrix = np.column_stack((np.zeros(len(groups)), *targets))
    states = np.zeros(len(groups), dtype=np.int64)
    for group in dict.fromkeys(groups.tolist()):
        indices = np.flatnonzero(groups == group)
        states[indices] = decode_multigate_sequence(emission[indices], switch_penalty)
    observed = target_matrix[np.arange(len(groups)), states]
    clip_values = np.asarray(
        [observed[groups == group].mean() for group in dict.fromkeys(groups.tolist())]
    )
    generator = np.random.default_rng(seed)
    bootstrap = generator.choice(
        clip_values, size=(10000, len(clip_values)), replace=True
    ).mean(1)
    state_count = len(hypothesis_metadata) + 1
    report = {
        "schema_version": "1.0",
        "method_name": "signal4d_m1_multiscale_gt_free_gate",
        "baseline_label": str(config.get("baseline_label", "legacy_full_fallback")),
        "hypotheses": hypothesis_metadata,
        "state_labels": [
            str(config.get("baseline_label", "legacy_full_fallback")),
            *[item["label"] for item in hypothesis_metadata],
        ],
        "switch_penalty_mm": switch_penalty,
        "development_selection_basis": "paired out_of_fold_predictions_only",
        "oof_clips": len(set(groups)),
        "oof_frames": len(groups),
        "oof_clip_macro_delta_mm": float(clip_values.mean()),
        "oof_frame_micro_delta_mm": float(observed.mean()),
        "oof_ci95_clip_bootstrap_mm": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "oof_state_frame_counts": {
            str(state): int((states == state).sum()) for state in range(state_count)
        },
        "oof_switches": int(
            sum(
                np.count_nonzero(np.diff(states[groups == group]))
                for group in dict.fromkeys(groups.tolist())
            )
        ),
        "config_sha256": sha256_file(config_path),
        "gt_used_at_inference": False,
        "seed": seed,
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_path / "oof_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "clip_id",
            "frame_id",
            "selected_state",
            "selected_label",
            "observed_candidate_minus_baseline_mm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, state, delta in zip(reference_keys, states, observed, strict=True):
            writer.writerow(
                {
                    "dataset": key[0],
                    "clip_id": key[1],
                    "frame_id": key[2],
                    "selected_state": int(state),
                    "selected_label": report["state_labels"][state],
                    "observed_candidate_minus_baseline_mm": float(delta),
                }
            )
    return report
