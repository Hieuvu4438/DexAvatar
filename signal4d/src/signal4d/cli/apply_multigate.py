from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..data.cache import ObservationBatch
from ..data.manifest import load_manifest
from ..io.predictions import PredictionArtifact
from ..models.gating import (
    ExtraTreesArtifact,
    decode_multigate_sequence,
    extract_gate_features,
    merge_multiple_predictions,
)
from ..utils.hashing import sha256_file


def _parse_hypotheses(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("hypothesis must be LABEL=PREDICTION_ROOT")
        label, root = value.split("=", 1)
        if not label or label in result:
            raise ValueError(f"invalid or duplicate hypothesis label: {label}")
        result[label] = Path(root)
    return result


def run(
    manifest_path: str,
    baseline_root: str,
    cache_root: str,
    bundle_root: str,
    hypothesis_values: list[str],
    output_root: str,
) -> dict[str, Any]:
    bundle_path = Path(bundle_root)
    bundle = json.loads((bundle_path / "metadata.json").read_text(encoding="utf-8"))
    runtime_roots = _parse_hypotheses(hypothesis_values)
    expected_labels = [item["label"] for item in bundle["hypotheses"]]
    if set(runtime_roots) != set(expected_labels):
        raise ValueError("runtime hypothesis labels differ from the frozen multi-gate bundle")
    artifacts = []
    for item in bundle["hypotheses"]:
        root = Path(item["artifact"])
        artifact = ExtraTreesArtifact.load(root)
        if artifact.metadata["forest_sha256"] != item["forest_sha256"]:
            raise ValueError(f"multi-gate forest mismatch for {item['label']}")
        if sha256_file(root / "metadata.json") != item["metadata_sha256"]:
            raise ValueError(f"multi-gate metadata mismatch for {item['label']}")
        artifacts.append(artifact)

    output = Path(output_root)
    rows: list[dict[str, Any]] = []
    clips = load_manifest(manifest_path)
    state_counts = np.zeros(len(artifacts) + 1, dtype=np.int64)
    switches = 0
    for item in clips:
        baseline_dir = Path(baseline_root) / item.clip_id
        baseline, baseline_meta = PredictionArtifact.load(baseline_dir)
        observations, _ = ObservationBatch.load(Path(cache_root) / item.clip_id)
        observations.validate_against(item)
        candidates: list[PredictionArtifact] = []
        candidate_metadata: list[dict[str, object]] = []
        emissions = [np.zeros(len(item.frame_ids), dtype=np.float64)]
        predicted_columns = []
        for label, artifact in zip(expected_labels, artifacts, strict=True):
            candidate_dir = runtime_roots[label] / item.clip_id
            candidate, candidate_meta = PredictionArtifact.load(candidate_dir)
            diagnostics = json.loads(
                (candidate_dir / "factor_diagnostics.json").read_text(encoding="utf-8")
            )
            features, names = extract_gate_features(
                candidate, baseline, observations, diagnostics
            )
            if tuple(names) != artifact.feature_names:
                raise ValueError(f"feature schema mismatch for hypothesis {label}")
            predicted = artifact.predict(features)
            emissions.append(predicted - artifact.decision_threshold_mm)
            predicted_columns.append(predicted)
            candidates.append(candidate)
            candidate_metadata.append(candidate_meta)
        states = decode_multigate_sequence(
            np.column_stack(emissions), float(bundle["switch_penalty_mm"])
        )
        state_counts += np.bincount(states, minlength=len(state_counts))
        switches += int(np.count_nonzero(np.diff(states)))
        merged = merge_multiple_predictions([baseline, *candidates], states)
        merged.save(
            output / "predictions" / item.clip_id,
            {
                "schema_version": "1.0",
                "method_name": bundle["method_name"],
                "clip_id": item.clip_id,
                "manifest_item_sha256": item.sha256,
                "smplx_model_sha256": baseline_meta.get("smplx_model_sha256"),
                "coordinate_convention": baseline_meta.get("coordinate_convention"),
                "baseline_artifact_sha256": baseline_meta["artifact_sha256"],
                "candidate_artifact_sha256": {
                    label: metadata["artifact_sha256"]
                    for label, metadata in zip(
                        expected_labels, candidate_metadata, strict=True
                    )
                },
                "multi_gate_metadata_sha256": sha256_file(bundle_path / "metadata.json"),
                "state_counts": {
                    label: int((states == state).sum())
                    for state, label in enumerate(bundle["state_labels"])
                },
                "gt_used_for_selection": False,
            },
        )
        for frame, frame_id in enumerate(item.frame_ids):
            row: dict[str, Any] = {
                "clip_id": item.clip_id,
                "frame_id": frame_id,
                "selected_state": int(states[frame]),
                "selected_label": bundle["state_labels"][states[frame]],
            }
            for label, predicted in zip(expected_labels, predicted_columns, strict=True):
                row[f"predicted_delta_mm_{label}"] = float(predicted[frame])
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "1.0",
        "method_name": bundle["method_name"],
        "clips": len(clips),
        "frames": len(rows),
        "manifest_sha256": sha256_file(manifest_path),
        "multi_gate_metadata_sha256": sha256_file(bundle_path / "metadata.json"),
        "state_counts": {
            label: int(state_counts[state])
            for state, label in enumerate(bundle["state_labels"])
        },
        "switches": switches,
        "gt_used_for_selection": False,
    }
    (output / "gate_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
