from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..data.cache import ObservationBatch
from ..data.manifest import load_manifest
from ..io.predictions import PredictionArtifact
from ..models.gating import (
    ExtraTreesArtifact,
    decode_gate_sequence,
    extract_gate_features,
    merge_predictions,
)
from ..utils.hashing import sha256_file


def run(
    manifest_path: str,
    candidate_root: str,
    baseline_root: str,
    cache_root: str,
    artifact_root: str,
    output_root: str,
) -> dict[str, Any]:
    artifact = ExtraTreesArtifact.load(artifact_root)
    output = Path(output_root)
    selection_rows: list[dict[str, Any]] = []
    clips = load_manifest(manifest_path)
    for item in clips:
        candidate_dir = Path(candidate_root) / item.clip_id
        baseline_dir = Path(baseline_root) / item.clip_id
        cache_dir = Path(cache_root) / item.clip_id
        candidate, candidate_meta = PredictionArtifact.load(candidate_dir)
        baseline, baseline_meta = PredictionArtifact.load(baseline_dir)
        observations, _ = ObservationBatch.load(cache_dir)
        observations.validate_against(item)
        diagnostics_path = candidate_dir / "factor_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        features, names = extract_gate_features(
            candidate, baseline, observations, diagnostics
        )
        if tuple(names) != artifact.feature_names:
            raise ValueError("runtime gate feature schema differs from frozen artifact")
        predicted_delta = artifact.predict(features)
        selected = decode_gate_sequence(
            predicted_delta,
            artifact.decision_threshold_mm,
            artifact.switch_penalty_mm,
        )
        merged = merge_predictions(candidate, baseline, selected)
        merged.save(
            output / "predictions" / item.clip_id,
            {
                "schema_version": "1.0",
                "method_name": "signal4d_m1_gt_free_gate",
                "clip_id": item.clip_id,
                "manifest_item_sha256": item.sha256,
                "smplx_model_sha256": candidate_meta.get("smplx_model_sha256"),
                "coordinate_convention": candidate_meta.get("coordinate_convention"),
                "candidate_artifact_sha256": candidate_meta["artifact_sha256"],
                "baseline_artifact_sha256": baseline_meta["artifact_sha256"],
                "gate_forest_sha256": artifact.metadata["forest_sha256"],
                "gate_metadata_sha256": sha256_file(Path(artifact_root) / "metadata.json"),
                "candidate_frames": int(selected.sum()),
                "baseline_frames": int((~selected).sum()),
                "gt_used_for_selection": False,
            },
        )
        for frame_id, prediction, choice in zip(
            item.frame_ids, predicted_delta, selected, strict=True
        ):
            selection_rows.append(
                {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "predicted_candidate_minus_baseline_mm": float(prediction),
                    "selected_candidate": int(choice),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    report = {
        "schema_version": "1.0",
        "clips": len(clips),
        "frames": len(selection_rows),
        "candidate_frames": sum(row["selected_candidate"] for row in selection_rows),
        "baseline_frames": sum(1 - row["selected_candidate"] for row in selection_rows),
        "manifest_sha256": sha256_file(manifest_path),
        "gate_forest_sha256": artifact.metadata["forest_sha256"],
        "gt_used_for_selection": False,
    }
    (output / "gate_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
