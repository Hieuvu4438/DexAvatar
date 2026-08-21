from __future__ import annotations

import json
from pathlib import Path

import torch

from ..data.manifest import load_manifest
from ..io.predictions import PredictionArtifact

PRIMARY_METRICS = (
    "tr_v2v_upper_body_mm",
    "tr_v2v_left_hand_mm",
    "tr_v2v_right_hand_mm",
    "velocity_error",
    "acceleration_error",
    "jerk_error",
    "coverage",
)


def run(
    manifest_path: str,
    first_predictions: str,
    second_predictions: str,
    first_summary: str,
    second_summary: str,
    output: str,
    metric_relative_tolerance: float = 0.05,
    tensor_absolute_tolerance: float = 1e-6,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    tensor_max_abs: dict[str, float] = {}
    exact_artifacts = 0
    for item in manifest:
        first, first_meta = PredictionArtifact.load(Path(first_predictions) / item.clip_id)
        second, second_meta = PredictionArtifact.load(Path(second_predictions) / item.clip_id)
        first_tensors = first.tensors()
        second_tensors = second.tensors()
        if first_tensors.keys() != second_tensors.keys():
            raise ValueError(f"prediction tensor keys differ for {item.clip_id}")
        for name in first_tensors:
            left = first_tensors[name]
            right = second_tensors[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"prediction tensor contract differs for {item.clip_id}/{name}")
            difference = float((left - right).abs().max()) if left.is_floating_point() else 0.0
            if not left.is_floating_point() and not torch.equal(left, right):
                difference = float("inf")
            key = f"{item.clip_id}/{name}"
            tensor_max_abs[key] = difference
        exact_artifacts += int(first_meta["artifact_sha256"] == second_meta["artifact_sha256"])

    first_metrics = json.loads(Path(first_summary).read_text(encoding="utf-8"))
    second_metrics = json.loads(Path(second_summary).read_text(encoding="utf-8"))
    metric_relative_error = {
        metric: abs(float(first_metrics[metric]) - float(second_metrics[metric]))
        / max(abs(float(first_metrics[metric])), 1e-12)
        for metric in PRIMARY_METRICS
    }
    maximum_tensor_error = max(tensor_max_abs.values(), default=0.0)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "clips": len(manifest),
        "exact_artifacts": exact_artifacts,
        "maximum_tensor_absolute_error": maximum_tensor_error,
        "metric_relative_error": metric_relative_error,
        "tensor_absolute_tolerance": tensor_absolute_tolerance,
        "metric_relative_tolerance": metric_relative_tolerance,
        "passed": maximum_tensor_error <= tensor_absolute_tolerance
        and max(metric_relative_error.values()) <= metric_relative_tolerance,
    }
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
