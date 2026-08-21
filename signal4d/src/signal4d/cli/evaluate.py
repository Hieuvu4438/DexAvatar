from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from ..data.manifest import load_manifest
from ..evaluation.evaluator import evaluate_prediction
from ..evaluation.tables import write_results
from ..io.predictions import PredictionArtifact


def run(manifest_path: str, prediction_root: str, output: str) -> None:
    manifest = load_manifest(manifest_path)
    rows = []
    for item in manifest:
        prediction, _ = PredictionArtifact.load(Path(prediction_root) / item.clip_id)
        if not item.gt_relpath:
            raise ValueError(f"manifest has no ground truth for {item.clip_id}")
        ground_truth = load_file(Path(manifest_path).parent / item.gt_relpath)
        joint_count = ground_truth["joints_3d"].shape[1]
        first = max(1, joint_count // 3)
        second = max(first + 1, 2 * joint_count // 3)
        regions = {
            "body": torch.arange(0, first),
            "left_hand": torch.arange(first, second),
            "right_hand": torch.arange(second, joint_count),
        }
        rows.append(evaluate_prediction(prediction, item, ground_truth, regions))
    write_results(rows, output)
