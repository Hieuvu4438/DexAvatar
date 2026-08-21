import json

import torch

from signal4d.cli.verify_repro import PRIMARY_METRICS, run
from signal4d.io.predictions import PredictionArtifact


def test_verify_repro_accepts_identical_artifacts(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "synthetic",
                "clip_id": "clip",
                "split": "development",
                "frame_ids": [0, 1],
                "fps": 25.0,
                "image_relpaths": ["0.png", "1.png"],
                "is_contiguous": True,
                "allowed_for_calibration": False,
                "allowed_for_hparam_selection": True,
                "allowed_for_final_reporting": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prediction = PredictionArtifact(
        frame_ids=torch.tensor([0, 1]),
        joints_3d=torch.zeros(2, 55, 3),
        rotations=None,
        translation=torch.zeros(2, 3),
        vertices=None,
        risk_score=torch.zeros(2, 3),
        abstain=torch.zeros(2, 3, dtype=torch.bool),
        uncertainty=torch.ones(2, 55),
    )
    metadata = {"schema_version": "1.0"}
    prediction.save(tmp_path / "first/clip", metadata)
    prediction.save(tmp_path / "second/clip", metadata)
    summary = {metric: 1.0 for metric in PRIMARY_METRICS}
    (tmp_path / "first.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "second.json").write_text(json.dumps(summary), encoding="utf-8")

    report = run(
        str(manifest),
        str(tmp_path / "first"),
        str(tmp_path / "second"),
        str(tmp_path / "first.json"),
        str(tmp_path / "second.json"),
        str(tmp_path / "report.json"),
    )

    assert report["passed"] is True
    assert report["maximum_tensor_absolute_error"] == 0.0
