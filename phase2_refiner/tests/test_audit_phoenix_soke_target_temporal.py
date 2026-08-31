import json
import pickle
from pathlib import Path

import numpy as np

from phase2_refiner.data.audit_phoenix_soke_target_temporal import audit


def _target(angle_degrees: float) -> dict:
    body = np.zeros((21, 3), dtype=np.float32)
    body[0, 2] = np.deg2rad(angle_degrees)
    return {
        "smplx_body_pose": body.reshape(-1),
        "smplx_lhand_pose": np.zeros(45, dtype=np.float32),
        "smplx_rhand_pose": np.zeros(45, dtype=np.float32),
    }


def test_temporal_audit_detects_middle_frame_rotation_spike(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    for frame, angle in enumerate((0.0, 90.0, 0.0), start=1):
        with (target_dir / f"images{frame:04d}.pkl").open("wb") as handle:
            pickle.dump(_target(angle), handle)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "official_split": "train",
                "clips": [
                    {
                        "target_dir": str(target_dir),
                        "target_frame_indices_one_based": [1, 2, 3],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = audit(selection, sample_clips=1)
    body = report["regions"]["body"]["triangle_excess"]
    assert body["count"] == 21
    assert body["above_threshold"]["gt_45_degrees"]["count"] == 1
    assert report["regions"]["left_hand"]["triangle_excess"][
        "above_threshold"
    ]["gt_45_degrees"]["count"] == 0
    assert report["automatic_quality_mask_recommended"] is False
