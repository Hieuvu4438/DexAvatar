import json
import pickle
import time
from pathlib import Path

import numpy as np
import pytest

from phase2_refiner.data.audit_phoenix_h32_frontend import audit
from phase2_refiner.data.audit_phoenix_h32_incremental import audit_incremental


def _write_h32(path: Path, frames: int = 2) -> None:
    payload = {
        "total_valid_index": np.arange(frames, dtype=np.int64),
        "smplx": np.zeros((frames, 182), dtype=np.float32),
        "unsmooth_smplx": np.zeros((frames, 169), dtype=np.float32),
        "pred_2d": np.zeros((frames, 106, 2), dtype=np.float32),
        "bb2img_trans": np.zeros((frames, 2, 3), dtype=np.float32),
        "width": 210,
        "height": 260,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_audit_phoenix_h32_frontend_validates_payloads(tmp_path: Path) -> None:
    selections = tmp_path / "selections"
    h32 = tmp_path / "h32"
    h32.mkdir()
    for split in ("train", "dev"):
        root = selections / split
        root.mkdir(parents=True)
        name = f"clip_{split}"
        (root / "selection.json").write_text(
            json.dumps(
                {
                    "clips": [
                        {
                            "source_clip": name,
                            "source_contract": {"frame_count": 3},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        _write_h32(h32 / f"{name}.pkl")
    report = audit(selections, h32, ("train", "dev"), 0.0)
    assert report["clips"] == 2
    assert report["target_fields_opened"] is False
    assert report["splits"]["train"]["h32_retained_frames"] == 2

    recent = h32 / "clip_train.pkl"
    recent.touch()
    with pytest.raises(ValueError, match="not yet stable"):
        audit(selections, h32, ("train", "dev"), 60.0)


def test_incremental_h32_audit_reuses_unchanged_verified_payloads(
    tmp_path: Path,
) -> None:
    selections = tmp_path / "selections"
    h32 = tmp_path / "h32"
    h32.mkdir()
    for split in ("train", "dev"):
        root = selections / split
        root.mkdir(parents=True)
        name = f"clip_{split}"
        (root / "selection.json").write_text(
            json.dumps(
                {
                    "clips": [
                        {
                            "source_clip": name,
                            "source_contract": {"frame_count": 3},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        if split == "train":
            _write_h32(h32 / f"{name}.pkl")
    first = audit_incremental(
        selections, h32, ("train", "dev"), minimum_age_seconds=0.0
    )
    assert first["verified_clips"] == 1
    assert first["pending_clips"] == 1
    assert first["newly_validated"] == 1
    second = audit_incremental(
        selections,
        h32,
        ("train", "dev"),
        minimum_age_seconds=0.0,
        previous=first,
    )
    assert second["verified_clips"] == 1
    assert second["newly_validated"] == 0
    assert second["reused"] == 1
