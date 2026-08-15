import json
from pathlib import Path

import numpy as np
import pytest

from phase2_refiner.data.render_signavatars_target_audit import (
    _load_sample,
    _validate_candidate,
)
from phase2_refiner.tests.test_cache import make_clip


def test_signavatars_renderer_requires_disjoint_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "schema": "signavatars-target-audit-sample-v1",
                "source_group_disjoint": False,
                "clips": [{"cache_path": "unused"}],
            }
        )
    )
    with pytest.raises(ValueError, match="not source-group disjoint"):
        _load_sample(sample)


def test_signavatars_renderer_requires_ineligible_candidate_contract() -> None:
    clip = make_clip(2)
    clip.target_axis_angle = np.zeros((2, 51, 3), dtype=np.float32)
    clip.target_rotation_valid = np.ones((2, 51), dtype=bool)
    clip.metadata_json = json.dumps(
        {
            "target_type": "licensed_released_3d_smplx_pose_audit_candidate",
            "target_contract": {
                "audit_passed": False,
                "exact_frame_count_match": True,
            },
        }
    )
    metadata = _validate_candidate(clip)
    assert metadata["target_contract"]["audit_passed"] is False
    metadata["target_contract"]["audit_passed"] = True
    clip.metadata_json = json.dumps(metadata)
    with pytest.raises(ValueError, match="already claims"):
        _validate_candidate(clip)
