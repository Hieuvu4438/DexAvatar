import json
from pathlib import Path
import pickle

import numpy as np

from phase2_refiner.data.materialize_a1r_cache import attach_initializer
from phase2_refiner.tests.test_cache import make_clip
from phase2_refiner.tests.test_exact_a1_stack import valid_payload


DIGEST = "a" * 64


def test_a1r_materialization_binds_results_and_portable_contract(
    tmp_path: Path,
) -> None:
    clip = make_clip(2)
    clip.clip_id = "portable-clip"
    results = tmp_path / "results"
    results.mkdir()
    for index, name in enumerate(clip.frame_names):
        payload = valid_payload()
        payload["body_pose"][:] = index + 1
        with (results / f"{name}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps({"clip_name": clip.clip_id, "sign_class": "1"}) + "\n"
    )
    provider = {
        "provider_id": "portable-a1r",
        "weights": {"weights.pt": DIGEST},
        "configuration": {"config.yaml": DIGEST},
        "provider_code": {"runner.py": DIGEST},
        "manifest_sha256": DIGEST,
    }

    updated = attach_initializer(clip, results, decision, provider)

    np.testing.assert_array_equal(updated.init_axis_angle[:, 0, 0], [1, 2])
    assert all(len(value) == 64 for value in updated.source_sha256)
    assert updated.initializer_component.tolist() == ["portable-a1r"] * 2
    contract = json.loads(updated.metadata_json)["initializer_contract"]
    assert contract["portable"]
    assert contract["benchmark_conditioning"] is False
    assert len(contract["weights_sha256"]) == 64
