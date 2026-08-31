import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from phase2_refiner.data.build_phoenix_soke_full_cache import (
    _target_pose,
    _verified_artifacts,
    build,
)
from phase2_refiner.provenance import sha256_file


def _write_target(path, *, body=63, left=45, right=45) -> None:
    payload = {
        "smplx_body_pose": np.arange(body, dtype=np.float32),
        "smplx_lhand_pose": np.arange(left, dtype=np.float32),
        "smplx_rhand_pose": np.arange(right, dtype=np.float32),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_phoenix_target_pose_requires_exact_released_shapes(tmp_path) -> None:
    target = tmp_path / "images0001.pkl"
    _write_target(target, body=62)
    with pytest.raises(ValueError, match="62 values, expected 63"):
        _target_pose([target])


def test_phoenix_target_pose_preserves_nonfinite_validity(tmp_path) -> None:
    target = tmp_path / "images0001.pkl"
    _write_target(target)
    with target.open("rb") as handle:
        payload = pickle.load(handle)
    payload["smplx_lhand_pose"][4] = np.nan
    with target.open("wb") as handle:
        pickle.dump(payload, handle)

    pose, valid = _target_pose([target])
    assert pose.shape == (1, 51, 3)
    assert valid.shape == (1, 51)
    assert not valid[0, 22]
    assert np.isfinite(pose).all()


def test_verified_wilor_artifacts_reject_hamer_keys_outside_manifest(
    tmp_path,
) -> None:
    manifest = tmp_path / "shard_0000.json"
    manifest.write_text(json.dumps({"records": [{"image_key": "frame.png"}]}))
    shard = tmp_path / "outputs" / "shard_0000"
    (shard / "hamer").mkdir(parents=True)
    (shard / "wilor").mkdir()
    with (shard / "hamer" / "hamer.pkl").open("wb") as handle:
        pickle.dump({"frame.png": {}, "outside.png": {}}, handle)
    with (shard / "wilor" / "wilor.pkl").open("wb") as handle:
        pickle.dump(
            {
                "images": {"frame.png": {}},
                "meta": {
                    "frame_manifest_sha256": sha256_file(manifest),
                    "frame_manifest_sources_verified": True,
                },
            },
            handle,
        )

    with pytest.raises(ValueError, match="HaMeR keys outside shard manifest"):
        _verified_artifacts(tmp_path / "outputs", manifest, 0)


def test_phoenix_cache_publishes_final_path_only_after_completion(
    tmp_path,
) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "split": "train",
                "official_split": "train",
                "clips": [],
            }
        )
    )
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    (shard_root / "shard_report.json").write_text(
        json.dumps({"shards": []})
    )
    output = tmp_path / "cache"
    staging = tmp_path / ".cache.staging"
    staging.mkdir()
    (staging / "stale").write_text("interrupted prior attempt")

    report = build(
        SimpleNamespace(
            selection=selection,
            wilor_shard_manifest_root=shard_root,
            wilor_root=tmp_path / "wilor",
            smplerx_root=tmp_path / "h32",
            output=output,
        )
    )

    assert output.is_dir()
    assert not staging.exists()
    assert (output / "splits" / "train.json").is_file()
    assert report["manifest"] == str(output / "splits" / "train.json")
