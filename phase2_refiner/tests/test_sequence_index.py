import csv
import hashlib
import json
from pathlib import Path

import pytest

from phase2_refiner.data.build_sequence_index import build_indices
from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.dataset import (
    LengthBucketBatchSampler,
    SequenceCacheDataset,
)
from phase2_refiner.data.cache_schema import save_cache_clip
from phase2_refiner.tests.test_cache import make_clip


def test_explicit_split_manifest_loads_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    clip = make_clip()
    save_cache_clip(cache_root / "clips" / "synthetic.npz", clip)
    assignments = tmp_path / "assignments.csv"
    with assignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("clip_id", "split", "source", "signer")
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "synthetic",
                "split": "train",
                "source": "video_001",
                "signer": "signer_001",
            }
        )
    split_root = cache_root / "splits"
    counts = build_indices(cache_root, assignments, split_root)
    assert counts["train"] == 1
    assert sum(counts.values()) == 1
    dataset = SequenceCacheDataset(str(split_root / "train.json"), identity_target=True)
    assert len(dataset) == 1


def test_evaluation_all_windows_covers_every_frame_once(tmp_path: Path) -> None:
    cache = tmp_path / "clip.npz"
    save_cache_clip(cache, make_clip(frames=5))
    dataset = SequenceCacheDataset(
        str(cache), max_frames=2, identity_target=True, all_windows=True
    )
    assert len(dataset) == 3
    assert [dataset[index]["frame_names"] for index in range(3)] == [
        ["low_000", "low_001"],
        ["low_002", "low_003"],
        ["low_004"],
    ]


def test_training_window_is_stateless_by_epoch_and_index(tmp_path: Path) -> None:
    cache = tmp_path / "clip.npz"
    save_cache_clip(cache, make_clip(frames=8))
    dataset = SequenceCacheDataset(
        str(cache), max_frames=2, training=True, identity_target=True, seed=17
    )

    reference = dataset[(0, 3)]["frame_names"]
    assert dataset[(0, 3)]["frame_names"] == reference
    assert len(
        {
            tuple(dataset[(0, epoch)]["frame_names"])
            for epoch in range(8)
        }
    ) > 1


def test_bucket_sampler_carries_epoch_into_dataset_index(tmp_path: Path) -> None:
    cache = tmp_path / "clip.npz"
    save_cache_clip(cache, make_clip(frames=8))
    dataset = SequenceCacheDataset(
        str(cache), max_frames=2, training=True, identity_target=True
    )
    sampler = LengthBucketBatchSampler(
        dataset, batch_size=1, shuffle=False, seed=42
    )

    assert list(iter(sampler)) == [[(0, 0)]]
    assert list(iter(sampler)) == [[(0, 1)]]


def test_dataset_enforces_declared_split_before_training(tmp_path: Path) -> None:
    cache = tmp_path / "clip.npz"
    clip = make_clip()
    clip.metadata_json = '{"official_split":"dev","phase2_split":"val"}'
    save_cache_clip(cache, clip)

    matching = SequenceCacheDataset(
        str(cache), identity_target=True, expected_split="val"
    )
    assert len(matching) == 1
    with pytest.raises(ValueError, match="does not match expected 'train'"):
        SequenceCacheDataset(
            str(cache), identity_target=True, expected_split="train"
        )


def test_dataset_verifies_manifest_cache_hashes(tmp_path: Path) -> None:
    cache = tmp_path / "clip.npz"
    save_cache_clip(cache, make_clip())
    relative = "clip.npz"
    manifest = tmp_path / "train.json"
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps({"clips": [relative], "clip_sha256": {relative: digest}}),
        encoding="utf-8",
    )
    assert len(SequenceCacheDataset(str(manifest), identity_target=True)) == 1
    assert _manifest_paths(manifest) == [cache.resolve()]

    manifest.write_text(
        json.dumps({"clips": [relative], "clip_sha256": {relative: "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cache SHA-256 mismatch"):
        SequenceCacheDataset(str(manifest), identity_target=True)
    with pytest.raises(ValueError, match="Cache SHA-256 mismatch"):
        _manifest_paths(manifest)
