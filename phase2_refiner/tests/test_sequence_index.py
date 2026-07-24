import csv
from pathlib import Path

from phase2_refiner.data.build_sequence_index import build_indices
from phase2_refiner.data.dataset import SequenceCacheDataset
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
