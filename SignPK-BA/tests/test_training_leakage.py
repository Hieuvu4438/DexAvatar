import json
from pathlib import Path

from signpk.data.signavatars_dataset import SignAvatarWindowDataset


def test_sgnify_cache_is_forbidden(tmp_path: Path):
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "cache_path": "sgnify/window.pt",
                "signer_id": "s1",
                "sequence_id": "q1",
                "split": "train",
                "quality_weight": 1.0,
                "source_dataset": "SignAvatars",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        SignAvatarWindowDataset(index)
    except ValueError as error:
        assert "leakage" in str(error)
    else:
        raise AssertionError("benchmark cache was accepted for training")


def test_signer_disjointness_is_enforced(tmp_path: Path):
    rows = []
    for split in ("train", "val"):
        rows.append(
            {
                "cache_path": f"cache/{split}.pt",
                "signer_id": "same",
                "sequence_id": split,
                "split": split,
                "quality_weight": 0.8,
                "source_dataset": "SignAvatars",
            }
        )
    index = tmp_path / "index.jsonl"
    index.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    try:
        SignAvatarWindowDataset(index)
    except ValueError as error:
        assert "multiple splits" in str(error)
    else:
        raise AssertionError("signer leakage was accepted")

