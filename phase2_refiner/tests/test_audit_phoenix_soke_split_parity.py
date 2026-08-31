import gzip
import json
import pickle
from pathlib import Path

from phase2_refiner.data.audit_phoenix_soke_split_parity import audit


def test_audit_phoenix_soke_split_parity(tmp_path: Path) -> None:
    selections = tmp_path / "selections"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for split in ("train", "dev", "test"):
        clip_name = f"example_{split}"
        target = tmp_path / "targets" / split / clip_name
        target.mkdir(parents=True)
        (target / "images0001.pkl").touch()
        (target / "images0003.pkl").touch()
        row = {
            "name": f"{split}/{clip_name}",
            "gloss": "HELLO",
            "text": "hello",
            "signer": "Signer01",
            "src": "phoenix",
            "num_frames": 3,
        }
        with gzip.open(annotations / f"phoenix14t.{split}", "wb") as handle:
            pickle.dump([row], handle)
        split_root = selections / split
        split_root.mkdir(parents=True)
        (split_root / "selection.json").write_text(
            json.dumps(
                {
                    "clips": [
                        {
                            "official_name": row["name"],
                            "source_clip": clip_name,
                            "gloss": row["gloss"],
                            "text": row["text"],
                            "signer_id": row["signer"],
                            "source_contract": {"frame_count": 3},
                            "target_dir": str(target),
                            "target_frame_indices_one_based": [1, 3],
                            "frame_indices": [0, 2],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    report = audit(selections, annotations)
    assert report["exact_soke_loader_parity"] is True
    assert report["pose_payloads_opened"] is False
    assert report["no_cross_split_clip_overlap"] is True
    assert report["source_group_disjoint"] is True
    assert report["cross_split"]["train_test"]["clip_id_overlap_count"] == 0
    assert report["splits"]["train"]["soke_loader_pose_frames"] == 2
    assert report["splits"]["test"]["missing_fitted_frames"] == 1


def test_audit_reports_official_source_group_context_overlap(tmp_path: Path) -> None:
    selections = tmp_path / "selections"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    suffix = {"train": "100", "dev": "101", "test": "200"}
    for split in ("train", "dev", "test"):
        clip_name = f"shared_news-{suffix[split]}"
        target = tmp_path / "targets" / split / clip_name
        target.mkdir(parents=True)
        (target / "images0001.pkl").touch()
        row = {
            "name": f"{split}/{clip_name}",
            "gloss": "HELLO",
            "text": "hello",
            "signer": "Signer01",
            "src": "phoenix",
            "num_frames": 1,
        }
        with gzip.open(annotations / f"phoenix14t.{split}", "wb") as handle:
            pickle.dump([row], handle)
        split_root = selections / split
        split_root.mkdir(parents=True)
        (split_root / "selection.json").write_text(
            json.dumps(
                {
                    "clips": [
                        {
                            "clip_id": f"phoenix_{clip_name}",
                            "source_group": "shared_news",
                            "official_name": row["name"],
                            "source_clip": clip_name,
                            "gloss": row["gloss"],
                            "text": row["text"],
                            "signer_id": row["signer"],
                            "source_contract": {"frame_count": 1},
                            "target_dir": str(target),
                            "target_frame_indices_one_based": [1],
                            "frame_indices": [0],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    report = audit(selections, annotations)
    assert report["no_cross_split_clip_overlap"] is True
    assert report["source_group_disjoint"] is False
    assert report["cross_split"]["train_dev"]["source_group_overlap_count"] == 1
    assert report["cross_split"]["train_test"]["source_group_overlap_count"] == 1
