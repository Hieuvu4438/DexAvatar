import argparse
import json
from pathlib import Path

import pytest

from phase2_refiner.data.cache_schema import save_cache_clip
from phase2_refiner.data.repartition_phase2r_by_signer import repartition
from phase2_refiner.tests.test_cache import make_clip


def _inputs(tmp_path: Path) -> tuple[list[Path], Path]:
    entries = []
    signers = {}
    for index, signer in enumerate(("a", "b", "c", "a", "b", "c")):
        clip = make_clip(3)
        clip.clip_id = f"clip-{index}"
        source_clip = f"source-{index}"
        clip.metadata_json = json.dumps(
            {
                "source_clip": source_clip,
                "source_group": f"group-{index}",
                "official_split": "train",
            }
        )
        path = tmp_path / f"clip-{index}.npz"
        save_cache_clip(path, clip)
        entries.append(str(path))
        signers[source_clip] = signer
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"clips": entries[:3]}))
    second = tmp_path / "second.json"
    second.write_text(json.dumps({"clips": entries[3:]}))
    signer_map = tmp_path / "signers.json"
    signer_map.write_text(json.dumps(signers))
    return [first, second], signer_map


def test_repartition_is_signer_and_source_disjoint(tmp_path: Path) -> None:
    manifests, signer_map = _inputs(tmp_path)
    assignment = tmp_path / "assignment.json"
    assignment.write_text(
        json.dumps({"train": ["a"], "val": ["b"], "calibration": ["c"]})
    )
    report = repartition(
        argparse.Namespace(
            manifest=manifests,
            signer_map=signer_map,
            assignment=assignment,
            output=tmp_path / "output",
        )
    )
    assert report["passed"] is True
    assert report["checks"]["signer_disjoint"] is True
    assert report["checks"]["source_group_disjoint"] is True
    assert [
        report["splits"][split]["clips"] for split in ("train", "val", "calibration")
    ] == [2, 2, 2]


def test_repartition_rejects_signer_reuse(tmp_path: Path) -> None:
    manifests, signer_map = _inputs(tmp_path)
    assignment = tmp_path / "assignment.json"
    assignment.write_text(
        json.dumps({"train": ["a"], "val": ["a", "b"], "calibration": ["c"]})
    )
    with pytest.raises(ValueError, match="multiple splits"):
        repartition(
            argparse.Namespace(
                manifest=manifests,
                signer_map=signer_map,
                assignment=assignment,
                output=tmp_path / "output",
            )
        )
