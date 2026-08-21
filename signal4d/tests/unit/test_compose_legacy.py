import json
from pathlib import Path

from signal4d.cli import compose_legacy


def test_compose_legacy_uses_atomic_fallback(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "clip_id": "clip",
                "dataset": "test",
                "split": "test",
                "frame_ids": [1, 3],
                "fps": 15.0,
                "image_relpaths": ["one.png", "three.png"],
                "gt_relpath": "gt/clip",
                "signer_id": "unknown",
                "language": "unknown",
                "sign_type": "unknown",
                "allowed_for_calibration": False,
                "allowed_for_hparam_selection": False,
                "allowed_for_final_reporting": True,
                "is_contiguous": False,
                "frame_start": None,
                "frame_end_exclusive": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    primary = tmp_path / "primary/clip/smplifyx/results"
    fallback = tmp_path / "fallback/clip/smplifyx/results"
    primary.mkdir(parents=True)
    fallback.mkdir(parents=True)
    (primary / "low_001.pkl").write_bytes(b"primary")
    (fallback / "low_003.pkl").write_bytes(b"fallback")
    output = tmp_path / "composed"
    report = compose_legacy.run(
        str(manifest), str(tmp_path / "primary"), str(tmp_path / "fallback"), str(output)
    )
    assert report["source_counts"] == {"primary": 1, "fallback": 1}
    assert (output / "clip/smplifyx/results/low_001.pkl").read_bytes() == b"primary"
    assert (output / "clip/smplifyx/results/low_003.pkl").read_bytes() == b"fallback"
    second = compose_legacy.run(
        str(manifest), str(tmp_path / "primary"), str(tmp_path / "fallback"), str(output)
    )
    assert second["source_counts"] == report["source_counts"]
