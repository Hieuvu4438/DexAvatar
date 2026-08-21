import json
import pickle

from signal4d.cli.build_extended_manifest import run
from signal4d.data.manifest import load_manifest


def test_extended_manifest_uses_intersection_after_central_segment(tmp_path) -> None:
    (tmp_path / "frames/clip").mkdir(parents=True)
    (tmp_path / "body/clip/smplerx/smplx").mkdir(parents=True)
    (tmp_path / "wilor/clip/wilor").mkdir(parents=True)
    (tmp_path / "gt/clip").mkdir(parents=True)
    for frame_id in (1, 3, 5, 7, 9):
        (tmp_path / f"frames/clip/low_{frame_id:03d}.png").write_bytes(b"image")
        (tmp_path / f"body/clip/smplerx/smplx/low_{frame_id:03d}.pkl").write_bytes(b"body")
        (tmp_path / f"gt/clip/{frame_id * 2:05d}.obj").write_text("v 0 0 0\n")
    with (tmp_path / "wilor/clip/wilor/wilor.pkl").open("wb") as handle:
        pickle.dump({"images": {f"low_{frame_id:03d}.png": {} for frame_id in (1, 5, 7)}}, handle)
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps({"clip": [1, 3]}), encoding="utf-8")
    output = tmp_path / "extended.jsonl"

    report = run(
        str(segments),
        str(tmp_path / "frames"),
        str(tmp_path / "body"),
        str(tmp_path / "wilor"),
        str(tmp_path / "gt"),
        str(output),
    )

    row = load_manifest(output)[0]
    assert row.frame_ids == [5, 7]
    assert row.allowed_for_final_reporting
    assert report["ground_truth_values_read"] is False
