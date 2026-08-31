import json

from signpccx.data.manifest import build_sign_manifest, evaluator_gt_ids


def test_manifest_matches_author_positional_protocol(tmp_path):
    images = tmp_path / "images"
    gt = tmp_path / "gt"
    images.mkdir()
    gt.mkdir()
    for frame in (9, 11, 13, 15, 17):
        (images / f"low_{frame}.png").write_bytes(b"x")
    for frame in (22, 26, 30):
        (gt / f"{frame:05d}.obj").write_text("", encoding="utf-8")
    assert evaluator_gt_ids(gt, (11, 15)) == [22, 26, 30]
    records = build_sign_manifest("S", "~0", images, gt, (11, 15))
    assert [record.source_frame_id for record in records] == [11, 13, 15]
    assert [record.gt_frame_id for record in records] == [22, 26, 30]
    assert [record.evaluator_index for record in records] == [0, 1, 2]

