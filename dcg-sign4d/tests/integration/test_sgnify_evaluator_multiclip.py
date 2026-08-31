import json
import pickle

import numpy as np

from dcg_sign4d.evaluation.sgnify import evaluate_sgnify_obj


def _obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x} {y} {z}\n" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces]
    path.write_text("".join(lines), encoding="utf-8")


def test_multiclip_evaluator_keeps_prediction_root_path(tmp_path):
    assets = tmp_path / "assets"
    region = assets / "sgnify_part_segm_above_pelvis_joint"
    region.mkdir(parents=True)
    faces = np.array([[0, 1, 2], [3, 4, 5], [0, 6, 7]], dtype=np.int64)
    regressor = np.zeros((22, 8))
    regressor[1, 0] = 1
    regressor[2, 1] = 1
    regressor[20, 2] = 1
    regressor[21, 4] = 1
    np.savez(assets / "SMPLX_NEUTRAL.npz", J_regressor=regressor, f=faces)
    np.save(region / "upper_body.npy", np.arange(8))
    with (assets / "MANO_SMPLX_vertex_ids.pkl").open("wb") as handle:
        pickle.dump({"left_hand": np.array([2, 3]), "right_hand": np.array([4, 5])}, handle)
    manifest = []
    prediction_root = tmp_path / "predictions"
    gt_root = tmp_path / "gt"
    for clip in ("first", "second"):
        frame_ids = [1, 2, 3, 4]
        manifest.append({"clip_id": clip, "frame_ids": frame_ids, "fps": 15})
        for frame in frame_ids:
            vertices = np.zeros((8, 3))
            vertices[:, 0] = np.arange(8) * 0.01
            vertices[2, 1] = frame * 0.01
            _obj(
                prediction_root / clip / "smplifyx/meshes" / f"low_{frame}.obj",
                vertices,
                faces,
            )
            _obj(gt_root / clip / f"{frame * 2:05d}.obj", vertices, faces)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    signs = tmp_path / "signs.txt"
    signs.write_text("first 1\nsecond 1\n", encoding="utf-8")
    summary = evaluate_sgnify_obj(
        manifest_path=manifest_path,
        prediction_root=prediction_root,
        gt_root=gt_root,
        author_asset_root=assets,
        author_sign_file=signs,
        output_root=tmp_path / "evaluation",
        trusted_author_assets=True,
    )
    assert summary["clips"] == 2
    assert summary["root_aligned_hand_pve_mm"] == 0
    assert summary["temporal_hand_velocity_error_mm_per_s"] == 0
