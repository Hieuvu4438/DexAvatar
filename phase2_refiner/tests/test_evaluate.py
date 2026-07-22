import csv
import pickle
from pathlib import Path

import numpy as np

from phase2_refiner.evaluate import evaluate


def write_obj(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        handle.write("f 1 2 3\n")


def test_strict_evaluator_identity(tmp_path: Path) -> None:
    vertices = np.zeros((10475, 3), np.float32)
    vertices[:, 0] = np.linspace(0, 1, len(vertices))
    gt = tmp_path / "data/smplx_gt/Test/00002.obj"
    prediction = tmp_path / "prediction/Test/smplifyx/meshes/low_001.obj"
    baseline = tmp_path / "baseline/Test/smplifyx/meshes/low_001.obj"
    for path in (gt, prediction, baseline):
        write_obj(path, vertices)
    assets = tmp_path / "assets"
    (assets / "sgnify_part_segm_above_pelvis_joint").mkdir(parents=True)
    np.save(
        assets / "sgnify_part_segm_above_pelvis_joint/upper_body_minus_face.npy",
        np.arange(100),
    )
    with (assets / "MANO_SMPLX_vertex_ids.pkl").open("wb") as handle:
        pickle.dump(
            {"left_hand": np.arange(100, 110), "right_hand": np.arange(110, 120)},
            handle,
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sign",
                "class",
                "ordinal",
                "gt_path",
                "prediction_path",
                "left_evaluated",
                "right_evaluated",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sign": "Test",
                "class": "~0",
                "ordinal": 0,
                "gt_path": "data/smplx_gt/Test/00002.obj",
                "prediction_path": "outputs/method_hamer/Test/smplifyx/meshes/low_001.obj",
                "left_evaluated": "True",
                "right_evaluated": "True",
            }
        )
    summary = evaluate(
        manifest,
        tmp_path / "prediction",
        tmp_path / "evaluation",
        tmp_path,
        tmp_path / "baseline",
        assets,
        bootstrap_samples=100,
        seed=1,
        overwrite=False,
    )
    assert summary["frames"] == 1
    assert summary["prediction"] == {"ubody": 0.0, "lhand": 0.0, "rhand": 0.0}
