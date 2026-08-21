from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


def load_author(path: Path):
    spec = importlib.util.spec_from_file_location("strict_original_author_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--author-asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--sign-file", type=Path, required=True)
    parser.add_argument("--segment-file", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    args = parser.parse_args()

    author = load_author(args.author_source)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    clip_ids = sorted(row["clip_id"] for row in rows)
    classes = {}
    for line in args.sign_file.read_text().splitlines():
        if line.strip():
            clip_id, sign_class = line.split()
            classes[clip_id] = sign_class
    with (args.author_asset_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle, encoding="latin1")
    model = np.load(args.author_asset_root / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    region_root = args.author_asset_root / "sgnify_part_segm_above_pelvis_joint"

    author.class_sign = {clip_id: classes[clip_id] for clip_id in clip_ids}
    author.left_hand_ids = np.asarray(mano["left_hand"])
    author.right_hand_ids = np.asarray(mano["right_hand"])
    vertex_indices = {
        "all": np.arange(10475),
        "left hand": author.left_hand_ids,
        "right hand": author.right_hand_ids,
        "above pelvis upper body": np.load(region_root / "upper_body.npy"),
        "above pelvis minus head": np.load(region_root / "upper_body_minus_head.npy"),
        "above pelvis minus face": np.load(region_root / "upper_body_minus_face.npy"),
    }
    mapping = {clip_id: "slrt" for clip_id in clip_ids}
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    author.logger.remove()
    with args.output_log.open("w", encoding="utf-8") as handle:
        author.logger.add(handle, level="INFO", format="{message}")
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            author.main(
                mapping,
                vertex_indices,
                clip_ids,
                clip_ids,
                str(args.gt_root),
                str(args.prediction_root),
                J_regressor=np.asarray(model["J_regressor"]),
                method="slrt",
                sign_seg_file=str(args.segment_file),
            )


if __name__ == "__main__":
    main()
