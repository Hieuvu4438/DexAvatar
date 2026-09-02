#!/usr/bin/env python3
"""Compare the extracted hand method with one frozen pre-refactor artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from signeft.hand.parallel import refine_hands_parallel
from signeft.io.obj import load_obj
from signeft.io_utils import sha256_file
from signeft.manifest import HandFrameRecord, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    project = workspace / "SignEFT-X"
    legacy_manifest = project / "_archive/research_history/manifests/trv2v.jsonl"
    legacy = [json.loads(line) for line in legacy_manifest.read_text().splitlines()[:16]]
    records = []
    comparisons = []
    for item in legacy:
        sign = item["sign_id"]
        frame = item["source_frame_id"]
        state = (
            project
            / "_archive/research_history/baseline_states/a3f"
            / sign
            / f"{frame:06d}.npz"
        )
        canonical_obj = Path(item["a3f_obj_path"])
        rgb = Path(item["rgb_path"])
        expected_obj = (
            project
            / "_archive/research_history/runs/paper_ablation_native_radius12_full57/frames"
            / sign
            / f"{frame:06d}.obj"
        )
        expected_state = expected_obj.with_suffix(".npz")
        required = (state, canonical_obj, rgb, expected_obj, expected_state)
        if missing := [path for path in required if not path.is_file()]:
            raise FileNotFoundError(missing[0])
        records.append(
            HandFrameRecord(
                record_id=f"{sign}/{frame}",
                sign=sign,
                sign_class=item["sign_class"],
                frame_index=item["frame_index"],
                source_frame_id=frame,
                rgb_path=str(rgb),
                canonical_state_path=str(state),
                canonical_obj_path=str(canonical_obj),
                rgb_sha256=sha256_file(rgb),
                state_sha256=sha256_file(state),
                obj_sha256=sha256_file(canonical_obj),
            )
        )
        comparisons.append((sign, frame, expected_obj, expected_state))
    with tempfile.TemporaryDirectory(prefix="signeft-refactor-") as folder:
        root = Path(folder)
        manifest = root / "manifest.jsonl"
        write_jsonl(records, manifest)
        refine_hands_parallel(
            manifest,
            root / "output",
            workspace / "SMPLer-X/common/utils/human_model_files",
            project / "inputs/wilor_full1493_v1",
            workers=2,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=8,
            radius_deg=12.0,
            steps=40,
            learning_rate=0.03,
            residual_prior=0.2,
        )
        maxima = {"vertex": 0.0, "left_pose": 0.0, "right_pose": 0.0}
        for sign, frame, expected_obj, expected_state in comparisons:
            actual_obj = root / "output/meshes" / sign / f"{frame:06d}.obj"
            actual_state = root / "output/states" / sign / f"{frame:06d}.npz"
            actual_vertices, actual_faces = load_obj(actual_obj)
            expected_vertices, expected_faces = load_obj(expected_obj)
            vertex_error = np.abs(actual_vertices - expected_vertices)
            if not np.array_equal(actual_faces, expected_faces):
                raise AssertionError("face topology changed during extraction")
            with np.load(actual_state, allow_pickle=False) as actual, np.load(
                expected_state, allow_pickle=False
            ) as expected:
                left_error = np.abs(actual["left_hand_pose"] - expected["left_hand_pose"])
                right_error = np.abs(actual["right_hand_pose"] - expected["right_hand_pose"])
            maxima["vertex"] = max(maxima["vertex"], float(vertex_error.max()))
            maxima["left_pose"] = max(maxima["left_pose"], float(left_error.max()))
            maxima["right_pose"] = max(maxima["right_pose"], float(right_error.max()))
        tolerance = 2e-6
        maximum = max(maxima.values())
        print(
            "REFRACTOR_DIAGNOSTIC",
            f"max_vertex={maxima['vertex']:.3e}",
            f"max_left_pose={maxima['left_pose']:.3e}",
            f"max_right_pose={maxima['right_pose']:.3e}",
        )
        if maximum > tolerance:
            raise AssertionError(f"refactor mismatch {maximum:.3e} > {tolerance:.3e}")
        print(
            "REFRACTOR_EQUIVALENT",
            f"max_vertex={maxima['vertex']:.3e}",
            f"max_left_pose={maxima['left_pose']:.3e}",
            f"max_right_pose={maxima['right_pose']:.3e}",
        )


if __name__ == "__main__":
    main()
