from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dcg_sign4d.evaluation.sgnify import read_obj
from dcg_sign4d.geometry.smplx_adapter import SMPLXAdapter
from dcg_sign4d.initialization.dexavatar_adapter import DexAvatarPklInitializer
from dcg_sign4d.utils.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one real DexAvatar PKL to SMPL-X replay")
    parser.add_argument("--result-pkl", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--reference-obj", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--trusted-local-assets", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.trusted_local_assets:
        raise PermissionError("replay audit requires --trusted-local-assets")
    result_path = Path(args.result_pkl)
    state, metadata = DexAvatarPklInitializer(args.fps).reconstruct_from_directory(
        result_path.parent,
        expected_hashes={result_path.name: args.expected_result_sha256},
        trusted=True,
        include_names={result_path.name},
    )
    # The converter directory may contain more files; isolate the requested frame
    # contract by selecting its exact index after all source hashes are verified.
    frame_id = int(result_path.stem.split("_")[-1])
    index = metadata["frame_ids"].index(frame_id)
    state = replace(
        state,
        root_rot6d=state.root_rot6d[:, index : index + 1],
        root_translation=state.root_translation[:, index : index + 1],
        root_velocity=state.root_velocity[:, index : index + 1],
        body_rot6d=state.body_rot6d[:, index : index + 1],
        left_hand_rot6d=state.left_hand_rot6d[:, index : index + 1],
        right_hand_rot6d=state.right_hand_rot6d[:, index : index + 1],
        face_state=state.face_state[:, index : index + 1],
        valid_mask=state.valid_mask[:, index : index + 1],
    )
    adapter = SMPLXAdapter(
        args.model,
        expected_sha256=args.expected_model_sha256,
        trusted_model=True,
    )
    with torch.inference_mode():
        replay = adapter(state).vertices[0, 0].cpu().numpy()
    reference, _ = read_obj(args.reference_obj)
    errors_mm = np.linalg.norm(replay - reference, axis=-1) * 1000
    report = {
        "result_pkl": str(result_path),
        "result_pkl_sha256": file_sha256(result_path),
        "reference_obj": args.reference_obj,
        "reference_obj_sha256": file_sha256(args.reference_obj),
        "model_sha256": args.expected_model_sha256,
        "mean_vertex_error_mm": float(errors_mm.mean()),
        "max_vertex_error_mm": float(errors_mm.max()),
        "vertices": int(replay.shape[0]),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable replay report exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
