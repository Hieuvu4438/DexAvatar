"""Materialize external-trained NLF V2 on target-free inference observations."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import smplx
from scipy.spatial.transform import Rotation

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file
from phase2_refiner.render import render_source_anchored_directory
from signal4d_external.nlf_v2_core import (
    FEATURE_COLUMNS,
    cache_pose_from_full,
    frame_features,
    full_initializer_rotations,
    nlf_body_candidate,
    nlf_observation_contract,
    viterbi_benefit_selection,
)


def _load_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    with (root / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["clip_id"]), int(row["frame_id"]))
            if key in result:
                raise ValueError(f"Duplicate NLF inference observation: {key}")
            result[key] = row
    return result


def _load_params(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _selected_params(baseline: dict[str, Any], candidate_full: np.ndarray) -> dict[str, Any]:
    result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in baseline.items()}
    rotvec = Rotation.from_matrix(candidate_full).as_rotvec().astype(np.float32)
    result["body_pose"] = rotvec[1:22].reshape(1, 63)
    # The fusion contract keeps these exactly on the baseline expert.
    for key in (
        "global_orient",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "betas",
        "expression",
        "transl",
    ):
        if key in baseline:
            result[key] = np.asarray(baseline[key]).copy()
    if "body_pose_fore" in result:
        result["body_pose_fore"] = result["body_pose"][:, :45].copy()
    if "body_pose_op" in result:
        result["body_pose_op"] = result["body_pose"][:, 45:].copy()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    calibration_path = args.router_root / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("decision") != "PASS":
        raise ValueError("External NLF calibration did not PASS")
    if int(calibration.get("sgnify_training_or_selection_reads", -1)) != 0:
        raise ValueError("Router calibration does not prove zero SGNify reads")
    router_path = args.router_root / "router.joblib"
    if sha256_file(router_path) != calibration["router_sha256"]:
        raise ValueError("Router hash does not match external calibration")
    router = joblib.load(router_path)
    selected_config = calibration["selected"]
    alpha = float(selected_config["alpha"])
    margin = float(selected_config["margin_deg"])
    transition = float(selected_config["transition_penalty_deg"])
    observation_index = _load_index(args.observation_root)
    target_observation_metadata = json.loads(
        (args.observation_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    target_observation_contract = nlf_observation_contract(
        target_observation_metadata
    )
    if target_observation_contract != calibration.get("nlf_observation_contract"):
        raise ValueError(
            "Target NLF observations do not match the externally calibrated "
            "model/source/settings contract"
        )
    baseline_manifest_path = args.baseline_root / "run_manifest.json"
    baseline_manifest = json.loads(
        baseline_manifest_path.read_text(encoding="utf-8")
    )
    if int(baseline_manifest.get("sgnify_target_reads_before_evaluation", -1)) != 0:
        raise ValueError("Baseline does not prove target-free materialization")
    forbidden_model_parts = {"evaluation_from_author", "smplx_gt", "sgnify"}
    if forbidden_model_parts & {
        part.lower() for part in args.model_path.resolve().parts
    }:
        raise ValueError(f"Forbidden evaluator-owned SMPL-X model: {args.model_path}")

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True)
    incomplete = args.output_root / ".incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    model = smplx.SMPLX(
        str(args.model_path.resolve()),
        gender="neutral",
        ext=args.model_path.suffix.lstrip("."),
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    ).eval()
    parents = model.parents[:55].detach().cpu().numpy().astype(np.int64)

    summaries = []
    total_frames = 0
    total_selected = 0
    cache_paths = sorted((args.cache_root / "clips").glob("*.npz"))
    if len(cache_paths) != 57:
        raise ValueError(f"Expected 57 target-free cache clips, got {len(cache_paths)}")
    for cache_path in cache_paths:
        cache = load_cache_clip(cache_path)
        if cache.target_axis_angle is not None or cache.target_joint_positions is not None:
            raise ValueError(f"Inference cache contains targets: {cache_path}")
        candidate_full = []
        feature_rows = []
        previous_initial = None
        previous_candidate = None
        previous_frame = None
        records = []
        for frame, frame_id in enumerate(cache.frame_numbers):
            key = (cache.clip_id, int(frame_id))
            record = observation_index.get(key)
            if record is None or record.get("status") not in {"ok", "existing"}:
                raise FileNotFoundError(f"Missing successful NLF observation: {key}")
            with np.load(args.observation_root / record["output_relpath"]) as observation:
                elapsed_seconds = (
                    None
                    if previous_frame is None
                    else float(cache.timestamps[frame] - cache.timestamps[previous_frame])
                )
                candidate = nlf_body_candidate(
                    cache, frame, observation["pose"], parents, alpha
                )
                features = frame_features(
                    cache,
                    frame,
                    observation,
                    candidate,
                    previous_initializer=previous_initial,
                    previous_nlf_body=previous_candidate,
                    elapsed_seconds=elapsed_seconds,
                )
            initial = cache_pose_from_full(full_initializer_rotations(cache, frame))
            candidate_pose = cache_pose_from_full(candidate)
            candidate_full.append(candidate)
            feature_rows.append([features[name] for name in FEATURE_COLUMNS])
            records.append(record)
            previous_initial = initial
            previous_candidate = candidate_pose
            previous_frame = frame
        predicted = router.predict(np.asarray(feature_rows, dtype=np.float32))
        gap = np.asarray(
            [row[FEATURE_COLUMNS.index("time_gap_reference_units")] for row in feature_rows],
            dtype=np.float64,
        )
        transition_scales = np.minimum(1.0, 1.0 / np.maximum(gap, 1.0))
        selected = viterbi_benefit_selection(
            predicted, margin, transition, transition_scales
        )

        result_dir = args.output_root / cache.clip_id / "smplifyx" / "results"
        result_dir.mkdir(parents=True)
        source_paths = []
        selection_rows = []
        for frame, frame_name in enumerate(cache.frame_names.astype(str)):
            source = args.baseline_root / cache.clip_id / "smplifyx" / "results" / f"{frame_name}.pkl"
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = result_dir / source.name
            source_paths.append(str(source))
            if selected[frame]:
                params = _selected_params(_load_params(source), candidate_full[frame])
                with destination.open("xb") as handle:
                    pickle.dump(params, handle, protocol=2)
            else:
                shutil.copyfile(source, destination)
            selection_rows.append(
                {
                    "frame_id": int(cache.frame_numbers[frame]),
                    "frame_name": frame_name,
                    "predicted_delta_deg": float(predicted[frame]),
                    "selected_nlf_body": bool(selected[frame]),
                    "nlf_output_relpath": records[frame]["output_relpath"],
                }
            )
        mesh_count = 0
        if args.render:
            mesh_count = render_source_anchored_directory(
                result_dir,
                args.output_root / cache.clip_id / "smplifyx" / "meshes",
                source_paths,
                args.model_folder,
                args.device,
            )
        diagnostics = args.output_root / cache.clip_id / "external_nlf_v2"
        diagnostics.mkdir(parents=True)
        (diagnostics / "selection.json").write_text(
            json.dumps(selection_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "clip_id": cache.clip_id,
            "frames": len(cache.frame_names),
            "meshes": mesh_count,
            "selected_frames": int(selected.sum()),
            "selection_fraction": float(selected.mean()),
            "cache_sha256": sha256_file(cache_path),
        }
        (diagnostics / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        total_frames += summary["frames"]
        total_selected += summary["selected_frames"]
        print(
            f"[external-nlf-v2] {len(summaries)}/{len(cache_paths)} "
            f"{cache.clip_id} selected={summary['selected_frames']}",
            flush=True,
        )
    if total_frames != 1493:
        raise ValueError(f"Expected 1493 frames, got {total_frames}")
    report = {
        "schema_version": "signal4d.external_nlf_v2_run.v1",
        "method": "SIGNAL4D_EXTERNAL_NLF_BODY_ROUTER_V2",
        "baseline": str(args.baseline_root.resolve()),
        "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
        "router_calibration": str(calibration_path.resolve()),
        "router_calibration_sha256": sha256_file(calibration_path),
        "router_sha256": sha256_file(router_path),
        "nlf_observation_metadata_sha256": sha256_file(
            args.observation_root / "run_metadata.json"
        ),
        "nlf_observation_contract": target_observation_contract,
        "smplx_model_sha256": sha256_file(args.model_path),
        "sgnify_training_or_selection_reads": 0,
        "sgnify_inference_images_used": True,
        "sgnify_ground_truth_used": False,
        "alpha": alpha,
        "margin_deg": margin,
        "transition_penalty_deg": transition,
        "global_wrist_preserved": True,
        "hand_pose_preserved": True,
        "rendered_during_materialization": bool(args.render),
        "clips": summaries,
        "frames": total_frames,
        "selected_frames": total_selected,
        "selection_fraction": total_selected / total_frames,
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-root", required=True, type=Path)
    parser.add_argument("--observation-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
