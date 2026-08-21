#!/usr/bin/env python3
"""Train a GT-free-at-inference NLF/V6 body router and materialize SMPL-X meshes.

The router is fitted only on the development partition. Its conservative
selection margin is chosen only on the calibration partition. Test target
values are discarded before feature/target materialization and are never used
for fitting or selection. The selected NLF body rotations are fused with the
V6 wrist and hand rotations, then forwarded through SMPL-X; no vertex splicing
is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import smplx
import torch
from safetensors.numpy import load_file, save_file
from scipy.spatial.transform import Rotation
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLUMNS = (
    "nlf_unc_body",
    "nlf_unc_all",
    "nlf_fit_body",
    "nlf_fit_all",
    "disagree_ubody",
    "disagree_joints",
    "nlf_velocity",
    "v6_velocity",
    "v6_unc_body",
    "risk0",
    "risk1",
    "risk2",
    "box_score",
    "box_area",
)
MARGINS_MM = (0.0, 0.5, 1.0, 2.0)
CAMERA_X_180 = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_index(path: Path) -> Dict[tuple[str, int], Dict[str, Any]]:
    result: Dict[tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[(str(row["clip_id"]), int(row["frame_id"]))] = row
    return result


def load_sign_classes(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields:
                result[fields[0]] = fields[1]
    return result


def centered_mean_distance_mm(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(first - second, axis=-1).mean() * 1000.0)


def build_features(
    observation_root: Path,
    v6_root: Path,
    nlf_error_csv: Path,
    v6_error_csv: Path,
    upper_body_indices: np.ndarray,
    left_hand_indices: np.ndarray,
    classes: Dict[str, str],
) -> pd.DataFrame:
    index = load_jsonl_index(observation_root / "index.jsonl")
    split_table = pd.DataFrame(
        {
            "sign": key[0],
            "frame": key[1],
            "split": record["split"],
        }
        for key, record in index.items()
    )
    nlf_errors = pd.read_csv(nlf_error_csv)
    v6_errors = pd.read_csv(v6_error_csv).rename(
        columns={"clip_id": "sign", "frame_id": "frame"}
    )
    # Materialize supervision only for development/calibration. In particular,
    # no test error is merged into or saved with the inference features.
    target_splits = ("development", "calibration")
    nlf_targets = nlf_errors.loc[
        nlf_errors["split"].isin(target_splits),
        ["sign", "frame", "tr_upper_body_minus_face_mm"],
    ].copy()
    v6_targets = v6_errors.merge(
        split_table,
        on=["sign", "frame"],
        validate="one_to_one",
    )
    v6_targets = v6_targets[v6_targets["split"].isin(target_splits)]
    error_columns = v6_targets[
        ["sign", "frame", "tr_upper_body_minus_face_mm"]
    ]
    targets = nlf_targets.merge(
        error_columns,
        on=["sign", "frame"],
        suffixes=("_nlf", "_v6"),
        validate="one_to_one",
    )
    rows: List[Dict[str, Any]] = []

    for sign in sorted({key[0] for key in index}):
        v6 = load_file(v6_root / sign / "prediction.safetensors")
        frame_ids = v6["frame_ids"].astype(np.int64)
        previous_nlf_joints = None
        previous_v6_joints = None
        for frame_index, frame_id in enumerate(frame_ids):
            record = index[(sign, int(frame_id))]
            observation = np.load(observation_root / str(record["output_relpath"]))
            nlf_vertices = observation["vertices3d"] / 1000.0 * CAMERA_X_180
            nlf_joints = observation["joints3d"] / 1000.0 * CAMERA_X_180
            nlf_joints_nonparam = (
                observation["joints3d_nonparam"] / 1000.0 * CAMERA_X_180
            )
            v6_vertices = v6["vertices"][frame_index]
            v6_joints = v6["joints_3d"][frame_index]
            indices = upper_body_indices
            if classes[sign] == "0":
                indices = np.setdiff1d(indices, left_hand_indices)

            fit_residual = np.linalg.norm(
                (nlf_joints - nlf_joints.mean(axis=0, keepdims=True))
                - (
                    nlf_joints_nonparam
                    - nlf_joints_nonparam.mean(axis=0, keepdims=True)
                ),
                axis=-1,
            )
            nlf_velocity = 0.0
            v6_velocity = 0.0
            if previous_nlf_joints is not None:
                nlf_velocity = centered_mean_distance_mm(nlf_joints, previous_nlf_joints)
                v6_velocity = centered_mean_distance_mm(v6_joints, previous_v6_joints)
            joint_uncertainty = observation["joint_uncertainties"]
            rows.append(
                {
                    "sign": sign,
                    "frame": int(frame_id),
                    "split": record["split"],
                    "nlf_unc_body": float(joint_uncertainty[:22].mean()),
                    "nlf_unc_all": float(joint_uncertainty.mean()),
                    "nlf_fit_body": float(fit_residual[:22].mean() * 1000.0),
                    "nlf_fit_all": float(fit_residual.mean() * 1000.0),
                    "disagree_ubody": centered_mean_distance_mm(
                        nlf_vertices[indices], v6_vertices[indices]
                    ),
                    "disagree_joints": centered_mean_distance_mm(
                        nlf_joints[:22], v6_joints[:22]
                    ),
                    "nlf_velocity": nlf_velocity,
                    "v6_velocity": v6_velocity,
                    "v6_unc_body": float(v6["uncertainty"][frame_index, :22].mean()),
                    "risk0": float(v6["risk_score"][frame_index, 0]),
                    "risk1": float(v6["risk_score"][frame_index, 1]),
                    "risk2": float(v6["risk_score"][frame_index, 2]),
                    "box_score": float(record["box"][4]),
                    "box_area": float(record["box"][2] * record["box"][3]),
                }
            )
            previous_nlf_joints = nlf_joints
            previous_v6_joints = v6_joints

    features = pd.DataFrame(rows).merge(
        targets,
        on=["sign", "frame"],
        how="left",
        validate="one_to_one",
    )
    features["delta_nlf_minus_v6_mm"] = (
        features["tr_upper_body_minus_face_mm_nlf"]
        - features["tr_upper_body_minus_face_mm_v6"]
    )
    test = features[features["split"] == "test"]
    target_columns = (
        "tr_upper_body_minus_face_mm_nlf",
        "tr_upper_body_minus_face_mm_v6",
        "delta_nlf_minus_v6_mm",
    )
    if not test[list(target_columns)].isna().all().all():
        raise AssertionError("test targets must not be materialized")
    return features


def fit_router(features: pd.DataFrame) -> tuple[RandomForestRegressor, float, Dict[str, Any]]:
    train = features[features["split"] == "development"]
    calibration = features[features["split"] == "calibration"].copy()
    if train.empty or calibration.empty:
        raise ValueError("development and calibration partitions are required")
    router = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=8,
        max_features=0.8,
        random_state=12345,
        n_jobs=-1,
    )
    router.fit(train[list(FEATURE_COLUMNS)], train["delta_nlf_minus_v6_mm"])
    calibration["predicted_delta_mm"] = router.predict(
        calibration[list(FEATURE_COLUMNS)]
    )
    margin_rows = []
    for margin in MARGINS_MM:
        choose = calibration["predicted_delta_mm"] < -margin
        hybrid = np.where(
            choose,
            calibration["tr_upper_body_minus_face_mm_nlf"],
            calibration["tr_upper_body_minus_face_mm_v6"],
        )
        margin_rows.append(
            {
                "margin_mm": margin,
                "selection_fraction": float(choose.mean()),
                "frame_macro_mm": float(hybrid.mean()),
                "gain_vs_v6_mm": float(
                    calibration["tr_upper_body_minus_face_mm_v6"].mean()
                    - hybrid.mean()
                ),
            }
        )
    selected = max(margin_rows, key=lambda row: row["gain_vs_v6_mm"])
    report = {
        "training_partition": "development",
        "training_frames": int(len(train)),
        "threshold_partition": "calibration",
        "calibration_frames": int(len(calibration)),
        "candidate_margins": margin_rows,
        "selected_margin_mm": selected["margin_mm"],
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, router.feature_importances_.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "test_targets_materialized": False,
        "test_labels_used_for_training_or_selection": False,
    }
    return router, float(selected["margin_mm"]), report


def geodesic_blend(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Move from ``first`` toward ``second`` along SO(3) geodesics."""
    delta = second @ np.swapaxes(first, -1, -2)
    tangent = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec()
    step = Rotation.from_rotvec(alpha * tangent).as_matrix().reshape(delta.shape)
    return (step @ first).astype(np.float32)


def global_rotations(local_rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Compose SMPL-X local rotations along the kinematic tree."""
    global_values = np.empty_like(local_rotations)
    for joint_index, parent_index in enumerate(parents):
        if int(parent_index) < 0:
            global_values[joint_index] = local_rotations[joint_index]
        else:
            global_values[joint_index] = (
                global_values[int(parent_index)] @ local_rotations[joint_index]
            )
    return global_values


def preserve_global_rotations(
    reference_local: np.ndarray,
    candidate_local: np.ndarray,
    parents: np.ndarray,
    joint_indices: Sequence[int],
) -> np.ndarray:
    """Compensate local rotations so selected joints retain reference globals."""
    result = candidate_local.copy()
    reference_global = global_rotations(reference_local, parents)
    for joint_index in joint_indices:
        parent_index = int(parents[joint_index])
        candidate_global = global_rotations(result, parents)
        if parent_index < 0:
            result[joint_index] = reference_global[joint_index]
        else:
            result[joint_index] = (
                candidate_global[parent_index].T @ reference_global[joint_index]
            )
    return result.astype(np.float32)


def materialize(
    features: pd.DataFrame,
    router: RandomForestRegressor,
    margin_mm: float,
    alpha: float,
    observation_root: Path,
    v6_root: Path,
    baseline_parameter_root: Path,
    model_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    features = features.copy()
    features["predicted_delta_mm"] = router.predict(features[list(FEATURE_COLUMNS)])
    features["select_nlf_body"] = features["predicted_delta_mm"] < -margin_mm
    output_root.mkdir(parents=True, exist_ok=True)
    index = load_jsonl_index(observation_root / "index.jsonl")
    model_hash = sha256_file(model_path)
    model = smplx.SMPLX(
        str(model_path),
        gender="neutral",
        ext=model_path.suffix.lstrip("."),
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    ).eval()
    parents = model.parents[:55].detach().cpu().numpy().astype(np.int64)
    selection_rows: List[Dict[str, Any]] = []
    selected_count = 0

    for sign, sign_features in features.groupby("sign", sort=True):
        v6 = load_file(v6_root / sign / "prediction.safetensors")
        frame_ids = v6["frame_ids"].astype(np.int64)
        row_by_frame = sign_features.set_index("frame")
        output = {key: value.copy() for key, value in v6.items()}
        selected_indices = [
            index_in_clip
            for index_in_clip, frame_id in enumerate(frame_ids)
            if bool(row_by_frame.loc[int(frame_id), "select_nlf_body"])
        ]

        if selected_indices:
            poses = []
            betas = []
            expressions = []
            translations = []
            blended_rotations = []
            for index_in_clip in selected_indices:
                frame_id = int(frame_ids[index_in_clip])
                record = index[(sign, frame_id)]
                observation = np.load(observation_root / str(record["output_relpath"]))
                nlf_rotations = Rotation.from_rotvec(
                    observation["pose"].reshape(55, 3)
                ).as_matrix().astype(np.float32)
                fused_rotations = geodesic_blend(
                    v6["rotations"][index_in_clip], nlf_rotations, alpha
                )
                # Preserve all local face/hand articulation, then compensate the
                # wrist locals so their *global* orientations remain exactly V6.
                # Copying the local wrists alone is insufficient after changing
                # the upstream shoulder/elbow rotations.
                fused_rotations[22:55] = v6["rotations"][index_in_clip, 22:55]
                fused_rotations = preserve_global_rotations(
                    v6["rotations"][index_in_clip],
                    fused_rotations,
                    parents,
                    (20, 21),
                )

                parameter_path = (
                    baseline_parameter_root
                    / sign
                    / "smplifyx"
                    / "results"
                    / f"low_{frame_id}.pkl"
                )
                with parameter_path.open("rb") as handle:
                    baseline = pickle.load(handle)
                baseline_betas = np.asarray(baseline["betas"], dtype=np.float32).reshape(10)
                # NLF shape is frame-varying and changes hand geometry. Keep the
                # frozen V6/DexAvatar identity while refining body articulation.
                fused_betas = baseline_betas

                poses.append(
                    Rotation.from_matrix(fused_rotations).as_rotvec().astype(np.float32)
                )
                betas.append(fused_betas.astype(np.float32))
                expressions.append(
                    np.asarray(baseline["expression"], dtype=np.float32).reshape(10)
                )
                translations.append(v6["translation"][index_in_clip])
                blended_rotations.append(fused_rotations)

            poses_tensor = torch.from_numpy(np.stack(poses))
            with torch.inference_mode():
                body = model(
                    global_orient=poses_tensor[:, 0],
                    body_pose=poses_tensor[:, 1:22].flatten(1),
                    jaw_pose=poses_tensor[:, 22],
                    leye_pose=poses_tensor[:, 23],
                    reye_pose=poses_tensor[:, 24],
                    left_hand_pose=poses_tensor[:, 25:40].flatten(1),
                    right_hand_pose=poses_tensor[:, 40:55].flatten(1),
                    betas=torch.from_numpy(np.stack(betas)),
                    expression=torch.from_numpy(np.stack(expressions)),
                    transl=torch.from_numpy(np.stack(translations)),
                    return_verts=True,
                )
            vertices = body.vertices.detach().cpu().numpy() * CAMERA_X_180
            joints = body.joints[:, :55].detach().cpu().numpy() * CAMERA_X_180
            output["vertices"][selected_indices] = vertices.astype(np.float32)
            output["joints_3d"][selected_indices] = joints.astype(np.float32)
            output["rotations"][selected_indices] = np.stack(blended_rotations)

        clip_output = output_root / "predictions" / sign
        clip_output.mkdir(parents=True, exist_ok=True)
        save_file(output, clip_output / "prediction.safetensors")
        metadata = {
            "schema_version": "1.0",
            "method_name": "SIGNAL4D_V7_NLFBodyRouter",
            "clip_id": sign,
            "frames": int(len(frame_ids)),
            "frame_ids": frame_ids.tolist(),
            "artifact_sha256": sha256_file(
                clip_output / "prediction.safetensors"
            ),
            "smplx_model_sha256": model_hash,
            "coordinate_convention": "opencv_x_right_y_down_z_forward",
            "length_unit": "meter",
            "selected_frames": int(len(selected_indices)),
            "gt_used_for_inference": False,
            "alpha": alpha,
            "margin_mm": margin_mm,
            "preserved_rotation_indices": (
                "global wrists 20:21; local face, eyes, hands 22:55"
            ),
            "shape_policy": "preserve_v6_betas",
        }
        (clip_output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for frame_id in frame_ids:
            row = row_by_frame.loc[int(frame_id)]
            selection_rows.append(
                {
                    "sign": sign,
                    "frame": int(frame_id),
                    "split": row["split"],
                    "predicted_delta_mm": float(row["predicted_delta_mm"]),
                    "selected_nlf_body": bool(row["select_nlf_body"]),
                }
            )
        selected_count += len(selected_indices)

    with (output_root / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    return {
        "frames": len(features),
        "selected_frames": selected_count,
        "selection_fraction": selected_count / len(features),
        "alpha": alpha,
        "margin_mm": margin_mm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-root", required=True, type=Path)
    parser.add_argument("--v6-root", required=True, type=Path)
    parser.add_argument("--baseline-parameter-root", required=True, type=Path)
    parser.add_argument("--nlf-error-csv", required=True, type=Path)
    parser.add_argument("--v6-error-csv", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--sign-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    with (args.asset_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle)
    upper_body = np.load(
        args.asset_root
        / "sgnify_part_segm_above_pelvis_joint"
        / "upper_body_minus_face.npy"
    ).astype(np.int64)
    classes = load_sign_classes(args.sign_file)
    started = time.time()
    features = build_features(
        args.observation_root,
        args.v6_root,
        args.nlf_error_csv,
        args.v6_error_csv,
        upper_body,
        np.asarray(mano["left_hand"], dtype=np.int64),
        classes,
    )
    router, margin_mm, router_report = fit_router(features)
    args.output_root.mkdir(parents=True, exist_ok=True)
    features[["sign", "frame", "split", *FEATURE_COLUMNS]].to_csv(
        args.output_root / "router_features.csv", index=False
    )
    joblib.dump(router, args.output_root / "router.joblib")
    materialization = materialize(
        features,
        router,
        margin_mm,
        args.alpha,
        args.observation_root,
        args.v6_root,
        args.baseline_parameter_root,
        args.model_path,
        args.output_root,
    )
    report = {
        "schema_version": "signal4d.v7_nlf_body_router.v1",
        "router": router_report,
        "materialization": materialization,
        "runtime_seconds": time.time() - started,
        "inputs": {
            "nlf_observation_metadata_sha256": sha256_file(
                args.observation_root / "run_metadata.json"
            ),
            "nlf_error_csv_sha256": sha256_file(args.nlf_error_csv),
            "v6_error_csv_sha256": sha256_file(args.v6_error_csv),
            "smplx_model_sha256": sha256_file(args.model_path),
        },
        "claim_boundary": (
            "Exploratory: development labels train the router and calibration labels "
            "select the margin; test target values are not materialized or used for "
            "training/selection. The test result has nevertheless been inspected "
            "during method development."
        ),
    }
    (args.output_root / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
