from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
import torch
import yaml

from signdart.geometry.arm_ik import (
    ARM_IDS,
    enumerate_body_pose_candidates,
    enumerate_three_link_body_pose_candidates,
    internal_intrinsics,
)
from signdart.geometry.ray_sphere import (
    enumerate_arm_branches,
    enumerate_three_link_branches,
    project,
)
from signdart.io.h1_state import H1State, read_manifest, sha256_file, state_path
from signdart.model import (
    create_model,
    forward_state_batch,
    rigid_transport_hand_vertices,
)


METRIC_NAMES = (
    "target_joint_error_mm",
    "reprojection_error_px",
    "bone_length_error_mm",
    "global_wrist_error_deg",
    "centered_hand_rms_mm",
)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def centered_rms_mm(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.square(first - second))) * 1000.0)


def incumbent_root_recovered(
    joints: np.ndarray,
    K_evaluator: np.ndarray,
    side: str,
    candidate_space: str,
) -> bool:
    ids = ARM_IDS[side]
    collar, shoulder, elbow, wrist = [
        joints[ids[key]] for key in ("collar", "shoulder", "elbow", "wrist")
    ]
    K = internal_intrinsics(K_evaluator)
    if candidate_space == "collar_shoulder_elbow_wrist":
        branches = enumerate_three_link_branches(
            collar,
            project(K, shoulder),
            project(K, elbow),
            project(K, wrist),
            float(np.linalg.norm(shoulder - collar)),
            float(np.linalg.norm(elbow - shoulder)),
            float(np.linalg.norm(wrist - elbow)),
            K,
        )
    else:
        branches = enumerate_arm_branches(
            shoulder,
            project(K, elbow),
            project(K, wrist),
            float(np.linalg.norm(elbow - shoulder)),
            float(np.linalg.norm(wrist - elbow)),
            K,
        )
    return any(
        (
            candidate_space != "collar_shoulder_elbow_wrist"
            or np.linalg.norm(np.asarray(branch["shoulder"]) - shoulder) <= 1e-5
        )
        and np.linalg.norm(np.asarray(branch["elbow"]) - elbow) <= 1e-5
        and np.linalg.norm(np.asarray(branch["wrist"]) - wrist) <= 1e-5
        for branch in branches
    )


def validate_candidates(
    model,
    state: H1State,
    joints0: np.ndarray,
    vertices0: np.ndarray,
    candidates,
    side: str,
    hand_ids: np.ndarray,
    device: str,
    limits: dict,
    hand_transport: bool,
):
    if len(candidates) == 1:
        return candidates, np.zeros((1, len(METRIC_NAMES)), dtype=np.float64), []
    poses = np.stack([candidate.body_pose for candidate in candidates])
    vertices, joints = forward_state_batch(model, state, poses, device)
    ids = ARM_IDS[side]
    K = internal_intrinsics(state.K_evaluator)
    original_uv = np.stack(
        [project(K, joints0[ids["elbow"]]), project(K, joints0[ids["wrist"]])]
    )
    collar0 = np.linalg.norm(joints0[ids["shoulder"]] - joints0[ids["collar"]])
    upper0 = np.linalg.norm(joints0[ids["elbow"]] - joints0[ids["shoulder"]])
    fore0 = np.linalg.norm(joints0[ids["wrist"]] - joints0[ids["elbow"]])
    metrics = []
    accepted = []
    rejected = []
    for index, candidate in enumerate(candidates):
        if index == 0:
            row = np.zeros(len(METRIC_NAMES), dtype=np.float64)
            accepted.append(candidate)
            metrics.append(row)
            continue
        if hand_transport:
            vertices[index] = rigid_transport_hand_vertices(
                vertices[index],
                joints[index],
                vertices0,
                joints0,
                hand_ids,
                ids["wrist"],
            )
        target_error = max(
            np.linalg.norm(joints[index, ids["shoulder"]] - candidate.shoulder_target),
            np.linalg.norm(joints[index, ids["elbow"]] - candidate.elbow_target),
            np.linalg.norm(joints[index, ids["wrist"]] - candidate.wrist_target),
        ) * 1000.0
        candidate_uv = np.stack([
            project(K, joints[index, ids["shoulder"]]),
            project(K, joints[index, ids["elbow"]]),
            project(K, joints[index, ids["wrist"]]),
        ])
        original_uv_full = np.concatenate(
            (project(K, joints0[ids["shoulder"]])[None], original_uv), axis=0
        )
        reprojection = float(np.max(np.linalg.norm(candidate_uv - original_uv_full, axis=-1)))
        collar = np.linalg.norm(
            joints[index, ids["shoulder"]] - joints[index, ids["collar"]]
        )
        upper = np.linalg.norm(joints[index, ids["elbow"]] - joints[index, ids["shoulder"]])
        fore = np.linalg.norm(joints[index, ids["wrist"]] - joints[index, ids["elbow"]])
        bone_error = float(
            max(abs(collar - collar0), abs(upper - upper0), abs(fore - fore0))
            * 1000.0
        )
        hand_error = centered_rms_mm(vertices[index, hand_ids], vertices0[hand_ids])
        row = np.asarray(
            [
                target_error,
                reprojection,
                bone_error,
                candidate.global_wrist_error_deg,
                hand_error,
            ],
            dtype=np.float64,
        )
        passed = bool(
            target_error <= float(limits["target_joint_abs_mm"])
            and reprojection <= float(limits["reprojection_max_px"])
            and bone_error <= float(limits["bone_length_abs_mm"])
            and candidate.global_wrist_error_deg <= float(limits["global_wrist_angle_deg"])
            and hand_error <= float(limits["centered_hand_rms_mm"])
        )
        if passed:
            accepted.append(candidate)
            metrics.append(row)
        else:
            rejected.append({
                "name": candidate.name,
                **{name: float(value) for name, value in zip(METRIC_NAMES, row)},
            })
    return accepted, np.stack(metrics), rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    device = str(config["runtime"]["device"])
    records = read_manifest(paths["manifest"])
    model = create_model(paths["model_root"], device)
    parents = model.parents[:22].detach().cpu().numpy().astype(np.int64)
    with (paths["model_root"] / "smplx" / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle)
    hand_ids = {
        side: np.asarray(mano[f"{side}_hand"], dtype=np.int64)
        for side in ("left", "right")
    }
    report_rows = []
    candidate_space = str(config.get("candidate_space", "shoulder_elbow_wrist"))
    max_baseline_error = 0.0
    for ordinal, record in enumerate(records, start=1):
        state_file = state_path(paths["h1_state_root"], record)
        state = H1State.load(state_file)
        vertices_batch, joints_batch = forward_state_batch(
            model, state, state.arrays["body_pose"], device
        )
        vertices0, joints0 = vertices_batch[0], joints_batch[0]
        baseline_error = float(
            np.max(np.linalg.norm(vertices0 - state.vertices_evaluator, axis=-1)) * 1000.0
        )
        max_baseline_error = max(max_baseline_error, baseline_error)
        if baseline_error > float(config["invariants"]["baseline_vertex_max_mm"]):
            raise RuntimeError(
                f"H1 forward reproduction failed for {record['record_id']}: {baseline_error:.6f} mm"
            )
        output_arrays = {
            "record_id": np.asarray(record["record_id"]),
            "h1_state_sha256": np.asarray(sha256_file(state_file)),
            "K_evaluator": state.K_evaluator,
        }
        row = {
            "record_id": record["record_id"],
            "baseline_vertex_max_mm": baseline_error,
            "sides": {},
        }
        for side in ("left", "right"):
            recovered = incumbent_root_recovered(
                joints0, state.K_evaluator, side, candidate_space
            )
            enumerate_candidates = (
                enumerate_three_link_body_pose_candidates
                if candidate_space == "collar_shoulder_elbow_wrist"
                else enumerate_body_pose_candidates
            )
            candidates = enumerate_candidates(
                state.arrays["global_orient"],
                state.arrays["body_pose"],
                parents,
                joints0,
                state.K_evaluator,
                side,
            )
            accepted, metrics, rejected = validate_candidates(
                model,
                state,
                joints0,
                vertices0,
                candidates,
                side,
                hand_ids[side],
                device,
                config["invariants"],
                config.get("hand_preservation", {}).get("mode")
                == "rigid_h1_surface_transport",
            )
            prefix = side
            output_arrays[f"{prefix}_names"] = np.asarray([item.name for item in accepted])
            output_arrays[f"{prefix}_body_pose"] = np.stack([item.body_pose for item in accepted])
            output_arrays[f"{prefix}_metrics"] = metrics
            row["sides"][side] = {
                "incumbent_root_recovered": recovered,
                "generated_including_c0": len(candidates),
                "valid_including_c0": len(accepted),
                "valid_alternatives": len(accepted) - 1,
                "rejected": rejected,
            }
        output = (
            paths["candidate_root"]
            / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        atomic_npz(output, **output_arrays)
        report_rows.append(row)
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"[G1] {ordinal}/{len(records)}", flush=True)

    side_rows = [side for row in report_rows for side in row["sides"].values()]
    root_rate = float(np.mean([item["incumbent_root_recovered"] for item in side_rows]))
    alternative_rate = float(np.mean([item["valid_alternatives"] > 0 for item in side_rows]))
    all_rejected_metrics = [
        reject
        for item in side_rows
        for reject in item["rejected"]
    ]
    report = {
        "schema_version": "signdart.g1_candidate_coverage.v1",
        "status": "pass" if root_rate >= 0.95 and alternative_rate >= 0.60 else "fail",
        "frames": len(records),
        "arm_sides": len(side_rows),
        "incumbent_root_recovery_rate": root_rate,
        "valid_alternative_rate": alternative_rate,
        "max_h1_forward_vertex_error_mm": max_baseline_error,
        "invariant_limits": config["invariants"],
        "hand_preservation": config.get("hand_preservation", {"mode": "smplx_lbs_only"}),
        "candidate_space": candidate_space,
        "rejected_candidate_count": len(all_rejected_metrics),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "config_sha256": sha256_file(args.config),
        "items": report_rows,
    }
    report_path = paths["report_root"] / "g1_candidate_invariants.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "frames", "arm_sides", "incumbent_root_recovery_rate",
        "valid_alternative_rate", "max_h1_forward_vertex_error_mm",
        "rejected_candidate_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
