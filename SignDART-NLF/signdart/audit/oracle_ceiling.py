from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import yaml

from signdart.io.h1_state import H1State, read_manifest, sha256_file, state_path
from signdart.model import (
    create_model,
    forward_state_batch,
    rigid_transport_hand_vertices,
)


REPORT_NAMES = {
    "All": "all",
    "UBody": "upper_body",
    "UBody-F": "upper_body_minus_face",
    "UBody-H": "upper_body_minus_head",
    "LHand": "left_hand",
    "RHand": "right_hand",
}
ARM_SLOTS = {
    "shoulder_elbow_wrist": {
        "left": (15, 17, 19), "right": (16, 18, 20)
    },
    "collar_shoulder_elbow_wrist": {
        "left": (12, 15, 17, 19), "right": (13, 16, 18, 20)
    },
}


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    result = np.asarray(vertices, dtype=np.float32)
    if result.shape != (10475, 3):
        raise RuntimeError(f"invalid OBJ vertex array: {path} {result.shape}")
    return result


def centered_error(prediction: np.ndarray, target: np.ndarray, ids: np.ndarray) -> np.ndarray:
    prediction = prediction[ids] - prediction[ids].mean(axis=0, keepdims=True)
    target = target[ids] - target[ids].mean(axis=0, keepdims=True)
    return np.linalg.norm(prediction - target, axis=-1)


def compose_body_poses(
    base: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    candidate_space: str = "shoulder_elbow_wrist",
) -> np.ndarray:
    output = np.repeat(np.asarray(base, dtype=np.float32).reshape(1, 21, 3), len(left) * len(right), axis=0)
    slots = ARM_SLOTS[candidate_space]
    index = 0
    for left_pose in left.reshape(-1, 21, 3):
        for right_pose in right.reshape(-1, 21, 3):
            output[index, slots["left"]] = left_pose[list(slots["left"])]
            output[index, slots["right"]] = right_pose[list(slots["right"])]
            index += 1
    return output.reshape(-1, 63)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    records = read_manifest(paths["manifest"])
    protocol = json.loads(paths["protocol_lock"].read_text(encoding="utf-8"))
    pairing = {
        (item["sign_id"], int(frame["source_frame_id"])): int(frame["gt_frame_id"])
        for item in protocol["items"]
        for frame in item["frames"]
    }
    segment_root = paths["author_assets"] / "sgnify_part_segm_above_pelvis_joint"
    with (paths["author_assets"] / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle)
    base_regions = {
        "All": np.arange(10475, dtype=np.int64),
        "UBody": np.load(segment_root / "upper_body.npy"),
        "UBody-F": np.load(segment_root / "upper_body_minus_face.npy"),
        "UBody-H": np.load(segment_root / "upper_body_minus_head.npy"),
        "LHand": np.asarray(mano["left_hand"], dtype=np.int64),
        "RHand": np.asarray(mano["right_hand"], dtype=np.int64),
    }
    device = str(config["runtime"]["device"])
    model = create_model(paths["model_root"], device)
    baseline_vectors = {name: [] for name in base_regions}
    selected_vectors = {name: [] for name in base_regions}
    metric_oracle_vectors = {name: [] for name in base_regions}
    decisions = []
    target_name = str(config["oracle_gate"]["target_metric"])

    for ordinal, record in enumerate(records, start=1):
        state = H1State.load(state_path(paths["h1_state_root"], record))
        candidate_path = (
            paths["candidate_root"]
            / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        with np.load(candidate_path, allow_pickle=False) as archive:
            left_names = archive["left_names"].astype(str)
            right_names = archive["right_names"].astype(str)
            body_poses = compose_body_poses(
                state.arrays["body_pose"],
                archive["left_body_pose"],
                archive["right_body_pose"],
                str(config.get("candidate_space", "shoulder_elbow_wrist")),
            )
        vertices, joints = forward_state_batch(model, state, body_poses, device)
        # The exact incumbent is authoritative, including its serialized precision.
        incumbent_index = int(
            np.where(
                np.asarray([
                    left == "c0" and right == "c0"
                    for left in left_names
                    for right in right_names
                ])
            )[0][0]
        )
        vertices[incumbent_index] = state.vertices_evaluator
        if config.get("hand_preservation", {}).get("mode") == "rigid_h1_surface_transport":
            incumbent_joints = joints[incumbent_index].copy()
            for candidate_index in range(len(vertices)):
                for side, wrist_id, region_name in (
                    ("left", 20, "LHand"), ("right", 21, "RHand")
                ):
                    vertices[candidate_index] = rigid_transport_hand_vertices(
                        vertices[candidate_index],
                        joints[candidate_index],
                        state.vertices_evaluator,
                        incumbent_joints,
                        base_regions[region_name],
                        wrist_id,
                    )
        gt_frame = pairing[(record["sign_id"], int(record["source_frame_id"]))]
        target = load_obj_vertices(paths["gt_root"] / record["sign_id"] / f"{gt_frame:05d}.obj")
        regions = dict(base_regions)
        if record["sign_class"] == "0":
            left_ids = base_regions["LHand"]
            regions = {
                name: ids if name == "LHand" else np.setdiff1d(ids, left_ids)
                for name, ids in regions.items()
            }
        candidate_errors = {}
        for name, ids in regions.items():
            if name == "LHand" and record["sign_class"] == "0":
                continue
            candidate_errors[name] = np.stack(
                [centered_error(candidate, target, ids) for candidate in vertices]
            )
        target_means = candidate_errors[target_name].mean(axis=1)
        selected_index = int(np.argmin(target_means))
        names = [
            (str(left), str(right))
            for left in left_names
            for right in right_names
        ]
        decisions.append({
            "record_id": record["record_id"],
            "candidate_count": len(vertices),
            "selected_left": names[selected_index][0],
            "selected_right": names[selected_index][1],
            "selected_non_incumbent": selected_index != incumbent_index,
            "target_gain_mm": float(
                (target_means[incumbent_index] - target_means[selected_index]) * 1000.0
            ),
        })
        for name, errors in candidate_errors.items():
            baseline_vectors[name].append(errors[incumbent_index])
            selected_vectors[name].append(errors[selected_index])
            metric_oracle_vectors[name].append(errors[int(np.argmin(errors.mean(axis=1)))])
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"[G2] {ordinal}/{len(records)}", flush=True)

    def aggregate(vectors):
        return {
            name: float(np.concatenate(items).mean() * 1000.0)
            for name, items in vectors.items()
            if items
        }

    baseline = aggregate(baseline_vectors)
    selected = aggregate(selected_vectors)
    metric_oracle = aggregate(metric_oracle_vectors)
    delta = {name: selected[name] - baseline[name] for name in selected}
    gain = {name: -value for name, value in delta.items()}
    thresholds = config["oracle_gate"]["min_gain_mm"]
    hand_limit = float(config["oracle_gate"]["max_hand_regression_mm"])
    passed = bool(
        all(gain[name] >= float(threshold) for name, threshold in thresholds.items())
        and all(delta[name] <= hand_limit for name in ("LHand", "RHand") if name in delta)
    )
    report = {
        "schema_version": "signdart.g2_oracle_ceiling.v1",
        "banner": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "status": "pass" if passed else "fail",
        "selection_target": target_name,
        "frames": len(records),
        "selected_non_incumbent_frames": int(sum(item["selected_non_incumbent"] for item in decisions)),
        "baseline_metrics_mm": baseline,
        "target_selected_metrics_mm": selected,
        "target_selected_delta_mm": delta,
        "target_selected_gain_mm": gain,
        "independent_metric_oracle_mm": metric_oracle,
        "gate_min_gain_mm": thresholds,
        "gate_max_hand_regression_mm": hand_limit,
        "hand_preservation": config.get("hand_preservation", {"mode": "smplx_lbs_only"}),
        "candidate_space": str(config.get("candidate_space", "shoulder_elbow_wrist")),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "config_sha256": sha256_file(args.config),
        "uses_gt": True,
        "inference_artifact": False,
        "decisions": decisions,
    }
    output = paths["report_root"] / "g2_oracle_ceiling.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "banner", "status", "frames", "selected_non_incumbent_frames",
        "baseline_metrics_mm", "target_selected_metrics_mm",
        "target_selected_gain_mm", "independent_metric_oracle_mm",
    )}, indent=2))


if __name__ == "__main__":
    main()
