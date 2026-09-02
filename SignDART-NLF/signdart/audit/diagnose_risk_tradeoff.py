from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import yaml

from signdart.audit.oracle_ceiling import (
    REPORT_NAMES,
    centered_error,
    compose_body_poses,
    load_obj_vertices,
)
from signdart.io.h1_state import H1State, read_manifest, state_path
from signdart.model import create_model, forward_state_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--thresholds-mm",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 5.0],
    )
    parser.add_argument("--output", type=Path, required=True)
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
    thresholds = [float(value) for value in args.thresholds_mm]
    vectors = {
        threshold: {"baseline": {name: [] for name in base_regions},
                    "selected": {name: [] for name in base_regions},
                    "changed": 0}
        for threshold in thresholds
    }
    model = create_model(paths["model_root"], str(config["runtime"]["device"]))
    for ordinal, record in enumerate(records, start=1):
        state = H1State.load(state_path(paths["h1_state_root"], record))
        candidate_path = (
            paths["candidate_root"] / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        with np.load(candidate_path, allow_pickle=False) as archive:
            left_names = archive["left_names"].astype(str)
            right_names = archive["right_names"].astype(str)
            left_risk = archive["left_metrics"][:, 4]
            right_risk = archive["right_metrics"][:, 4]
            poses = compose_body_poses(
                state.arrays["body_pose"], archive["left_body_pose"],
                archive["right_body_pose"],
                str(config.get("candidate_space", "shoulder_elbow_wrist")),
            )
        vertices, _ = forward_state_batch(
            model, state, poses, str(config["runtime"]["device"])
        )
        incumbent = int(np.where(np.asarray([
            left == "c0" and right == "c0"
            for left in left_names for right in right_names
        ]))[0][0])
        vertices[incumbent] = state.vertices_evaluator
        gt_frame = pairing[(record["sign_id"], int(record["source_frame_id"]))]
        target = load_obj_vertices(
            paths["gt_root"] / record["sign_id"] / f"{gt_frame:05d}.obj"
        )
        regions = dict(base_regions)
        if record["sign_class"] == "0":
            left_ids = base_regions["LHand"]
            regions = {
                name: ids if name == "LHand" else np.setdiff1d(ids, left_ids)
                for name, ids in regions.items()
            }
        errors = {
            name: np.stack([centered_error(candidate, target, ids) for candidate in vertices])
            for name, ids in regions.items()
            if not (name == "LHand" and record["sign_class"] == "0")
        }
        target_means = errors["UBody-H"].mean(axis=1)
        pair_risk = np.asarray([
            max(float(left_risk[left_index]), float(right_risk[right_index]))
            for left_index in range(len(left_names))
            for right_index in range(len(right_names))
        ])
        for threshold in thresholds:
            eligible = np.where(pair_risk <= threshold + 1e-12)[0]
            selected = int(eligible[np.argmin(target_means[eligible])])
            vectors[threshold]["changed"] += int(selected != incumbent)
            for name, array in errors.items():
                vectors[threshold]["baseline"][name].append(array[incumbent])
                vectors[threshold]["selected"][name].append(array[selected])
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"[risk-diagnostic] {ordinal}/{len(records)}", flush=True)

    results = []
    for threshold in thresholds:
        item = vectors[threshold]
        baseline = {
            name: float(np.concatenate(values).mean() * 1000.0)
            for name, values in item["baseline"].items() if values
        }
        selected = {
            name: float(np.concatenate(values).mean() * 1000.0)
            for name, values in item["selected"].items() if values
        }
        results.append({
            "threshold_mm": threshold,
            "selected_non_incumbent_frames": item["changed"],
            "gain_mm": {name: baseline[name] - selected[name] for name in selected},
        })
    report = {
        "schema_version": "signdart.development_risk_tradeoff.v1",
        "banner": "RETROSPECTIVE_ENGINEERING12_DESIGN_DIAGNOSTIC",
        "uses_gt": True,
        "frames": len(records),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
