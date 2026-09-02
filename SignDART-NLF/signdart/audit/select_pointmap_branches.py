from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from signdart.io.h1_state import H1State, read_manifest, sha256_file, state_path
from signdart.model import create_model, forward_state_batch
from signdart.pointmap import pointmap_bootstrap_decision
from signdart.selector import compose_selected_pose


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--axes-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    forbidden = {"gt_root", "protocol_lock", "author_assets"}.intersection(
        config["paths"]
    )
    if forbidden:
        raise ValueError("selector config exposes evaluation-only paths")
    paths = {key: Path(value) for key, value in config["paths"].items()}
    records = read_manifest(paths["manifest"])
    if args.limit is not None:
        records = records[: args.limit]
    model = create_model(paths["model_root"], str(config["runtime"]["device"]))
    decisions = []
    for ordinal, record in enumerate(records, start=1):
        state = H1State.load(state_path(paths["h1_state_root"], record))
        candidate_path = (
            paths["candidate_root"] / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        axes_path = (
            args.axes_root / "frames" / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        selected_poses = {}
        side_rows = {}
        with np.load(candidate_path, allow_pickle=False) as candidates, np.load(
            axes_path, allow_pickle=False
        ) as evidence:
            for side in ("left", "right"):
                poses = candidates[f"{side}_body_pose"]
                names = candidates[f"{side}_names"].astype(str)
                incumbent = int(np.where(names == "c0")[0][0])
                part_names = (f"{side}_upper", f"{side}_forearm")
                if not all(bool(evidence[f"{name}_valid"]) for name in part_names):
                    selected = incumbent
                    diagnostics = {"reason": "pointmap_part_invalid"}
                else:
                    _, joints = forward_state_batch(
                        model, state, poses, str(config["runtime"]["device"])
                    )
                    axes = np.stack([evidence[f"{name}_axis"] for name in part_names])
                    boot = [evidence[f"{name}_bootstrap_axes"] for name in part_names]
                    ci = np.deg2rad(np.asarray([
                        float(evidence[f"{name}_ci95_deg"]) for name in part_names
                    ]))
                    gap = np.asarray([
                        float(evidence[f"{name}_eigen_gap"]) for name in part_names
                    ])
                    reliability = np.maximum(gap, 0.0) / np.square(
                        ci + np.deg2rad(5.0)
                    )
                    selected, diagnostics = pointmap_bootstrap_decision(
                        joints, axes, boot, reliability, side, incumbent
                    )
                selected_poses[side] = poses[selected]
                side_rows[side] = {
                    "selected_name": str(names[selected]),
                    "selected_index": selected,
                    "candidate_count": len(names),
                    "selected_non_incumbent": selected != incumbent,
                    **diagnostics,
                }
        body_pose = compose_selected_pose(
            state.arrays["body_pose"], selected_poses["left"], selected_poses["right"]
        )
        destination = (
            args.output_root / "frames" / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        atomic_npz(
            destination, record_id=np.asarray(record["record_id"]),
            body_pose=body_pose,
            left_name=np.asarray(side_rows["left"]["selected_name"]),
            right_name=np.asarray(side_rows["right"]["selected_name"]),
        )
        decisions.append({"record_id": record["record_id"], "sides": side_rows})
        if ordinal % 50 == 0 or ordinal == len(records):
            print(f"[pointmap-select] {ordinal}/{len(records)}", flush=True)
    sides = [side for row in decisions for side in row["sides"].values()]
    changed_frames = sum(
        any(side["selected_non_incumbent"] for side in row["sides"].values())
        for row in decisions
    )
    report = {
        "schema_version": "signray.pointmap_selection.v1",
        "status": "prediction_complete_and_locked_before_evaluation",
        "frames": len(records),
        "selected_non_incumbent_frames": int(changed_frames),
        "selected_non_incumbent_sides": int(sum(
            side["selected_non_incumbent"] for side in sides
        )),
        "selection_fraction": changed_frames / max(len(records), 1),
        "trained_parameters": 0,
        "uses_gt": False,
        "uses_sgnify_for_training_or_tuning": False,
        "decision_reason_counts": dict(Counter(side["reason"] for side in sides)),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "config_sha256": sha256_file(args.config),
        "axes_run_sha256": sha256_file(args.axes_root / "run.json"),
        "decisions": decisions,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: report[key] for key in (
            "status", "frames", "selected_non_incumbent_frames",
            "selected_non_incumbent_sides", "selection_fraction",
            "trained_parameters", "uses_gt", "decision_reason_counts",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
