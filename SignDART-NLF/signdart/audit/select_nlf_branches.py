from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from signdart.io.h1_state import H1State, read_manifest, sha256_file, state_path
from signdart.model import create_model, forward_state_batch
from signdart.selector import branch_scores, compose_selected_pose


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
    parser.add_argument("--nlf-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    records = read_manifest(paths["manifest"])
    nlf_index = {}
    with (args.nlf_root / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            nlf_index[(str(row["clip_id"]), int(row["frame_id"]))] = row
    model = create_model(paths["model_root"], str(config["runtime"]["device"]))
    decisions = []
    for ordinal, record in enumerate(records, start=1):
        state = H1State.load(state_path(paths["h1_state_root"], record))
        candidate_path = (
            paths["candidate_root"] / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        key = (record["sign_id"], int(record["source_frame_id"]))
        nlf_record = nlf_index[key]
        with np.load(args.nlf_root / nlf_record["output_relpath"]) as archive:
            nlf_parametric = archive["joints3d"].copy()
            nlf_nonparametric = archive["joints3d_nonparam"].copy()
            nlf_uncertainty = archive["joint_uncertainties"].copy()
        selected = {}
        side_rows = {}
        with np.load(candidate_path, allow_pickle=False) as archive:
            for side in ("left", "right"):
                poses = archive[f"{side}_body_pose"]
                names = archive[f"{side}_names"].astype(str)
                _, joints = forward_state_batch(
                    model, state, poses, str(config["runtime"]["device"])
                )
                scores = branch_scores(
                    joints, nlf_parametric, nlf_nonparametric, nlf_uncertainty, side
                )
                index = int(np.argmin(scores))
                selected[side] = poses[index]
                c0_index = int(np.where(names == "c0")[0][0])
                side_rows[side] = {
                    "selected_name": str(names[index]),
                    "selected_index": index,
                    "candidate_count": len(names),
                    "selected_score": float(scores[index]),
                    "c0_score": float(scores[c0_index]),
                    "score_gain_vs_c0": float(scores[c0_index] - scores[index]),
                    "selected_non_incumbent": index != c0_index,
                }
        body_pose = compose_selected_pose(
            state.arrays["body_pose"], selected["left"], selected["right"]
        )
        destination = (
            args.output_root / "frames" / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"]),
            body_pose=body_pose,
            left_name=np.asarray(side_rows["left"]["selected_name"]),
            right_name=np.asarray(side_rows["right"]["selected_name"]),
        )
        decisions.append({"record_id": record["record_id"], "sides": side_rows})
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"[G4-select] {ordinal}/{len(records)}", flush=True)
    side_rows = [side for item in decisions for side in item["sides"].values()]
    changed_frames = sum(
        any(side["selected_non_incumbent"] for side in item["sides"].values())
        for item in decisions
    )
    report = {
        "schema_version": "signdart.nlf_branch_selector.v1",
        "status": "complete",
        "frames": len(records),
        "selected_non_incumbent_frames": changed_frames,
        "selected_non_incumbent_sides": int(sum(
            side["selected_non_incumbent"] for side in side_rows
        )),
        "selection_fraction": changed_frames / len(records),
        "trained_parameters": 0,
        "uses_gt": False,
        "selector": "uncertainty_weighted_normalized_bone_direction_argmin",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "candidate_config_sha256": sha256_file(args.config),
        "nlf_metadata_sha256": sha256_file(args.nlf_root / "run_metadata.json"),
        "decisions": decisions,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "status", "frames", "selected_non_incumbent_frames",
        "selected_non_incumbent_sides", "selection_fraction", "trained_parameters",
        "uses_gt",
    )}, indent=2))


if __name__ == "__main__":
    main()
