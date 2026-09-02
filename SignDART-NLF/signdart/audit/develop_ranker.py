from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import joblib
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signdart.audit.oracle_ceiling import centered_error, compose_body_poses, load_obj_vertices
from signdart.features import FEATURE_NAMES, candidate_features
from signdart.io.h1_state import H1State, read_manifest, sha256_file, state_path
from signdart.model import create_model, forward_state_batch


MARGINS_MM = (
    0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0,
    1.5, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 4.0, 5.0,
)
TEMPORAL_WINDOWS = (1, 3, 5)
PROBABILITY_MARGINS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
CLASSIFIER_TARGETS_MM = {
    "clf_pos0": 0.0,
    "clf_gain05": 0.5,
    "clf_gain10": 1.0,
    "clf_gain20": 2.0,
}


def model_factories():
    return {
        "ridge_a1": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_a10": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist_l7_r1": lambda: HistGradientBoostingRegressor(
            max_iter=250, max_leaf_nodes=7, min_samples_leaf=20,
            l2_regularization=1.0, random_state=20260902,
        ),
        "hist_l15_r10": lambda: HistGradientBoostingRegressor(
            max_iter=250, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=10.0, random_state=20260902,
        ),
        **{
            name: (lambda: HistGradientBoostingClassifier(
                max_iter=250, max_leaf_nodes=15, min_samples_leaf=20,
                l2_regularization=10.0, random_state=20260902,
            ))
            for name in CLASSIFIER_TARGETS_MM
        },
    }


def aggregate_selection(
    frames, predictions: np.ndarray, margin: float, temporal_window: int = 1
):
    metric_names = sorted({
        name for frame in frames for name in frame["pair_sums"]
    })
    sums = {name: 0.0 for name in metric_names}
    baseline_sums = {name: 0.0 for name in sums}
    counts = {name: 0 for name in sums}
    changed_frames = 0
    selected_sides = 0
    raw = []
    for frame in frames:
        item = {"sign": frame["sign"], "sides": {}}
        for side in ("left", "right"):
            rows = frame["rows"][side]
            side_predictions = predictions[rows].copy()
            side_predictions[0] = 0.0
            best = int(np.argmax(side_predictions))
            item["sides"][side] = (best, float(side_predictions[best]))
        raw.append(item)
    selected_by_frame = [{"left": 0, "right": 0} for _ in frames]
    radius = temporal_window // 2
    for side in ("left", "right"):
        for sign in sorted({item["sign"] for item in raw}):
            indices = [i for i, item in enumerate(raw) if item["sign"] == sign]
            confidence = np.asarray([raw[i]["sides"][side][1] for i in indices])
            for local_index, frame_index in enumerate(indices):
                start = max(0, local_index - radius)
                stop = min(len(indices), local_index + radius + 1)
                supported = float(np.median(confidence[start:stop]))
                best, current = raw[frame_index]["sides"][side]
                if best != 0 and current > 0.0 and supported > margin:
                    selected_by_frame[frame_index][side] = best
    for frame_index, frame in enumerate(frames):
        selected = selected_by_frame[frame_index]
        selected_sides += int(selected["left"] != 0) + int(selected["right"] != 0)
        changed_frames += int(selected["left"] != 0 or selected["right"] != 0)
        pair_index = selected["left"] * frame["right_count"] + selected["right"]
        for name in sums:
            if name not in frame["pair_sums"]:
                continue
            sums[name] += float(frame["pair_sums"][name][pair_index])
            baseline_sums[name] += float(frame["pair_sums"][name][0])
            counts[name] += int(frame["counts"][name])
    selected_metrics = {name: sums[name] / counts[name] * 1000.0 for name in sums}
    baseline_metrics = {
        name: baseline_sums[name] / counts[name] * 1000.0 for name in sums
    }
    gain = {name: baseline_metrics[name] - selected_metrics[name] for name in sums}
    return {
        "margin_mm": margin,
        "temporal_window_frames": temporal_window,
        "selected_non_incumbent_frames": changed_frames,
        "selected_non_incumbent_sides": selected_sides,
        "selection_fraction": changed_frames / len(frames),
        "gain_mm": gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nlf-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(model_factories()),
        default=list(model_factories()),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    records = read_manifest(paths["manifest"])
    protocol = json.loads(paths["protocol_lock"].read_text(encoding="utf-8"))
    pairing = {
        (item["sign_id"], int(frame["source_frame_id"])): int(frame["gt_frame_id"])
        for item in protocol["items"] for frame in item["frames"]
    }
    nlf_index = {}
    with (args.nlf_root / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            nlf_index[(str(row["clip_id"]), int(row["frame_id"]))] = row
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
    model = create_model(paths["model_root"], str(config["runtime"]["device"]))
    feature_rows = []
    targets = []
    groups = []
    frames = []
    row_index = 0
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
        with np.load(candidate_path, allow_pickle=False) as archive:
            left_pose = archive["left_body_pose"].copy()
            right_pose = archive["right_body_pose"].copy()
            side_data = {}
            for side, poses in (("left", left_pose), ("right", right_pose)):
                _, joints = forward_state_batch(
                    model, state, poses, str(config["runtime"]["device"])
                )
                side_data[side] = {
                    "features": candidate_features(
                        joints, archive[f"{side}_metrics"], nlf_parametric,
                        nlf_nonparametric, nlf_uncertainty, side,
                        np.asarray(nlf_record["box"]), int(record["width"]),
                        int(record["height"]),
                    )
                }
            pair_poses = compose_body_poses(
                state.arrays["body_pose"], left_pose, right_pose,
                str(config["candidate_space"]),
            )
        pair_vertices, _ = forward_state_batch(
            model, state, pair_poses, str(config["runtime"]["device"])
        )
        pair_vertices[0] = state.vertices_evaluator
        gt_frame = pairing[key]
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
        pair_sums = {
            name: np.asarray([
                centered_error(vertices, target, ids).sum() for vertices in pair_vertices
            ])
            for name, ids in regions.items()
            if not (name == "LHand" and record["sign_class"] == "0")
        }
        counts = {name: len(regions[name]) for name in pair_sums}
        rows_by_side = {}
        right_count = len(right_pose)
        baseline_ubody_h = pair_sums["UBody-H"][0] / counts["UBody-H"]
        for side, poses in (("left", left_pose), ("right", right_pose)):
            count = len(poses)
            rows = np.arange(row_index, row_index + count, dtype=np.int64)
            rows_by_side[side] = rows
            row_index += count
            feature_rows.extend(side_data[side]["features"])
            if side == "left":
                pair_indices = np.arange(count) * right_count
            else:
                pair_indices = np.arange(count)
            side_gain = (
                baseline_ubody_h
                - pair_sums["UBody-H"][pair_indices] / counts["UBody-H"]
            ) * 1000.0
            targets.extend(side_gain.tolist())
            groups.extend([record["sign_id"]] * count)
        frames.append({
            "record_id": record["record_id"], "sign": record["sign_id"],
            "rows": rows_by_side,
            "right_count": right_count, "pair_sums": pair_sums, "counts": counts,
        })
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"[v8-develop-data] {ordinal}/{len(records)}", flush=True)
    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    groups_array = np.asarray(groups)
    logo = LeaveOneGroupOut()
    candidates = []
    oof_by_model = {}
    factories = model_factories()
    for name in args.models:
        factory = factories[name]
        oof = np.zeros(len(X), dtype=np.float64)
        classification = name in CLASSIFIER_TARGETS_MM
        fit_target = (
            (y > CLASSIFIER_TARGETS_MM[name]).astype(np.int64)
            if classification else y
        )
        for train, test in logo.split(X, y, groups_array):
            model_fold = factory()
            model_fold.fit(X[train], fit_target[train])
            oof[test] = (
                model_fold.predict_proba(X[test])[:, 1]
                if classification else model_fold.predict(X[test])
            )
        oof_by_model[name] = oof
        margin_grid = PROBABILITY_MARGINS if classification else MARGINS_MM
        for temporal_window in TEMPORAL_WINDOWS:
            for margin in margin_grid:
                result = aggregate_selection(
                    frames, oof, margin, temporal_window
                )
                result["model"] = name
                result["prediction_type"] = (
                    "benefit_probability" if classification else "gain_mm"
                )
                result["target_threshold_mm"] = (
                    CLASSIFIER_TARGETS_MM[name] if classification else None
                )
                if classification:
                    result["oof_brier"] = float(np.mean(np.square(oof - fit_target)))
                else:
                    result["oof_mae_mm"] = float(np.mean(np.abs(oof - y)))
                result["eligible"] = bool(
                    result["gain_mm"]["UBody-H"] >= 0.15
                    and result["gain_mm"]["UBody"] >= -0.02
                    and result["gain_mm"]["All"] >= -0.02
                    and result["gain_mm"]["LHand"] >= -0.02
                    and result["gain_mm"]["RHand"] >= -0.02
                    and 0.02 <= result["selection_fraction"] <= 0.80
                )
                candidates.append(result)
    eligible = [item for item in candidates if item["eligible"]]
    if eligible:
        selected = max(eligible, key=lambda item: item["gain_mm"]["UBody-H"])
        status = "pass"
    else:
        selected = max(candidates, key=lambda item: item["gain_mm"]["UBody-H"])
        status = "fail"
    final_model = model_factories()[selected["model"]]()
    final_is_classifier = selected["model"] in CLASSIFIER_TARGETS_MM
    final_target = (
        (y > CLASSIFIER_TARGETS_MM[selected["model"]]).astype(np.int64)
        if final_is_classifier else y
    )
    final_model.fit(X, final_target)
    args.output_root.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "model_name": selected["model"],
        "margin_mm": selected["margin_mm"],
        "prediction_type": selected["prediction_type"],
        "target_threshold_mm": selected["target_threshold_mm"],
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(bundle, args.output_root / "ranker.joblib")
    report = {
        "schema_version": "signdart.v8_ranker_development.v1",
        "status": status,
        "frames": len(records), "signs": len(set(groups)),
        "candidate_rows": len(X), "features": list(FEATURE_NAMES),
        "validation": "leave_one_sign_out",
        "selected": selected,
        "candidate_models_and_margins": candidates,
        "uses_gt_for_training": True,
        "inference_uses_gt": False,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "config_sha256": sha256_file(args.config),
    }
    (args.output_root / "development_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status, "frames": len(records), "candidate_rows": len(X),
        "selected": selected,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
