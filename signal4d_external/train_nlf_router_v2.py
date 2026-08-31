"""Train and calibrate an NLF body-benefit router on external data only."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import smplx
from sklearn.ensemble import RandomForestRegressor

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file
from signal4d_external.nlf_v2_core import (
    FEATURE_COLUMNS,
    cache_pose_from_full,
    external_upper_body_error_degrees,
    frame_features,
    full_initializer_rotations,
    nlf_body_candidate,
    nlf_observation_contract,
    viterbi_benefit_selection,
)


ALPHAS = (0.25, 0.5, 0.75, 1.0)
MARGINS_DEG = (0.0, 0.25, 0.5, 1.0, 2.0)
TRANSITION_PENALTIES_DEG = (0.0, 0.25, 0.5, 1.0)


def _load_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    with (root / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[(str(row["clip_id"]), int(row["frame_id"]))] = row
    return result


def _load_manifest(path: Path, split: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != split:
        raise ValueError(f"Expected {split} manifest, got {payload.get('split')}: {path}")
    if int(payload.get("sgnify_training_reads", -1)) != 0:
        raise ValueError(f"Manifest does not prove zero SGNify reads: {path}")
    return payload


def _rows_for_alpha(
    payload: dict[str, Any],
    split: str,
    alpha: float,
    observations: dict[tuple[str, int], dict[str, Any]],
    observation_root: Path,
    parents: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for item in payload["clips"]:
        cache = load_cache_clip(item["cache_path"])
        previous_initial = None
        previous_candidate = None
        previous_frame = None
        for frame, frame_id in enumerate(item["frame_ids"]):
            record = observations.get((str(item["clip_id"]), int(frame_id)))
            if record is None or record.get("status") not in {"ok", "existing"}:
                continue
            with np.load(observation_root / record["output_relpath"]) as observation:
                elapsed_seconds = (
                    None
                    if previous_frame is None
                    else float(cache.timestamps[frame] - cache.timestamps[previous_frame])
                )
                candidate_full = nlf_body_candidate(
                    cache, frame, observation["pose"], parents, alpha
                )
                candidate_pose = cache_pose_from_full(candidate_full)
                features = frame_features(
                    cache,
                    frame,
                    observation,
                    candidate_full,
                    previous_initializer=previous_initial,
                    previous_nlf_body=previous_candidate,
                    elapsed_seconds=elapsed_seconds,
                )
            initial_pose = cache_pose_from_full(full_initializer_rotations(cache, frame))
            baseline_error = external_upper_body_error_degrees(cache, frame, initial_pose)
            candidate_error = external_upper_body_error_degrees(cache, frame, candidate_pose)
            if math.isfinite(baseline_error) and math.isfinite(candidate_error):
                rows.append(
                    {
                        "split": split,
                        "clip_id": str(item["clip_id"]),
                        "signer": str(item["signer"]),
                        "frame_id": int(frame_id),
                        **features,
                        "baseline_error_deg": baseline_error,
                        "candidate_error_deg": candidate_error,
                        "delta_candidate_minus_baseline_deg": candidate_error
                        - baseline_error,
                    }
                )
            previous_initial = initial_pose
            previous_candidate = candidate_pose
            previous_frame = frame
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No usable external NLF rows for {split}, alpha={alpha}")
    return table


def _fit(table: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=12,
        max_features=0.8,
        random_state=4202,
        n_jobs=-1,
    )
    model.fit(
        table[list(FEATURE_COLUMNS)],
        table["delta_candidate_minus_baseline_deg"],
    )
    return model


def _evaluate_policy(
    table: pd.DataFrame,
    predicted: np.ndarray,
    margin: float,
    transition: float,
) -> dict[str, Any]:
    work = table[
        [
            "clip_id",
            "signer",
            "time_gap_reference_units",
            "baseline_error_deg",
            "candidate_error_deg",
        ]
    ].copy()
    work["predicted_delta_deg"] = predicted
    chosen_parts = []
    for _, clip in work.groupby("clip_id", sort=False):
        gap = clip["time_gap_reference_units"].to_numpy(dtype=np.float64)
        transition_scales = np.minimum(1.0, 1.0 / np.maximum(gap, 1.0))
        chosen_parts.append(
            pd.Series(
                viterbi_benefit_selection(
                    clip["predicted_delta_deg"].to_numpy(),
                    margin,
                    transition,
                    transition_scales,
                ),
                index=clip.index,
            )
        )
    chosen = pd.concat(chosen_parts).sort_index().astype(bool)
    hybrid = np.where(
        chosen,
        work["candidate_error_deg"],
        work["baseline_error_deg"],
    )
    work["hybrid_error_deg"] = hybrid
    signer_gain = (
        work.groupby("signer")["baseline_error_deg"].mean()
        - work.groupby("signer")["hybrid_error_deg"].mean()
    )
    baseline = float(work["baseline_error_deg"].mean())
    candidate = float(np.mean(hybrid))
    return {
        "margin_deg": margin,
        "transition_penalty_deg": transition,
        "frames": len(work),
        "selection_fraction": float(chosen.mean()),
        "baseline_error_deg": baseline,
        "hybrid_error_deg": candidate,
        "gain_deg": baseline - candidate,
        "signer_gain_deg": {str(key): float(value) for key, value in signer_gain.items()},
        "worst_signer_gain_deg": float(signer_gain.min()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {
        "train": _load_manifest(args.train_manifest.resolve(), "train"),
        "validation": _load_manifest(args.validation_manifest.resolve(), "validation"),
        "calibration": _load_manifest(args.calibration_manifest.resolve(), "calibration"),
    }
    signer_sets = {name: set(value["signers"]) for name, value in manifests.items()}
    if any(
        signer_sets[first] & signer_sets[second]
        for first, second in (
            ("train", "validation"),
            ("train", "calibration"),
            ("validation", "calibration"),
        )
    ):
        raise ValueError("External V2 manifests are not signer-disjoint")
    metadata = json.loads(
        (args.observation_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    if int(metadata.get("sgnify_training_reads", -1)) != 0:
        raise ValueError("NLF observations do not prove zero SGNify reads")
    observations = _load_index(args.observation_root)
    body_model = smplx.SMPLX(
        str(args.model_path.resolve()),
        gender="neutral",
        ext=args.model_path.suffix.lstrip("."),
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    ).eval()
    parents = body_model.parents[:55].detach().cpu().numpy().astype(np.int64)

    models = {}
    tables = defaultdict(dict)
    validation_candidates = []
    for alpha in ALPHAS:
        for split, payload in manifests.items():
            tables[alpha][split] = _rows_for_alpha(
                payload,
                split,
                alpha,
                observations,
                args.observation_root,
                parents,
            )
        model = _fit(tables[alpha]["train"])
        models[alpha] = model
        validation = tables[alpha]["validation"]
        predicted = model.predict(validation[list(FEATURE_COLUMNS)])
        for transition in TRANSITION_PENALTIES_DEG:
            validation_candidates.append(
                {
                    "alpha": alpha,
                    **_evaluate_policy(validation, predicted, 0.0, transition),
                }
            )
    selected_validation = max(
        validation_candidates,
        key=lambda row: (row["gain_deg"], -row["alpha"], -row["transition_penalty_deg"]),
    )
    alpha = float(selected_validation["alpha"])
    transition = float(selected_validation["transition_penalty_deg"])
    model = models[alpha]
    calibration = tables[alpha]["calibration"]
    predicted = model.predict(calibration[list(FEATURE_COLUMNS)])
    calibration_candidates = [
        _evaluate_policy(calibration, predicted, margin, transition)
        for margin in MARGINS_DEG
    ]
    eligible = [
        row
        for row in calibration_candidates
        if row["selection_fraction"] >= 0.01 and row["worst_signer_gain_deg"] >= -0.25
    ]
    selected_calibration = max(
        eligible or calibration_candidates,
        key=lambda row: (row["gain_deg"], row["margin_deg"]),
    )
    decision = (
        "PASS"
        if selected_calibration["gain_deg"] > 0.0
        and selected_calibration["selection_fraction"] >= 0.01
        and selected_calibration["worst_signer_gain_deg"] >= -0.25
        else "FAIL"
    )

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True)
    router_path = args.output_root / "router.joblib"
    joblib.dump(model, router_path)
    # Save inference features only; labels remain summarized in the report.
    for split in ("train", "validation", "calibration"):
        tables[alpha][split][
            ["split", "clip_id", "signer", "frame_id", *FEATURE_COLUMNS]
        ].to_csv(args.output_root / f"{split}_features.csv", index=False)
    report = {
        "schema_version": "signal4d.external_nlf_router_v2.v1",
        "decision": decision,
        "training_data": "How2Sign external-only",
        "target_metric": "weighted external pseudo-target upper-body SO(3) error",
        "sgnify_training_or_selection_reads": 0,
        "feature_columns": list(FEATURE_COLUMNS),
        "candidate_alphas": list(ALPHAS),
        "candidate_margins_deg": list(MARGINS_DEG),
        "candidate_transition_penalties_deg": list(TRANSITION_PENALTIES_DEG),
        "validation_candidates": validation_candidates,
        "selected_validation": selected_validation,
        "calibration_candidates": calibration_candidates,
        "selected_calibration": selected_calibration,
        "selected": {
            "alpha": alpha,
            "margin_deg": float(selected_calibration["margin_deg"]),
            "transition_penalty_deg": transition,
        },
        "training_frames": len(tables[alpha]["train"]),
        "validation_frames": len(tables[alpha]["validation"]),
        "calibration_frames": len(tables[alpha]["calibration"]),
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "router_sha256": sha256_file(router_path),
        "model_path": str(args.model_path.resolve()),
        "model_sha256": sha256_file(args.model_path),
        "observation_metadata_sha256": sha256_file(
            args.observation_root / "run_metadata.json"
        ),
        "nlf_observation_contract": nlf_observation_contract(metadata),
        "manifests": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "train": args.train_manifest,
                "validation": args.validation_manifest,
                "calibration": args.calibration_manifest,
            }.items()
        },
    }
    report_path = args.output_root / "calibration.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--observation-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
