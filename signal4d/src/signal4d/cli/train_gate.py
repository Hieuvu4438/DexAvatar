from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from ..data.cache import ObservationBatch
from ..data.manifest import load_manifest
from ..io.predictions import PredictionArtifact
from ..models.gating import ExtraTreesArtifact, decode_gate_sequence, extract_gate_features
from ..utils.hashing import sha256_file
from ..utils.seed import seed_everything


def _read_frame_errors(path: str | Path, metric: str) -> dict[tuple[str, int], float]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["clip_id"], int(row["frame_id"])): float(row[metric]) for row in rows}


def _decode_all(
    predictions: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    switch_penalty: float,
) -> np.ndarray:
    selected = np.zeros(len(predictions), dtype=bool)
    for group in dict.fromkeys(groups.tolist()):
        indices = np.flatnonzero(groups == group)
        selected[indices] = decode_gate_sequence(
            predictions[indices], threshold, switch_penalty
        )
    return selected


def _bootstrap_clip_mean(values: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float]:
    clip_values = np.asarray(
        [values[groups == group].mean() for group in dict.fromkeys(groups.tolist())]
    )
    generator = np.random.default_rng(seed)
    samples = generator.choice(clip_values, size=(10000, len(clip_values)), replace=True).mean(1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def run(config_path: str, output: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), list):
        raise ValueError("gate training config requires a datasets list")
    seed = int(config.get("seed", 12345))
    seed_everything(seed)
    metric = str(config.get("target_metric", "tr_v2v_left_hand_mm"))
    features: list[np.ndarray] = []
    targets: list[float] = []
    groups: list[str] = []
    keys: list[tuple[str, str, int]] = []
    partitions: list[str] = []
    feature_names: list[str] | None = None
    source_hashes: dict[str, str] = {config_path: sha256_file(config_path)}

    for dataset in config["datasets"]:
        name = str(dataset["name"])
        partition = str(dataset.get("partition", name))
        manifest_path = Path(dataset["manifest"])
        candidate_root = Path(dataset["candidate_root"])
        baseline_root = Path(dataset["baseline_root"])
        cache_root = Path(dataset["cache_root"])
        candidate_csv = Path(dataset["candidate_frame_csv"])
        baseline_csv = Path(dataset["baseline_frame_csv"])
        candidate_errors = _read_frame_errors(candidate_csv, metric)
        baseline_errors = _read_frame_errors(baseline_csv, metric)
        for path in (manifest_path, candidate_csv, baseline_csv):
            source_hashes[str(path)] = sha256_file(path)
        for item in load_manifest(manifest_path):
            candidate_dir = candidate_root / item.clip_id
            baseline_dir = baseline_root / item.clip_id
            cache_dir = cache_root / item.clip_id
            candidate, _ = PredictionArtifact.load(candidate_dir)
            baseline, _ = PredictionArtifact.load(baseline_dir)
            observations, _ = ObservationBatch.load(cache_dir)
            observations.validate_against(item)
            diagnostics_path = candidate_dir / "factor_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            matrix, names = extract_gate_features(
                candidate, baseline, observations, diagnostics
            )
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("gate feature schemas differ between clips")
            for path in (
                candidate_dir / "prediction.safetensors",
                candidate_dir / "metadata.json",
                diagnostics_path,
                baseline_dir / "prediction.safetensors",
                baseline_dir / "metadata.json",
                cache_dir / "observations.safetensors",
                cache_dir / "metadata.json",
            ):
                source_hashes[str(path)] = sha256_file(path)
            for frame, frame_id in enumerate(item.frame_ids):
                key = (item.clip_id, frame_id)
                if key not in candidate_errors or key not in baseline_errors:
                    raise ValueError(f"missing paired training target for {name}/{key}")
                features.append(matrix[frame])
                targets.append(candidate_errors[key] - baseline_errors[key])
                groups.append(f"{name}:{item.clip_id}")
                keys.append((name, item.clip_id, frame_id))
                partitions.append(partition)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float64)
    group_array = np.asarray(groups)
    partition_array = np.asarray(partitions)
    if len(set(group_array)) < 5:
        raise ValueError("gate training requires at least five independent clips")
    folds = min(int(config.get("folds", 5)), len(set(group_array)))
    tree_count = int(config.get("trees", 200))
    min_samples_grid = [int(value) for value in config.get("min_samples_leaf", [5, 10, 20])]
    max_features_grid = [float(value) for value in config.get("max_features", [0.5, 0.8])]
    thresholds = [float(value) for value in config.get("decision_threshold_mm", [-0.5, 0.0, 0.5])]
    switch_penalties = [float(value) for value in config.get("switch_penalty_mm", [0.0, 0.5, 2.0])]
    split = list(GroupKFold(folds).split(x, y, group_array))
    model_results: list[dict[str, Any]] = []
    best: tuple[float, int, float, float, float, np.ndarray, np.ndarray] | None = None
    for minimum_leaf in min_samples_grid:
        for maximum_features in max_features_grid:
            oof = np.zeros(len(y), dtype=np.float64)
            for train, validation in split:
                model = ExtraTreesRegressor(
                    n_estimators=tree_count,
                    min_samples_leaf=minimum_leaf,
                    max_features=maximum_features,
                    n_jobs=int(config.get("n_jobs", 4)),
                    random_state=seed,
                )
                model.fit(x[train], y[train])
                oof[validation] = model.predict(x[validation])
            for threshold in thresholds:
                for switch_penalty in switch_penalties:
                    selected = _decode_all(
                        oof, group_array, threshold, switch_penalty
                    )
                    observed_delta = np.where(selected, y, 0.0)
                    clip_delta = float(
                        np.mean(
                            [
                                observed_delta[group_array == group].mean()
                                for group in dict.fromkeys(group_array.tolist())
                            ]
                        )
                    )
                    result = {
                        "min_samples_leaf": minimum_leaf,
                        "max_features": maximum_features,
                        "decision_threshold_mm": threshold,
                        "switch_penalty_mm": switch_penalty,
                        "oof_clip_macro_delta_mm": clip_delta,
                        "oof_frame_micro_delta_mm": float(observed_delta.mean()),
                        "candidate_fraction": float(selected.mean()),
                        "switches": int(
                            sum(
                                np.count_nonzero(
                                    np.diff(selected[group_array == group].astype(np.int8))
                                )
                                for group in dict.fromkeys(group_array.tolist())
                            )
                        ),
                    }
                    model_results.append(result)
                    key = (
                        clip_delta,
                        minimum_leaf,
                        maximum_features,
                        threshold,
                        switch_penalty,
                        oof,
                        selected,
                    )
                    if best is None or key[:5] < best[:5]:
                        best = key
    if best is None or feature_names is None:
        raise AssertionError("gate model selection produced no candidate")
    _, minimum_leaf, maximum_features, threshold, switch_penalty, oof, selected = best
    observed_delta = np.where(selected, y, 0.0)
    ci_low, ci_high = _bootstrap_clip_mean(observed_delta, group_array, seed)
    final_model = ExtraTreesRegressor(
        n_estimators=tree_count,
        min_samples_leaf=minimum_leaf,
        max_features=maximum_features,
        n_jobs=int(config.get("n_jobs", 4)),
        random_state=seed,
    )
    final_model.fit(x, y)

    historical: dict[str, Any] | None = None
    train_partition = config.get("historical_train_partition")
    test_partition = config.get("historical_test_partition")
    if train_partition is not None and test_partition is not None:
        train_mask = partition_array == str(train_partition)
        test_mask = partition_array == str(test_partition)
        historical_model = ExtraTreesRegressor(
            n_estimators=tree_count,
            min_samples_leaf=minimum_leaf,
            max_features=maximum_features,
            n_jobs=int(config.get("n_jobs", 4)),
            random_state=seed,
        )
        historical_model.fit(x[train_mask], y[train_mask])
        historical_prediction = historical_model.predict(x[test_mask])
        historical_selection = _decode_all(
            historical_prediction,
            group_array[test_mask],
            threshold,
            switch_penalty,
        )
        historical_delta = np.where(historical_selection, y[test_mask], 0.0)
        historical = {
            "train_partition": str(train_partition),
            "test_partition": str(test_partition),
            "clip_macro_delta_mm": float(
                np.mean(
                    [
                        historical_delta[group_array[test_mask] == group].mean()
                        for group in dict.fromkeys(group_array[test_mask].tolist())
                    ]
                )
            ),
            "frame_micro_delta_mm": float(historical_delta.mean()),
            "candidate_fraction": float(historical_selection.mean()),
            "clips": len(set(group_array[test_mask])),
        }

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed,
        "target_metric": metric,
        "target_definition": "candidate_minus_strongest_legacy_baseline_per_frame_mm",
        "training_frames": len(y),
        "training_clips": len(set(group_array)),
        "folds": folds,
        "trees": tree_count,
        "min_samples_leaf": minimum_leaf,
        "max_features": maximum_features,
        "oof_clip_macro_delta_mm": float(
            np.mean(
                [
                    observed_delta[group_array == group].mean()
                    for group in dict.fromkeys(group_array.tolist())
                ]
            )
        ),
        "oof_frame_micro_delta_mm": float(observed_delta.mean()),
        "oof_ci95_clip_bootstrap_mm": [ci_low, ci_high],
        "oof_candidate_fraction": float(selected.mean()),
        "historical_temporal_holdout": historical,
        "source_hashes": source_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    artifact = ExtraTreesArtifact.from_sklearn(
        final_model,
        feature_names,
        threshold,
        switch_penalty,
        metadata,
    )
    artifact.save(output_path)
    safe_prediction = artifact.predict(x)
    sklearn_prediction = final_model.predict(x)
    serialization_error = float(np.max(np.abs(safe_prediction - sklearn_prediction)))
    if serialization_error > 1e-10:
        raise RuntimeError(f"safe forest serialization mismatch: {serialization_error}")
    metadata_path = output_path / "metadata.json"
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_metadata["safe_serialization_max_abs_error"] = serialization_error
    metadata_path.write_text(
        json.dumps(saved_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_path / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "clip_id",
                "frame_id",
                "observed_candidate_minus_baseline_mm",
                "predicted_candidate_minus_baseline_mm",
                "selected_candidate",
            ],
        )
        writer.writeheader()
        for key, target, prediction, choice in zip(keys, y, oof, selected, strict=True):
            writer.writerow(
                {
                    "dataset": key[0],
                    "clip_id": key[1],
                    "frame_id": key[2],
                    "observed_candidate_minus_baseline_mm": target,
                    "predicted_candidate_minus_baseline_mm": prediction,
                    "selected_candidate": int(choice),
                }
            )
    (output_path / "model_selection.json").write_text(
        json.dumps(
            {"selected": saved_metadata, "grid": model_results},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return saved_metadata
