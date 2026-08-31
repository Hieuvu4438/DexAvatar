from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle

import numpy as np

from signpccx.data.manifest import read_jsonl
from signpccx.export.preflight import load_obj_minimal
from signpccx.io import atomic_write_json, sha256_file


REGION_FILES = {
    "tr above pelvis upper body": "upper_body.npy",
    "tr above pelvis minus head": "upper_body_minus_head.npy",
    "tr above pelvis minus face": "upper_body_minus_face.npy",
}


def translation_aligned_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    return np.linalg.norm(prediction - target, axis=-1)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty audited CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_regions(asset_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with (asset_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle)
    left = np.asarray(mano["left_hand"], dtype=np.int64)
    right = np.asarray(mano["right_hand"], dtype=np.int64)
    segment_root = asset_root / "sgnify_part_segm_above_pelvis_joint"
    regions = {
        "tr all": np.arange(10475, dtype=np.int64),
        "tr left hand": left,
        "tr right hand": right,
    }
    regions.update({key: np.load(segment_root / filename) for key, filename in REGION_FILES.items()})
    return regions, left


def _prediction_paths(mesh_dir: Path, expected: int) -> list[Path]:
    import re

    def first_int(path: Path) -> int:
        match = re.search(r"\d+", path.stem)
        if match is None:
            raise ValueError(f"prediction filename has no integer: {path}")
        return int(match.group())

    paths = sorted(mesh_dir.glob("*.obj"), key=first_int)
    if len(paths) != expected:
        raise RuntimeError(f"{mesh_dir}: {len(paths)} predictions != {expected}")
    return paths


def evaluate_audited(
    manifest_root: Path,
    prediction_root: Path,
    gt_root: Path,
    asset_root: Path,
    output_root: Path,
    method: str,
    official_result: Path | None = None,
    signs: set[str] | None = None,
) -> dict[str, object]:
    regions, left_ids = _load_regions(asset_root)
    vector_errors: dict[str, list[np.ndarray]] = {key: [] for key in regions}
    frame_rows: list[dict[str, object]] = []
    sign_rows: list[dict[str, object]] = []
    manifest_paths = sorted(manifest_root.glob("*.jsonl"))
    if signs is not None:
        unknown = signs - {path.stem for path in manifest_paths}
        if unknown:
            raise ValueError(f"unknown audited signs: {sorted(unknown)}")
        manifest_paths = [path for path in manifest_paths if path.stem in signs]
    for manifest_path in manifest_paths:
        records = read_jsonl(manifest_path)
        paths = _prediction_paths(
            prediction_root / manifest_path.stem / "smplifyx" / "meshes", len(records)
        )
        per_sign: dict[str, list[float]] = {key: [] for key in regions}
        for record, prediction_path in zip(records, paths, strict=True):
            prediction, _ = load_obj_minimal(prediction_path)
            if record.gt_frame_id is None:
                raise ValueError(f"manifest has no GT pairing: {manifest_path}")
            target_path = gt_root / record.sign / f"{record.gt_frame_id:05d}.obj"
            target, _ = load_obj_minimal(target_path)
            if prediction.shape != target.shape or prediction.shape != (10475, 3):
                raise ValueError(f"mesh shape mismatch: {prediction_path}/{target_path}")
            one_handed = record.sign_class == "0"
            row: dict[str, object] = {
                "sign": record.sign,
                "source_frame_id": record.source_frame_id,
                "gt_frame_id": record.gt_frame_id,
                "sequence_index": record.sequence_index,
                "one_handed_class_0": one_handed,
            }
            for name, base_ids in regions.items():
                if name == "tr left hand" and one_handed:
                    row[name] = ""
                    continue
                ids = base_ids
                if name != "tr left hand" and one_handed:
                    ids = np.setdiff1d(ids, left_ids)
                errors = translation_aligned_error(prediction[ids], target[ids])
                if not np.isfinite(errors).all():
                    raise FloatingPointError(f"non-finite audited error: {prediction_path}")
                vector_errors[name].append(errors)
                value = float(errors.mean() * 1000.0)
                row[name] = value
                per_sign[name].append(value)
            frame_rows.append(row)
        sign_row: dict[str, object] = {
            "sign": manifest_path.stem,
            "frames": len(records),
            "one_handed_class_0": records[0].sign_class == "0",
        }
        for name, values in per_sign.items():
            sign_row[name] = "" if not values else float(np.mean(values))
        sign_rows.append(sign_row)
    metrics = {
        name: float(np.concatenate(values).mean() * 1000.0)
        for name, values in vector_errors.items()
    }
    _write_csv(output_root / "per_frame.csv", frame_rows)
    _write_csv(output_root / "per_sign.csv", sign_rows)
    report: dict[str, object] = {
        "schema_version": "signpccx.audited-evaluation.v1",
        "method": method,
        "aggregation": "author_vertex_micro",
        "signs": len(sign_rows),
        "frames": len(frame_rows),
        "prediction_root": str(prediction_root.resolve()),
        "manifest_summary_sha256": sha256_file(manifest_root / "summary.json"),
        "evaluated_signs": [row["sign"] for row in sign_rows],
        "evaluator_assets": {
            "smplx": sha256_file(asset_root / "SMPLX_NEUTRAL.npz"),
            "mano_ids": sha256_file(asset_root / "MANO_SMPLX_vertex_ids.pkl"),
            **{
                filename: sha256_file(asset_root / "sgnify_part_segm_above_pelvis_joint" / filename)
                for filename in REGION_FILES.values()
            },
        },
        "metrics_mm": metrics,
    }
    if official_result is not None:
        official = json.loads(official_result.read_text(encoding="utf-8"))["metrics_mm"]
        differences = {key: metrics[key] - float(official[key]) for key in metrics}
        if max(abs(value) for value in differences.values()) > 5e-5:
            raise RuntimeError(f"audited/official aggregate mismatch: {differences}")
        report["official_result"] = str(official_result.resolve())
        report["official_rounded_parity"] = True
        report["official_differences_mm"] = differences
    atomic_write_json(output_root / "summary.json", report)
    return report


def _read_sign_metrics(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["sign"]] = {
                key: float(value) if value not in (None, "") else float("nan")
                for key, value in row.items()
                if key.startswith("tr ")
            }
    return result


def paired_sign_bootstrap(
    candidate_csv: Path,
    baseline_csv: Path,
    output: Path,
    replicates: int = 10000,
    seed: int = 20260830,
) -> dict[str, object]:
    candidate = _read_sign_metrics(candidate_csv)
    baseline = _read_sign_metrics(baseline_csv)
    if candidate.keys() != baseline.keys():
        raise ValueError("paired sign sets differ")
    rng = np.random.default_rng(seed)
    results = []
    for metric in (
        "tr all", "tr above pelvis upper body", "tr above pelvis minus face",
        "tr left hand", "tr right hand",
    ):
        delta = np.asarray([candidate[key][metric] - baseline[key][metric] for key in candidate])
        delta = delta[np.isfinite(delta)]
        samples = rng.choice(delta, size=(replicates, len(delta)), replace=True).mean(axis=1)
        results.append({
            "metric": metric,
            "eligible_signs": len(delta),
            "mean_sign_delta_mm": float(delta.mean()),
            "median_sign_delta_mm": float(np.median(delta)),
            "ci95_percentile_mm": np.quantile(samples, [0.025, 0.975]).tolist(),
            "probability_nonnegative": float(np.mean(samples >= 0)),
            "improved_signs": int(np.sum(delta < 0)),
            "worse_signs": int(np.sum(delta > 0)),
        })
    report = {
        "schema_version": "signpccx.paired-sign-bootstrap.v1",
        "candidate": str(candidate_csv.resolve()),
        "baseline": str(baseline_csv.resolve()),
        "replicates": replicates,
        "seed": seed,
        "results": results,
    }
    atomic_write_json(output, report)
    return report
