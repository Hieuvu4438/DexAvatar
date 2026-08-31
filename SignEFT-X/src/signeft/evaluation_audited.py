from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle

import numpy as np

from signeft.data.manifest import read_manifest
from signeft.io.obj import load_obj
from signeft.io_utils import atomic_write_json, sha256_file


REGION_FILES = {
    "tr above pelvis upper body": "upper_body.npy",
    "tr above pelvis minus face": "upper_body_minus_face.npy",
    "tr above pelvis minus head": "upper_body_minus_head.npy",
}
REPORT_NAMES = {
    "tr all": "All",
    "tr above pelvis upper body": "UBody",
    "tr above pelvis minus face": "UBody-F",
    "tr above pelvis minus head": "UBody-H",
    "tr left hand": "LHand",
    "tr right hand": "RHand",
}


def translation_aligned_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    return np.linalg.norm(prediction - target, axis=-1)


def _regions(asset_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with (asset_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle)
    left = np.asarray(mano["left_hand"], dtype=np.int64)
    right = np.asarray(mano["right_hand"], dtype=np.int64)
    segment_root = asset_root / "sgnify_part_segm_above_pelvis_joint"
    result = {
        "tr all": np.arange(10475, dtype=np.int64),
        "tr left hand": left,
        "tr right hand": right,
    }
    result.update({name: np.load(segment_root / filename) for name, filename in REGION_FILES.items()})
    return result, left


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_audited(
    manifest_path: Path,
    protocol_lock: Path,
    prediction_root: Path,
    gt_root: Path,
    asset_root: Path,
    output_root: Path,
    *,
    official_result: Path | None = None,
) -> dict[str, object]:
    records = read_manifest(manifest_path)
    protocol = json.loads(protocol_lock.read_text(encoding="utf-8"))
    pairing = {
        (item["sign_id"], int(frame["source_frame_id"])): int(frame["gt_frame_id"])
        for item in protocol["items"] for frame in item["frames"]
    }
    regions, left_ids = _regions(asset_root)
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in regions}
    frame_rows = []
    per_sign: dict[str, dict[str, list[float]]] = {}
    for record in records:
        key = (record.sign_id, record.source_frame_id)
        if key not in pairing:
            raise RuntimeError(f"protocol pairing missing: {key}")
        prediction_path = (
            prediction_root / record.sign_id / "smplifyx" / "meshes" / f"{record.frame_index:03d}.obj"
        )
        target_path = gt_root / record.sign_id / f"{pairing[key]:05d}.obj"
        prediction, _ = load_obj(prediction_path)
        target, _ = load_obj(target_path)
        if prediction.shape != (10475, 3) or target.shape != prediction.shape:
            raise RuntimeError(f"audited mesh mismatch: {prediction_path}/{target_path}")
        one_handed = record.sign_class == "0"
        row: dict[str, object] = {
            "sign": record.sign_id,
            "source_frame_id": record.source_frame_id,
            "gt_frame_id": pairing[key],
            "sequence_index": record.frame_index,
            "one_handed_class_0": one_handed,
        }
        per_sign.setdefault(record.sign_id, {name: [] for name in regions})
        for name, region_ids in regions.items():
            if name == "tr left hand" and one_handed:
                row[name] = ""
                continue
            ids = region_ids
            if name != "tr left hand" and one_handed:
                ids = np.setdiff1d(ids, left_ids)
            error = translation_aligned_error(prediction[ids], target[ids])
            if not np.isfinite(error).all():
                raise FloatingPointError(f"non-finite audited metric: {prediction_path}")
            vectors[name].append(error)
            value = float(error.mean() * 1000.0)
            row[name] = value
            per_sign[record.sign_id][name].append(value)
        frame_rows.append(row)
    sign_rows = []
    for sign, values_by_metric in per_sign.items():
        row: dict[str, object] = {"sign": sign, "frames": sum(r.sign_id == sign for r in records)}
        for name, values in values_by_metric.items():
            row[name] = "" if not values else float(np.mean(values))
        sign_rows.append(row)
    metrics = {name: float(np.concatenate(values).mean() * 1000.0) for name, values in vectors.items()}
    distribution = {
        name: {
            "median_mm": float(np.median(np.concatenate(values)) * 1000.0),
            "p90_mm": float(np.quantile(np.concatenate(values), 0.90) * 1000.0),
            "p95_mm": float(np.quantile(np.concatenate(values), 0.95) * 1000.0),
        }
        for name, values in vectors.items()
    }
    _write_csv(output_root / "per_frame.csv", frame_rows)
    _write_csv(output_root / "per_sign.csv", sign_rows)
    report: dict[str, object] = {
        "schema_version": "signeft.audited-evaluation.v1",
        "aggregation": "author_vertex_micro",
        "signs": len(sign_rows),
        "frames": len(frame_rows),
        "metrics_mm": metrics,
        "display_metrics_mm": {REPORT_NAMES[name]: value for name, value in metrics.items()},
        "distribution": distribution,
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_lock_sha256": sha256_file(protocol_lock),
        "evaluator_assets_used_only_after_fitting": True,
        "asset_hashes": {
            "mano_ids": sha256_file(asset_root / "MANO_SMPLX_vertex_ids.pkl"),
            **{
                filename: sha256_file(asset_root / "sgnify_part_segm_above_pelvis_joint" / filename)
                for filename in REGION_FILES.values()
            },
        },
    }
    if official_result is not None:
        official = json.loads(official_result.read_text(encoding="utf-8"))["metrics_mm"]
        differences = {name: metrics[name] - float(official[name]) for name in metrics}
        if max(abs(value) for value in differences.values()) > 5e-5:
            raise RuntimeError(f"official/audited mismatch: {differences}")
        report["official_rounded_parity"] = True
        report["official_differences_mm"] = differences
    atomic_write_json(output_root / "summary.json", report)
    return report


def paired_sign_bootstrap(
    candidate_csv: Path,
    baseline_csv: Path,
    output: Path,
    *,
    replicates: int = 10000,
    seed: int = 20260831,
) -> dict[str, object]:
    def read(path: Path) -> dict[str, dict[str, float]]:
        result = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                result[row["sign"]] = {
                    name: float(row[name]) if row[name] else float("nan") for name in REPORT_NAMES
                }
        return result

    candidate = read(candidate_csv)
    baseline = read(baseline_csv)
    if candidate.keys() != baseline.keys():
        raise RuntimeError("paired bootstrap sign sets differ")
    rng = np.random.default_rng(seed)
    rows = []
    for name in REPORT_NAMES:
        delta = np.asarray([candidate[sign][name] - baseline[sign][name] for sign in candidate])
        delta = delta[np.isfinite(delta)]
        samples = rng.choice(delta, size=(replicates, len(delta)), replace=True).mean(1)
        rows.append({
            "metric": name,
            "display_name": REPORT_NAMES[name],
            "eligible_signs": len(delta),
            "mean_sign_delta_mm": float(delta.mean()),
            "median_sign_delta_mm": float(np.median(delta)),
            "ci95_percentile_mm": np.quantile(samples, (0.025, 0.975)).tolist(),
            "probability_nonnegative": float(np.mean(samples >= 0)),
            "improved_signs": int(np.sum(delta < 0)),
            "worse_signs": int(np.sum(delta > 0)),
        })
    report = {
        "schema_version": "signeft.paired-sign-bootstrap.v1",
        "candidate": str(candidate_csv.resolve()),
        "baseline": str(baseline_csv.resolve()),
        "replicates": replicates,
        "seed": seed,
        "results": rows,
    }
    atomic_write_json(output, report)
    return report

