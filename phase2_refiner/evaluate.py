"""Strict common-manifest regional TR-V2V evaluation for Phase 2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase2_refiner.provenance import sha256_file


REGIONS = ("ubody", "lhand", "rhand")


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append(
                    [int(value.split("/")[0]) - 1 for value in line.split()[1:4]]
                )
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    if vertices_array.shape != (10475, 3):
        raise ValueError(
            f"Expected 10,475 vertices in {path}, got {vertices_array.shape}"
        )
    return vertices_array, faces_array


def regional_trv2v(
    prediction: np.ndarray, target: np.ndarray, indices: np.ndarray
) -> float:
    prediction = prediction[indices]
    target = target[indices]
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(prediction - target, axis=-1).mean() * 1000.0)


def _prediction_path(root: Path, sign: str, manifest_prediction: str) -> Path:
    name = Path(manifest_prediction).name
    return root / sign / "smplifyx" / "meshes" / name


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pooled_mean(rows: list[dict], value_key: str, count_key: str) -> float | str:
    """Match the author evaluator's mean over all retained vertices."""
    values = []
    counts = []
    for row in rows:
        if row[value_key] == "":
            continue
        values.append(float(row[value_key]))
        counts.append(int(row[count_key]))
    if not values:
        return ""
    return float(np.average(values, weights=counts))


def _bootstrap_difference(
    per_sign: list[dict], region: str, samples: int, seed: int
) -> dict[str, float | int | None]:
    differences = np.asarray(
        [
            row[f"prediction_{region}"] - row[f"baseline_{region}"]
            for row in per_sign
            if row[f"prediction_{region}"] != "" and row[f"baseline_{region}"] != ""
        ],
        dtype=np.float64,
    )
    if len(differences) == 0:
        return {
            "signs": 0,
            "mean_delta_mm": None,
            "ci95_low_mm": None,
            "ci95_high_mm": None,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    distribution = differences[indices].mean(axis=1)
    return {
        "signs": int(len(differences)),
        "mean_delta_mm": float(differences.mean()),
        "ci95_low_mm": float(np.quantile(distribution, 0.025)),
        "ci95_high_mm": float(np.quantile(distribution, 0.975)),
    }


def evaluate(
    manifest_path: Path,
    prediction_root: Path,
    output_dir: Path,
    repository_root: Path,
    baseline_root: Path | None,
    assets_root: Path,
    bootstrap_samples: int,
    seed: int,
    overwrite: bool,
) -> dict:
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {summary_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError("Evaluation manifest is empty")
    manifest_ids = [
        (row["sign"], Path(row["prediction_path"]).name) for row in manifest_rows
    ]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("Evaluation manifest contains duplicate sign/frame rows")
    with (assets_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        import pickle

        hand_ids = pickle.load(handle, encoding="latin1")
    left_ids = np.asarray(hand_ids["left_hand"], dtype=np.int64)
    right_ids = np.asarray(hand_ids["right_hand"], dtype=np.int64)
    upper_ids = np.load(
        assets_root
        / "sgnify_part_segm_above_pelvis_joint"
        / "upper_body_minus_face.npy"
    ).astype(np.int64)

    frame_rows = []
    failures = []
    expected_paths = set()
    for row in manifest_rows:
        sign = row["sign"]
        prediction_path = _prediction_path(
            prediction_root, sign, row["prediction_path"]
        )
        expected_paths.add(prediction_path.resolve())
        gt_path = repository_root / row["gt_path"]
        baseline_path = (
            _prediction_path(baseline_root, sign, row["prediction_path"])
            if baseline_root is not None
            else None
        )
        required = [prediction_path, gt_path] + (
            [baseline_path] if baseline_path else []
        )
        missing = [
            str(path) for path in required if path is not None and not path.exists()
        ]
        if missing:
            failures.append(
                {
                    "sign": sign,
                    "frame": Path(row["prediction_path"]).stem,
                    "missing": missing,
                }
            )
            continue
        prediction, prediction_faces = load_obj(prediction_path)
        target, target_faces = load_obj(gt_path)
        if not np.array_equal(prediction_faces, target_faces):
            raise ValueError(f"Topology mismatch: {prediction_path} vs {gt_path}")
        baseline = baseline_faces = None
        if baseline_path is not None:
            baseline, baseline_faces = load_obj(baseline_path)
            if not np.array_equal(baseline_faces, target_faces):
                raise ValueError(f"Topology mismatch: {baseline_path} vs {gt_path}")
        left_evaluated = row["left_evaluated"].lower() == "true"
        right_evaluated = row.get("right_evaluated", "true").lower() == "true"
        effective_upper = (
            upper_ids if left_evaluated else np.setdiff1d(upper_ids, left_ids)
        )
        regions = {"ubody": effective_upper, "lhand": left_ids, "rhand": right_ids}
        frame = {"sign": sign, "frame": Path(row["prediction_path"]).stem}
        for region, indices in regions.items():
            if region == "lhand" and not left_evaluated:
                frame[f"prediction_{region}"] = ""
                frame[f"baseline_{region}"] = ""
                frame[f"vertices_{region}"] = ""
                continue
            if region == "rhand" and not right_evaluated:
                frame[f"prediction_{region}"] = ""
                frame[f"baseline_{region}"] = ""
                frame[f"vertices_{region}"] = ""
                continue
            frame[f"vertices_{region}"] = int(len(indices))
            frame[f"prediction_{region}"] = regional_trv2v(prediction, target, indices)
            frame[f"baseline_{region}"] = (
                regional_trv2v(baseline, target, indices)
                if baseline is not None
                else ""
            )
        frame_rows.append(frame)
    if failures:
        raise RuntimeError(
            f"Coverage failure: {len(failures)} manifest rows are incomplete. First: {failures[0]}"
        )
    actual_paths = set(
        path.resolve() for path in prediction_root.glob("*/smplifyx/meshes/*.obj")
    )
    extras = sorted(str(path) for path in actual_paths - expected_paths)
    if extras:
        raise RuntimeError(
            f"Prediction contains {len(extras)} stale/extra meshes; first: {extras[0]}"
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame_rows:
        grouped[row["sign"]].append(row)
    per_sign = []
    for sign, rows in sorted(grouped.items()):
        result: dict[str, str | float] = {"sign": sign}
        for region in REGIONS:
            for prefix in ("prediction", "baseline"):
                result[f"{prefix}_{region}"] = _pooled_mean(
                    rows, f"{prefix}_{region}", f"vertices_{region}"
                )
        per_sign.append(result)

    summary: dict = {
        "frames": len(frame_rows),
        "signs": len(grouped),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_root": str(prediction_root.resolve()),
        "baseline_root": str(baseline_root.resolve()) if baseline_root else None,
        "prediction": {},
    }
    if baseline_root:
        summary["baseline"] = {}
        summary["paired_bootstrap"] = {}
    for region in REGIONS:
        summary["prediction"][region] = _pooled_mean(
            frame_rows, f"prediction_{region}", f"vertices_{region}"
        )
        if baseline_root:
            summary["baseline"][region] = _pooled_mean(
                frame_rows, f"baseline_{region}", f"vertices_{region}"
            )
            summary["paired_bootstrap"][region] = _bootstrap_difference(
                per_sign, region, bootstrap_samples, seed
            )
            differences = np.asarray(
                [
                    float(row[f"prediction_{region}"])
                    - float(row[f"baseline_{region}"])
                    for row in per_sign
                    if row[f"prediction_{region}"] != ""
                    and row[f"baseline_{region}"] != ""
                ],
                dtype=np.float64,
            )
            summary["paired_bootstrap"][region].update(
                {
                    "median_delta_mm": (
                        float(np.median(differences)) if len(differences) else None
                    ),
                    "worst_decile_delta_mm": (
                        float(np.quantile(differences, 0.9))
                        if len(differences)
                        else None
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        ["sign", "frame"]
        + [
            f"{prefix}_{region}"
            for region in REGIONS
            for prefix in ("prediction", "baseline")
        ]
        + [f"vertices_{region}" for region in REGIONS]
    )
    _write_csv(output_dir / "per_frame.csv", fields, frame_rows)
    sign_fields = ["sign"] + [
        f"{prefix}_{region}"
        for region in REGIONS
        for prefix in ("prediction", "baseline")
    ]
    _write_csv(output_dir / "per_sign.csv", sign_fields, per_sign)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("data/evaluation_from_author/data/data"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(
        args.manifest,
        args.prediction,
        args.output,
        args.repository_root.resolve(),
        args.baseline,
        args.assets_root,
        args.bootstrap_samples,
        args.seed,
        args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
