"""Strict frame-ID audit plus paired sign-cluster uncertainty analysis."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from cusp_sl.config import load_config


REGIONS = ("ubody_minus_face", "left_hand", "right_hand")


def load_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(value.split("/")[0]) - 1 for value in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def translation_removed(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(prediction - target, axis=-1).mean() * 1000.0)


def bootstrap(values: np.ndarray, replicates: int, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    distribution = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(distribution, (0.025, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    config = load_config(args.config)
    args.output.mkdir(parents=True)
    rows = list(csv.DictReader(args.frame_manifest.open(encoding="utf-8")))
    if len(rows) != config.protocol.expected_frames:
        raise ValueError(f"Manifest has {len(rows)} rows")
    with Path(config.protocol.signs_file).open(encoding="utf-8") as handle:
        sign_class = {line.split()[0]: line.split()[1] for line in handle if line.strip()}
    assets = Path(config.protocol.assets_root)
    with (assets / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        hand = pickle.load(handle)
    ubody = np.load(assets / "sgnify_part_segm_above_pelvis_joint" / "upper_body_minus_face.npy")
    indices = {"ubody_minus_face": ubody, "left_hand": np.asarray(hand["left_hand"]), "right_hand": np.asarray(hand["right_hand"])}
    frame_records, failures = [], []
    face_reference = None
    for row in rows:
        sign, name = row["sign"], row["frame_name"]
        prediction_path = args.prediction_root / sign / "smplifyx" / "meshes" / f"{name}.obj"
        baseline_path, target_path = Path(row["baseline_mesh"]), Path(row["gt_path_author_pairing"])
        if not prediction_path.is_file():
            failures.append({"sign": sign, "frame_name": name, "reason": "missing_prediction"})
            continue
        prediction, faces = load_vertices_faces(prediction_path)
        baseline, baseline_faces = load_vertices_faces(baseline_path)
        target, target_faces = load_vertices_faces(target_path)
        if prediction.shape != (10475, 3) or not np.isfinite(prediction).all():
            failures.append({"sign": sign, "frame_name": name, "reason": "invalid_prediction"})
            continue
        if not np.array_equal(faces, target_faces) or not np.array_equal(baseline_faces, target_faces):
            raise ValueError(f"Topology mismatch: {sign}/{name}")
        if face_reference is None:
            face_reference = faces
        elif not np.array_equal(faces, face_reference):
            raise ValueError(f"Prediction topology changes at {sign}/{name}")
        record = {"sign": sign, "frame_name": name, "status": "ok"}
        for region in REGIONS:
            if region == "left_hand" and sign_class[sign] == "0":
                record[f"method_{region}_mm"] = ""
                record[f"baseline_{region}_mm"] = ""
                record[f"vertices_{region}"] = ""
                continue
            selected = indices[region]
            if region == "ubody_minus_face" and sign_class[sign] == "0":
                selected = np.setdiff1d(selected, indices["left_hand"])
            record[f"method_{region}_mm"] = translation_removed(prediction[selected], target[selected])
            record[f"baseline_{region}_mm"] = translation_removed(baseline[selected], target[selected])
            record[f"vertices_{region}"] = int(len(selected))
        frame_records.append(record)
    if failures:
        (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Audited evaluation refuses {len(failures)} missing/invalid predictions")
    frame_csv = args.output / "per_frame.csv"
    with frame_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_records[0]))
        writer.writeheader()
        writer.writerows(frame_records)
    summary = {"coverage": {"expected": len(rows), "valid": len(frame_records), "failures": 0}, "regions": {}}
    signs = sorted({row["sign"] for row in frame_records})
    for region in REGIONS:
        method, baseline, counts, sign_delta = [], [], [], []
        for sign in signs:
            selected = [row for row in frame_records if row["sign"] == sign and row[f"method_{region}_mm"] != ""]
            if not selected:
                continue
            method.extend(float(row[f"method_{region}_mm"]) for row in selected)
            baseline.extend(float(row[f"baseline_{region}_mm"]) for row in selected)
            counts.extend(int(row[f"vertices_{region}"]) for row in selected)
            sign_delta.append(np.mean([float(row[f"method_{region}_mm"]) - float(row[f"baseline_{region}_mm"]) for row in selected]))
        delta = np.asarray(sign_delta)
        summary["regions"][region] = {
            "method_frame_mean_mm": float(np.mean(method)),
            "baseline_frame_mean_mm": float(np.mean(baseline)),
            "method_vertex_frame_mean_mm": float(np.average(method, weights=counts)),
            "baseline_vertex_frame_mean_mm": float(np.average(baseline, weights=counts)),
            "paired_sign_mean_delta_mm": float(delta.mean()),
            "paired_sign_delta_95ci_mm": bootstrap(delta, config.protocol.bootstrap_replicates),
            "signs": len(delta),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
