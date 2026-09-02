from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import yaml

from signdart.audit.oracle_ceiling import centered_error, load_obj_vertices
from signdart.io.h1_state import H1State, read_manifest, sha256_file, state_path
from signdart.model import create_model, forward_state_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    records = read_manifest(paths["manifest"])
    protocol = json.loads(paths["protocol_lock"].read_text(encoding="utf-8"))
    pairing = {
        (item["sign_id"], int(frame["source_frame_id"])): int(frame["gt_frame_id"])
        for item in protocol["items"] for frame in item["frames"]
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
    model = create_model(paths["model_root"], str(config["runtime"]["device"]))
    vectors = {
        "baseline": {name: [] for name in base_regions},
        "selected": {name: [] for name in base_regions},
    }
    sign_vectors = {
        method: {} for method in ("baseline", "selected")
    }
    for ordinal, record in enumerate(records, start=1):
        state = H1State.load(state_path(paths["h1_state_root"], record))
        selected_path = (
            args.selection_root / "frames" / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        with np.load(selected_path, allow_pickle=False) as archive:
            selected_pose = archive["body_pose"]
        vertices, _ = forward_state_batch(
            model, state, selected_pose, str(config["runtime"]["device"])
        )
        selected_vertices = vertices[0]
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
        for name, ids in regions.items():
            if name == "LHand" and record["sign_class"] == "0":
                continue
            baseline_error = centered_error(state.vertices_evaluator, target, ids)
            selected_error = centered_error(selected_vertices, target, ids)
            vectors["baseline"][name].append(baseline_error)
            vectors["selected"][name].append(selected_error)
            for method, error in (
                ("baseline", baseline_error), ("selected", selected_error)
            ):
                sign_vectors[method].setdefault(record["sign_id"], {}).setdefault(
                    name, []
                ).append(error)
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"[G4-eval] {ordinal}/{len(records)}", flush=True)
    metrics = {}
    for method, regions in vectors.items():
        metrics[method] = {
            name: float(np.concatenate(items).mean() * 1000.0)
            for name, items in regions.items() if items
        }
    gain = {
        name: metrics["baseline"][name] - metrics["selected"][name]
        for name in metrics["selected"]
    }
    signs = sorted(sign_vectors["baseline"])
    sign_gain_ubody_h = np.asarray([
        (
            np.concatenate(sign_vectors["baseline"][sign]["UBody-H"]).mean()
            - np.concatenate(sign_vectors["selected"][sign]["UBody-H"]).mean()
        ) * 1000.0
        for sign in signs
    ])
    bootstrap_replicates = 100000
    bootstrap_seed = 20260902
    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0, len(signs), size=(bootstrap_replicates, len(signs))
    )
    bootstrap = sign_gain_ubody_h[sample_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
    paired_bootstrap = {
        "unit": "sign",
        "signs": len(signs),
        "replicates": bootstrap_replicates,
        "seed": bootstrap_seed,
        "mean_sign_macro_gain_mm": float(sign_gain_ubody_h.mean()),
        "ci95_gain_mm": [float(ci_low), float(ci_high)],
    }
    run = json.loads((args.selection_root / "run.json").read_text(encoding="utf-8"))
    passed = bool(
        gain["UBody-H"] >= 0.15
        and gain["UBody"] >= -0.02
        and gain["All"] >= -0.02
        and gain["LHand"] >= -0.02
        and gain["RHand"] >= -0.02
        and 0.02 <= run["selection_fraction"] <= 0.80
        and ci_low > 0.0
    )
    report = {
        "schema_version": "signdart.g4_zero_training_evaluation.v1",
        "status": "pass" if passed else "fail",
        "frames": len(records),
        "baseline_metrics_mm": metrics["baseline"],
        "selected_metrics_mm": metrics["selected"],
        "gain_mm": gain,
        "paired_sign_bootstrap_ubody_h": paired_bootstrap,
        "per_sign_ubody_h_gain_mm": dict(zip(signs, sign_gain_ubody_h.tolist())),
        "selection_fraction": run["selection_fraction"],
        "selector_trained_parameters": run["trained_parameters"],
        "selector_uses_gt": run["uses_gt"],
        "evaluation_uses_gt": True,
        "selection_run_sha256": sha256_file(args.selection_root / "run.json"),
        "manifest_sha256": sha256_file(paths["manifest"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
