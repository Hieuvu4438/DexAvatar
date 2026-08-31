"""Recompute target-free image evidence for frozen development predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config
from cusp_sl.evidence import candidate_evidence_terms
from cusp_sl.evaluate_frontend_evidence import cluster_delta_interval
from cusp_sl.evaluate_selection_evidence import target_reads_prohibited
from cusp_sl.geometry import axis_angle_to_matrix
from cusp_sl.training import resolve_device
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.render import create_smplx_model


TERM_NAMES = ("observation", "motion", "physical", "form")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if args.bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    config = load_config(args.config)
    input_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    prediction_manifest_path = args.predictions / "manifest.json"
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    if input_manifest.get("role") != "development_targetless_inference":
        raise ValueError("Prediction evidence requires targetless development input")
    if not target_reads_prohibited(input_manifest):
        raise ValueError("Input manifest does not explicitly prohibit target reads")
    if prediction_manifest.get("protocol_role") != "development_validation":
        raise ValueError("Prediction evidence is development-only")
    input_hash = sha256(args.input_manifest)
    if prediction_manifest.get("input_manifest_sha256") != input_hash:
        raise ValueError("Prediction/input manifest hash mismatch")
    input_summaries = {
        str(item["clip_id"]): item for item in input_manifest.get("summaries", [])
    }
    prediction_summaries = {
        str(item["clip_id"]): item
        for item in prediction_manifest.get("summaries", [])
    }
    entries = input_manifest.get("clips", [])
    if not (
        len(entries) == len(input_summaries) == len(prediction_summaries)
    ):
        raise ValueError("Input/prediction coverage differs")

    device = resolve_device(args.device)
    model = create_smplx_model(config.protocol.smplx_model_folder, device)
    model.requires_grad_(False)
    records: list[dict[str, object]] = []
    for entry in entries:
        cache_path = Path(entry)
        if not cache_path.is_absolute():
            cache_path = args.input_manifest.parent / cache_path
        clip = load_cache_clip(cache_path)
        clip_id = str(clip.clip_id)
        input_item = input_summaries.get(clip_id)
        prediction_item = prediction_summaries.get(clip_id)
        if input_item is None or prediction_item is None:
            raise ValueError(f"Missing declared clip: {clip_id}")
        if sha256(cache_path) != input_item.get("targetless_cache_sha256"):
            raise ValueError(f"Targetless cache hash mismatch: {cache_path}")
        if (
            clip.target_axis_angle is not None
            or clip.target_joint_positions is not None
            or bool(np.any(clip.target_quality))
        ):
            raise ValueError(f"Evidence audit refuses target-bearing cache: {cache_path}")
        prediction_path = args.predictions / "clips" / f"{clip_id}.npz"
        if sha256(prediction_path) != prediction_item.get("prediction_sha256"):
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as payload:
            if str(payload["clip_id"].item()) != clip_id or not np.array_equal(
                payload["frame_names"].astype(str), clip.frame_names.astype(str)
            ):
                raise ValueError(f"Prediction/cache identity mismatch: {clip_id}")
            selected = torch.as_tensor(
                payload["selected_rotation"], device=device
            ).float()
        base = axis_angle_to_matrix(
            torch.as_tensor(clip.init_axis_angle, device=device).float()
        )
        terms = candidate_evidence_terms(
            model,
            torch.stack((base, selected)),
            clip,
            device,
            huber_delta=config.selection.huber_delta,
            rom_threshold_degrees=config.selection.rom_threshold_degrees,
        ).cpu().numpy()
        metadata = json.loads(clip.metadata_json)
        source_group = str(metadata.get("source_group", ""))
        if not source_group:
            raise ValueError(f"Targetless cache lacks source_group: {cache_path}")
        record: dict[str, object] = {
            "clip_id": clip_id,
            "source_group": source_group,
            "frames": len(clip.frame_names),
        }
        for term_index, term in enumerate(TERM_NAMES):
            record[f"base_{term}"] = float(terms[0, term_index])
            record[f"selected_{term}"] = float(terms[1, term_index])
        records.append(record)
        print(f"[prediction-evidence] {clip_id}: {len(clip.frame_names)} frames")
    if set(prediction_summaries) != {str(row["clip_id"]) for row in records}:
        raise ValueError("Prediction clip set differs from targetless input")

    args.output.mkdir(parents=True)
    with (args.output / "per_clip.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    weights = np.asarray([row["frames"] for row in records], dtype=np.float64)

    def aggregate(name: str) -> float:
        return float(np.average([row[name] for row in records], weights=weights))

    summary: dict[str, object] = {
        "role": "target_free_frozen_prediction_image_evidence_audit",
        "target_reads": 0,
        "variant": prediction_manifest["variant"],
        "generator_kind": prediction_manifest["generator_kind"],
        "clips": len(records),
        "frames": int(weights.sum()),
        "source_groups": len({str(row["source_group"]) for row in records}),
        "config_sha256": sha256(args.config),
        "input_manifest_sha256": input_hash,
        "prediction_manifest_sha256": sha256(prediction_manifest_path),
    }
    for term in TERM_NAMES:
        base = f"base_{term}"
        selected = f"selected_{term}"
        summary[base] = aggregate(base)
        summary[selected] = aggregate(selected)
        summary[f"clustered_selected_minus_base_{term}"] = cluster_delta_interval(
            records,
            selected,
            base,
            "frames",
            replicates=args.bootstrap_replicates,
            seed=args.seed + len(summary),
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
