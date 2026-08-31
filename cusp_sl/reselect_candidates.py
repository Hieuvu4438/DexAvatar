"""Re-select a frozen candidate set with development-fitted energy statistics.

This is deliberately a post-processing step: it cannot generate, remove, or
modify candidates.  It exists so the development candidate set used to fit
robust energy normalization is exactly the set used by the selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config
from cusp_sl.geometry import matrix_to_axis_angle
from cusp_sl.selection import EnergyStatistics, candidate_energy, select_candidates


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_statistics(
    statistics: np.lib.npyio.NpzFile,
    source_manifest: dict,
    source_manifest_path: Path,
    config_path: Path,
) -> None:
    required = {
        "schema_version",
        "role",
        "median",
        "mad",
        "source_candidate_manifest_sha256",
        "config_sha256",
        "reliability_checkpoint_sha256",
        "flow_checkpoint_sha256",
        "gate_calibration_sha256",
    }
    missing = required.difference(statistics.files)
    if missing:
        raise ValueError(f"Energy statistics omit fields: {sorted(missing)}")
    if (
        str(statistics["schema_version"].item())
        != "cusp_sl_energy_statistics_v2"
        or str(statistics["role"].item())
        != "development_candidate_energy_normalization"
    ):
        raise ValueError("Unsupported energy-statistics artifact")
    expected = {
        "source_candidate_manifest_sha256": sha256(source_manifest_path),
        "config_sha256": sha256(config_path),
        "reliability_checkpoint_sha256": source_manifest[
            "reliability_checkpoint_sha256"
        ],
        "flow_checkpoint_sha256": source_manifest["flow_checkpoint_sha256"],
        "gate_calibration_sha256": source_manifest.get(
            "gate_calibration_sha256"
        )
        or "",
    }
    for field, value in expected.items():
        if str(statistics[field].item()) != value:
            raise ValueError(f"Energy-statistics {field} mismatch")


def reselect_payload(
    arrays: dict[str, np.ndarray],
    statistics: EnergyStatistics,
    weights: torch.Tensor,
    temperature: float,
) -> tuple[dict[str, np.ndarray], int]:
    rotations = torch.from_numpy(arrays["candidate_rotation"]).float()
    terms = torch.from_numpy(arrays["energy_terms"]).float()
    valid = torch.from_numpy(arrays["candidate_valid"]).bool()
    if rotations.shape[0] != terms.shape[0] or valid.shape != (rotations.shape[0],):
        raise ValueError("Candidate rotations, terms, and validity disagree")
    energy = candidate_energy(terms, statistics, weights)
    selected = select_candidates(
        rotations[None], energy[None], valid[None], temperature
    )
    index = int(selected["index"][0])
    result = dict(arrays)
    result.update(
        selected_rotation=selected["rotation"][0].numpy(),
        selected_axis_angle=matrix_to_axis_angle(
            selected["rotation"][0]
        ).numpy(),
        energy=energy.numpy(),
        selected_index=np.asarray(index),
        energy_weights=selected["weights"][0].numpy(),
        disagreement=selected["disagreement"][0].numpy(),
        energy_margin=selected["energy_margin"][0].numpy(),
    )
    return result, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--energy-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    source_manifest_path = args.candidate_root / "manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    if source_manifest.get("protocol_role") != "development_validation":
        raise ValueError("Re-selection requires development_validation candidates")
    if source_manifest.get("variant") != "a7_geometry":
        raise ValueError("Re-selection currently accepts only frozen A7 geometry sets")
    if not all(
        item.get("energy_computed") is True
        for item in source_manifest["summaries"]
    ):
        raise ValueError("Source candidates do not contain computed evidence")
    config = load_config(args.config)
    with np.load(args.energy_statistics, allow_pickle=False) as payload:
        validate_statistics(
            payload, source_manifest, source_manifest_path, args.config
        )
        statistics = EnergyStatistics(
            torch.from_numpy(payload["median"]).float(),
            torch.from_numpy(payload["mad"]).float(),
        )
    weights = torch.tensor(
        [
            config.selection.observation_weight,
            config.selection.motion_weight,
            config.selection.physical_weight,
            0.0,
        ],
        dtype=torch.float32,
    )
    declared = {
        str(item["clip_id"]): item for item in source_manifest["summaries"]
    }
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    summaries = []
    for clip_id in sorted(declared):
        clip_started = time.perf_counter()
        source = args.candidate_root / "clips" / f"{clip_id}.npz"
        if sha256(source) != declared[clip_id]["prediction_sha256"]:
            raise ValueError(f"Candidate hash mismatch: {source}")
        with np.load(source, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        selected, index = reselect_payload(
            arrays,
            statistics,
            weights,
            config.selection.energy_temperature,
        )
        destination = args.output / "clips" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **selected)
        summary = dict(declared[clip_id])
        summary.update(
            selected_index=index,
            source_prediction_sha256=declared[clip_id]["prediction_sha256"],
            prediction_sha256=sha256(destination),
            reselection_runtime_seconds=time.perf_counter() - clip_started,
        )
        summaries.append(summary)
        print(f"[reselect] {clip_id}: candidate={index}")
    if set(declared) != {item["clip_id"] for item in summaries}:
        raise ValueError("Re-selection coverage differs from source candidates")
    runtime = time.perf_counter() - started
    report = {
        **{key: value for key, value in source_manifest.items() if key != "summaries"},
        "variant": "a7_geometry",
        "selection_stage": "frozen_candidate_cpu_reselection",
        "source_candidate_manifest_sha256": sha256(source_manifest_path),
        "energy_statistics_sha256": sha256(args.energy_statistics),
        "source_inference_runtime_seconds": source_manifest.get(
            "runtime_seconds"
        ),
        "source_inference_frames_per_second": source_manifest.get(
            "frames_per_second"
        ),
        "runtime_seconds": runtime,
        "frames_per_second": source_manifest["frames"] / runtime,
        "summaries": summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "clips": len(summaries),
        "frames": report["frames"],
        "runtime_seconds": runtime,
    }, indent=2))


if __name__ == "__main__":
    main()
