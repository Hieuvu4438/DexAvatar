"""Materialize portable A1R results into a formal Phase 2R cache."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.build_observation_cache import _array, _pose_from_params
from phase2_refiner.data.cache_schema import load_cache_clip, save_cache_clip
from phase2_refiner.data.run_exact_a1_stack import validate_exact_result_file
from phase2_refiner.provenance import sha256_file


A1R_CONTRACT_VERSION = "a1r-portable-v1"


def _aggregate_digest(values: dict[str, str]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_provider_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("contract_version") != A1R_CONTRACT_VERSION:
        raise ValueError("Unsupported A1R provider contract")
    for key, expected in (
        ("portable", True),
        ("frozen", True),
        ("benchmark_conditioning", False),
    ):
        if payload.get(key) is not expected:
            raise ValueError(f"A1R provider manifest requires {key}={expected}")
    if not str(payload.get("provider_id", "")).strip():
        raise ValueError("A1R provider manifest lacks provider_id")
    repository_root = (path.parent / str(payload.get("repository_root", ""))).resolve()
    if not repository_root.is_dir():
        raise FileNotFoundError(
            f"A1R provider repository_root is missing: {repository_root}"
        )
    for category in ("weights", "configuration", "provider_code"):
        entries = payload.get(category)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"A1R provider manifest lacks {category}")
        for raw_path, expected in entries.items():
            source = (repository_root / raw_path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"A1R {category} file is missing: {source}")
            actual = sha256_file(source)
            if actual != expected:
                raise ValueError(
                    f"A1R {category} hash mismatch for {source}: "
                    f"expected={expected}, actual={actual}"
                )
    return payload


def _load_result(path: Path) -> dict[str, Any]:
    validate_exact_result_file(path)
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def attach_initializer(
    template,
    results_dir: Path,
    decision_path: Path,
    provider: dict[str, Any],
) -> Any:
    if not decision_path.is_file():
        raise FileNotFoundError(
            f"A1R observation-derived decision is missing: {decision_path}"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("clip_name") != template.clip_id:
        raise ValueError(
            f"A1R decision clip_name mismatch: expected={template.clip_id!r}, "
            f"actual={decision.get('clip_name')!r}"
        )
    paths = [results_dir / f"{name}.pkl" for name in template.frame_names]
    expected = {path.name for path in paths}
    actual = {path.name for path in results_dir.glob("*.pkl") if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"A1R result coverage mismatch for {template.clip_id}: "
            f"missing={sorted(expected - actual)[:1]} extra={sorted(actual - expected)[:1]}"
        )
    params = [_load_result(path) for path in paths]
    poses = np.stack([_pose_from_params(item) for item in params]).astype(np.float32)
    globals_ = np.stack([_array(item.get("global_orient"), 3) for item in params])
    translations = np.stack([_array(item.get("transl"), 3) for item in params])
    jaws = np.stack([_array(item.get("jaw_pose"), 3) for item in params])
    leyes = np.stack([_array(item.get("leye_pose"), 3) for item in params])
    reyes = np.stack([_array(item.get("reye_pose"), 3) for item in params])
    expressions = np.stack([_array(item.get("expression"), 10) for item in params])
    betas = np.median(
        np.stack([_array(item.get("betas"), 10) for item in params]), axis=0
    ).astype(np.float32)
    metadata = json.loads(template.metadata_json)
    metadata["initializer_expert"] = provider["provider_id"]
    metadata["initializer_matches_locked_lane_a1"] = False
    metadata["initializer_contract"] = {
        "portable": True,
        "frozen": True,
        "benchmark_conditioning": False,
        "provider_id": provider["provider_id"],
        "weights_sha256": _aggregate_digest(provider["weights"]),
        "configuration_sha256": _aggregate_digest(provider["configuration"]),
        "provider_code_sha256": _aggregate_digest(provider["provider_code"]),
        "provider_manifest_sha256": provider["manifest_sha256"],
        "observation_derived_contract_sha256": sha256_file(decision_path),
    }
    source_hashes = np.asarray([sha256_file(path) for path in paths])
    return replace(
        template,
        init_axis_angle=poses,
        betas=betas,
        global_orient=globals_.astype(np.float32),
        transl=translations.astype(np.float32),
        jaw_pose=jaws.astype(np.float32),
        leye_pose=leyes.astype(np.float32),
        reye_pose=reyes.astype(np.float32),
        expression=expressions.astype(np.float32),
        source_paths=np.asarray([str(path.resolve()) for path in paths]),
        source_sha256=source_hashes,
        initializer_component=np.asarray(
            [provider["provider_id"]] * len(paths), dtype=str
        ),
        fallback_reason=np.asarray([""] * len(paths), dtype=str),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    template_root = args.template_root.resolve()
    results_root = args.results_root.resolve()
    contracts_root = args.contracts_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Append-only output exists: {output_root}")
    provider = validate_provider_manifest(args.provider_manifest.resolve())
    provider = {**provider, "manifest_sha256": sha256_file(args.provider_manifest)}
    report: dict[str, Any] = {
        "contract_version": A1R_CONTRACT_VERSION,
        "provider_id": provider["provider_id"],
        "provider_manifest": str(args.provider_manifest.resolve()),
        "provider_manifest_sha256": provider["manifest_sha256"],
        "splits": {},
    }
    output_root.mkdir(parents=True)
    try:
        (output_root / "splits").mkdir()
        for split in ("train", "val", "calibration"):
            source_manifest = template_root / "splits" / f"{split}.json"
            clip_dir = output_root / "clips" / split
            clip_dir.mkdir(parents=True)
            entries = []
            frames = 0
            for source in _manifest_paths(source_manifest):
                template = load_cache_clip(source)
                results = results_root / template.clip_id / "smplifyx" / "results"
                decision = contracts_root / template.clip_id / "decision.json"
                clip = attach_initializer(template, results, decision, provider)
                destination = clip_dir / source.name
                temporary = destination.with_name(destination.stem + ".tmp.npz")
                save_cache_clip(temporary, clip)
                os.replace(temporary, destination)
                entries.append(f"../clips/{split}/{destination.name}")
                frames += len(clip.frame_names)
            output_manifest = output_root / "splits" / f"{split}.json"
            output_manifest.write_text(
                json.dumps({"clips": entries}, indent=2) + "\n", encoding="utf-8"
            )
            report["splits"][split] = {
                "clips": len(entries),
                "frames": frames,
                "manifest": str(output_manifest.resolve()),
                "manifest_sha256": sha256_file(output_manifest),
            }
        (output_root / "materialization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output_root)
        raise
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--contracts-root", type=Path, required=True)
    parser.add_argument("--provider-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(materialize(parse_args()), indent=2, sort_keys=True))
