"""Fail-closed provenance audit for formal Phase 2R training caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import (
    load_cache_clip,
    validate_phase2r_semantics,
)
from phase2_refiner.provenance import sha256_file


FORMAL_CONTRACT_VERSION = "phase2r-formal-v1"
FORMAL_TARGET_GEOMETRIES = {
    "mocap_3d",
    "multiview_3d",
    "multiview_mesh",
    "released_3d_mesh",
    "released_3d_smplx_pose",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def formal_clip_failures(metadata: dict[str, Any]) -> list[str]:
    """Return all formal-contract failures instead of trusting prose labels."""
    failures = []
    initializer = metadata.get("initializer_contract")
    if not isinstance(initializer, dict):
        failures.append("missing initializer_contract")
    else:
        for key in ("portable", "frozen"):
            if initializer.get(key) is not True:
                failures.append(f"initializer_contract.{key} must be true")
        if initializer.get("benchmark_conditioning") is not False:
            failures.append("initializer_contract.benchmark_conditioning must be false")
        if not str(initializer.get("provider_id", "")).strip():
            failures.append("initializer_contract.provider_id is missing")
        if not _is_sha256(initializer.get("weights_sha256")):
            failures.append("initializer_contract.weights_sha256 is invalid")
        if not _is_sha256(initializer.get("configuration_sha256")):
            failures.append("initializer_contract.configuration_sha256 is invalid")
        if not _is_sha256(initializer.get("provider_code_sha256")):
            failures.append("initializer_contract.provider_code_sha256 is invalid")

    target = metadata.get("target_contract")
    if not isinstance(target, dict):
        failures.append("missing target_contract")
    else:
        required_true = (
            "independent_from_initializer",
            "not_same_view_2d_only",
            "initializer_outputs_used_false",
            "release_benchmark_excluded",
            "audit_passed",
            "license_verified",
            "exact_frame_count_match",
            "shared_geometry_decode",
        )
        for key in required_true:
            if target.get(key) is not True:
                failures.append(f"target_contract.{key} must be true")
        if target.get("geometry") not in FORMAL_TARGET_GEOMETRIES:
            failures.append("target_contract.geometry is not formal 3D evidence")
        if not str(target.get("provider_id", "")).strip():
            failures.append("target_contract.provider_id is missing")
        hashes = target.get("source_sha256")
        if (
            not isinstance(hashes, list)
            or not hashes
            or not all(_is_sha256(value) for value in hashes)
        ):
            failures.append("target_contract.source_sha256 is invalid or empty")
        if not _is_sha256(target.get("provider_code_sha256")):
            failures.append("target_contract.provider_code_sha256 is invalid")
        if not _is_sha256(target.get("audit_report_sha256")):
            failures.append("target_contract.audit_report_sha256 is invalid")

    adapter = metadata.get("phase2r_adapter")
    if isinstance(adapter, dict) and str(
        adapter.get("target_independence", "")
    ).upper().startswith("NO"):
        failures.append("proxy adapter explicitly marks target independence NO")
    return failures


def audit_formal_manifests(manifests: dict[str, Path]) -> dict[str, Any]:
    """Audit every clip and enforce source-group-disjoint split boundaries."""
    splits: dict[str, Any] = {}
    groups_by_split: dict[str, set[str]] = {}
    signers_by_split: dict[str, set[str]] = {}
    all_failures: list[dict[str, Any]] = []
    for split, manifest in manifests.items():
        resolved = manifest.resolve()
        paths = _manifest_paths(resolved)
        groups: set[str] = set()
        signers: set[str] = set()
        frames = 0
        providers: set[str] = set()
        geometries: set[str] = set()
        for path in paths:
            clip = load_cache_clip(path)
            validate_phase2r_semantics(clip)
            metadata = json.loads(clip.metadata_json)
            failures = formal_clip_failures(metadata)
            if clip.target_axis_angle is None or clip.target_rotation_valid is None:
                failures.append("formal cache has no rotation target")
            elif not all(
                clip.target_rotation_valid[:, region].any()
                for region in (slice(0, 21), slice(21, 36), slice(36, 51))
            ):
                failures.append("formal cache lacks a supervised body/hand region")
            if not (clip.target_quality > 0).any():
                failures.append("formal cache has no positive target-quality evidence")
            if any(component == "unknown" for component in clip.initializer_component):
                failures.append("initializer_component contains unknown")
            if not all(_is_sha256(value) for value in clip.source_sha256):
                failures.append("cached initializer source_sha256 is invalid")
            group = metadata.get("source_group")
            if not isinstance(group, str) or not group:
                failures.append("metadata.source_group is missing")
            else:
                groups.add(group)
            signer = metadata.get("signer_id")
            if not isinstance(signer, str) or not signer:
                failures.append("metadata.signer_id is missing")
            else:
                signers.add(signer)
            if failures:
                all_failures.append(
                    {"split": split, "clip_id": clip.clip_id, "failures": failures}
                )
            initializer = metadata.get("initializer_contract", {})
            target = metadata.get("target_contract", {})
            providers.add(str(initializer.get("provider_id", "missing")))
            geometries.add(str(target.get("geometry", "missing")))
            frames += len(clip.frame_names)
        groups_by_split[split] = groups
        signers_by_split[split] = signers
        splits[split] = {
            "manifest": str(resolved),
            "manifest_sha256": sha256_file(resolved),
            "clips": len(paths),
            "frames": frames,
            "source_groups": len(groups),
            "signers": len(signers),
            "initializer_providers": sorted(providers),
            "target_geometries": sorted(geometries),
        }

    overlaps = {}
    signer_overlaps = {}
    names = sorted(groups_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            values = sorted(groups_by_split[left] & groups_by_split[right])
            overlaps[f"{left}__{right}"] = values
            signer_overlaps[f"{left}__{right}"] = sorted(
                signers_by_split[left] & signers_by_split[right]
            )
    checks = {
        "all_clips_satisfy_formal_contract": not all_failures,
        "source_group_disjoint_splits": not any(overlaps.values()),
        "signer_disjoint_splits": not any(signer_overlaps.values()),
    }
    return {
        "contract_version": FORMAL_CONTRACT_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "splits": splits,
        "source_group_overlaps": overlaps,
        "signer_overlaps": signer_overlaps,
        "failure_count": len(all_failures),
        "failures": all_failures[:100],
        "failure_list_truncated": len(all_failures) > 100,
    }


def validate_formal_audit_report(
    report_path: str | Path, expected_manifests: dict[str, str | Path]
) -> dict[str, Any]:
    """Bind a passing audit to the exact manifests used by an experiment."""
    path = Path(report_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("contract_version") != FORMAL_CONTRACT_VERSION:
        raise ValueError("Formal Phase 2R audit has an unsupported contract version")
    if report.get("passed") is not True:
        raise ValueError("Formal Phase 2R audit did not pass")
    audited = report.get("splits", {})
    for split, manifest in expected_manifests.items():
        resolved = Path(manifest).resolve()
        item = audited.get(split)
        if not isinstance(item, dict):
            raise ValueError(f"Formal Phase 2R audit lacks split {split!r}")
        if Path(str(item.get("manifest", ""))).resolve() != resolved:
            raise ValueError(f"Formal Phase 2R audit manifest mismatch for {split}")
        if item.get("manifest_sha256") != sha256_file(resolved):
            raise ValueError(f"Formal Phase 2R manifest hash mismatch for {split}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = audit_formal_manifests(
        {
            "train": args.train_manifest,
            "validation": args.val_manifest,
            "calibration": args.calibration_manifest,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
