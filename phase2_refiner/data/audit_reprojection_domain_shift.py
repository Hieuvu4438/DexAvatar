"""Quantify source/target reprojection-feature shift without target labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file


REGIONS = {"ubody": slice(0, 21), "lhand": slice(21, 36), "rhand": slice(36, 51)}


def _statistics(paths: list[Path]) -> dict:
    residuals = {name: [] for name in REGIONS}
    reliabilities = {name: [] for name in REGIONS}
    valid_counts = {name: 0 for name in REGIONS}
    total_counts = {name: 0 for name in REGIONS}
    frames = 0
    for path in paths:
        clip = load_cache_clip(path)
        frames += len(clip.frame_names)
        for name, indices in REGIONS.items():
            active = clip.refine_mask[indices]
            valid = clip.keypoint_valid[:, indices] & active[None]
            norm = np.linalg.vector_norm(
                clip.reprojection_residual_2d[:, indices], axis=-1
            )
            residuals[name].append(norm[valid])
            reliabilities[name].append(clip.u0_reliability[:, indices][valid])
            valid_counts[name] += int(valid.sum())
            total_counts[name] += int(valid.size)
    result = {"clips": len(paths), "frames": frames, "regions": {}}
    for name in REGIONS:
        residual = np.concatenate(residuals[name])
        reliability = np.concatenate(reliabilities[name])
        if not len(residual):
            raise ValueError(f"No valid reprojection residuals for {name}")
        result["regions"][name] = {
            "residual_norm_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("q01", "q10", "q50", "q90", "q99"),
                    np.quantile(residual, (0.01, 0.10, 0.50, 0.90, 0.99)),
                    strict=True,
                )
            },
            "residual_norm_mean": float(residual.mean()),
            "u0_reliability_mean": float(reliability.mean()),
            "valid_fraction": valid_counts[name] / max(total_counts[name], 1),
        }
    return result


def audit(source_manifest: Path, target_root: Path) -> dict:
    source_paths = _manifest_paths(source_manifest)
    target_paths = sorted((target_root / "clips").glob("*.npz"))
    if not target_paths:
        raise ValueError(f"No target cache clips under {target_root}")
    source = _statistics(source_paths)
    target = _statistics(target_paths)
    ratios = {}
    shifted = []
    for name in REGIONS:
        source_median = source["regions"][name]["residual_norm_quantiles"]["q50"]
        target_median = target["regions"][name]["residual_norm_quantiles"]["q50"]
        ratio = target_median / max(source_median, 1e-12)
        ratios[name] = ratio
        if ratio < 0.5 or ratio > 2.0:
            shifted.append(name)
    return {
        "schema_version": 1,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "target_root": str(target_root.resolve()),
        "target_manifest_sha256": sha256_file(target_root / "manifest.json"),
        "source": source,
        "target": target,
        "target_over_source_median_ratio": ratios,
        "shift_threshold": "median ratio outside [0.5, 2.0]",
        "shifted_regions": shifted,
        "reprojection_domain_shift_detected": bool(shifted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = audit(args.source_manifest.resolve(), args.target_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
