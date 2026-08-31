"""Target-free image-evidence audit for a frozen strong-A1 derived cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cusp_sl.evidence import DIRECT_BODY_OBSERVATIONS
from phase2_refiner.data.cache_schema import load_cache_clip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_huber(
    residual: np.ndarray, confidence: np.ndarray, delta: float
) -> tuple[float, float]:
    distance = np.linalg.norm(residual, axis=-1)
    robust = np.where(
        distance <= delta,
        0.5 * distance**2 / max(delta, 1e-8),
        distance - 0.5 * delta,
    )
    weight = float(confidence.sum())
    return float((robust * confidence).sum() / max(weight, 1.0)), weight


def visible_motion(
    residual: np.ndarray, confidence: np.ndarray
) -> tuple[float, float]:
    if len(residual) < 2:
        return 0.0, 0.0
    distance = np.linalg.norm(residual[1:] - residual[:-1], axis=-1)
    weight = np.minimum(confidence[1:], confidence[:-1])
    denominator = float(weight.sum())
    return float((distance * weight).sum() / max(denominator, 1.0)), denominator


def cluster_delta_interval(
    records: list[dict], method: str, base: str, weight: str,
    *, replicates: int, seed: int,
) -> dict[str, float | int]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["source_group"]), []).append(record)
    if len(grouped) < 2:
        raise ValueError("Cluster bootstrap requires two source groups")
    numerators, denominators = [], []
    for values in grouped.values():
        weights = np.asarray([row[weight] for row in values], dtype=np.float64)
        deltas = np.asarray(
            [row[method] - row[base] for row in values], dtype=np.float64
        )
        numerators.append(float((weights * deltas).sum()))
        denominators.append(float(weights.sum()))
    numerator = np.asarray(numerators)
    denominator = np.asarray(denominators)
    observed = float(numerator.sum() / denominator.sum())
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(grouped), size=(replicates, len(grouped))
    )
    sampled = numerator[indices].sum(1) / denominator[indices].sum(1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "clusters": len(grouped),
        "replicates": replicates,
        "delta": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_improvement": float(np.mean(sampled < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--a1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if args.huber_delta <= 0 or args.bootstrap_replicates < 1:
        raise ValueError("Huber delta and bootstrap replicates must be positive")
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    a1_manifest = json.loads(args.a1_manifest.read_text(encoding="utf-8"))
    if a1_manifest.get("role") != "frozen_strong_a1_derived_cache":
        raise ValueError("Candidate manifest is not a frozen strong-A1 cache")
    if a1_manifest.get("source_manifest_sha256") != sha256(args.base_manifest):
        raise ValueError("A1/source-manifest hash mismatch")
    declared = {
        str(item["clip_id"]): item for item in a1_manifest["summaries"]
    }
    entries = a1_manifest["clips"]
    if len(entries) != len(base_manifest["clips"]) or len(entries) != len(declared):
        raise ValueError("Base/A1 manifest coverage differs")
    observable = np.zeros(51, dtype=bool)
    observable[list(DIRECT_BODY_OBSERVATIONS)] = True
    observable[21:] = True
    group_slices = {
        "overall": (0, 51),
        "body": (0, 21),
        "hands": (21, 51),
        "left": (21, 36),
        "right": (36, 51),
    }
    records = []
    for base_entry, a1_entry in zip(
        base_manifest["clips"], entries, strict=True
    ):
        base_path = Path(base_entry["cache"] if isinstance(base_entry, dict) else base_entry)
        if not base_path.is_absolute():
            base_path = args.base_manifest.parent / base_path
        a1_path = Path(a1_entry["cache"] if isinstance(a1_entry, dict) else a1_entry)
        if not a1_path.is_absolute():
            a1_path = args.a1_manifest.parent / a1_path
        base = load_cache_clip(base_path)
        a1 = load_cache_clip(a1_path)
        if base.clip_id != a1.clip_id or not np.array_equal(
            base.frame_names.astype(str), a1.frame_names.astype(str)
        ):
            raise ValueError(f"Base/A1 clip identity mismatch: {base_path}")
        item = declared.get(base.clip_id)
        if item is None or sha256(base_path) != item["source_cache_sha256"]:
            raise ValueError(f"Base cache hash mismatch: {base_path}")
        if sha256(a1_path) != item["derived_cache_sha256"]:
            raise ValueError(f"A1 cache hash mismatch: {a1_path}")
        metadata = json.loads(base.metadata_json)
        source_group = str(metadata.get("source_group", ""))
        if not source_group:
            raise ValueError(f"Cache lacks source_group: {base_path}")
        record: dict[str, object] = {
            "clip_id": base.clip_id,
            "source_group": source_group,
            "frames": len(base.frame_names),
        }
        common_valid = (
            base.keypoint_valid
            & base.refine_mask[None]
            & observable[None]
        )
        common_confidence = np.clip(base.raw_confidence, 0.0, 1.0) * common_valid
        for group, (start, stop) in group_slices.items():
            confidence = common_confidence[:, start:stop]
            base_obs, obs_weight = weighted_huber(
                base.reprojection_residual_2d[:, start:stop],
                confidence,
                args.huber_delta,
            )
            a1_obs, _ = weighted_huber(
                a1.reprojection_residual_2d[:, start:stop],
                confidence,
                args.huber_delta,
            )
            base_motion, motion_weight = visible_motion(
                base.reprojection_residual_2d[:, start:stop], confidence
            )
            a1_motion, _ = visible_motion(
                a1.reprojection_residual_2d[:, start:stop], confidence
            )
            record.update({
                f"{group}_observation_weight": obs_weight,
                f"{group}_motion_weight": motion_weight,
                f"base_{group}_observation": base_obs,
                f"a1_{group}_observation": a1_obs,
                f"base_{group}_motion": base_motion,
                f"a1_{group}_motion": a1_motion,
            })
        records.append(record)
    if set(declared) != {str(row["clip_id"]) for row in records}:
        raise ValueError("Derived-cache/source clip sets differ")
    args.output.mkdir(parents=True)
    with (args.output / "per_clip.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def aggregate(value: str, weight: str) -> float:
        weights = np.asarray([row[weight] for row in records], dtype=np.float64)
        return float(np.average([row[value] for row in records], weights=weights))

    summary: dict[str, object] = {
        "role": "target_free_frozen_frontend_image_evidence_audit",
        "target_reads": 0,
        "clips": len(records),
        "frames": int(sum(row["frames"] for row in records)),
        "huber_delta_normalized_image": args.huber_delta,
        "base_manifest_sha256": sha256(args.base_manifest),
        "a1_manifest_sha256": sha256(args.a1_manifest),
        "wilor_pickle_sha256": a1_manifest["wilor_pickle_sha256"],
    }
    for group in group_slices:
        for term in ("observation", "motion"):
            weight = f"{group}_{term}_weight"
            summary[f"base_{group}_{term}"] = aggregate(
                f"base_{group}_{term}", weight
            )
            summary[f"a1_{group}_{term}"] = aggregate(
                f"a1_{group}_{term}", weight
            )
            summary[f"clustered_a1_minus_base_{group}_{term}"] = (
                cluster_delta_interval(
                    records,
                    f"a1_{group}_{term}",
                    f"base_{group}_{term}",
                    weight,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + len(summary),
                )
            )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
