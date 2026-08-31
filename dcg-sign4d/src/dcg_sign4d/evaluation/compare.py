"""Paired evaluator comparison with explicit cluster semantics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .bootstrap import paired_cluster_bootstrap


def _read_rows(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["clip_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate clip rows in {path}")
    return result


def compare_per_clip(
    baseline_csv: str | Path,
    candidate_csv: str | Path,
    *,
    metrics: tuple[str, ...],
    output_path: str | Path,
    replicates: int = 10_000,
    seed: int = 12345,
    cluster_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    baseline_rows = _read_rows(baseline_csv)
    candidate_rows = _read_rows(candidate_csv)
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("baseline/candidate clip coverage differs")
    if not metrics:
        raise ValueError("at least one metric is required")
    baseline_columns = set(next(iter(baseline_rows.values()))) if baseline_rows else set()
    candidate_columns = set(next(iter(candidate_rows.values()))) if candidate_rows else set()
    missing = {
        metric: {
            "baseline": metric not in baseline_columns,
            "candidate": metric not in candidate_columns,
        }
        for metric in metrics
        if metric not in baseline_columns or metric not in candidate_columns
    }
    if missing:
        details = ", ".join(
            f"{metric} ({'/'.join(side for side, absent in sides.items() if absent)})"
            for metric, sides in missing.items()
        )
        raise ValueError(f"requested metric columns are missing: {details}")
    if cluster_map is None:
        cluster_map = {clip_id: clip_id for clip_id in baseline_rows}
        cluster_unit = "clip_sensitivity_not_registered_signer_bootstrap"
    else:
        cluster_unit = "signer"
    result: dict[str, Any] = {
        "baseline": str(baseline_csv),
        "candidate": str(candidate_csv),
        "cluster_unit": cluster_unit,
        "metrics": {},
    }
    for metric in metrics:
        baseline = {
            clip_id: float(row[metric])
            for clip_id, row in baseline_rows.items()
            if row.get(metric) not in {None, ""}
        }
        candidate = {
            clip_id: float(candidate_rows[clip_id][metric])
            for clip_id in baseline
            if candidate_rows[clip_id].get(metric) not in {None, ""}
        }
        if baseline.keys() != candidate.keys():
            raise ValueError(f"baseline/candidate valid coverage differs for metric {metric}")
        if len(baseline) < 2:
            raise ValueError(f"metric {metric} has fewer than two paired valid clips")
        clusters = {clip_id: cluster_map[clip_id] for clip_id in baseline}
        result["metrics"][metric] = paired_cluster_bootstrap(
            baseline,
            candidate,
            clusters,
            replicates=replicates,
            seed=seed,
        )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"immutable comparison exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result
