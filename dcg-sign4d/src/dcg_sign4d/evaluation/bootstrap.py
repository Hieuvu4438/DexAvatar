"""Deterministic paired cluster bootstrap; signer is required for final inference."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def paired_cluster_bootstrap(
    baseline: dict[str, float],
    candidate: dict[str, float],
    clusters: dict[str, str],
    *,
    replicates: int = 10_000,
    seed: int = 12345,
) -> dict[str, float | int]:
    if baseline.keys() != candidate.keys() or baseline.keys() != clusters.keys():
        raise ValueError("paired values and cluster assignments must have identical item IDs")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    grouped: dict[str, list[str]] = defaultdict(list)
    for item_id, cluster_id in clusters.items():
        if not cluster_id or cluster_id.lower() == "unknown":
            raise ValueError("unknown clusters cannot support a cluster bootstrap")
        grouped[cluster_id].append(item_id)
    cluster_ids = sorted(grouped)
    if len(cluster_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    deltas = np.asarray(
        [candidate[item_id] - baseline[item_id] for item_id in sorted(baseline)], dtype=np.float64
    )
    point = float(deltas.mean())
    cluster_delta = {
        cluster_id: np.asarray(
            [candidate[item_id] - baseline[item_id] for item_id in grouped[cluster_id]],
            dtype=np.float64,
        )
        for cluster_id in cluster_ids
    }
    rng = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        distribution[index] = np.concatenate(
            [cluster_delta[cluster_id] for cluster_id in sample]
        ).mean()
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "mean_delta": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "clusters": len(cluster_ids),
        "replicates": replicates,
        "seed": seed,
    }


def cluster_bootstrap(
    values: dict[str, float],
    clusters: dict[str, str],
    *,
    replicates: int = 10_000,
    seed: int = 12345,
) -> dict[str, float | int]:
    """Clip-macro point estimate with signer-cluster-resampled uncertainty."""

    if values.keys() != clusters.keys() or not values:
        raise ValueError("values and cluster assignments must have identical non-empty IDs")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    grouped: dict[str, list[float]] = defaultdict(list)
    for item_id, cluster_id in clusters.items():
        if not cluster_id or cluster_id.lower() == "unknown":
            raise ValueError("unknown clusters cannot support a cluster bootstrap")
        grouped[cluster_id].append(float(values[item_id]))
    cluster_ids = sorted(grouped)
    if len(cluster_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    rng = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        distribution[index] = np.concatenate(
            [np.asarray(grouped[cluster_id], dtype=np.float64) for cluster_id in sample]
        ).mean()
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "mean": float(np.mean(list(values.values()))),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "clusters": len(cluster_ids),
        "items": len(values),
        "replicates": replicates,
        "seed": seed,
    }
