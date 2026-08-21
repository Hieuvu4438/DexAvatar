from __future__ import annotations

from ..evaluation.sgnify import compare_clip_metrics


def run(
    candidate_csv: str,
    baseline_csv: str,
    metric: str,
    output: str,
    replicates: int,
) -> dict[str, float]:
    return compare_clip_metrics(candidate_csv, baseline_csv, metric, output, replicates=replicates)
