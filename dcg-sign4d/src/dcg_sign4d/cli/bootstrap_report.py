"""Unpaired signer-cluster bootstrap over already clip-aggregated metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dcg_sign4d.evaluation.bootstrap import cluster_bootstrap
from dcg_sign4d.utils.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="per-clip CSV")
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--cluster", default="signer_id")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.metrics)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "clip_id" not in rows[0] or args.cluster not in rows[0]:
        raise ValueError("metrics CSV lacks clip_id or requested cluster column")
    result = {
        "schema_version": "dcg_cluster_bootstrap_v1",
        "source_sha256": file_sha256(source),
        "cluster_column": args.cluster,
        "metrics": {},
    }
    for metric in args.metric:
        if metric not in rows[0]:
            raise ValueError(f"missing metric column: {metric}")
        active = [row for row in rows if row.get(metric) not in {None, ""}]
        values = {row["clip_id"]: float(row[metric]) for row in active}
        clusters = {row["clip_id"]: row[args.cluster] for row in active}
        result["metrics"][metric] = cluster_bootstrap(
            values,
            clusters,
            replicates=args.replicates,
            seed=args.seed,
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable bootstrap report exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
