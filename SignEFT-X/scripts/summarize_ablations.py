#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = {
    "All": "tr all",
    "UBody": "tr above pelvis upper body",
    "UBody-F": "tr above pelvis minus face",
    "UBody-H": "tr above pelvis minus head",
    "LHand": "tr left hand",
    "RHand": "tr right hand",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", action="append", nargs=2, metavar=("ID", "RUN"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for row_id, run_value in args.row:
        run = Path(run_value).resolve()
        refinement = json.loads((run / "refinement_summary.json").read_text(encoding="utf-8"))
        official = json.loads((run / "metrics" / "official_result.json").read_text(encoding="utf-8"))
        values = official["metrics_mm"]
        row = {"row": row_id, "run_root": str(run)}
        row.update({name: values[key] for name, key in METRICS.items()})
        row.update({
            "accepted": refinement["accepted"],
            "fallback": refinement["fallback"],
            "acceptance_rate": refinement.get(
                "acceptance_rate", refinement["accepted"] / max(refinement["frames"], 1),
            ),
            "evaluator_sha256": official["evaluator_sha256"],
            "implementation_sha256": refinement.get("implementation_sha256"),
        })
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(args.out.resolve()), "json": str(json_path.resolve()), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
