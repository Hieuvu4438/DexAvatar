from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_results(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
