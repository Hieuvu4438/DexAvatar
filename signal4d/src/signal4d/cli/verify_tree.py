from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.hashing import sha256_file


def _relative_hashes(root: str | Path) -> dict[str, str]:
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(base)
    return {
        item.relative_to(base).as_posix(): sha256_file(item)
        for item in sorted(base.rglob("*"))
        if item.is_file()
    }


def run(first: str, second: str, output: str) -> dict[str, Any]:
    first_hashes = _relative_hashes(first)
    second_hashes = _relative_hashes(second)
    first_names = set(first_hashes)
    second_names = set(second_hashes)
    mismatched = sorted(
        name
        for name in first_names & second_names
        if first_hashes[name] != second_hashes[name]
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "first_root": str(first),
        "second_root": str(second),
        "first_files": len(first_hashes),
        "second_files": len(second_hashes),
        "missing_from_first": sorted(second_names - first_names),
        "missing_from_second": sorted(first_names - second_names),
        "hash_mismatches": mismatched,
    }
    report["passed"] = not (
        report["missing_from_first"]
        or report["missing_from_second"]
        or report["hash_mismatches"]
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
