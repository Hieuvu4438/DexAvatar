"""Incrementally validate stable PHOENIX H32 artifacts while extraction runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2_refiner.data.audit_phoenix_h32_frontend import validate_payload
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-phoenix-h32-incremental-audit-v1"


def audit_incremental(
    selection_root: Path,
    h32_root: Path,
    splits: tuple[str, ...],
    *,
    minimum_age_seconds: float,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection_root = selection_root.resolve()
    h32_root = h32_root.resolve()
    previous = previous if previous and previous.get("schema") == SCHEMA else {}
    previous_verified = previous.get("verified", {})
    now = time.time()
    selections = {}
    selection_hashes = {}
    declared: list[tuple[str, str, dict[str, Any]]] = []
    seen = set()
    for split in splits:
        path = selection_root / split / "selection.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        selections[split] = payload
        selection_hashes[split] = sha256_file(path)
        for clip in payload["clips"]:
            name = str(clip["source_clip"])
            if name in seen:
                raise ValueError(f"H32 clip occurs in multiple audited splits: {name}")
            seen.add(name)
            declared.append((split, name, clip))
    can_reuse = previous.get("selection_sha256") == selection_hashes
    verified = {}
    unstable = []
    newly_validated = reused = 0
    for split, name, clip in declared:
        path = h32_root / f"{name}.pkl"
        if not path.is_file():
            continue
        stat = path.stat()
        if now - stat.st_mtime < minimum_age_seconds:
            unstable.append(name)
            continue
        prior = previous_verified.get(name) if can_reuse else None
        if (
            prior
            and prior.get("split") == split
            and int(prior.get("size", -1)) == stat.st_size
            and int(prior.get("mtime_ns", -1)) == stat.st_mtime_ns
        ):
            item = dict(prior)
            reused += 1
        else:
            item = validate_payload(
                path,
                clip,
                now=now,
                minimum_age_seconds=minimum_age_seconds,
            )
            item["split"] = split
            newly_validated += 1
        verified[name] = item
    split_reports = {}
    for split in splits:
        items = [item for item in verified.values() if item["split"] == split]
        declared_count = len(selections[split]["clips"])
        split_reports[split] = {
            "declared_clips": declared_count,
            "verified_clips": len(items),
            "pending_clips": declared_count - len(items),
            "source_video_frames": sum(
                int(item["source_video_frames"]) for item in items
            ),
            "h32_retained_frames": sum(
                int(item["h32_retained_frames"]) for item in items
            ),
        }
    combined = hashlib.sha256()
    for name in sorted(verified):
        combined.update(
            name.encode("utf-8")
            + b"\0"
            + str(verified[name]["sha256"]).encode("ascii")
            + b"\n"
        )
    return {
        "schema": SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_fields_opened": False,
        "selection_sha256": selection_hashes,
        "minimum_age_seconds": minimum_age_seconds,
        "declared_clips": len(declared),
        "verified_clips": len(verified),
        "pending_clips": len(declared) - len(verified),
        "unstable_clips": sorted(unstable),
        "newly_validated": newly_validated,
        "reused": reused,
        "all_verified": len(verified) == len(declared),
        "h32_verified_content_set_sha256": combined.hexdigest(),
        "splits": split_reports,
        "verified": verified,
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--h32-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--minimum-age-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.minimum_age_seconds < 0 or args.interval_seconds <= 0:
        parser.error("age must be non-negative and interval must be positive")
    output = args.output.resolve()
    while True:
        previous = (
            json.loads(output.read_text(encoding="utf-8"))
            if output.is_file()
            else None
        )
        report = audit_incremental(
            args.selection_root,
            args.h32_root,
            tuple(args.splits),
            minimum_age_seconds=args.minimum_age_seconds,
            previous=previous,
        )
        _write_atomic(output, report)
        print(
            json.dumps(
                {
                    "timestamp_utc": report["timestamp_utc"],
                    "verified": report["verified_clips"],
                    "declared": report["declared_clips"],
                    "new": report["newly_validated"],
                    "reused": report["reused"],
                    "unstable": len(report["unstable_clips"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.watch or report["all_verified"]:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
