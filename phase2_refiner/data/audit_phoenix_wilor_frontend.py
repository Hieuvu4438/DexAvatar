"""Incrementally audit immutable PHOENIX WiLoR shard artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2_refiner.data.build_phoenix_soke_full_cache import (
    _artifact_paths,
    _verified_artifacts,
)
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-phoenix-wilor-frontend-audit-v1"


def _artifact_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_previous(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("schema") == SCHEMA else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit_once(
    *,
    run_root: Path,
    output: Path,
    splits: tuple[str, ...],
    minimum_age_seconds: float,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    previous = _load_previous(output).get("verified", {})
    verified: dict[str, Any] = {}
    split_reports: dict[str, Any] = {}
    for split in splits:
        shard_root = run_root / "wilor_shards" / split
        report_path = shard_root / "shard_report.json"
        shard_report = json.loads(report_path.read_text(encoding="utf-8"))
        output_root = run_root / "wilor_outputs" / split
        complete_artifacts = 0
        for index, shard in enumerate(shard_report["shards"]):
            manifest = shard_root / f"shard_{index:04d}.json"
            hamer_path, wilor_path = _artifact_paths(output_root, index)
            if not (hamer_path.is_file() and wilor_path.is_file()):
                continue
            if min(
                now - hamer_path.stat().st_mtime,
                now - wilor_path.stat().st_mtime,
            ) < minimum_age_seconds:
                continue
            complete_artifacts += 1
            key = f"{split}/shard_{index:04d}"
            states = {
                "hamer": _artifact_state(hamer_path),
                "wilor": _artifact_state(wilor_path),
            }
            manifest_sha256 = sha256_file(manifest)
            prior = previous.get(key)
            if (
                isinstance(prior, dict)
                and prior.get("artifact_state") == states
                and prior.get("manifest_sha256") == manifest_sha256
            ):
                verified[key] = prior
                continue

            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            expected = {
                str(record["image_key"])
                for record in manifest_payload["records"]
            }
            hamer, wilor = _verified_artifacts(output_root, manifest, index)
            verified[key] = {
                "split": split,
                "shard": index,
                "records": len(manifest_payload["records"]),
                "raw_keys": len(wilor["images"]),
                "hamer_keys": len(hamer),
                "hamer_dropouts": len(expected - set(hamer)),
                "manifest_sha256": manifest_sha256,
                "hamer_sha256": sha256_file(hamer_path),
                "wilor_sha256": sha256_file(wilor_path),
                "artifact_state": states,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        declared = len(shard_report["shards"])
        verified_count = sum(
            item.get("split") == split for item in verified.values()
        )
        split_reports[split] = {
            "declared_shards": declared,
            "complete_artifacts": complete_artifacts,
            "verified_shards": verified_count,
            "all_verified": verified_count == declared,
        }
    payload = {
        "schema": SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root.resolve()),
        "splits": split_reports,
        "verified": verified,
        "all_verified": all(item["all_verified"] for item in split_reports.values()),
    }
    _atomic_json(output, payload)
    return payload


def require_all_verified(payload: dict[str, Any]) -> None:
    if not payload.get("all_verified"):
        incomplete = sorted(
            split
            for split, item in payload.get("splits", {}).items()
            if not item.get("all_verified")
        )
        raise RuntimeError(
            f"Not every declared WiLoR shard is verified: {incomplete}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--minimum-age-seconds", type=float, default=5.0)
    parser.add_argument(
        "--require-all-verified",
        action="store_true",
        help="Fail a one-shot audit unless every declared shard is verified",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    arguments = parser.parse_args()
    while True:
        payload = audit_once(
            run_root=arguments.run_root,
            output=arguments.output,
            splits=tuple(arguments.splits),
            minimum_age_seconds=arguments.minimum_age_seconds,
        )
        summary = {
            split: {
                "verified": item["verified_shards"],
                "declared": item["declared_shards"],
            }
            for split, item in payload["splits"].items()
        }
        print(
            f"[{datetime.now().astimezone().isoformat()}] {json.dumps(summary, sort_keys=True)}",
            flush=True,
        )
        if not arguments.watch:
            if arguments.require_all_verified:
                require_all_verified(payload)
            return
        if payload["all_verified"]:
            return
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    main()
