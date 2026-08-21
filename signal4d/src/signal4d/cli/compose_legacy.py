from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.manifest import load_manifest
from ..utils.hashing import sha256_file


def run(
    manifest_path: str,
    primary_root: str,
    fallback_root: str,
    output_root: str,
    primary_subpath: str = "smplifyx/results",
    fallback_subpath: str = "smplifyx/results",
    method_name: str = "legacy_primary_fallback_composition",
) -> dict[str, Any]:
    primary = Path(primary_root)
    fallback = Path(fallback_root)
    output = Path(output_root)
    source_counts = {"primary": 0, "fallback": 0}
    rows: list[dict[str, Any]] = []
    for item in load_manifest(manifest_path):
        destination = output / item.clip_id / "smplifyx/results"
        destination.mkdir(parents=True, exist_ok=True)
        for frame_id in item.frame_ids:
            source = primary / item.clip_id / primary_subpath / f"low_{frame_id:03d}.pkl"
            source_name = "primary"
            if not source.is_file():
                source = fallback / item.clip_id / fallback_subpath / f"low_{frame_id:03d}.pkl"
                source_name = "fallback"
            if not source.is_file():
                raise FileNotFoundError(
                    f"no primary/fallback legacy parameters for {item.clip_id}/{frame_id}"
                )
            target = destination / source.name
            resolved = source.resolve()
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != resolved:
                    raise FileExistsError(f"composition target conflicts: {target}")
            else:
                target.symlink_to(resolved)
            digest = sha256_file(resolved)
            source_counts[source_name] += 1
            rows.append(
                {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "source": source_name,
                    "source_path": str(resolved),
                    "source_sha256": digest,
                    "composed_path": str(target),
                }
            )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "method_name": method_name,
        "selection_policy": "primary_else_atomic_fallback_per_manifest_frame",
        "manifest_sha256": sha256_file(manifest_path),
        "clips": len({row["clip_id"] for row in rows}),
        "frames": len(rows),
        "source_counts": source_counts,
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "composition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
