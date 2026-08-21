from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ..utils.hashing import sha256_file
from .manifest import ClipManifest, load_manifest, write_manifest


def _rank(clip_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()


def freeze_clip_splits(
    source_manifest: str | Path,
    output_dir: str | Path,
    calibration_clips: int,
    development_clips: int,
    seed: int,
    extra_development: tuple[str, ...] = (),
) -> dict[str, object]:
    """Create deterministic, clip-disjoint manifests without inspecting targets."""
    source_manifest = Path(source_manifest)
    rows = load_manifest(source_manifest)
    if calibration_clips < 1 or development_clips < 1:
        raise ValueError("calibration and development must each contain at least one clip")
    if calibration_clips + development_clips >= len(rows):
        raise ValueError("at least one clip must remain in the test split")

    ranked = sorted(rows, key=lambda row: (_rank(row.clip_id, seed), row.clip_id))
    names = (
        ["calibration"] * calibration_clips
        + ["development"] * development_clips
        + ["test"] * (len(rows) - calibration_clips - development_clips)
    )
    split_rows: dict[str, list[ClipManifest]] = {
        "calibration": [],
        "development": [],
        "test": [],
    }
    for row, split_name in zip(ranked, names, strict=True):
        split_rows[split_name].append(
            row.model_copy(
                update={
                    "split": split_name,
                    "allowed_for_calibration": split_name == "calibration",
                    "allowed_for_hparam_selection": split_name == "development",
                    "allowed_for_final_reporting": split_name == "test",
                }
            )
        )

    known = {row.clip_id for row in rows}
    unknown_extra = set(extra_development) - known
    if unknown_extra:
        raise ValueError(f"unknown extra development clips: {sorted(unknown_extra)}")
    for clip_id in extra_development:
        for source_split in ("calibration", "test"):
            match = next((row for row in split_rows[source_split] if row.clip_id == clip_id), None)
            if match is not None:
                if source_split == "calibration":
                    raise ValueError("a calibration clip cannot also be quarantined to development")
                split_rows[source_split].remove(match)
                split_rows["development"].append(
                    match.model_copy(
                        update={
                            "split": "development",
                            "allowed_for_calibration": False,
                            "allowed_for_hparam_selection": True,
                            "allowed_for_final_reporting": False,
                        }
                    )
                )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for split_name, split in split_rows.items():
        path = output_dir / f"sgnify_{split_name}.jsonl"
        write_manifest(sorted(split, key=lambda row: row.clip_id), path)
        hashes[split_name] = sha256_file(path)
        counts[split_name] = {
            "clips": len(split),
            "frames": sum(len(row.frame_ids) for row in split),
        }

    all_signers = Counter(row.signer_id for row in rows)
    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "strategy": "sha256(seed:clip_id), then fixed clip counts",
        "seed": seed,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "counts": counts,
        "manifest_sha256": hashes,
        "clip_disjoint": True,
        "extra_development_quarantine": list(extra_development),
        "signer_disjoint": len(all_signers) == len(rows),
        "signer_limitation": (
            None
            if len(all_signers) == len(rows)
            else "SGNify release lacks usable signer identifiers; signer_id is unknown, so "
            "the frozen split is clip-disjoint but signer-disjointness cannot be verified."
        ),
    }
    metadata_path = output_dir / "split_freeze.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
