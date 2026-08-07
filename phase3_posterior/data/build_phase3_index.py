"""Build append-only Phase 3 indexes that reference immutable Phase 2 clips."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
import torch

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.render import create_smplx_model
from phase3_posterior.data.cache_schema import (
    SCHEMA_VERSION,
    Phase3IndexEntry,
    reject_forbidden_path,
)
from phase3_posterior.data.build_relation_targets import (
    InterHandJointProvider,
    build_sidecar,
)
from phase3_posterior.data.cache_schema import save_relation_sidecar
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.data.quality_filter import assess_clip


def _manifest_paths(path: Path) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("clips", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"Invalid source manifest: {path}")
    return [
        (path.parent / value).resolve()
        if not Path(value).is_absolute()
        else Path(value)
        for value in values
    ]


def _metadata(clip: Any) -> dict[str, Any]:
    try:
        value = json.loads(clip.metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid clip metadata for {clip.clip_id}") from error
    return value if isinstance(value, dict) else {}


def _resolve_signer(source: dict[str, Any], metadata: dict[str, Any]) -> str:
    signer = str(metadata.get("signer", metadata.get("subject", "unknown")))
    if signer != "unknown":
        return signer
    if source.get("signer_resolver") != "how2sign_filename_v1":
        return signer
    source_clip = str(metadata.get("source_clip", ""))
    match = re.search(r"-(\d+)-rgb_front$", source_clip)
    if match is None:
        raise ValueError(f"Cannot parse How2Sign signer from {source_clip!r}")
    return f"how2sign_signer_{int(match.group(1)):02d}"


def build_index(
    sources_path: Path,
    output: Path,
    model_folder: Path | None = None,
    device: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite Phase 3 cache: {output}")
    with sources_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    sources = config.get("sources", [])
    if not sources:
        raise ValueError("data source configuration contains no sources")
    output.mkdir(parents=True, exist_ok=resume)
    by_split: dict[str, list[Phase3IndexEntry]] = defaultdict(list)
    group_splits: dict[str, str] = {}
    signer_splits: dict[str, str] = {}
    unknown_signers: list[str] = []
    source_report: list[dict[str, Any]] = []
    relation_sources: dict[Path, str] = {}
    model = (
        create_smplx_model(model_folder, torch.device(device))
        if model_folder is not None
        else None
    )
    interhand_provider = InterHandJointProvider()
    for source in sources:
        license_id = str(source.get("license_id", "")).strip()
        if not license_id:
            raise ValueError(f"Missing license_id for source {source.get('name')}")
        source_count = 0
        quality_counts: dict[str, int] = defaultdict(int)
        for split, manifest_value in source.get("splits", {}).items():
            manifest = Path(manifest_value).resolve()
            reject_forbidden_path(manifest)
            for clip_path in _manifest_paths(manifest):
                reject_forbidden_path(clip_path)
                clip = load_cache_clip(clip_path)
                metadata = _metadata(clip)
                group = str(
                    metadata.get("source_group", metadata.get("video_id", clip.clip_id))
                )
                signer = _resolve_signer(source, metadata)
                group_identity = f"{source['name']}:{group}"
                previous_split = group_splits.setdefault(group_identity, split)
                if previous_split != split:
                    raise ValueError(
                        f"Source-group leakage: {group_identity} occurs in "
                        f"{previous_split} and {split}"
                    )
                if signer == "unknown":
                    unknown_signers.append(clip.clip_id)
                else:
                    signer_identity = f"{source['name']}:{signer}"
                    signer_split = signer_splits.setdefault(signer_identity, split)
                    if signer_split != split:
                        raise ValueError(
                            f"Signer leakage: {signer_identity} occurs in "
                            f"{signer_split} and {split}"
                        )
                relation = (
                    output / "relations" / str(source["name"]) / f"{clip.clip_id}.npz"
                )
                clip_digest = sha256_file(clip_path)
                previous_digest = relation_sources.setdefault(relation, clip_digest)
                if previous_digest != clip_digest:
                    raise ValueError(
                        f"Non-unique clip_id maps different clips to {relation}"
                    )
                if not relation.exists():
                    save_relation_sidecar(
                        relation,
                        build_sidecar(
                            clip_path,
                            model=model,
                            device=device,
                            interhand_provider=interhand_provider,
                        ),
                    )
                quality = assess_clip(clip_path)
                quality_path = (
                    output / "quality" / str(source["name"]) / f"{clip.clip_id}.json"
                )
                if quality_path.exists() and not resume:
                    raise FileExistsError(f"Duplicate quality artifact: {quality_path}")
                if not quality_path.exists():
                    atomic_json(quality_path, quality)
                quality_counts[str(quality["band"])] += 1
                entry = Phase3IndexEntry(
                    clip_id=clip.clip_id,
                    clip_path=str(clip_path),
                    source=str(source["name"]),
                    domain=str(source.get("domain", "unknown")),
                    split=str(split),
                    source_group=group,
                    signer=signer,
                    target_weight=(
                        0.0
                        if quality["catastrophic"]
                        else float(source.get("target_weight", 1.0))
                    ),
                    license_id=license_id,
                    clip_sha256=clip_digest,
                    relation_path=str(relation) if relation.exists() else "",
                    relation_sha256=sha256_file(relation) if relation.exists() else "",
                )
                entry.validate()
                by_split[str(split)].append(entry)
                source_count += 1
                if source_count == 1 or source_count % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "phase3_cache_progress",
                                "source": source["name"],
                                "clips": source_count,
                                "split": split,
                            }
                        ),
                        flush=True,
                    )
        source_report.append(
            {
                "name": source["name"],
                "license_id": license_id,
                "clips": source_count,
                "quality_bands": dict(sorted(quality_counts.items())),
            }
        )
    split_hashes: dict[str, str] = {}
    for split, entries in sorted(by_split.items()):
        target = output / "splits" / f"{split}.json"
        atomic_json(
            target,
            {
                "schema_version": SCHEMA_VERSION,
                "clips": [entry.__dict__ for entry in entries],
            },
        )
        split_hashes[split] = sha256_file(target)
    report = {
        "schema_version": SCHEMA_VERSION,
        "sources_config": str(sources_path.resolve()),
        "sources_config_sha256": sha256_file(sources_path),
        "sources": source_report,
        "split_counts": {key: len(value) for key, value in sorted(by_split.items())},
        "split_sha256": split_hashes,
        "forbidden_source_scan": "passed",
        "source_signer_disjoint": not unknown_signers,
        "unknown_signer_clips": unknown_signers,
    }
    atomic_json(output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_index(
                args.sources,
                args.output,
                model_folder=args.model_folder,
                device=args.device,
                resume=args.resume,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
