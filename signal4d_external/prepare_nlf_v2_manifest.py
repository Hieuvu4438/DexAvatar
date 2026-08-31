"""Create deterministic, signer-disjoint How2Sign subsets for NLF V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from phase2_refiner.provenance import sha256_file
from signal4d_external.leakage import FORBIDDEN_PATH_PARTS


def _rank(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}:{row['signer']}:{row['source_group']}:{row['clip_id']}"
    return hashlib.sha256(value.encode()).hexdigest()


def _load_how2sign(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("clips", []) if row.get("source") == "how2sign"]
    if not rows:
        raise ValueError(f"No How2Sign rows in {index_path}")
    return rows


def _diverse_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin signers while preferring distinct source videos."""
    if count < 1:
        raise ValueError("requested clip count must be positive")
    by_signer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_signer[str(row["signer"])].append(row)
    for signer in by_signer:
        by_signer[signer].sort(key=lambda row: _rank(seed, row))

    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    signer_names = sorted(by_signer)
    # Two passes: first maximize source-video diversity, then fill if necessary.
    for require_new_group in (True, False):
        cursors = {signer: 0 for signer in signer_names}
        progressed = True
        while len(selected) < count and progressed:
            progressed = False
            for signer in signer_names:
                values = by_signer[signer]
                while cursors[signer] < len(values):
                    row = values[cursors[signer]]
                    cursors[signer] += 1
                    if row in selected:
                        continue
                    group = str(row["source_group"])
                    if require_new_group and group in seen_groups:
                        continue
                    selected.append(row)
                    seen_groups.add(group)
                    progressed = True
                    break
                if len(selected) >= count:
                    break
    if len(selected) != min(count, len(rows)):
        raise AssertionError("deterministic sampler did not reach requested size")
    return selected


def build_manifest(index_path: Path, split: str, count: int, seed: int) -> dict[str, Any]:
    selected = _diverse_sample(_load_how2sign(index_path), count, seed)
    clips = []
    for row in selected:
        cache_path = Path(row["clip_path"]).resolve()
        normalized = cache_path.as_posix().lower()
        if any(part in normalized for part in FORBIDDEN_PATH_PARTS):
            raise ValueError(f"Forbidden SGNify path: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            sources = cache["source_paths"].astype(str)
            frame_ids = cache["frame_numbers"].astype(np.int64)
            frame_names = cache["frame_names"].astype(str)
            if not (len(sources) == len(frame_ids) == len(frame_names)):
                raise ValueError(f"Frame contract mismatch: {cache_path}")
            videos = {source.rsplit("#frame=", 1)[0] for source in sources}
            if len(videos) != 1:
                raise ValueError(f"Expected one video per clip: {cache_path}")
            video_path = Path(videos.pop()).resolve()
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            clips.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "split": split,
                    "signer": str(row["signer"]),
                    "source_group": str(row["source_group"]),
                    "cache_path": str(cache_path),
                    "cache_sha256": sha256_file(cache_path),
                    "video_path": str(video_path),
                    "frame_ids": frame_ids.tolist(),
                    "frame_names": frame_names.tolist(),
                }
            )
    return {
        "schema_version": "signal4d.external_nlf_v2_manifest.v1",
        "dataset": "How2Sign",
        "split": split,
        "seed": seed,
        "source_index": str(index_path.resolve()),
        "source_index_sha256": sha256_file(index_path),
        "forbidden_sgnify_scan": "passed",
        "sgnify_training_reads": 0,
        "clip_count": len(clips),
        "frame_count": sum(len(row["frame_ids"]) for row in clips),
        "signers": sorted({row["signer"] for row in clips}),
        "source_groups": len({row["source_group"] for row in clips}),
        "clips": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--validation-index", required=True, type=Path)
    parser.add_argument("--calibration-index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-clips", type=int, default=512)
    parser.add_argument("--validation-clips", type=int, default=192)
    parser.add_argument("--calibration-clips", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True)
    specifications = (
        ("train", args.train_index, args.train_clips),
        ("validation", args.validation_index, args.validation_clips),
        ("calibration", args.calibration_index, args.calibration_clips),
    )
    manifests = {}
    signer_sets = {}
    for split, source, count in specifications:
        payload = build_manifest(source.resolve(), split, count, args.seed)
        destination = args.output_root / f"{split}.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifests[split] = {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "clips": payload["clip_count"],
            "frames": payload["frame_count"],
        }
        signer_sets[split] = set(payload["signers"])
    for first, second in (("train", "validation"), ("train", "calibration"), ("validation", "calibration")):
        overlap = signer_sets[first] & signer_sets[second]
        if overlap:
            raise ValueError(f"Signer overlap {first}/{second}: {sorted(overlap)}")
    summary = {
        "schema_version": "signal4d.external_nlf_v2_protocol.v1",
        "seed": args.seed,
        "signer_disjoint": True,
        "sgnify_training_reads": 0,
        "manifests": manifests,
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
