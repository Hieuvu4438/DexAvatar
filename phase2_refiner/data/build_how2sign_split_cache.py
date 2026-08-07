"""Build one append-only Phase 2 cache split from How2Sign teacher outputs.

This is the single-split companion to :mod:`build_how2sign_cache`.  It exists so
an official held-out split can be materialized without changing or copying the
legacy train/validation cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from phase2_refiner.data.build_how2sign_cache import (
    _load_selection,
    _make_clip,
    _pose,
    _quality,
    _source_group,
)
from phase2_refiner.data.cache_schema import save_cache_clip


def build(args: argparse.Namespace) -> dict:
    teacher_root = args.teacher_root.resolve()
    output = args.output.resolve()
    split = args.split
    if output.exists():
        raise FileExistsError(f"Append-only cache output already exists: {output}")
    selection_payload, selected = _load_selection(teacher_root, split)
    clip_dir = output / "clips" / split
    split_dir = output / "splits"
    clip_dir.mkdir(parents=True)
    split_dir.mkdir()
    entries: list[str] = []
    groups: set[str] = set()
    rejected: list[dict] = []
    missing: list[str] = []
    try:
        for index, item in enumerate(selected, start=1):
            teacher_path = teacher_root / split / "clips" / f"{item['clip_id']}.npz"
            if not teacher_path.is_file():
                missing.append(item["clip_id"])
                continue
            with np.load(teacher_path, allow_pickle=False) as payload:
                quality = _quality(_pose(payload))
            if not quality["passed"]:
                rejected.append({"clip_id": item["clip_id"], **quality})
                continue
            clip = _make_clip(teacher_path, item, split, quality)
            destination = clip_dir / f"{clip.clip_id}.npz"
            save_cache_clip(destination, clip)
            entries.append(f"../clips/{split}/{destination.name}")
            groups.add(_source_group(item["clip_id"]))
            if index % 100 == 0 or index == len(selected):
                print(
                    f"[how2sign-cache] split={split} processed={index}/{len(selected)} "
                    f"accepted={len(entries)} rejected={len(rejected)}",
                    flush=True,
                )
        if missing:
            raise ValueError(
                f"Incomplete teacher coverage: {len(missing)}/{len(selected)} missing; "
                f"first={missing[:3]}"
            )
        if not entries:
            raise RuntimeError(f"No {split} clips passed quality filtering")
        manifest = {
            "dataset": "How2Sign",
            "official_split": split,
            "clips": entries,
            "source_groups": sorted(groups),
            "motion_domain": "sign_language_asl",
            "training_target_scope": "complete-body-and-hands",
            "target_type": "SMPLer-X H32 pseudo-3D",
            "teacher_selection": str(
                (teacher_root / split / "selection.json").resolve()
            ),
            "sgnify_excluded": True,
        }
        manifest_path = split_dir / f"{split}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = {
            "official_split": split,
            "selected": len(selected),
            "accepted_clips": len(entries),
            "accepted_frames": len(entries)
            * int(selection_payload["frames_per_clip"]),
            "source_groups": len(groups),
            "rejected_quality": rejected,
            "complete_teacher_coverage": True,
        }
        (output / "audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    except Exception:
        shutil.rmtree(output)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
