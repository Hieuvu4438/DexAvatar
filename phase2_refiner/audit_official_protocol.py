"""Reconcile the paper's 2,872 frames with the released 1,493-mesh protocol."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from phase2_refiner.provenance import sha256_file


def audit(segment_path: Path, gt_root: Path, manifest_path: Path) -> dict:
    with segment_path.open("r", encoding="utf-8") as handle:
        segments = json.load(handle)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    manifest_counts = Counter(row["sign"] for row in manifest)
    signs = []
    source_span_frames = inclusive_source_frames = released_frames = 0
    all_cadences: set[int] = set()
    for sign, bounds in sorted(segments.items()):
        start, end = map(int, bounds)
        if end <= start:
            raise ValueError(f"Invalid segment for {sign}: {bounds}")
        numbers = sorted(int(path.stem) for path in (gt_root / sign).glob("*.obj"))
        selected = [number for number in numbers if 2 * start <= number <= 2 * end]
        cadence = sorted(set(np.diff(numbers).tolist())) if len(numbers) > 1 else []
        all_cadences.update(cadence)
        source_span_frames += end - start
        inclusive_source_frames += end - start + 1
        released_frames += len(selected)
        signs.append(
            {
                "sign": sign,
                "segment_start": start,
                "segment_end": end,
                "paper_span_frames_end_exclusive": end - start,
                "released_selected_meshes": len(selected),
                "released_mesh_cadence": cadence,
                "manifest_rows": manifest_counts[sign],
                "manifest_matches_released": manifest_counts[sign] == len(selected),
            }
        )
    paper_count = 2872
    released_count = 1493
    report = {
        "schema_version": 1,
        "segment_file": str(segment_path.resolve()),
        "segment_file_sha256": sha256_file(segment_path),
        "gt_root": str(gt_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "signs": len(signs),
        "paper_reported_frames": paper_count,
        "sum_segment_end_minus_start": source_span_frames,
        "sum_segment_inclusive_lengths": inclusive_source_frames,
        "released_selected_gt_meshes": released_frames,
        "released_manifest_rows": len(manifest),
        "released_gt_cadences": sorted(all_cadences),
        "paper_count_exactly_explained_by_end_exclusive_segment_spans": (
            source_span_frames == paper_count
        ),
        "released_count_reproduced": (
            released_frames == released_count == len(manifest)
        ),
        "every_sign_manifest_matches_released_selection": all(
            item["manifest_matches_released"] for item in signs
        ),
        "interpretation": (
            "2,872 is exactly the sum of the 57 end-exclusive source segment "
            "spans. The released evaluator doubles those segment bounds but the "
            "released GT meshes have cadence 4, producing 1,493 inclusive meshes."
        ),
        "official_comparison_go": False,
        "official_comparison_blocker": (
            "The intermediate meshes needed for the paper's 2,872-frame population "
            "are not present; released 1,493-frame metrics remain Lane L only."
        ),
        "per_sign": signs,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("data/evaluation_from_author/segment.json"),
    )
    parser.add_argument("--gt-root", type=Path, default=Path("data/smplx_gt"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("probes/results/phase0/frame_manifest.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = audit(
        args.segments.resolve(), args.gt_root.resolve(), args.manifest.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_sign"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
