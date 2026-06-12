#!/usr/bin/env python3
"""Build the paper-formula segment JSON for the SGNify central-frames protocol.

For each sign, this script reads the input video length T from
`data/frames/{sign}/low_*.png` and computes the paper's central window:

    new_start = min_idx + round(0.5 * T / 8)
    new_end   = min_idx + round(7 * T / 8)

where T = max_idx - min_idx + 1 and indices are 30 fps PNG frame ids.

The output JSON has the same schema as `data/segment.json`:
    { "Ablehnen": [128, 222], "Akzeptieren": [105, 221], ... }

This file is consumed by `dexavatar_fitting/cfg_files/fit_smplx_vposer_x_paper.yaml`
(via `sign_segment: 'segment_paper.json'`), giving SMPLify-X a wider fitting
window that matches the SGNify paper's "core part of a sign" definition
(Appendix C: 0.5 × T/8 < t < 7 × T/8).

This script is idempotent — re-running overwrites the output file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signs_file", default="data/signs.txt",
                   help="Path to signs.txt (one sign per line, with class label)")
    p.add_argument("--frames_root", default="data/frames",
                   help="Root directory containing per-sign subdirs of low_*.png")
    p.add_argument("--out_path", default="dexavatar_fitting/cfg_files/segment_paper.json",
                   help="Output JSON path")
    return p.parse_args()


def load_signs(signs_file: Path) -> list[str]:
    signs: list[str] = []
    with open(signs_file) as f:
        for line in f:
            line = line.strip()
            if line:
                signs.append(line.split()[0])
    return signs


def input_frame_range(frames_dir: Path) -> tuple[int, int] | None:
    """Return (min_idx, max_idx) from low_*.png filenames, or None if empty.

    Mirrors data_parser.py:145-158 (data_parser.py filename_to_int).
    """
    ids: list[int] = []
    for p in frames_dir.glob("low_*.png"):
        try:
            ids.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    if not ids:
        return None
    return min(ids), max(ids)


def paper_central_window(min_idx: int, max_idx: int) -> tuple[int, int]:
    """Return (new_start, new_end) per SGNify paper Appendix C.

    new_start = min + round(0.5 * T / 8)
    new_end   = min + round(7.0 * T / 8)
    where T = max - min + 1. Strict-inequality boundaries (caller enforces).
    """
    T = max_idx - min_idx + 1
    new_start = min_idx + round(0.5 * T / 8.0)
    new_end   = min_idx + round(7.0 * T / 8.0)
    return new_start, new_end


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent  # repo root

    signs_file = Path(args.signs_file)
    if not signs_file.is_absolute():
        signs_file = project_root / signs_file
    frames_root = Path(args.frames_root)
    if not frames_root.is_absolute():
        frames_root = project_root / frames_root
    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    if not signs_file.exists():
        print(f"[ERROR] signs file not found: {signs_file}", file=sys.stderr)
        return 1
    if not frames_root.exists():
        print(f"[ERROR] frames root not found: {frames_root}", file=sys.stderr)
        return 1

    signs = load_signs(signs_file)
    print(f"Loaded {len(signs)} signs from {signs_file}")

    segment_paper: dict[str, list[int]] = {}
    skipped: list[str] = []
    for sign in signs:
        sign_dir = frames_root / sign
        r = input_frame_range(sign_dir)
        if r is None:
            print(f"[WARN] {sign}: no low_*.png in {sign_dir}; skipping", file=sys.stderr)
            skipped.append(sign)
            continue
        min_idx, max_idx = r
        new_start, new_end = paper_central_window(min_idx, max_idx)
        segment_paper[sign] = [new_start, new_end]
        T = max_idx - min_idx + 1
        print(f"  {sign}: T={T} (min={min_idx}, max={max_idx}) "
              f"→ keep (t > {new_start}, t < {new_end})  [{new_end - new_start - 1} frames]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(segment_paper, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(segment_paper)} entries to {out_path}")
    if skipped:
        print(f"[WARN] skipped {len(skipped)} signs (no input frames): {skipped}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
