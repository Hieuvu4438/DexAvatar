#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.data.frame_manifest import build_all_manifests, load_sign_classes
from signpk.utils.config import load_yaml, project_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict SGNify frame manifests")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data/sgnify.yaml")
    parser.add_argument("--allow-missing-gt", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)["data"]
    segment_path = project_path(config["segment_file"], PROJECT_ROOT)
    signs_path = project_path(config["signs_file"], PROJECT_ROOT)
    segments = json.loads(segment_path.read_text(encoding="utf-8"))
    signs = load_sign_classes(signs_path)
    manifests = build_all_manifests(
        segments,
        signs,
        frames_root=project_path(config["frames_root"], PROJECT_ROOT),
        gt_root=project_path(config["gt_root"], PROJECT_ROOT),
        image_glob=config.get("image_glob", "low_*.png"),
        image_id_regex=config.get("image_id_regex", r"(?:low_)?(\d+)"),
        gt_id_multiplier=int(config.get("gt_id_multiplier", 2)),
        fps=float(config.get("fps", 25.0)),
        boundary_padding=config.get("boundary_padding", "reflect"),
        strict_gt=bool(config.get("strict_gt", True)) and not args.allow_missing_gt,
    )
    root = project_path(config["manifest_root"], PROJECT_ROOT)
    for manifest in manifests:
        manifest.save(root / manifest.sign_name / "manifest.json")
    print(
        f"wrote {len(manifests)} manifests / {sum(len(item.records) for item in manifests)} "
        f"central frames to {root}"
    )


if __name__ == "__main__":
    main()
