#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.data.frame_manifest import SignManifest
from signpk.geometry.topology import load_reference_faces, validate_topology
from signpk.observers.h4w_wrapper import load_h4w_cache
from signpk.observers.omnihands_wrapper import load_omnihands_cache
from signpk.utils.config import load_yaml, project_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate observer caches against manifests")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data/sgnify.yaml")
    parser.add_argument("--require-omni", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    data, observers, geometry = config["data"], config["observers"], config["geometry"]
    manifest_root = project_path(data["manifest_root"], PROJECT_ROOT)
    reference_faces = load_reference_faces(project_path(geometry["smplx_model"], PROJECT_ROOT))
    h4w_root = project_path(observers["h4w"]["cache_root"], PROJECT_ROOT)
    omni_root = project_path(observers["omnihands"]["cache_root"], PROJECT_ROOT)
    validated = 0
    omni_validated = 0
    for manifest_path in sorted(manifest_root.glob("*/manifest.json")):
        manifest = SignManifest.load(manifest_path, validate_paths=True)
        body, left, right, _ = load_h4w_cache(
            h4w_root,
            manifest,
            expected_commit=observers["h4w"].get("expected_commit"),
        )
        for vertices in body.vertices:
            validate_topology(vertices, reference_faces, reference_faces)
        if not (left.valid | right.valid).any():
            raise ValueError(f"no valid H4W++ hands for {manifest.sign_name}")
        omni_path = omni_root / manifest.sign_name / "omni.pt"
        if omni_path.is_file():
            load_omnihands_cache(omni_path, manifest)
            omni_validated += 1
        elif args.require_omni:
            raise FileNotFoundError(omni_path)
        validated += 1
    if not validated:
        raise FileNotFoundError(f"no manifests in {manifest_root}; run build_sgnify_manifest.py")
    print(f"validated H4W++ for {validated} signs; OmniHands for {omni_validated} signs")


if __name__ == "__main__":
    main()
