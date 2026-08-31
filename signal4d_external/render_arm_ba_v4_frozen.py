"""Render frozen arm BA V4 results using exact V1 mesh anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phase2_refiner.provenance import sha256_file
from phase2_refiner.render import render_source_anchored_directory


EXPECTED_FRAMES = 1493


def run(args: argparse.Namespace) -> dict:
    root = args.output_root.resolve()
    baseline = args.baseline_root.resolve()
    output = root / "render_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    freeze_path = root / "freeze_audit.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("decision") != "PASS" or int(freeze.get("sgnify_target_reads", -1)) != 0:
        raise ValueError("V4 result is not frozen target-free output")
    if freeze.get("baseline_root") != str(baseline):
        raise ValueError("Render baseline does not match freeze audit")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("frames", -1)) != EXPECTED_FRAMES:
        raise ValueError("Frozen manifest coverage mismatch")
    incomplete = root / ".render_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    tree = hashlib.sha256()
    clips = []
    total = 0
    for index, row in enumerate(manifest["clips"], start=1):
        clip_id = row["clip_id"]
        result_dir = root / clip_id / "smplifyx" / "results"
        result_paths = sorted(result_dir.glob("*.pkl"))
        if len(result_paths) != int(row["frames"]):
            raise ValueError(f"Result coverage mismatch: {clip_id}")
        sources = [
            str(baseline / clip_id / "smplifyx" / "results" / path.name)
            for path in result_paths
        ]
        mesh_dir = root / clip_id / "smplifyx" / "meshes"
        count = render_source_anchored_directory(
            result_dir, mesh_dir, sources, args.model_folder, args.device
        )
        mesh_paths = sorted(mesh_dir.glob("*.obj"))
        if count != len(result_paths) or len(mesh_paths) != len(result_paths):
            raise ValueError(f"Mesh coverage mismatch: {clip_id}")
        for path in mesh_paths:
            relative = path.relative_to(root).as_posix()
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(sha256_file(path).encode("ascii"))
            tree.update(b"\n")
        clips.append({"clip_id": clip_id, "meshes": count})
        total += count
        print(f"[arm-ba-v4-render] {index}/{len(manifest['clips'])} {clip_id}", flush=True)
    if total != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} meshes, got {total}")
    report = {
        "schema_version": "signal4d.external_arm_ba_v4_render.v1",
        "method": "SIGNAL4D_EXTERNAL_ARM_BA_V4_FROZEN_RENDER",
        "output_root": str(root),
        "baseline_root": str(baseline),
        "freeze_audit_sha256": sha256_file(freeze_path),
        "result_tree_sha256": freeze["result_tree_sha256"],
        "mesh_tree_sha256": tree.hexdigest(),
        "model_folder": str(args.model_folder.resolve()),
        "device": str(args.device),
        "clips": clips,
        "frames": total,
        "sgnify_target_reads": 0,
    }
    temporary = root / ".render_manifest.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    incomplete.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--model-folder", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
