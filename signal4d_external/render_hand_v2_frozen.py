"""Render a frozen V2H result tree while preserving V1 mesh anchoring."""

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
    render_manifest_path = root / "render_manifest.json"
    if render_manifest_path.exists():
        raise FileExistsError(render_manifest_path)
    freeze_path = root / "freeze_audit.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("decision") != "PASS" or int(freeze.get("sgnify_target_reads", -1)) != 0:
        raise ValueError("V2H result is not frozen target-free output")
    if freeze.get("baseline_root") != str(baseline):
        raise ValueError("Render baseline does not match freeze audit")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_frames = int(manifest.get("frames", -1))
    clip_frames = sum(int(row["frames"]) for row in manifest["clips"])
    if manifest_frames != EXPECTED_FRAMES or clip_frames != EXPECTED_FRAMES:
        raise ValueError(
            "Frozen manifest coverage mismatch: "
            f"manifest={manifest_frames} clips={clip_frames} expected={EXPECTED_FRAMES}"
        )
    incomplete = root / ".render_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    tree = hashlib.sha256()
    clips = []
    total = 0
    for index, row in enumerate(manifest["clips"], start=1):
        clip_id = row["clip_id"]
        result_dir = root / clip_id / "smplifyx" / "results"
        result_paths = sorted(result_dir.glob("*.pkl"))
        expected_count = int(row["frames"])
        if len(result_paths) != expected_count:
            raise ValueError(
                f"Result coverage mismatch for {clip_id}: "
                f"got={len(result_paths)} expected={expected_count}"
            )
        source_path_objects = [
            baseline / clip_id / "smplifyx" / "results" / path.name
            for path in result_paths
        ]
        missing_sources = [str(path) for path in source_path_objects if not path.is_file()]
        if missing_sources:
            raise FileNotFoundError(
                f"Missing {len(missing_sources)} V1 source anchors for {clip_id}: "
                f"{missing_sources[:3]}"
            )
        source_paths = [str(path) for path in source_path_objects]
        mesh_dir = root / clip_id / "smplifyx" / "meshes"
        count = render_source_anchored_directory(
            result_dir,
            mesh_dir,
            source_paths,
            args.model_folder,
            args.device,
        )
        mesh_paths = sorted(mesh_dir.glob("*.obj"))
        expected_names = {path.with_suffix(".obj").name for path in result_paths}
        actual_names = {path.name for path in mesh_paths}
        if count != expected_count or actual_names != expected_names:
            raise ValueError(
                f"Mesh coverage mismatch for {clip_id}: count={count} "
                f"expected={expected_count} missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}"
            )
        for path in mesh_paths:
            relative = path.relative_to(root).as_posix()
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(sha256_file(path).encode("ascii"))
            tree.update(b"\n")
        clips.append({"clip_id": clip_id, "meshes": count})
        total += count
        print(f"[hand-v2-render] {index}/{len(manifest['clips'])} {clip_id}", flush=True)
    if total != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} meshes, got {total}")
    report = {
        "schema_version": 1,
        "method": "SIGNAL4D_EXTERNAL_HAND_V2_FROZEN_RENDER",
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
    temporary_manifest = root / ".render_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(render_manifest_path)
    incomplete.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
