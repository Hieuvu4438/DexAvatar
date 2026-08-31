"""Freeze and audit a materialized external-only V2H result tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from phase2_refiner.provenance import sha256_file


HAND_KEYS = {"lhand": "left_hand_pose", "rhand": "right_hand_pose"}


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed result: {path}")
    return payload


def _equal(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return np.array_equal(np.asarray(first), np.asarray(second))
    return first == second


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    baseline_root = args.baseline_root.resolve()
    run_manifest_path = root / "run_manifest.json"
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("method") != "SIGNAL4D_EXTERNAL_HAND_V2_RANK_SOFT_RESIDUAL":
        raise ValueError("Unexpected V2H run manifest")
    if int(manifest.get("sgnify_target_reads_before_evaluation", -1)) != 0:
        raise ValueError("V2H manifest does not prove target-free materialization")

    expected_clips = {row["clip_id"]: row for row in manifest["clips"]}
    tree = hashlib.sha256()
    frames = 0
    selected = {region: 0 for region in HAND_KEYS}
    changed = {region: 0 for region in HAND_KEYS}
    fallback = {region: 0 for region in HAND_KEYS}
    preserved_nonhand_fields = 0
    preserved_unselected_hands = 0
    for clip_id in sorted(expected_clips):
        diagnostics_path = root / clip_id / "external_hand_v2_diagnostics" / "sequence.npz"
        with np.load(diagnostics_path) as diagnostics:
            frame_names = diagnostics["frame_names"].astype(str)
            masks = {
                region: np.asarray(diagnostics[f"selected_{region}"], dtype=bool)
                for region in HAND_KEYS
            }
            fallback_masks = {
                region: np.asarray(diagnostics[f"fallback_{region}"], dtype=bool)
                for region in HAND_KEYS
            }
        if len(frame_names) != expected_clips[clip_id]["frames"]:
            raise ValueError(f"Frame count mismatch: {clip_id}")
        for region in HAND_KEYS:
            selected[region] += int(masks[region].sum())
            fallback[region] += int(fallback_masks[region].sum())
        result_dir = root / clip_id / "smplifyx" / "results"
        result_paths = sorted(result_dir.glob("*.pkl"))
        if {path.stem for path in result_paths} != set(frame_names):
            raise ValueError(f"Result coverage mismatch: {clip_id}")
        for frame, frame_name in enumerate(frame_names):
            output_path = result_dir / f"{frame_name}.pkl"
            baseline_path = (
                baseline_root / clip_id / "smplifyx" / "results" / f"{frame_name}.pkl"
            )
            output = _load_pickle(output_path)
            baseline = _load_pickle(baseline_path)
            if set(output) != set(baseline):
                raise ValueError(f"Result keys changed: {output_path}")
            for key in output:
                if key not in HAND_KEYS.values():
                    if not _equal(output[key], baseline[key]):
                        raise ValueError(f"Non-hand field changed: {output_path}:{key}")
                    preserved_nonhand_fields += 1
            for region, key in HAND_KEYS.items():
                same = _equal(output[key], baseline[key])
                if not masks[region][frame] and not same:
                    raise ValueError(f"Unselected hand changed: {output_path}:{key}")
                if not masks[region][frame]:
                    preserved_unselected_hands += 1
                elif not same:
                    changed[region] += 1
            relative = output_path.relative_to(root).as_posix()
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(sha256_file(output_path).encode("ascii"))
            tree.update(b"\n")
            frames += 1
    checks = {
        "57_clips": len(expected_clips) == 57,
        "1493_frames": frames == 1493,
        "zero_fallbacks": sum(fallback.values()) == 0,
        "nonhand_fields_exact_v1": True,
        "unselected_hands_exact_v1": True,
        "target_reads_zero": True,
    }
    report = {
        "schema_version": 1,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "output_root": str(root),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "result_tree_sha256": tree.hexdigest(),
        "baseline_root": str(baseline_root),
        "baseline_manifest_sha256": sha256_file(baseline_root / "run_manifest.json"),
        "clips": len(expected_clips),
        "frames": frames,
        "selected_frames": selected,
        "selected_fraction": {
            region: count / frames for region, count in selected.items()
        },
        "changed_selected_frames": changed,
        "fallback_frames": fallback,
        "preserved_nonhand_field_arrays": preserved_nonhand_fields,
        "preserved_unselected_hand_arrays": preserved_unselected_hands,
        "sgnify_target_reads": 0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
