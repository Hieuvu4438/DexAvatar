"""Convert the frozen legacy SGNify split into the strict DCG manifest schema."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from dcg_sign4d.data.manifest import ManifestItem, load_manifest
from dcg_sign4d.utils.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--raw-observation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-count", type=int, default=5)
    args = parser.parse_args()
    legacy_root = Path(args.legacy_root)
    raw_root = Path(args.raw_observation_root).resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable converted manifest exists: {output}")
    source_paths = {
        "calibration": legacy_root / "sgnify_calibration.jsonl",
        "development": legacy_root / "sgnify_development.jsonl",
        "test": legacy_root / "sgnify_test.jsonl",
    }
    source_rows: dict[str, list[dict[str, object]]] = {}
    for name, path in source_paths.items():
        source_rows[name] = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    development_ids = sorted(row["clip_id"] for row in source_rows["development"])
    if not 0 < args.validation_count < len(development_ids):
        raise ValueError("validation count must leave at least one development training clip")
    validation_ids = set(development_ids[-args.validation_count :])
    items: list[ManifestItem] = []
    for source_split, rows in source_rows.items():
        for row in rows:
            frame_ids = tuple(int(value) for value in row["frame_ids"])
            if source_split == "development":
                split = "validation" if row["clip_id"] in validation_ids else "train"
            else:
                split = source_split
            items.append(
                ManifestItem(
                    clip_id=row["clip_id"],
                    # The legacy release contains selected frame sequences, not
                    # source videos. This existing immutable NPZ is only the
                    # manifest path anchor; cache identity uses the frozen raw
                    # frame-hash registry and records that policy explicitly.
                    video_path=raw_root / row["clip_id"] / "raw_keypoints.npz",
                    # Legacy frame IDs are already indexed on the released
                    # 15 fps timeline (confirmed by the raw artifact timestamps).
                    fps_native=15.0,
                    frame_count=max(frame_ids) + 1,
                    width=1341,
                    height=804,
                    signer_id="sgnify_signer_metadata_unavailable",
                    split=split,
                    camera_id="sgnify_camera_metadata_unavailable",
                    dataset_name="SGNify",
                    dataset_version="legacy_a1_extended_post_v1",
                    license_id="SGNify_local_research_release",
                    fps_effective=float(row["fps"]),
                    frame_mapping=frame_ids,
                )
            )
    items.sort(key=lambda item: item.clip_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for split in ("all", "train", "calibration", "validation", "test"):
            selected = items if split == "all" else [item for item in items if item.split == split]
            path = temporary / f"sgnify_{split}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
                    for item in selected
                ),
                "utf-8",
            )
            if selected:
                load_manifest(path, require_existing_video=True)
        report = {
            "schema_version": "dcg_sgnify_manifest_conversion_v1",
            "development_only": True,
            "scientific_status": "SIGNER_METADATA_UNAVAILABLE_NOT_SIGNER_DISJOINT",
            "source_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
            "split_policy": "legacy development sorted clip IDs; final N assigned validation",
            "validation_count": args.validation_count,
            "counts": {
                split: sum(item.split == split for item in items)
                for split in ("train", "calibration", "validation", "test")
            },
            "clips": len(items),
            "frames": sum(item.effective_frame_count for item in items),
            "source_identity_policy": "frozen raw per-frame image hash registry",
        }
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", "utf-8"
        )
        (temporary / "MANIFEST_CONVERSION_COMPLETE").write_text("complete\n", "utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
