"""Split a locked WiLoR frame manifest into append-only, clip-aligned shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2_refiner.provenance import sha256_file


def _clip_id(image_key: str) -> str:
    stem = Path(image_key).stem
    clip_id, separator, frame = stem.rpartition("_")
    if not separator or not clip_id or not frame.isdigit():
        raise ValueError(f"Cannot recover clip id from image key: {image_key}")
    return clip_id


def shard(input_path: Path, output_dir: Path, max_frames: int) -> dict:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Append-only shard directory exists: {output_dir}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if payload.get("frame_count") != len(records):
        raise ValueError("Manifest frame_count does not match records")

    clip_records: list[list[dict]] = []
    current_clip = None
    for record in records:
        clip_id = _clip_id(str(record["image_key"]))
        if clip_id != current_clip:
            clip_records.append([])
            current_clip = clip_id
        clip_records[-1].append(record)
    if any(len(items) > max_frames for items in clip_records):
        raise ValueError("A single clip exceeds --max-frames")

    groups: list[list[dict]] = []
    for items in clip_records:
        if not groups or len(groups[-1]) + len(items) > max_frames:
            groups.append([])
        groups[-1].extend(items)

    output_dir.mkdir(parents=True)
    shard_reports = []
    input_sha256 = sha256_file(input_path)
    for index, items in enumerate(groups):
        videos = sorted({str(item["video_path"]) for item in items})
        video_sha256 = payload.get("video_sha256", {})
        missing = [video for video in videos if video not in video_sha256]
        if missing:
            raise ValueError(f"Missing source video hashes: {missing[:3]}")
        output = output_dir / f"shard_{index:04d}.json"
        shard_payload = {
            "schema_version": "cusp_sl_wilor_frame_manifest_v2",
            "source_selection": payload.get("source_selection"),
            "source_selection_sha256": payload.get("source_selection_sha256"),
            "parent_manifest": str(input_path),
            "parent_manifest_sha256": input_sha256,
            "shard_index": index,
            "shard_count": len(groups),
            "frame_count": len(items),
            "records": items,
            "image_sha256": {},
            "video_sha256": {video: video_sha256[video] for video in videos},
        }
        output.write_text(
            json.dumps(shard_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shard_reports.append(
            {
                "manifest": str(output),
                "manifest_sha256": sha256_file(output),
                "clips": len({_clip_id(str(item["image_key"])) for item in items}),
                "frames": len(items),
                "videos": len(videos),
            }
        )

    report = {
        "schema": "signal4d-wilor-frame-shards-v1",
        "input_manifest": str(input_path),
        "input_manifest_sha256": input_sha256,
        "max_frames": max_frames,
        "shards": shard_reports,
        "total_frames": sum(item["frames"] for item in shard_reports),
    }
    (output_dir / "shard_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=640)
    arguments = parser.parse_args()
    print(
        json.dumps(
            shard(arguments.input, arguments.output_dir, arguments.max_frames),
            indent=2,
            sort_keys=True,
        )
    )
