"""Bind a locked sign-domain component selection to WiLoR video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2_refiner.provenance import sha256_file


def build(selection: Path, output: Path) -> dict:
    selection = selection.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only frame manifest exists: {output}")
    payload = json.loads(selection.read_text(encoding="utf-8"))
    if payload.get("schema") != "signal4d-sign-domain-smplerx-selection-v1":
        raise ValueError(f"Unsupported component selection: {selection}")
    records = []
    videos: dict[str, str] = {}
    for entry in payload["clips"]:
        video = str(Path(entry["video"]).resolve())
        expected_width = int(entry["source_contract"]["width"])
        expected_height = int(entry["source_contract"]["height"])
        videos.setdefault(video, sha256_file(video))
        for frame in entry["frame_indices"]:
            records.append(
                {
                    "image_key": f"{entry['clip_id']}_{int(frame):06d}.png",
                    "video_path": video,
                    "frame_number": int(frame),
                    "expected_width": expected_width,
                    "expected_height": expected_height,
                }
            )
    manifest = {
        "schema_version": "cusp_sl_wilor_frame_manifest_v2",
        "source_selection": str(selection),
        "source_selection_sha256": sha256_file(selection),
        "frame_count": len(records),
        "records": records,
        "image_sha256": {},
        "video_sha256": dict(sorted(videos.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
        "clips": len(payload["clips"]),
        "frames": len(records),
        "videos": len(videos),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.selection, arguments.output), indent=2, sort_keys=True))

