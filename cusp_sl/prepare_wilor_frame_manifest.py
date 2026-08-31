"""Create an immutable WiLoR video-frame manifest from CUSP cache clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_image_size_order(
    cached_size: tuple[int, int], actual_size: tuple[int, int]
) -> str:
    """Identify whether a cache stores image size as width-height or height-width."""
    actual_width, actual_height = actual_size
    if (actual_width, actual_height) == cached_size:
        return "width_height"
    if (actual_height, actual_width) == cached_size:
        return "height_width"
    raise ValueError(
        f"Raw image {(actual_width, actual_height)} is inconsistent "
        f"with cache image_size {cached_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Raw-frame root for cache sources that are initializer PKLs",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if args.limit_clips is not None and args.limit_clips < 1:
        raise ValueError("--limit-clips must be positive")

    source_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    entries = source_manifest["clips"]
    if args.limit_clips is not None:
        entries = entries[: args.limit_clips]
    records: list[dict[str, object]] = []
    cache_hashes: dict[str, str] = {}
    video_paths: set[Path] = set()
    image_paths: set[Path] = set()
    image_keys: set[str] = set()
    for entry in entries:
        cache = Path(entry) if Path(entry).is_absolute() else args.input_manifest.parent / entry
        cache = cache.resolve()
        with np.load(cache, allow_pickle=False) as payload:
            clip_id = str(payload["clip_id"].item())
            frame_names = payload["frame_names"].astype(str).tolist()
            source_paths = payload["source_paths"].astype(str).tolist()
            frame_numbers = payload["frame_numbers"].astype(np.int64).tolist()
            image_sizes = payload["image_size"].astype(np.int64)
            metadata = json.loads(str(payload["metadata_json"].item()))
        if not (
            len(frame_names)
            == len(source_paths)
            == len(frame_numbers)
            == len(image_sizes)
        ):
            raise ValueError(f"Inconsistent frame arrays in {cache}")
        cache_hashes[str(cache)] = sha256(cache)
        for name, source, frame_number, image_size in zip(
            frame_names, source_paths, frame_numbers, image_sizes
        ):
            marker = "#frame="
            record_source: dict[str, object]
            if marker in source:
                video_text, source_frame_text = source.rsplit(marker, 1)
                if int(source_frame_text) != int(frame_number):
                    raise ValueError(f"Frame-number mismatch in {cache}: {source}")
                video = Path(video_text).resolve()
                if not video.is_file():
                    raise FileNotFoundError(video)
                video_paths.add(video)
                image_key = f"{name}.png"
                record_source = {
                    "video_path": str(video),
                    "frame_number": int(frame_number),
                    "cache_image_size_order": "width_height",
                }
                expected_width = int(image_size[0])
                expected_height = int(image_size[1])
            else:
                # The locked SGNify cache points at the initializer result PKL.
                # Require an explicit raw-frame root. The sibling ``images/``
                # directory contains fitting visualizations at a different
                # resolution and must never be mistaken for model input RGB.
                result_path = Path(source).resolve()
                if result_path.suffix != ".pkl" or not result_path.is_file():
                    raise ValueError(
                        "Non-video cache source must be an existing initializer PKL: "
                        f"{source}"
                    )
                if args.image_root is None:
                    raise ValueError(
                        "--image-root is required for initializer-PKL cache sources"
                    )
                image = args.image_root.resolve() / clip_id / f"{name}.png"
                if not image.is_file():
                    raise FileNotFoundError(image)
                with Image.open(image) as decoded:
                    actual_width, actual_height = decoded.size
                cached_size = (int(image_size[0]), int(image_size[1]))
                try:
                    cache_size_order = cache_image_size_order(
                        cached_size, (actual_width, actual_height)
                    )
                except ValueError as error:
                    raise ValueError(f"{error}: {image}") from error
                image_paths.add(image)
                image_key = f"{clip_id}__{name}.png"
                record_source = {
                    "image_path": str(image),
                    "cache_image_size_order": cache_size_order,
                }
                expected_width = actual_width
                expected_height = actual_height
            if image_key in image_keys:
                raise ValueError(f"Duplicate WiLoR image key: {image_key}")
            image_keys.add(image_key)
            records.append(
                {
                    "image_key": image_key,
                    "clip_id": clip_id,
                    **record_source,
                    "expected_width": expected_width,
                    "expected_height": expected_height,
                    "initializer_expert": metadata.get("initializer_expert"),
                    "initializer_matches_locked_lane_a1": metadata.get(
                        "initializer_matches_locked_lane_a1"
                    ),
                }
            )

    payload = {
        "schema_version": (
            "cusp_sl_wilor_frame_manifest_v2"
            if image_paths
            else "cusp_sl_wilor_frame_manifest_v1"
        ),
        "role": "benchmark_subset" if args.limit_clips else "frozen_frontend_input",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "clip_count": len(entries),
        "frame_count": len(records),
        "cache_sha256": cache_hashes,
        "video_sha256": {str(path): sha256(path) for path in sorted(video_paths)},
        "image_sha256": {str(path): sha256(path) for path in sorted(image_paths)},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"clips": len(entries), "frames": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
