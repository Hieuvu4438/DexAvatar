"""Create immutable full-PHOENIX/SOKE split selections.

The official PHOENIX-2014T annotation gzip files are the split authority.  The
released SOKE frame-pose tree is used only as the reconstruction target.  This
command validates clip/frame bindings without opening any pose payload and
writes one append-only selection per official split.  Downstream training must
consume only ``train/selection.json``; ``dev`` is checkpoint selection and
``test`` remains evaluation-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import cv2

from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-phoenix-soke-full-selection-v1"
COMPATIBLE_SELECTION_SCHEMA = "signal4d-sign-domain-smplerx-selection-v1"
SPLITS = ("train", "dev", "test")
PHASE2_SPLIT = {"train": "train", "dev": "val", "test": "test"}
TARGET_PATTERN = re.compile(r"images(\d+)\.pkl$")


def _load_official(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, list):
        raise ValueError(f"Official annotations must be a list: {path}")
    return payload


def _video_contract(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"Cannot open PHOENIX video: {path}")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(
            f"Invalid video contract for {path}: "
            f"frames={frames} size={width}x{height} fps={fps}"
        )
    return {"frame_count": frames, "width": width, "height": height, "fps": fps}


def _target_indices(directory: Path) -> list[int]:
    indices = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = TARGET_PATTERN.fullmatch(path.name)
            if match:
                indices.append(int(match.group(1)))
    indices.sort()
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate SOKE target frame indices: {directory}")
    return indices


def _source_group(clip: str) -> str:
    return re.sub(r"-\d+$", "", clip)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    official_root = args.official_root.resolve()
    video_root = args.video_root.resolve()
    target_root = args.target_root.resolve()
    output = args.output.resolve()
    archive = args.target_archive.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only PHOENIX selection exists: {output}")
    if not archive.is_file():
        raise FileNotFoundError(archive)

    output.mkdir(parents=True)
    split_reports: dict[str, Any] = {}
    admitted_names: dict[str, set[str]] = {}
    official_names: dict[str, set[str]] = {}
    try:
        for split in SPLITS:
            annotation = official_root / f"phoenix14t.pami0.{split}.annotations_only.gzip"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            rows = _load_official(annotation)
            clips = []
            excluded = []
            official_names[split] = set()
            frames = 0
            for row in rows:
                official_name = str(row["name"])
                prefix, separator, clip = official_name.partition("/")
                if not separator or prefix != split or not clip:
                    raise ValueError(
                        f"Official row has inconsistent split {split}: {official_name}"
                    )
                if clip in official_names[split]:
                    raise ValueError(f"Duplicate official clip: {official_name}")
                official_names[split].add(clip)
                video = video_root / split / f"{clip}.mp4"
                target_dir = target_root / split / clip
                if not video.is_file():
                    excluded.append(
                        {"clip": clip, "reason": "missing_video", "path": str(video)}
                    )
                    continue
                contract = _video_contract(video)
                target_indices = _target_indices(target_dir)
                if not target_indices:
                    excluded.append(
                        {
                            "clip": clip,
                            "reason": "missing_soke_target",
                            "path": str(target_dir),
                        }
                    )
                    continue
                if target_indices[0] < 1 or target_indices[-1] > contract["frame_count"]:
                    raise ValueError(
                        f"SOKE target index outside video for {official_name}: "
                        f"video_frames={contract['frame_count']} "
                        f"target_range={target_indices[0]}..{target_indices[-1]}"
                    )
                clip_id = f"phoenix_{clip}"
                # SOKE frame names preserve the original video's one-based
                # frame number and legitimately omit failed fitting frames.
                # Bind to RGB with an explicit one-based -> zero-based map;
                # never compress missing frames into different RGB positions.
                frame_indices = [index - 1 for index in target_indices]
                clips.append(
                    {
                        "dataset": "SOKE",
                        "clip_id": clip_id,
                        "source_clip": clip,
                        "source_group": _source_group(clip),
                        "official_name": official_name,
                        "official_split": split,
                        "phase2_split": PHASE2_SPLIT[split],
                        "signer_id": str(row.get("signer", "")),
                        "gloss": str(row.get("gloss", "")),
                        "text": str(row.get("text", "")),
                        "video": str(video.resolve()),
                        "frame_indices": frame_indices,
                        "target_frame_indices_one_based": target_indices,
                        "target_dir": str(target_dir.resolve()),
                        "target_frame_pattern": "images{one_based_frame:04d}.pkl",
                        "target_provider": "released SOKE phoenix_poses.zip",
                        "source_contract": contract,
                    }
                )
                frames += len(target_indices)

            split_dir = output / split
            split_dir.mkdir()
            selection = {
                # Retain the established WiLoR selection schema so the locked
                # frame-manifest tool can consume this artifact unchanged.
                "schema": COMPATIBLE_SELECTION_SCHEMA,
                "selection_role": SCHEMA,
                "split": PHASE2_SPLIT[split],
                "official_split": split,
                "clips": clips,
                "target_fields_opened": False,
                "target_archive": str(archive),
                "target_archive_sha256": sha256_file(archive),
                "official_annotation": str(annotation.resolve()),
                "official_annotation_sha256": sha256_file(annotation),
            }
            selection_path = split_dir / "selection.json"
            _write_json(selection_path, selection)
            admitted_names[split] = {item["source_clip"] for item in clips}
            split_reports[split] = {
                "official_clips": len(rows),
                "admitted_clips": len(clips),
                "admitted_frames": frames,
                "excluded": excluded,
                "selection": str(selection_path),
                "selection_sha256": sha256_file(selection_path),
            }

        official_overlap = {
            f"{first}_{second}": sorted(official_names[first] & official_names[second])
            for index, first in enumerate(SPLITS)
            for second in SPLITS[index + 1 :]
        }
        admitted_overlap = {
            f"{first}_{second}": sorted(admitted_names[first] & admitted_names[second])
            for index, first in enumerate(SPLITS)
            for second in SPLITS[index + 1 :]
        }
        if any(official_overlap.values()) or any(admitted_overlap.values()):
            raise ValueError("PHOENIX split clip overlap detected")
        report = {
            "schema": SCHEMA,
            "split_authority": "official PHOENIX-2014T annotation gzip name prefix",
            "training_contract": {
                "gradient_split": "train only",
                "checkpoint_selection_split": "dev only",
                "final_evaluation_split": "test only after checkpoint freeze",
            },
            "official_clip_overlap": official_overlap,
            "admitted_clip_overlap": admitted_overlap,
            "splits": split_reports,
            "target_archive": str(archive),
            "target_archive_sha256": sha256_file(archive),
        }
        _write_json(output / "audit_report.json", report)
        return report
    except Exception:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root", type=Path, default=Path("/home/dongvk/datasets/phoenix14T")
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/home/dongvk/datasets/phoenix14T/videos_phoenix/videos"),
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/SignAvatar_SOKE/extracted/soke_phoenix_frame_poses"),
    )
    parser.add_argument(
        "--target-archive",
        type=Path,
        default=Path("data/SignAvatar_SOKE/phoenix_poses.zip"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
