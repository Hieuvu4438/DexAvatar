from __future__ import annotations

import json
import pickle
from pathlib import Path

from ..data.manifest import ClipManifest, write_manifest
from ..utils.hashing import sha256_file


def _frame_number(path_or_name: str | Path) -> int:
    stem = Path(path_or_name).stem
    return int(stem.split("_")[-1])


def run(
    segments_path: str,
    frames_root: str,
    body_root: str,
    wilor_root: str,
    gt_root: str,
    output: str,
    split: str = "test",
) -> dict[str, object]:
    """Freeze frames after the legacy segment without reading any GT vertex value."""
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    frames = Path(frames_root)
    body = Path(body_root)
    wilor = Path(wilor_root)
    ground_truth = Path(gt_root)
    rows: list[ClipManifest] = []
    for source_clip_id in sorted(segments):
        _, central_end = map(int, segments[source_clip_id])
        images = {_frame_number(path): path for path in (frames / source_clip_id).glob("low_*.png")}
        body_ids = {
            _frame_number(path)
            for path in (body / source_clip_id / "smplerx/smplx").glob("low_*.pkl")
        }
        gt_ids = {
            int(path.stem) // 2
            for path in (ground_truth / source_clip_id).glob("*.obj")
            if int(path.stem) % 2 == 0
        }
        wilor_path = wilor / source_clip_id / "wilor/wilor.pkl"
        with wilor_path.open("rb") as handle:
            wilor_payload = pickle.load(handle, encoding="latin1")
        wilor_ids = {_frame_number(name) for name in wilor_payload.get("images", {})}
        selected = sorted(
            frame_id
            for frame_id in images.keys() & body_ids & gt_ids & wilor_ids
            if frame_id > central_end
        )
        if not selected:
            continue
        rows.append(
            ClipManifest(
                dataset="sgnify_extended_post",
                clip_id=source_clip_id,
                split=split,  # type: ignore[arg-type]
                fps=15.0,
                frame_ids=selected,
                image_relpaths=[
                    str(images[frame_id].relative_to(frames.parent)) for frame_id in selected
                ],
                is_contiguous=False,
                gt_relpath=f"data/smplx_gt/{source_clip_id}",
                allowed_for_calibration=split == "calibration",
                allowed_for_hparam_selection=split == "development",
                allowed_for_final_reporting=split == "test",
            )
        )
    write_manifest(rows, output)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "selection": "strictly_after_frozen_central_segment",
        "ground_truth_values_read": False,
        "clips": len(rows),
        "frames": sum(len(row.frame_ids) for row in rows),
        "segments_sha256": sha256_file(segments_path),
        "manifest_sha256": sha256_file(output),
    }
    metadata_path = Path(output).with_suffix(Path(output).suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
