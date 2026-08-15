"""Render source-aligned overlays for a frozen SignAvatars audit-candidate sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.render_how2sign_audit import (
    _overlay_mesh,
    _read_frame,
    _side_view,
)
from phase2_refiner.geometry.rotations import axis_angle_to_matrix
from phase2_refiner.geometry.smplx_decode import decode_smplx_sequence
from phase2_refiner.provenance import sha256_file
from phase2_refiner.render import create_smplx_model


def _load_sample(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "signavatars-target-audit-sample-v1":
        raise ValueError("Unsupported SignAvatars target-audit sample schema")
    rows = payload.get("clips")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SignAvatars target-audit sample has no clips")
    if payload.get("source_group_disjoint") is not True:
        raise ValueError("Target-audit sample is not source-group disjoint")
    return rows


def _validate_candidate(clip) -> dict[str, Any]:
    metadata = json.loads(clip.metadata_json)
    contract = metadata.get("target_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Audit clip lacks target_contract: {clip.clip_id}")
    if not str(metadata.get("target_type", "")).endswith("audit_candidate"):
        raise ValueError(
            f"Audit renderer requires an audit-candidate cache: {clip.clip_id}"
        )
    if contract.get("audit_passed") is not False:
        raise ValueError(f"Audit candidate already claims audit_passed: {clip.clip_id}")
    if contract.get("exact_frame_count_match") is not True:
        raise ValueError(f"Audit candidate lacks exact frame binding: {clip.clip_id}")
    if clip.target_axis_angle is None or clip.target_rotation_valid is None:
        raise ValueError(f"Audit candidate lacks target rotations: {clip.clip_id}")
    return metadata


@torch.no_grad()
def _target_vertices(
    model, clip, positions: np.ndarray, device: torch.device
) -> np.ndarray:
    target = torch.from_numpy(clip.target_axis_angle[positions]).float().to(device)
    matrix = axis_angle_to_matrix(target)[None]

    def sequence(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value[positions]).float().to(device)[None]

    vertices, _ = decode_smplx_sequence(
        model,
        matrix,
        torch.from_numpy(clip.betas).float().to(device)[None],
        sequence(clip.global_orient),
        sequence(clip.transl),
        jaw_pose=sequence(clip.jaw_pose),
        leye_pose=sequence(clip.leye_pose),
        reye_pose=sequence(clip.reye_pose),
        expression=sequence(clip.expression),
    )
    return vertices[0].cpu().numpy()


def _draw_mapped_tracks(frame: np.ndarray, clip, position: int) -> np.ndarray:
    height, width = frame.shape[:2]
    points = clip.keypoints_2d[position].copy()
    points[:, 0] *= width
    points[:, 1] *= height
    for index in range(51):
        if not clip.track_valid[position, index]:
            continue
        if index < 21:
            color = (0, 215, 255)
        elif index < 36:
            color = (255, 80, 220)
        else:
            color = (255, 170, 40)
        cv2.circle(
            frame,
            tuple(np.rint(points[index]).astype(int)),
            2,
            color,
            -1,
            cv2.LINE_AA,
        )
    return frame


def _source_reference(clip, position: int) -> tuple[Path, int]:
    reference = str(clip.source_paths[position])
    if reference.count("#frame=") != 1:
        raise ValueError(f"Invalid exact source reference: {reference}")
    video, frame = reference.rsplit("#frame=", 1)
    return Path(video), int(frame)


def render_sample(
    sample: Path,
    output: Path,
    model_folder: Path,
    device: torch.device,
    frames_per_clip: int,
    clips_per_sheet: int,
    limit: int | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty audit output: {output}")
    rows = _load_sample(sample)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be positive")
    output.mkdir(parents=True, exist_ok=True)
    clip_dir = output / "clips"
    sheet_dir = output / "sheets"
    clip_dir.mkdir()
    sheet_dir.mkdir()
    model = create_smplx_model(model_folder, device)
    rendered = []
    target_hashes = set()
    for row_index, row in enumerate(rows):
        clip = load_cache_clip(Path(row["cache_path"]))
        metadata = _validate_candidate(clip)
        positions = np.linspace(
            0, len(clip.frame_names) - 1, frames_per_clip, dtype=np.int64
        )
        vertices = _target_vertices(model, clip, positions, device)
        teacher_path = Path(metadata["teacher_path"])
        with np.load(teacher_path, allow_pickle=False) as teacher:
            bboxes = teacher["bboxes"][positions]
        cells = []
        for local_index, position in enumerate(positions):
            video, frame_index = _source_reference(clip, int(position))
            frame = _read_frame(video, frame_index)
            frame = _overlay_mesh(frame, vertices[local_index], bboxes[local_index])
            frame = _draw_mapped_tracks(frame, clip, int(position))
            frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                f"released target pose frame {frame_index}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.extend(
                (
                    frame,
                    _side_view(
                        vertices[local_index], (180, 180), label="target side view"
                    ),
                )
            )
        strip = np.concatenate(cells, axis=1)
        label = np.full((34, strip.shape[1], 3), 25, dtype=np.uint8)
        cv2.putText(
            label,
            f"{row_index + 1:03d} {clip.clip_id} signer={row['signer']}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        image = np.concatenate((label, strip), axis=0)
        destination = clip_dir / f"{row_index + 1:03d}_{clip.clip_id}.jpg"
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError(f"Failed to write audit image: {destination}")
        rendered.append(destination)
        target_hashes.update(metadata["target_contract"]["source_sha256"])
        print(f"[signavatars-audit] {row_index + 1}/{len(rows)} {clip.clip_id}")

    sheets = []
    for start in range(0, len(rendered), clips_per_sheet):
        images = [
            cv2.imread(str(path)) for path in rendered[start : start + clips_per_sheet]
        ]
        sheet = np.concatenate(images, axis=0)
        destination = sheet_dir / f"sheet_{start // clips_per_sheet + 1:02d}.jpg"
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise IOError(f"Failed to write audit sheet: {destination}")
        sheets.append(destination)
    report = {
        "schema": "signavatars-target-audit-render-v1",
        "sample": str(sample.resolve()),
        "sample_manifest_sha256": sha256_file(sample),
        "clips": len(rows),
        "frames_per_clip": frames_per_clip,
        "target_source_sha256": sorted(target_hashes),
        "legend": {
            "green": "released SMPL-X target pose decoded in common geometry",
            "yellow": "independent How2Sign body observations",
            "magenta_blue": "independent left/right hand observations",
            "side_view": "target geometry colored by depth",
        },
        "clip_images": [str(path.resolve()) for path in rendered],
        "sheets": [str(path.resolve()) for path in sheets],
    }
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--clips-per-sheet", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = render_sample(
        args.sample.resolve(),
        args.output.resolve(),
        args.model_folder.resolve(),
        torch.device(args.device),
        args.frames_per_clip,
        args.clips_per_sheet,
        args.limit,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "clip_images"},
            indent=2,
            sort_keys=True,
        )
    )
