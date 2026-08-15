"""Render reproducible source/mesh/track evidence for How2Sign target review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file
from phase2_refiner.render import create_smplx_model


BODY_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
HAND_EDGES = tuple(
    (base + start + offset, base + start + offset + 1)
    for base in (91, 112)
    for start in (0, 5, 9, 13, 17)
    for offset in range(3)
)


def _read_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"Cannot open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise IOError(f"Cannot decode frame {frame_index} from {video}")
    return frame


def _draw_tracks(
    image: np.ndarray, points: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    height, width = image.shape[:2]
    pixels = points.copy()
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    for start, end in BODY_EDGES + HAND_EDGES:
        if scores[start] >= 0.20 and scores[end] >= 0.20:
            first = tuple(np.rint(pixels[start]).astype(int))
            second = tuple(np.rint(pixels[end]).astype(int))
            color = (0, 215, 255) if start < 91 else (255, 80, 220)
            cv2.line(image, first, second, color, 2, cv2.LINE_AA)
    for index in list(range(17)) + list(range(91, 133)):
        if scores[index] >= 0.20:
            color = (0, 215, 255) if index < 91 else (255, 80, 220)
            cv2.circle(
                image,
                tuple(np.rint(pixels[index]).astype(int)),
                2,
                color,
                -1,
                cv2.LINE_AA,
            )
    return image


def _project_vertices(vertices: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x, y, width, height = bbox.astype(np.float64)
    focal_x = 5000.0 / 192.0 * width
    focal_y = 5000.0 / 256.0 * height
    principal_x = x + width * 0.5
    principal_y = y + height * 0.5
    z = np.maximum(vertices[:, 2], 1e-5)
    return np.stack(
        (
            vertices[:, 0] / z * focal_x + principal_x,
            vertices[:, 1] / z * focal_y + principal_y,
        ),
        axis=-1,
    )


def _overlay_mesh(
    frame: np.ndarray, vertices: np.ndarray, bbox: np.ndarray
) -> np.ndarray:
    overlay = frame.copy()
    points = _project_vertices(vertices, bbox)[::4]
    height, width = frame.shape[:2]
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    for point in np.rint(points[valid]).astype(int):
        cv2.circle(overlay, tuple(point), 1, (40, 255, 40), -1)
    return cv2.addWeighted(frame, 0.58, overlay, 0.42, 0.0)


def _side_view(
    vertices: np.ndarray,
    size: tuple[int, int],
    label: str = "teacher side view",
) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    centered = vertices - np.median(vertices, axis=0, keepdims=True)
    vertical_span = float(
        np.quantile(centered[:, 1], 0.99) - np.quantile(centered[:, 1], 0.01)
    )
    scale = 0.82 * height / max(vertical_span, 1e-6)
    x = centered[:, 2] * scale + width * 0.5
    y = centered[:, 1] * scale + height * 0.5
    depth = centered[:, 0]
    order = np.argsort(depth)
    low, high = np.quantile(depth, (0.02, 0.98))
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    for index in order[::3]:
        point = (int(round(x[index])), int(round(y[index])))
        if 0 <= point[0] < width and 0 <= point[1] < height:
            color = (
                int(220 - 150 * normalized[index]),
                70,
                int(70 + 170 * normalized[index]),
            )
            cv2.circle(canvas, point, 1, color, -1)
    cv2.putText(
        canvas,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


@torch.no_grad()
def _vertices(model, teacher: dict[str, np.ndarray], positions: np.ndarray, device):
    def tensor(name: str) -> torch.Tensor:
        return torch.from_numpy(teacher[name][positions]).float().to(device)

    zeros = torch.zeros(len(positions), 3, device=device)
    output = model(
        betas=tensor("betas"),
        global_orient=tensor("global_orient"),
        body_pose=tensor("body_pose"),
        left_hand_pose=tensor("left_hand_pose"),
        right_hand_pose=tensor("right_hand_pose"),
        jaw_pose=tensor("jaw_pose"),
        leye_pose=zeros,
        reye_pose=zeros,
        expression=tensor("expression"),
        transl=tensor("transl"),
        return_verts=True,
    )
    return output.vertices.detach().cpu().numpy()


def render_queue(
    queue: Path,
    output: Path,
    model_folder: Path,
    device: torch.device,
    frames_per_clip: int,
    clips_per_sheet: int,
    limit: int | None = None,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty audit output: {output}")
    with queue.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Audit queue is empty")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    output.mkdir(parents=True, exist_ok=True)
    clip_dir = output / "clips"
    sheet_dir = output / "sheets"
    clip_dir.mkdir()
    sheet_dir.mkdir()
    model = create_smplx_model(model_folder, device)
    rendered = []
    for row_index, row in enumerate(rows):
        clip = load_cache_clip(Path(row["cache_path"]))
        metadata = json.loads(clip.metadata_json)
        teacher_path = Path(metadata["teacher_path"])
        with np.load(teacher_path, allow_pickle=False) as data:
            teacher = {key: data[key] for key in data.files}
        positions = np.linspace(
            0, len(teacher["sample_indices"]) - 1, frames_per_clip, dtype=np.int64
        )
        vertices = _vertices(model, teacher, positions, device)
        cells = []
        for local_index, position in enumerate(positions):
            frame_index = int(teacher["sample_indices"][position])
            frame = _read_frame(Path(row["video_path"]), frame_index)
            frame = _overlay_mesh(
                frame, vertices[local_index], teacher["bboxes"][position]
            )
            frame = _draw_tracks(
                frame,
                teacher["keypoints_2d"][position],
                teacher["keypoint_scores"][position],
            )
            frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                f"source+teacher frame {frame_index}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"source+teacher frame {frame_index}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.extend((frame, _side_view(vertices[local_index], (180, 180))))
        strip = np.concatenate(cells, axis=1)
        label = np.full((34, strip.shape[1], 3), 25, dtype=np.uint8)
        cv2.putText(
            label,
            f"{row_index + 1:03d} {row['clip_id']}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        image = np.concatenate((label, strip), axis=0)
        clip_path = clip_dir / f"{row_index + 1:03d}_{row['clip_id']}.jpg"
        cv2.imwrite(str(clip_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rendered.append(clip_path)
        print(
            f"[audit-render] {row_index + 1}/{len(rows)} {row['clip_id']}", flush=True
        )

    sheets = []
    for start in range(0, len(rendered), clips_per_sheet):
        images = [
            cv2.imread(str(path)) for path in rendered[start : start + clips_per_sheet]
        ]
        sheet = np.concatenate(images, axis=0)
        sheet_path = sheet_dir / f"sheet_{start // clips_per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        sheets.append(sheet_path)
    report = {
        "schema_version": 1,
        "queue": str(queue.resolve()),
        "queue_sha256": sha256_file(queue),
        "clips": len(rows),
        "frames_per_clip": frames_per_clip,
        "legend": {
            "green": "projected SMPL-X teacher vertices",
            "yellow": "independent How2Sign body track",
            "magenta": "independent How2Sign hand tracks",
            "side_view": "orthographic teacher geometry colored by depth",
        },
        "clip_images": [str(path.resolve()) for path in rendered],
        "sheets": [str(path.resolve()) for path in sheets],
    }
    manifest = output / "manifest.json"
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
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
    args = parser.parse_args()
    report = render_queue(
        args.queue.resolve(),
        args.output.resolve(),
        args.model_folder.resolve(),
        torch.device(args.device),
        args.frames_per_clip,
        args.clips_per_sheet,
        args.limit,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"clip_images", "sheets"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
