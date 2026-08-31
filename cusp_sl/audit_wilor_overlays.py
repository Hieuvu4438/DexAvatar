"""Create original-image WiLoR camera/chirality overlays and numeric audits."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from cusp_sl.evaluate_wilor_direct_development import select_detection
from cusp_sl.wilor_artifact import validate_wilor_raw_v3


def project_full_image(
    points: np.ndarray,
    camera_translation: np.ndarray,
    focal_length_px: float,
    image_size_wh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = np.asarray(points, dtype=np.float32) + np.asarray(
        camera_translation, dtype=np.float32
    ).reshape(1, 3)
    depth = points_camera[:, 2]
    center = np.asarray(image_size_wh, dtype=np.float32).reshape(2) / 2.0
    pixels = points_camera[:, :2] / depth[:, None] * float(focal_length_px)
    pixels += center
    return pixels, depth


def read_manifest_image(record: dict) -> np.ndarray:
    if "image_path" in record:
        image = cv2.imread(str(record["image_path"]))
    else:
        capture = cv2.VideoCapture(str(record["video_path"]))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_number"]))
            ok, image = capture.read()
            if not ok:
                image = None
        finally:
            capture.release()
    if image is None:
        raise RuntimeError(f"Could not decode manifest record {record['image_key']}")
    expected = (int(record["expected_width"]), int(record["expected_height"]))
    if (image.shape[1], image.shape[0]) != expected:
        raise ValueError(f"Image size mismatch for {record['image_key']}")
    return image


def draw_hand(image: np.ndarray, hand: dict, *, is_right: bool) -> dict[str, float]:
    points = np.asarray(hand["pred_keypoints_3d"], dtype=np.float32).copy()
    wrist_rotation = np.asarray(
        hand["pred_mano_global_orient_rotmat"], dtype=np.float32
    ).reshape(3, 3)
    if not is_right:
        points[:, 0] *= -1.0
        signs = np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
        wrist_rotation = wrist_rotation * signs[:, None] * signs[None, :]
    camera = np.asarray(hand["cam_t"], dtype=np.float32)
    focal = float(hand["focal_length_px"])
    size = np.asarray(hand["image_size_wh"], dtype=np.float32)
    pixels, depth = project_full_image(points, camera, focal, size)
    color = (60, 220, 60) if is_right else (255, 160, 50)
    height, width = image.shape[:2]
    in_frame = (
        (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    for pixel, visible in zip(pixels, in_frame):
        if visible:
            cv2.circle(image, tuple(np.rint(pixel).astype(int)), 3, color, -1)

    # Draw the three wrist-frame axes from the same global orientation consumed
    # by geometric A1.  This makes camera-basis and left-mirror mistakes visible.
    root = points[0]
    axis_points = np.concatenate(
        [root[None], root[None] + 0.04 * wrist_rotation.T], axis=0
    )
    axis_pixels, axis_depth = project_full_image(axis_points, camera, focal, size)
    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    if np.all(axis_depth > 0):
        origin = tuple(np.rint(axis_pixels[0]).astype(int))
        for endpoint, axis_color in zip(axis_pixels[1:], axis_colors):
            cv2.line(
                image,
                origin,
                tuple(np.rint(endpoint).astype(int)),
                axis_color,
                2,
            )
    box = np.rint(np.asarray(hand["detector_box_xyxy"])).astype(int)
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 1)
    return {
        "positive_depth_fraction": float((depth > 0).mean()),
        "in_frame_fraction": float(in_frame.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--wilor-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    args.output.mkdir(parents=True)

    manifest = json.loads(args.frame_manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    with args.wilor_pickle.open("rb") as handle:
        wilor = pickle.load(handle, encoding="latin1")
    images, wilor_meta = validate_wilor_raw_v3(
        wilor, expected_frame_manifest=args.frame_manifest
    )

    indices = np.unique(
        np.linspace(0, len(records) - 1, min(args.samples, len(records)), dtype=int)
    )
    panels = []
    hand_metrics = []
    detections = {"left": 0, "right": 0}
    for index in indices:
        record = records[int(index)]
        image = read_manifest_image(record)
        raw_hands = images[record["image_key"]].get("hands", [])
        for side, is_right in (("left", False), ("right", True)):
            hand = select_detection(raw_hands, is_right=is_right)
            if hand is None:
                continue
            detections[side] += 1
            hand_metrics.append(draw_hand(image, hand, is_right=is_right))
        cv2.putText(
            image,
            str(record["image_key"]),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        output = args.output / f"{int(index):05d}_{Path(record['image_key']).stem}.jpg"
        cv2.imwrite(str(output), image)
        scale = 320.0 / image.shape[1]
        panels.append(cv2.resize(image, (320, max(1, round(image.shape[0] * scale)))))

    panel_height = max(panel.shape[0] for panel in panels)
    normalized = [
        cv2.copyMakeBorder(panel, 0, panel_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for panel in panels
    ]
    columns = 4
    blank = np.zeros_like(normalized[0])
    rows = []
    for start in range(0, len(normalized), columns):
        row = normalized[start : start + columns]
        rows.append(np.concatenate(row + [blank] * (columns - len(row)), axis=1))
    cv2.imwrite(str(args.output / "montage.jpg"), np.concatenate(rows, axis=0))
    summary = {
        "role": "original_image_camera_chirality_audit",
        "frames_in_manifest": len(records),
        "sampled_frames": len(indices),
        "left_sample_detections": detections["left"],
        "right_sample_detections": detections["right"],
        "frame_manifest_sha256": wilor_meta["frame_manifest_sha256"],
        "wilor_checkpoint_sha256": wilor_meta["wilor_checkpoint_sha256"],
        "detector_checkpoint_sha256": wilor_meta[
            "detector_checkpoint_sha256"
        ],
        "mean_positive_depth_fraction": (
            float(np.mean([item["positive_depth_fraction"] for item in hand_metrics]))
            if hand_metrics
            else None
        ),
        "mean_in_frame_fraction": (
            float(np.mean([item["in_frame_fraction"] for item in hand_metrics]))
            if hand_metrics
            else None
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
