from __future__ import annotations

from pathlib import Path

import numpy as np

from signpccx.data.manifest import read_jsonl
from signpccx.io import atomic_write_json, sha256_file


def weighted_huber_line(
    coordinate: np.ndarray,
    pixels: np.ndarray,
    weights: np.ndarray,
    delta_px: float = 8.0,
    iterations: int = 12,
) -> tuple[float, float]:
    x = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    y = np.asarray(pixels, dtype=np.float64).reshape(-1)
    base_weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(base_weight) & (base_weight > 0)
    x, y, base_weight = x[valid], y[valid], base_weight[valid]
    if len(x) < 2:
        raise ValueError("At least two weighted camera observations are required")
    design = np.stack((x, np.ones_like(x)), axis=1)
    robust = np.ones_like(base_weight)
    solution = np.zeros(2, dtype=np.float64)
    for _ in range(iterations):
        weight = np.sqrt(base_weight * robust)
        solution = np.linalg.lstsq(design * weight[:, None], y * weight, rcond=None)[0]
        residual = design @ solution - y
        absolute = np.abs(residual)
        robust = np.where(absolute <= delta_px, 1.0, delta_px / np.maximum(absolute, 1e-12))
    return float(solution[0]), float(solution[1])


def calibrate_shared_camera(
    manifest_root: Path,
    h4wpp_frame_cache: Path,
    output_npz: Path,
    *,
    huber_delta_px: float = 8.0,
) -> dict[str, object]:
    xyz_items, uv_items, weight_items, frame_ids = [], [], [], []
    cache_hashes = []
    for manifest_path in sorted(manifest_root.glob("*.jsonl")):
        for record in read_jsonl(manifest_path):
            cache_path = h4wpp_frame_cache / "clips" / record.sign / f"{record.source_frame_id:06d}.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                xyz = np.asarray(cache["init_anchor_cam"], dtype=np.float64)
                uv_crop = np.asarray(cache["anchor_uv_observed"], dtype=np.float64)
                confidence = np.asarray(cache["anchor_uv_confidence"], dtype=np.float64)
                valid = np.asarray(cache["anchor_valid"], dtype=bool)
                affine = np.asarray(cache["crop_to_image"], dtype=np.float64)
            homogeneous = np.concatenate((uv_crop, np.ones((len(uv_crop), 1))), axis=1)
            uv_full = homogeneous @ affine.T
            xyz_items.append(xyz)
            uv_items.append(uv_full)
            weight_items.append(confidence * valid)
            frame_ids.append((record.sign, record.source_frame_id))
            cache_hashes.append(sha256_file(cache_path))
    xyz = np.concatenate(xyz_items)
    uv = np.concatenate(uv_items)
    weights = np.concatenate(weight_items)
    depth_valid = np.abs(xyz[:, 2]) > 1e-6
    weights = weights * depth_valid
    fx, cx = weighted_huber_line(xyz[:, 0] / xyz[:, 2], uv[:, 0], weights, huber_delta_px)
    fy, cy = weighted_huber_line(xyz[:, 1] / xyz[:, 2], uv[:, 1], weights, huber_delta_px)
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    predicted = np.stack((fx * xyz[:, 0] / xyz[:, 2] + cx, fy * xyz[:, 1] / xyz[:, 2] + cy), axis=1)
    residual = np.linalg.norm(predicted - uv, axis=1)
    valid = weights > 0
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        K_full=intrinsics.astype(np.float32),
        weighted_residual_px=residual[valid].astype(np.float32),
        frame_sign=np.asarray([item[0] for item in frame_ids]),
        frame_id=np.asarray([item[1] for item in frame_ids], dtype=np.int64),
    )
    digest = __import__("hashlib").sha256()
    for value in cache_hashes:
        digest.update(value.encode("ascii"))
    report = {
        "schema_version": "signpccx.shared-camera.v1",
        "scope": "camera",
        "frames": len(frame_ids),
        "observations": int(valid.sum()),
        "K_full": intrinsics.tolist(),
        "huber_delta_px": huber_delta_px,
        "weighted_mean_residual_px": float(np.average(residual[valid], weights=weights[valid])),
        "median_residual_px": float(np.median(residual[valid])),
        "p95_residual_px": float(np.quantile(residual[valid], 0.95)),
        "objective_uses_ground_truth": False,
        "h4wpp_cache_set_sha256": digest.hexdigest(),
        "sha256": sha256_file(output_npz),
    }
    atomic_write_json(output_npz.with_suffix(".json"), report)
    return report
