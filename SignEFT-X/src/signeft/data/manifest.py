from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np

from signeft.io.obj import load_obj, validate_mesh
from signeft.io_utils import array_sha256, atomic_write_json, atomic_write_text, sha256_file


STATE_KEYS = (
    "betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
    "jaw_pose", "leye_pose", "reye_pose", "expression", "transl",
)


@dataclass(frozen=True)
class ManifestRecord:
    record_id: str
    sign_id: str
    sign_class: str
    frame_index: int
    source_frame_id: int
    rgb_path: str
    a3f_state_path: str
    a3f_obj_path: str
    width: int
    height: int
    sha256_rgb: str
    sha256_a3f_state: str
    sha256_a3f_obj: str


def read_manifest(path: Path) -> list[ManifestRecord]:
    return [
        ManifestRecord(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _image_wh(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode RGB image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_manifest(
    protocol_path: Path,
    rgb_root: Path,
    a3f_run_root: Path,
    baseline_state_root: Path,
    canonical_model: Path,
    output: Path,
    *,
    expected_faces_sha256: str,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    camera_path = a3f_run_root / "camera" / "C1.npz"
    with np.load(camera_path, allow_pickle=False) as camera_archive:
        K = np.asarray(camera_archive["K_full"], dtype=np.float32)
    model_sha = sha256_file(canonical_model)
    records: list[ManifestRecord] = []
    seen: set[str] = set()
    sign_summaries = []
    for sign_item in protocol["items"]:
        sign = sign_item["sign_id"]
        sequence_path = a3f_run_root / "fit_sequences" / "clips" / sign / "mesh_parametric_final.npz"
        with np.load(sequence_path, allow_pickle=False) as archive:
            sequence = {key: np.asarray(archive[key]).copy() for key in archive.files}
        faces = np.asarray(sequence["faces"], dtype=np.int64)
        if array_sha256(faces) != expected_faces_sha256:
            raise RuntimeError(f"A3f face hash mismatch: {sequence_path}")
        protocol_frames = sign_item["frames"]
        if len(protocol_frames) != len(sequence["frame_ids"]):
            raise RuntimeError(f"A3f/protocol frame count mismatch: {sign}")
        for frame in protocol_frames:
            index = int(frame["sequence_index"])
            frame_id = int(frame["source_frame_id"])
            if int(sequence["frame_ids"][index]) != frame_id:
                raise RuntimeError(f"A3f frame ID mismatch: {sign}/{frame_id}")
            record_id = f"{sign}/{frame_id}"
            if record_id in seen:
                raise RuntimeError(f"duplicate manifest record: {record_id}")
            seen.add(record_id)
            rgb = (rgb_root / sign / f"low_{frame_id}.png").resolve()
            if not rgb.is_file():
                candidates = sorted((rgb_root / sign).glob(f"*{frame_id}*"))
                if len(candidates) != 1:
                    raise FileNotFoundError(rgb)
                rgb = candidates[0].resolve()
            baseline_obj = (
                a3f_run_root / "eval_layout" / sign / "smplifyx" / "meshes" / f"{index:03d}.obj"
            ).resolve()
            obj_vertices, obj_faces = load_obj(baseline_obj)
            validate_mesh(obj_vertices, obj_faces)
            if not np.array_equal(obj_faces, faces):
                raise RuntimeError(f"A3f OBJ topology mismatch: {baseline_obj}")
            vertices = np.asarray(sequence["mesh_parametric"][index], dtype=np.float32)
            if np.max(np.abs(vertices.astype(np.float64) - obj_vertices)) > 1.1e-8:
                raise RuntimeError(f"A3f state/OBJ coordinate mismatch: {record_id}")
            state_path = (baseline_state_root / sign / f"{frame_id:06d}.npz").resolve()
            arrays = {
                key: np.asarray(sequence[key][index:index + 1], dtype=np.float32)
                for key in STATE_KEYS
            }
            arrays.update({
                "K": K,
                "vertices": vertices,
                "faces_sha256": np.asarray(expected_faces_sha256),
                "coord_frame": np.asarray("evaluator_camera"),
                "unit": np.asarray("meter"),
                "model_sha256": np.asarray(model_sha),
                "source_sequence_sha256": np.asarray(sha256_file(sequence_path)),
            })
            if not state_path.is_file():
                _atomic_npz(state_path, arrays)
            else:
                with np.load(state_path, allow_pickle=False) as existing:
                    if set(existing.files) != set(arrays):
                        raise RuntimeError(f"baseline state contract mismatch: {state_path}")
            width, height = _image_wh(rgb)
            records.append(ManifestRecord(
                record_id=record_id,
                sign_id=sign,
                sign_class=str(sign_item["sign_class"]),
                frame_index=index,
                source_frame_id=frame_id,
                rgb_path=str(rgb),
                a3f_state_path=str(state_path),
                a3f_obj_path=str(baseline_obj),
                width=width,
                height=height,
                sha256_rgb=sha256_file(rgb),
                sha256_a3f_state=sha256_file(state_path),
                sha256_a3f_obj=sha256_file(baseline_obj),
            ))
        sign_summaries.append({"sign": sign, "frames": len(protocol_frames)})
    if len(records) != int(protocol["frame_count"]):
        raise RuntimeError("manifest frame count differs from protocol lock")
    content = "\n".join(json.dumps(asdict(record), sort_keys=True) for record in records) + "\n"
    atomic_write_text(output, content)
    report = {
        "schema_version": "signeft.manifest-summary.v1",
        "status": "ok",
        "signs": len(sign_summaries),
        "frames": len(records),
        "sha256": sha256_file(output),
        "protocol_sha256": sha256_file(protocol_path),
        "baseline_state_root": str(baseline_state_root.resolve()),
        "items": sign_summaries,
    }
    atomic_write_json(output.with_name("summary.json"), report)
    return report

