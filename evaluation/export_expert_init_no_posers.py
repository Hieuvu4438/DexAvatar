#!/usr/bin/env python3
"""Export the frozen DexAvatar expert initialization without learned posers.

This ablation composes the same inputs consumed by the original fitting code:
SMPLer-X supplies the SMPL-X body/camera/face parameters and HaMeR replaces
the detected hand poses.  Sapiens is used by the released data parser for the
one-handed active-side decision.  The script performs one SMPL-X forward pass
per frame and deliberately does not load or optimize SignBPoser/SignHPoser.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FITTING_ROOT = REPO_ROOT / "dexavatar_fitting"
SMPLIFYX_ROOT = FITTING_ROOT / "smplifyx"
for path in (str(SMPLIFYX_ROOT), str(FITTING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import data_parser  # noqa: E402
from data_parser import create_dataset  # noqa: E402
from rewrite_body_model import SMPLX  # noqa: E402


@dataclass(frozen=True)
class _ManifestItem:
    clip_id: str
    frame_ids: list[int]


def load_manifest(path: Path) -> list[_ManifestItem]:
    items: list[_ManifestItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            items.append(_ManifestItem(row["clip_id"], [int(value) for value in row["frame_ids"]]))
    return items


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def format_face_block(faces: np.ndarray) -> str:
    values = np.asarray(faces, dtype=np.int64) + 1
    return "\n".join(f"f {first} {second} {third}" for first, second, third in values)


def write_dexavatar_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    decimals: int,
    face_block: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# https://github.com/mikedh/trimesh\n")
            np.savetxt(handle, vertices, fmt=f"v %.{decimals}f %.{decimals}f %.{decimals}f")
            handle.write(face_block)
            handle.write("\n\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def read_simple_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(value.split("/", 1)[0]) - 1 for value in line.split()[1:4]])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


class _CPUUnpickler(pickle.Unpickler):
    """Load historical HaMeR tensors even when their pickle records CUDA."""

    def find_class(self, module: str, name: str):  # type: ignore[no-untyped-def]
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda payload: torch.load(io.BytesIO(payload), map_location="cpu")
        return super().find_class(module, name)


def _cpu_pickle_load(handle: BinaryIO, *args, **kwargs):  # type: ignore[no-untyped-def]
    return _CPUUnpickler(handle, *args, **kwargs).load()


def _read_sign_classes(path: Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens:
            classes[tokens[0]] = tokens[1]
    return classes


def _tensor(value: object, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).reshape(1, -1)


def _sha256_values(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl",
    )
    parser.add_argument("--expert-root", type=Path, default=REPO_ROOT / "outputs/method_hamer")
    parser.add_argument("--frames-root", type=Path, default=REPO_ROOT / "data/frames")
    parser.add_argument("--sign-file", type=Path, default=REPO_ROOT / "data/signs.txt")
    parser.add_argument("--segment-file", type=Path, default=REPO_ROOT / "data/segment.json")
    parser.add_argument(
        "--model-pkl",
        type=Path,
        default=REPO_ROOT / "SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl",
    )
    parser.add_argument(
        "--author-model",
        type=Path,
        default=REPO_ROOT / "data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/method_hamer_expert_init_no_posers",
    )
    parser.add_argument("--clips", nargs="*", help="Optional clip subset for a smoke test")
    parser.add_argument("--decimals", type=int, default=8)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.clips:
        requested = set(args.clips)
        manifest = [item for item in manifest if item.clip_id in requested]
        missing = requested - {item.clip_id for item in manifest}
        if missing:
            raise ValueError(f"Unknown requested clips: {sorted(missing)}")

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {args.output}")
    args.output.mkdir(parents=True)
    incomplete = args.output / ".export_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")

    sign_classes = _read_sign_classes(args.sign_file)
    segments = json.loads(args.segment_file.read_text(encoding="utf-8"))
    author_model = np.load(args.author_model, allow_pickle=True)
    faces = np.asarray(author_model["f"], dtype=np.int64)
    face_block = format_face_block(faces)

    # The released HaMeR pickle stores CUDA tensors.  Redirect only the data
    # parser's pickle loader; ordinary model loading retains its native loader.
    data_parser.pickle.load = _cpu_pickle_load

    dtype = torch.float64
    model = SMPLX(
        model_path=str(args.model_pkl),
        ext="pkl",
        use_face_contour=True,
        flat_hand_mean=True,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        create_body_pose=True,
        dtype=dtype,
    ).eval()

    parameter_keys = (
        "global_orient",
        "body_pose",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "betas",
        "expression",
        "transl",
    )
    rows: list[dict[str, object]] = []
    maximum_roundtrip_error_mm = 0.0

    for clip_index, item in enumerate(manifest, 1):
        clip = item.clip_id
        print(f"[{clip_index:02d}/{len(manifest):02d}] {clip}", flush=True)
        dataset = create_dataset(
            dataset="openpose",
            data_folder=str(args.expert_root / clip),
            indp_sign_segment=segments[clip],
            indp_sign_class=sign_classes[clip],
            img_path=str(args.frames_root / clip),
            use_hands=True,
            use_face=True,
            model_type="smplx",
            joints_to_ign=[-1],
            smplx_init_dir="smplerx/smplx",
        )
        dataset_frame_ids = [int(Path(path).stem.split("_")[-1]) for path in dataset.img_paths]
        if dataset_frame_ids != item.frame_ids:
            raise ValueError(
                f"Expert/parser frame mismatch for {clip}: "
                f"manifest={item.frame_ids}, parser={dataset_frame_ids}"
            )

        hamer_hash = sha256_file(args.expert_root / clip / "hamer/hamer.pkl")
        shape_hash = sha256_file(args.expert_root / clip / "mean_shape_smplx.npy")
        mesh_root = args.output / clip / "smplifyx" / "meshes"

        for frame_index, frame_id in enumerate(item.frame_ids):
            record = dataset[frame_index]
            params = record["smplx_param"]
            model_kwargs = {key: _tensor(params[key], dtype) for key in parameter_keys}
            with torch.no_grad():
                vertices = model(return_verts=True, **model_kwargs).vertices[0].cpu().numpy()

            # Match DexAvatar's saved OBJ convention: 180 degrees about x.
            vertices[:, 1:] *= -1.0
            target = mesh_root / f"low_{frame_id}.obj"
            write_dexavatar_obj(target, vertices, faces, decimals=args.decimals, face_block=face_block)

            roundtrip_vertices, roundtrip_faces = read_simple_obj(target)
            np.testing.assert_array_equal(roundtrip_faces, faces)
            if roundtrip_vertices.shape != (10475, 3) or not np.isfinite(roundtrip_vertices).all():
                raise ValueError(f"Invalid exported OBJ: {target}")
            roundtrip_error_mm = float(np.max(np.abs(roundtrip_vertices - vertices)) * 1000.0)
            maximum_roundtrip_error_mm = max(maximum_roundtrip_error_mm, roundtrip_error_mm)
            smplerx_path = args.expert_root / clip / "smplerx/smplx" / f"low_{frame_id}.pkl"
            source_hash = _sha256_values(sha256_file(smplerx_path), hamer_hash, shape_hash)
            rows.append(
                {
                    "clip_id": clip,
                    "frame_id": frame_id,
                    "obj_relpath": str(target.relative_to(args.output)),
                    "obj_sha256": sha256_file(target),
                    "source_artifact_sha256": source_hash,
                    "max_roundtrip_error_mm": roundtrip_error_mm,
                }
            )

    report = {
        "schema_version": "1.0",
        "method_name": "DexAvatar_HaMeR_expert_init_no_SignBPoser_no_SignHPoser",
        "ablation": {
            "smplerx_body_face_camera": True,
            "sapiens_active_hand_routing": True,
            "hamer_hand_pose_override": True,
            "smplx_forward_only": True,
            "signbposer_loaded": False,
            "signhposer_loaded": False,
            "optimization_steps": 0,
        },
        "format": "dexavatar_trimesh_obj",
        "header": "# https://github.com/mikedh/trimesh",
        "vertex_format": f"v %.{args.decimals}f %.{args.decimals}f %.{args.decimals}f",
        "face_format": "f %d %d %d (one-indexed)",
        "coordinate_convention": "opencv_x_right_y_down_z_forward",
        "length_unit": "meter",
        "vertices_per_mesh": int(faces.max()) + 1,
        "faces_per_mesh": int(len(faces)),
        "clips": len(manifest),
        "frames": len(rows),
        "manifest_sha256": sha256_file(args.manifest),
        "smplx_model_sha256": sha256_file(args.author_model),
        "expert_root": str(args.expert_root),
        "max_roundtrip_error_mm": maximum_roundtrip_error_mm,
        "files": rows,
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    print(json.dumps({key: report[key] for key in ("clips", "frames", "max_roundtrip_error_mm")}, indent=2))


if __name__ == "__main__":
    main()
