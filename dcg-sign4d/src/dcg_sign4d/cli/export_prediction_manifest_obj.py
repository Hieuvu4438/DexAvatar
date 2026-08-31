"""Resumably export selected DCG predictions to the author evaluator OBJ layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from dcg_sign4d.data.manifest import load_manifest
from dcg_sign4d.geometry.smplx_adapter import SMPLXAdapter
from dcg_sign4d.inference.artifacts import validate_prediction_artifact
from dcg_sign4d.initialization.trajectory_io import load_trajectory
from dcg_sign4d.utils.hashing import file_sha256


def _write_obj(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in vertices.tolist():
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for first, second, third in faces.tolist():
            handle.write(f"f {first + 1} {second + 1} {third + 1}\n")


def _validate_completed_clip(root: Path, frame_ids: tuple[int, ...]) -> None:
    if not (root / "OBJ_EXPORT_COMPLETE").is_file():
        raise ValueError(f"incomplete existing OBJ export: {root}")
    mesh_root = root / "smplifyx" / "meshes"
    observed = {path.name for path in mesh_root.glob("*.obj")}
    expected = {f"low_{frame_id}.obj" for frame_id in frame_ids}
    if observed != expected:
        raise ValueError(f"existing OBJ coverage mismatch: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trusted-local-assets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.trusted_local_assets:
        raise PermissionError("OBJ export requires explicit trust for the SMPL-X model")

    predictions = Path(args.predictions)
    manifest_path = Path(args.manifest)
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"immutable OBJ export exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    items = load_manifest(manifest_path, require_existing_video=False)
    device = torch.device(args.device)
    model = SMPLXAdapter(
        args.model,
        expected_sha256=args.expected_model_sha256,
        trusted_model=True,
    ).to(device)
    faces = model.model.faces_tensor.long().cpu()
    completed: list[str] = []
    skipped: list[str] = []

    for item in items:
        if item.frame_mapping is None:
            frame_ids = tuple(range(item.frame_count))
        else:
            frame_ids = item.frame_mapping
        destination = output / item.clip_id
        if destination.exists():
            if not args.resume:
                raise FileExistsError(f"immutable OBJ clip exists: {destination}")
            _validate_completed_clip(destination, frame_ids)
            completed.append(item.clip_id)
            skipped.append(item.clip_id)
            continue

        prediction_root = predictions / item.clip_id
        validation = validate_prediction_artifact(prediction_root)
        selected = int(
            (prediction_root / "selected_hypothesis.txt").read_text(encoding="utf-8")
        )
        state, trajectory_metadata = load_trajectory(
            prediction_root / f"hypothesis_{selected:03d}"
        )
        if state.valid_mask.shape != (1, len(frame_ids)) or not bool(state.valid_mask.all()):
            raise ValueError(f"{item.clip_id}: trajectory/frame manifest mismatch")
        state = replace(
            state,
            **{
                name: value.to(device)
                for name in state.__dataclass_fields__
                if isinstance((value := getattr(state, name)), torch.Tensor)
            },
        ).validate()
        with torch.inference_mode():
            vertices = model(state).vertices[0].cpu()

        temporary = Path(tempfile.mkdtemp(prefix=f".{item.clip_id}.", dir=output))
        try:
            mesh_root = temporary / "smplifyx" / "meshes"
            mesh_root.mkdir(parents=True)
            for frame_id, frame_vertices in zip(frame_ids, vertices, strict=True):
                _write_obj(mesh_root / f"low_{frame_id}.obj", frame_vertices, faces)
            report = {
                "schema_version": "dcg_author_evaluator_obj_clip_export_v1",
                "clip_id": item.clip_id,
                "frames": len(frame_ids),
                "frame_ids": list(frame_ids),
                "selected_hypothesis": selected,
                "prediction_validation": validation,
                "trajectory_sha256": trajectory_metadata["trajectory_sha256"],
                "smplx_model_sha256": args.expected_model_sha256,
                "coordinate_convention": model.coordinate_convention,
            }
            (temporary / "export_report.json").write_text(
                json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            (temporary / "OBJ_EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        completed.append(item.clip_id)
        print(
            json.dumps(
                {
                    "clip_id": item.clip_id,
                    "completed": len(completed),
                    "total": len(items),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    total_frames = sum(item.effective_frame_count for item in items)
    report = {
        "schema_version": "dcg_author_evaluator_obj_manifest_export_v1",
        "clips": len(items),
        "frames": total_frames,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "prediction_root": str(predictions.resolve()),
        "smplx_model_sha256": args.expected_model_sha256,
        "completed_clips": completed,
        "skipped_valid_clips": skipped,
    }
    (output / "export_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output / "OBJ_EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
