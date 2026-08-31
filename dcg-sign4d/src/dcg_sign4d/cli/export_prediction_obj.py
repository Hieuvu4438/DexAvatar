"""Export a DCG trajectory to the exact OBJ layout consumed by the author evaluator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from dcg_sign4d.geometry.smplx_adapter import SMPLXAdapter
from dcg_sign4d.initialization.trajectory_io import load_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--frame-ids", required=True, help="JSON array in source-frame order")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trusted-local-assets", action="store_true")
    args = parser.parse_args()
    if not args.trusted_local_assets:
        raise PermissionError("OBJ export requires explicit trust for the SMPL-X model")
    frame_ids = json.loads(args.frame_ids)
    state, metadata = load_trajectory(args.trajectory)
    if len(frame_ids) != state.valid_mask.shape[1] or not bool(state.valid_mask.all()):
        raise ValueError("frame IDs must match a fully valid trajectory")
    device = torch.device(args.device)
    state = replace(
        state,
        **{
            name: value.to(device)
            for name in state.__dataclass_fields__
            if isinstance((value := getattr(state, name)), torch.Tensor)
        },
    )
    model = SMPLXAdapter(
        args.model,
        expected_sha256=args.expected_model_sha256,
        trusted_model=True,
    ).to(device)
    with torch.inference_mode():
        vertices = model(state).vertices[0].cpu()
    faces = model.model.faces_tensor.long().cpu()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"immutable OBJ export exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        mesh_root = temporary / args.clip_id / "smplifyx" / "meshes"
        mesh_root.mkdir(parents=True)
        for frame_id, frame_vertices in zip(frame_ids, vertices, strict=True):
            path = mesh_root / f"low_{frame_id}.obj"
            with path.open("w", encoding="utf-8") as handle:
                for x, y, z in frame_vertices.tolist():
                    handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                for first, second, third in faces.tolist():
                    handle.write(f"f {first + 1} {second + 1} {third + 1}\n")
        report = {
            "schema_version": "dcg_author_evaluator_obj_export_v1",
            "clip_id": args.clip_id,
            "frames": len(frame_ids),
            "frame_ids": frame_ids,
            "trajectory_sha256": metadata["trajectory_sha256"],
            "smplx_model_sha256": args.expected_model_sha256,
            "coordinate_convention": model.coordinate_convention,
        }
        (temporary / "export_report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", "utf-8"
        )
        (temporary / "OBJ_EXPORT_COMPLETE").write_text("complete\n", "utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({**report, "output": str(destination.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
