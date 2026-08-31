from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dcg_sign4d.contact.labels import HysteresisPseudoLabeler
from dcg_sign4d.contact.ontology import EventState
from dcg_sign4d.geometry.contact_geometry import ContactGeometry
from dcg_sign4d.geometry.mesh import vertex_normals
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile explicitly development-only SGNify contact geometry"
    )
    parser.add_argument("--initialization-root", required=True)
    parser.add_argument("--patch-map", required=True)
    parser.add_argument("--model", required=True, help="SMPL-X NPZ providing the face topology")
    parser.add_argument("--enter-threshold-m", required=True, type=float)
    parser.add_argument("--exit-threshold-m", required=True, type=float)
    parser.add_argument("--n-enter", required=True, type=int)
    parser.add_argument("--n-exit", required=True, type=int)
    parser.add_argument("--uncertainty-margin-m", required=True, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.development_only:
        raise PermissionError("provisional patch/contact compilation requires --development-only")
    source = Path(args.initialization_root)
    if not (source / "CONVERSION_COMPLETE").is_file():
        raise ValueError("initialization root has no CONVERSION_COMPLETE marker")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable contact artifact exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".compilation_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")

    patch_path = Path(args.patch_map)
    patch_map = PatchMap.load(patch_path)
    if not patch_map.development_only:
        raise ValueError("this command accepts only a development patch map")
    model_path = Path(args.model)
    with np.load(model_path, allow_pickle=False) as model:
        faces = torch.from_numpy(np.asarray(model["f"], dtype=np.int64)).long()
    device = torch.device(args.device)
    faces = faces.to(device)
    labeler = HysteresisPseudoLabeler(
        enter_threshold=args.enter_threshold_m,
        exit_threshold=args.exit_threshold_m,
        n_enter=args.n_enter,
        n_exit=args.n_exit,
        uncertainty_margin=args.uncertainty_margin_m,
    )
    thresholds = {
        "enter_threshold_m": args.enter_threshold_m,
        "exit_threshold_m": args.exit_threshold_m,
        "n_enter": args.n_enter,
        "n_exit": args.n_exit,
        "uncertainty_margin_m": args.uncertainty_margin_m,
    }
    rows: list[dict[str, object]] = []
    clip_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    for clip_index, clip_dir in enumerate(clip_dirs, start=1):
        metadata_path = clip_dir / "metadata.json"
        forward_path = clip_dir / "smplx_forward.npz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if file_sha256(forward_path) != metadata["smplx_forward_sha256"]:
            raise ValueError(f"forward hash mismatch: {forward_path}")
        with np.load(forward_path, allow_pickle=False) as forward:
            vertices = torch.from_numpy(np.asarray(forward["vertices"])).to(device)
            frame_ids = np.asarray(forward["frame_ids"], dtype=np.int64)
        geometry_backend = ContactGeometry(
            patch_map,
            fps=float(metadata["fps"]),
            allow_missing_penetration=True,
        ).to(device)
        with torch.inference_mode():
            normals = vertex_normals(vertices, faces)
            geometry = geometry_backend.features(vertices, vertex_normals=normals)
            labels = labeler.compile(geometry.distance[0])
        clip_output = output / clip_dir.name
        clip_output.mkdir()
        artifact_path = clip_output / "contact_geometry.npz"
        np.savez_compressed(
            artifact_path,
            frame_ids=frame_ids,
            edge_names=np.asarray(
                [
                    f"{source_name}::{target_name}"
                    for source_name, target_name in patch_map.admissible_edges
                ]
            ),
            features=geometry.features[0].cpu().numpy(),
            distance_m=geometry.distance[0].cpu().numpy(),
            normal_compatibility=geometry.normal_compatibility[0].cpu().numpy(),
            relative_speed_m_per_s=geometry.relative_speed[0].cpu().numpy(),
            reliability=geometry.reliability[0].cpu().numpy(),
            pseudo_event_state=labels.event_state.cpu().numpy(),
            pseudo_uncertain_mask=labels.uncertain_mask.cpu().numpy(),
        )
        state_counts = {
            state.name.lower(): int((labels.event_state == int(state)).sum())
            for state in EventState
        }
        clip_metadata = {
            "schema_version": "development_contact_geometry_v1",
            "development_only": True,
            "scientific_status": "NOT_A_GOLD_CONTACT_RESULT",
            "clip_id": clip_dir.name,
            "frames": int(vertices.shape[1]),
            "edges": len(patch_map.admissible_edges),
            "patch_map_sha256": patch_map.content_hash,
            "source_forward_sha256": metadata["smplx_forward_sha256"],
            "penetration_available": geometry.penetration_available,
            "signed_distance_backend": None,
            "thresholds": thresholds,
            "threshold_status": "DEVELOPMENT_DEFAULT_NOT_AUTHOR_FROZEN",
            "state_counts": state_counts,
            "uncertain_count": int(labels.uncertain_mask.sum()),
            "minimum_distance_m": float(geometry.distance.min()),
            "contact_geometry_sha256": file_sha256(artifact_path),
        }
        (clip_output / "metadata.json").write_text(
            json.dumps(clip_metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(clip_metadata)
        print(f"[{clip_index:02d}/{len(clip_dirs):02d}] {clip_dir.name}", flush=True)

    report = {
        "schema_version": "development_contact_compilation_v1",
        "development_only": True,
        "scientific_status": "NOT_A_DCG_SIGN4D_RESULT",
        "patch_map": str(patch_path.resolve()),
        "patch_map_file_sha256": file_sha256(patch_path),
        "patch_map_content_sha256": patch_map.content_hash,
        "topology_model_name": model_path.name,
        "topology_model_sha256": file_sha256(model_path),
        "initialization_report_sha256": file_sha256(source / "conversion_report.json"),
        "thresholds": thresholds,
        "penetration_available": False,
        "clips": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "edges": len(patch_map.admissible_edges),
        "state_counts": {
            state.name.lower(): sum(int(row["state_counts"][state.name.lower()]) for row in rows)
            for state in EventState
        },
        "uncertain_count": sum(int(row["uncertain_count"]) for row in rows),
        "per_clip": rows,
    }
    report["configuration_sha256"] = canonical_hash(
        {
            "patch_map_content_sha256": patch_map.content_hash,
            "topology_model_sha256": report["topology_model_sha256"],
            "initialization_report_sha256": report["initialization_report_sha256"],
            "thresholds": thresholds,
        }
    )
    (output / "compilation_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(incomplete, output / "COMPILATION_COMPLETE")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_clip"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
