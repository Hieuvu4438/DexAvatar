from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import yaml

from signdart.io.h1_state import H1State, read_manifest, state_path
from signdart.model import (
    create_model,
    forward_state_batch,
    rigid_transport_hand_vertices,
)


def seam_topology(faces: np.ndarray, hand_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand = np.zeros(10475, dtype=bool)
    hand[np.asarray(hand_ids, dtype=np.int64)] = True
    seam_faces = faces[np.any(hand[faces], axis=1) & ~np.all(hand[faces], axis=1)]
    edges = np.concatenate(
        (seam_faces[:, [0, 1]], seam_faces[:, [1, 2]], seam_faces[:, [2, 0]]), axis=0
    )
    cross = hand[edges[:, 0]] != hand[edges[:, 1]]
    edges = np.unique(np.sort(edges[cross], axis=1), axis=0)
    return seam_faces, edges


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    first = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    second = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return np.cross(first, second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    records = read_manifest(paths["manifest"])
    device = str(config["runtime"]["device"])
    model = create_model(paths["model_root"], device)
    faces = np.asarray(model.faces, dtype=np.int64)
    with (paths["model_root"] / "smplx" / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        mano = pickle.load(handle)
    sides = {
        "left": (np.asarray(mano["left_hand"], dtype=np.int64), 20),
        "right": (np.asarray(mano["right_hand"], dtype=np.int64), 21),
    }
    seam = {
        side: seam_topology(faces, hand_ids)
        for side, (hand_ids, _) in sides.items()
    }
    edge_changes = []
    area_ratios = []
    normal_cosines = []
    candidate_count = 0
    for ordinal, record in enumerate(records, start=1):
        state = H1State.load(state_path(paths["h1_state_root"], record))
        candidate_path = (
            paths["candidate_root"]
            / record["sign_id"]
            / f"{int(record['source_frame_id']):06d}.npz"
        )
        with np.load(candidate_path, allow_pickle=False) as archive:
            for side, (hand_ids, wrist_id) in sides.items():
                poses = archive[f"{side}_body_pose"]
                if len(poses) <= 1:
                    continue
                pure_vertices, joints = forward_state_batch(model, state, poses, device)
                incumbent_vertices, incumbent_joints = forward_state_batch(
                    model, state, state.arrays["body_pose"], device
                )
                seam_faces, seam_edges = seam[side]
                for index in range(1, len(poses)):
                    transported = rigid_transport_hand_vertices(
                        pure_vertices[index],
                        joints[index],
                        state.vertices_evaluator,
                        incumbent_joints[0],
                        hand_ids,
                        wrist_id,
                    )
                    pure_edge = np.linalg.norm(
                        pure_vertices[index, seam_edges[:, 0]]
                        - pure_vertices[index, seam_edges[:, 1]], axis=-1,
                    )
                    transported_edge = np.linalg.norm(
                        transported[seam_edges[:, 0]]
                        - transported[seam_edges[:, 1]], axis=-1,
                    )
                    edge_changes.extend((np.abs(transported_edge - pure_edge) * 1000.0).tolist())
                    pure_normals = face_normals(pure_vertices[index], seam_faces)
                    transported_normals = face_normals(transported, seam_faces)
                    pure_area = np.linalg.norm(pure_normals, axis=-1)
                    transported_area = np.linalg.norm(transported_normals, axis=-1)
                    area_ratios.extend(
                        (transported_area / np.maximum(pure_area, 1e-12)).tolist()
                    )
                    normal_cosines.extend(
                        (
                            np.sum(pure_normals * transported_normals, axis=-1)
                            / np.maximum(pure_area * transported_area, 1e-12)
                        ).tolist()
                    )
                    candidate_count += 1
        if ordinal % 50 == 0 or ordinal == len(records):
            print(f"[SEAM] {ordinal}/{len(records)}", flush=True)

    edges = np.asarray(edge_changes)
    areas = np.asarray(area_ratios)
    cosines = np.asarray(normal_cosines)
    report = {
        "schema_version": "signdart.hand_transport_seam.v1",
        "status": "diagnostic_only",
        "frames": len(records),
        "non_incumbent_side_candidates": candidate_count,
        "cross_seam_edge_abs_change_mm": {
            "median": float(np.median(edges)),
            "p95": float(np.quantile(edges, 0.95)),
            "p99": float(np.quantile(edges, 0.99)),
            "max": float(np.max(edges)),
        },
        "seam_triangle_area_ratio": {
            "p01": float(np.quantile(areas, 0.01)),
            "median": float(np.median(areas)),
            "p99": float(np.quantile(areas, 0.99)),
            "max": float(np.max(areas)),
        },
        "seam_triangle_normal_cosine": {
            "min": float(np.min(cosines)),
            "p01": float(np.quantile(cosines, 0.01)),
            "median": float(np.median(cosines)),
            "flipped_fraction": float(np.mean(cosines < 0.0)),
        },
    }
    output = paths["report_root"] / "hand_transport_seam.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
