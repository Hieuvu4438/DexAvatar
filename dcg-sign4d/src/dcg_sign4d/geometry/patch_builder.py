"""Deterministic, development-only SMPL-X contact patch construction."""

from __future__ import annotations

import ast
import json
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def _literal_assignment(path: Path, variable: str) -> Any:
    """Read a literal assignment without importing third-party source code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"literal assignment {variable!r} not found in {path}")


def _nearest(vertices: np.ndarray, candidates: np.ndarray, seed: int, count: int) -> list[int]:
    if seed not in candidates:
        raise ValueError(f"seed vertex {seed} is outside its anatomical candidate segment")
    squared = np.square(vertices[candidates] - vertices[seed]).sum(axis=1)
    order = np.lexsort((candidates, squared))
    return candidates[order[:count]].astype(int).tolist()


def _farthest_point_sample(vertices: np.ndarray, candidates: np.ndarray, count: int) -> list[int]:
    """Deterministic Euclidean FPS; vertex id resolves every distance tie."""
    candidates = np.unique(candidates).astype(np.int64)
    if count > len(candidates):
        raise ValueError("FPS count exceeds candidate count")
    points = vertices[candidates]
    centroid_distance = np.square(points - points.mean(axis=0)).sum(axis=1)
    first_order = np.lexsort((candidates, -centroid_distance))
    selected_positions = [int(first_order[0])]
    minimum_distance = np.square(points - points[selected_positions[0]]).sum(axis=1)
    minimum_distance[selected_positions[0]] = -1.0
    for _ in range(1, count):
        order = np.lexsort((candidates, -minimum_distance))
        position = int(order[0])
        selected_positions.append(position)
        distance = np.square(points - points[position]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected_positions] = -1.0
    return sorted(candidates[selected_positions].astype(int).tolist())


def build_provisional_smplx_patch_map(
    *,
    model_path: str | Path,
    segmentation_path: str | Path,
    vertex_ids_path: str | Path,
) -> dict[str, Any]:
    """Build the declared provisional patch map from audited public assets.

    This compiler deliberately cannot emit a production asset. Patch counts and
    admissible edges are engineering defaults for development geometry checks,
    not substitutes for AUTHOR_REQUIRED scientific choices.
    """
    model_path = Path(model_path)
    segmentation_path = Path(segmentation_path)
    vertex_ids_path = Path(vertex_ids_path)
    with np.load(model_path, allow_pickle=False) as model:
        vertices = np.asarray(model["v_template"], dtype=np.float64)
    segmentation = json.loads(segmentation_path.read_text(encoding="utf-8"))
    vertex_ids = _literal_assignment(vertex_ids_path, "vertex_ids")["smplx"]
    if vertices.shape != (10475, 3):
        raise ValueError(f"expected SMPL-X topology [10475,3], got {vertices.shape}")

    hand_candidates = {
        "left": np.asarray(
            sorted(set(segmentation["leftHand"]) | set(segmentation["leftHandIndex1"])),
            dtype=np.int64,
        ),
        "right": np.asarray(
            sorted(set(segmentation["rightHand"]) | set(segmentation["rightHandIndex1"])),
            dtype=np.int64,
        ),
    }
    patches: dict[str, list[int]] = {}
    digit_names = ("thumb", "index", "middle", "ring", "pinky")
    for side, prefix in (("left", "l"), ("right", "r")):
        used: set[int] = set()
        for digit in digit_names:
            name = f"{side}_{digit}_tip"
            indices = _nearest(
                vertices, hand_candidates[side], int(vertex_ids[f"{prefix}{digit}"]), 16
            )
            patches[name] = indices
            used.update(indices)
        palm_candidates = np.asarray(
            [index for index in hand_candidates[side] if int(index) not in used],
            dtype=np.int64,
        )
        patches[f"{side}_palm"] = _farthest_point_sample(vertices, palm_candidates, 64)

    head_candidates = np.asarray(segmentation["head"], dtype=np.int64)
    face_seeds = np.asarray(
        [vertex_ids[name] for name in ("nose", "reye", "leye", "rear", "lear")],
        dtype=np.int64,
    )
    seed_distances = np.square(
        vertices[head_candidates, None, :] - vertices[face_seeds][None, :, :]
    ).sum(axis=-1)
    face_candidate_order = np.lexsort((head_candidates, seed_distances.min(axis=1)))
    face_candidates = head_candidates[face_candidate_order[:512]]
    patches["face"] = _farthest_point_sample(vertices, face_candidates, 128)

    torso_candidates = np.asarray(
        sorted(
            set(segmentation["spine"]) | set(segmentation["spine1"]) | set(segmentation["spine2"])
        ),
        dtype=np.int64,
    )
    patches["torso"] = _farthest_point_sample(vertices, torso_candidates, 128)

    left = [name for name in patches if name.startswith("left_")]
    right = [name for name in patches if name.startswith("right_")]
    admissible = [list(edge) for edge in product(left, right)]
    admissible.extend([list(edge) for edge in product(left + right, ("face", "torso"))])
    excluded = [list(edge) for edge in combinations(left, 2)]
    excluded.extend([list(edge) for edge in combinations(right, 2)])
    excluded.append(["face", "torso"])

    payload: dict[str, Any] = {
        "patch_map_version": "smplx_provisional_development_v1",
        "smplx_model_version": "SMPL-X neutral 10,475-vertex topology",
        "mesh_vertex_count": int(vertices.shape[0]),
        "development_only": True,
        "scientific_status": "UNFROZEN_AUTHOR_REVIEW_REQUIRED",
        "selection_method": {
            "fingertips": "16 Euclidean-nearest segment vertices per landmark",
            "palms": "64 deterministic Euclidean FPS vertices after fingertip exclusion",
            "face": "128 deterministic FPS vertices from 512 head vertices nearest face landmarks",
            "torso": "128 deterministic FPS vertices from spine/spine1/spine2 union",
            "warning": "Euclidean/FPS defaults are for development only; not geodesic patches",
        },
        "source_assets": {
            "neutral_model": {"name": model_path.name, "sha256": file_sha256(model_path)},
            "segmentation": {
                "name": segmentation_path.name,
                "sha256": file_sha256(segmentation_path),
            },
            "vertex_ids": {
                "name": vertex_ids_path.name,
                "sha256": file_sha256(vertex_ids_path),
            },
        },
        "patches": patches,
        "admissible_edges": admissible,
        "excluded_edges": excluded,
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def write_patch_map(payload: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable patch map already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output
