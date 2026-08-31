from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from signpk.data.frame_manifest import SignManifest
from signpk.evaluation.subgroup_metrics import aggregate_subgroups
from signpk.evaluation.temporal_metrics import acceleration_error, velocity_error
from signpk.geometry.topology import SMPLX_VERTEX_COUNT, load_obj
from signpk.utils.config_hash import sha256_file


@dataclass
class RegionSummary:
    mean_mm: float | None
    median_mm: float | None
    p95_mm: float | None
    frames: int
    vertices: int


@dataclass
class EvaluationResult:
    overall: dict[str, RegionSummary]
    per_sign: dict[str, dict[str, RegionSummary]]
    subgroups: dict[str, dict[str, float]]
    temporal: dict[str, object]
    protocol: dict[str, object]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def translation_aligned_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(0, keepdims=True)
    target = target - target.mean(0, keepdims=True)
    return np.linalg.norm(prediction - target, axis=-1)


def _mesh_id(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"mesh filename has no frame ID: {path}")
    return int(matches[-1])


def discover_meshes(root: str | Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in Path(root).glob("*.obj"):
        frame_id = _mesh_id(path)
        if frame_id in result:
            raise ValueError(f"duplicate mesh ID {frame_id}: {result[frame_id]} and {path}")
        result[frame_id] = path
    return result


def load_subsets(data_root: str | Path) -> dict[str, np.ndarray]:
    data_root = Path(data_root)
    with (data_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        hands = pickle.load(handle, encoding="latin1")
    upper = np.load(data_root / "sgnify_part_segm_above_pelvis_joint" / "upper_body_minus_face.npy")
    return {
        "UBody(-F)": np.asarray(upper, dtype=np.int64),
        "LHand": np.asarray(hands["left_hand"], dtype=np.int64),
        "RHand": np.asarray(hands["right_hand"], dtype=np.int64),
    }


def _summarize(values: list[np.ndarray]) -> RegionSummary:
    if not values:
        return RegionSummary(None, None, None, 0, 0)
    stacked = np.concatenate(values) * 1000.0
    return RegionSummary(
        mean_mm=float(stacked.mean()),
        median_mm=float(np.median(stacked)),
        p95_mm=float(np.percentile(stacked, 95)),
        frames=len(values),
        vertices=int(stacked.size),
    )


class AuditedTRV2VEvaluator:
    def __init__(
        self,
        subsets: Mapping[str, np.ndarray],
        *,
        vertex_count: int = SMPLX_VERTEX_COUNT,
        class0_exclude_left_hand: bool = True,
    ):
        self.subsets = {
            name: np.asarray(indices, dtype=np.int64) for name, indices in subsets.items()
        }
        self.vertex_count = vertex_count
        self.class0_exclude_left_hand = class0_exclude_left_hand

    def evaluate_sign(
        self,
        manifest: SignManifest,
        prediction_meshes: Mapping[int, Path],
        gt_meshes: Mapping[int, Path] | None = None,
    ) -> tuple[dict[str, RegionSummary], dict[str, list[np.ndarray]]]:
        expected = set(manifest.gt_ids)
        prediction_ids = set(prediction_meshes)
        if prediction_ids != expected:
            missing, extra = sorted(expected - prediction_ids), sorted(prediction_ids - expected)
            raise ValueError(
                f"prediction frame-ID mismatch for {manifest.sign_name}: missing={missing}, extra={extra}"
            )
        gt_meshes = gt_meshes or {
            record.gt_frame_id: record.gt_obj_path
            for record in manifest.records
            if record.gt_obj_path
        }
        if set(gt_meshes) != expected:
            raise ValueError(f"GT frame-ID mismatch for {manifest.sign_name}")
        errors: dict[str, list[np.ndarray]] = {name: [] for name in self.subsets}
        reference_faces: np.ndarray | None = None
        left_indices = self.subsets["LHand"]
        for frame_id in manifest.gt_ids:
            predicted_vertices, predicted_faces = load_obj(prediction_meshes[frame_id])
            target_vertices, target_faces = load_obj(gt_meshes[frame_id])
            if predicted_vertices.shape != target_vertices.shape or predicted_vertices.shape != (
                self.vertex_count,
                3,
            ):
                raise ValueError(f"vertex mismatch at {manifest.sign_name}/{frame_id}")
            np.testing.assert_array_equal(predicted_faces, target_faces)
            if reference_faces is None:
                reference_faces = target_faces
            else:
                np.testing.assert_array_equal(reference_faces, target_faces)
            if not np.isfinite(predicted_vertices).all():
                raise ValueError(f"NaN/Inf at {manifest.sign_name}/{frame_id}")
            for name, base_indices in self.subsets.items():
                if (
                    name == "LHand"
                    and self.class0_exclude_left_hand
                    and manifest.handedness_class == "0"
                ):
                    continue
                indices = base_indices
                if (
                    name == "UBody(-F)"
                    and self.class0_exclude_left_hand
                    and manifest.handedness_class == "0"
                ):
                    indices = np.setdiff1d(indices, left_indices)
                errors[name].append(
                    translation_aligned_errors(
                        predicted_vertices[indices], target_vertices[indices]
                    )
                )
        return {name: _summarize(values) for name, values in errors.items()}, errors

    def evaluate_root(
        self,
        manifest_root: str | Path,
        prediction_root: str | Path,
        signs: set[str] | None = None,
        official_evaluator_path: str | Path | None = None,
    ) -> EvaluationResult:
        manifest_paths = sorted(Path(manifest_root).glob("*/manifest.json"))
        if signs is not None:
            manifest_paths = [path for path in manifest_paths if path.parent.name in signs]
        if not manifest_paths:
            raise FileNotFoundError(f"no selected manifests below {manifest_root}")
        all_errors: dict[str, list[np.ndarray]] = {name: [] for name in self.subsets}
        per_sign: dict[str, dict[str, RegionSummary]] = {}
        per_frame_errors: dict[str, dict[str, np.ndarray]] = {}
        subgroup_attributes: dict[str, dict[str, str | bool]] = {}
        temporal_by_sign: dict[str, dict[str, dict[str, float]]] = {}
        for manifest_path in manifest_paths:
            manifest = SignManifest.load(manifest_path, validate_paths=True)
            meshes = discover_meshes(Path(prediction_root) / manifest.sign_name / "meshes")
            summaries, errors = self.evaluate_sign(manifest, meshes)
            per_sign[manifest.sign_name] = summaries
            for name, values in errors.items():
                all_errors[name].extend(values)
            diagnostics = _load_frame_diagnostics(
                Path(prediction_root) / manifest.sign_name / "frame_diagnostics.jsonl"
            )
            frame_count = len(manifest.records)
            for index, record in enumerate(manifest.records):
                frame_key = f"{manifest.sign_name}/{record.gt_frame_id}"
                per_frame_errors[frame_key] = {
                    region: values[index]
                    for region, values in errors.items()
                    if len(values) == frame_count
                }
                attributes: dict[str, str | bool] = {
                    "handedness": ("one_hand" if manifest.handedness_class == "0" else "two_hand"),
                    "segment": (
                        "early"
                        if index * 3 < frame_count
                        else "middle"
                        if index * 3 < 2 * frame_count
                        else "late"
                    ),
                }
                if record.prediction_frame_id in diagnostics:
                    attributes.update(
                        {
                            key: diagnostics[record.prediction_frame_id][key]
                            for key in ("interaction", "velocity", "disagreement")
                            if key in diagnostics[record.prediction_frame_id]
                        }
                    )
                subgroup_attributes[frame_key] = attributes
            temporal_by_sign[manifest.sign_name] = self._temporal_sign(
                manifest, Path(prediction_root) / manifest.sign_name / "meshes"
            )
        protocol: dict[str, object] = {
            "result_kind": "audited_strict",
            "name": "SGNify TR-V2V audited frame-identity protocol",
            "translation_alignment": "independent regional mean centering",
            "strict_frame_ids": True,
            "topology_assertion": True,
            "units_input": "meters",
            "units_output": "millimeters",
            "class0_exclude_left_hand": self.class0_exclude_left_hand,
            "official_numeric_formula_compatible": True,
            "original_official_evaluator_executed": False,
            "audited_evaluator_sha256": sha256_file(Path(__file__)),
            "distinction": (
                "Metrics use the official regional centering and class-0 convention, "
                "with stricter ID/topology checks; they are audited results, not output "
                "from the original ordinal-pairing script."
            ),
        }
        if official_evaluator_path is not None:
            official_path = Path(official_evaluator_path)
            protocol["original_official_evaluator_path"] = str(official_path)
            protocol["original_official_evaluator_sha256"] = sha256_file(official_path)
        return EvaluationResult(
            overall={name: _summarize(values) for name, values in all_errors.items()},
            per_sign=per_sign,
            subgroups=aggregate_subgroups(
                per_frame_errors,
                subgroup_attributes,
                keys=("handedness", "interaction", "velocity", "disagreement", "segment"),
            ),
            temporal={
                "per_sign": temporal_by_sign,
                "units": {"velocity": "mm/s", "acceleration": "mm/s^2"},
            },
            protocol=protocol,
        )

    def _temporal_sign(
        self, manifest: SignManifest, prediction_mesh_root: Path
    ) -> dict[str, dict[str, float]]:
        predictions = discover_meshes(prediction_mesh_root)
        predicted_sequence = []
        target_sequence = []
        for record in manifest.records:
            prediction, _ = load_obj(predictions[record.prediction_frame_id])
            target, _ = load_obj(record.gt_obj_path)
            predicted_sequence.append(prediction)
            target_sequence.append(target)
        predicted = np.stack(predicted_sequence)
        target = np.stack(target_sequence)
        timestamps = np.asarray([record.timestamp_sec for record in manifest.records])
        result: dict[str, dict[str, float]] = {}
        if len(timestamps) < 3:
            return result
        for name, base_indices in self.subsets.items():
            if (
                name == "LHand"
                and self.class0_exclude_left_hand
                and manifest.handedness_class == "0"
            ):
                continue
            indices = base_indices
            if (
                name == "UBody(-F)"
                and self.class0_exclude_left_hand
                and manifest.handedness_class == "0"
            ):
                indices = np.setdiff1d(indices, self.subsets["LHand"])
            predicted_region = predicted[:, indices]
            target_region = target[:, indices]
            predicted_region -= predicted_region.mean(1, keepdims=True)
            target_region -= target_region.mean(1, keepdims=True)
            result[name] = {
                "velocity_error_mm_per_s": 1000.0
                * velocity_error(predicted_region, target_region, timestamps),
                "acceleration_error_mm_per_s2": 1000.0
                * acceleration_error(predicted_region, target_region, timestamps),
            }
        return result


def _load_frame_diagnostics(path: Path) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {int(row["frame_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate diagnostic frame IDs in {path}")
    return result
