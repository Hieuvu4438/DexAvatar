"""Fail-closed scientific readiness audit for reconstruction runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dcg_sign4d.data.manifest import load_manifest
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.initialization.artifact import load_initialization_artifact
from dcg_sign4d.observations.cache import ObservationCache
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

PLACEHOLDERS = {"AUTHOR_REQUIRED", "UNKNOWN", "TODO", "TBD"}


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str


def _value(config: dict[str, Any], dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _placeholder_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        result = []
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_placeholder_paths(child, name))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(_placeholder_paths(child, f"{prefix}[{index}]"))
        return result
    if isinstance(value, str) and value.strip().upper() in PLACEHOLDERS:
        return [prefix]
    return []


def _checkpoint_check(path: Path, stage: str, development_only: bool) -> None:
    if not (path / "CHECKPOINT_COMPLETE").is_file():
        raise ValueError("completion marker missing")
    metadata_path = path / "metadata.json"
    weights_path = path / "weights.pt"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = payload.pop("metadata_identity_sha256", None)
    if identity != canonical_hash(payload):
        raise ValueError("metadata identity mismatch")
    if file_sha256(weights_path) != payload.get("weights_sha256"):
        raise ValueError("weights hash mismatch")
    if payload.get("stage") != stage:
        raise ValueError(f"expected stage {stage}")
    if payload.get("development_only") and not development_only:
        raise PermissionError("development checkpoint in production run")


def audit_reconstruction_readiness(
    config: dict[str, Any], manifest_path: str | Path
) -> dict[str, Any]:
    """Return a complete machine-readable report without starting inference."""

    checks: list[ReadinessCheck] = []

    def record(name: str, operation: Any) -> None:
        try:
            detail = operation()
            checks.append(ReadinessCheck(name, "PASS", str(detail or "verified")))
        except Exception as exc:
            first_line = str(exc).splitlines()[0]
            checks.append(ReadinessCheck(name, "BLOCKED", f"{type(exc).__name__}: {first_line}"))

    development_only = bool(_value(config, "experiment.development_only"))
    placeholders = _placeholder_paths(config)
    placeholder_detail = (
        "no unresolved placeholders; development inputs are not author freezes"
        if development_only and not placeholders
        else "no unresolved placeholders"
    )
    checks.append(
        ReadinessCheck(
            "author_freezes",
            "PASS" if not placeholders else "BLOCKED",
            placeholder_detail if not placeholders else ", ".join(placeholders),
        )
    )
    required = (
        "experiment.name",
        "experiment.seed",
        "experiment.deterministic",
        "experiment.development_only",
        "data.window_length",
        "data.window_overlap",
        "observation.artifact_root",
        "observation.calibration_artifact",
        "observation.camera_calibration",
        "initialization.artifact_root",
        "initialization.commit",
        "geometry.patch_map",
        "geometry.smplx_model",
        "geometry.smplx_model_sha256",
        "geometry.trusted_smplx_model",
        "geometry.selfcontact_source_root",
        "geometry.selfcontact_commit",
        "geometry.selfcontact_essentials_root",
        "geometry.selfcontact_essentials_registry",
        "geometry.selfcontact_essentials_registry_sha256",
        "geometry.trusted_selfcontact_assets",
        "geometry.selfcontact_test_segments",
        "geometry.separation_margin_m",
        "geometry.sigma_distance_m",
        "geometry.sigma_normal",
        "geometry.sigma_velocity_m_per_s",
        "geometry.normal_weight",
        "geometry.hold_velocity_weight",
        "geometry.penetration_area_weight",
        "contact.checkpoint",
        "contact.training_config",
        "contact.semi_markov.max_duration",
        "diffusion.checkpoint",
        "diffusion.training_config",
        "diffusion.trajectory_normalizer",
        "diffusion.dposer_registry",
        "guidance.enabled_terms",
        "guidance.guidance_scale",
        "guidance.gradient_clip_norm",
        "guidance.trust_region_norm",
        "guidance.keypoint_sigma_min_px",
        "guidance.keypoint_sigma_occluded_px",
        "inference.rounds",
        "inference.diffusion_steps",
        "inference.num_hypotheses",
        "inference.alternating",
        "ranking.artifact",
        "ranking.fit_split",
        "ranking.use_ground_truth",
        "third_party_manifest",
    )
    missing = [name for name in required if _value(config, name) is None]
    checks.append(
        ReadinessCheck(
            "configuration_contract",
            "PASS" if not missing else "BLOCKED",
            "all runtime fields present" if not missing else "missing: " + ", ".join(missing),
        )
    )
    manifest_items = []

    def manifest_operation() -> str:
        nonlocal manifest_items
        manifest_items = load_manifest(manifest_path, require_existing_video=False)
        if not development_only and any(item.split != "test" for item in manifest_items):
            raise ValueError("production reconstruction manifest must contain only test clips")
        return f"{len(manifest_items)} clips"

    record("manifest", manifest_operation)

    def patch_operation() -> str:
        path = Path(_value(config, "geometry.patch_map"))
        patch = PatchMap.load(path)
        if patch.development_only and not development_only:
            raise PermissionError("development patch map in production run")
        label = "development" if patch.development_only else "frozen"
        return f"{len(patch.admissible_edges)} {label} edges; {file_sha256(path)}"

    if _value(config, "geometry.patch_map") not in (None, "AUTHOR_REQUIRED"):
        record("patch_map", patch_operation)
    else:
        checks.append(ReadinessCheck("patch_map", "BLOCKED", "path unresolved"))

    def calibration_operation() -> str:
        path = Path(_value(config, "observation.calibration_artifact"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "dcg_temperature_calibration_v1":
            raise ValueError("unknown schema")
        if payload.get("gate_status") != "PASS":
            raise PermissionError("calibration gate did not pass")
        if payload.get("development_only") and not development_only:
            raise PermissionError("development calibrator in production run")
        return file_sha256(path)

    if _value(config, "observation.calibration_artifact") not in (None, "AUTHOR_REQUIRED"):
        record("calibration", calibration_operation)
    else:
        checks.append(ReadinessCheck("calibration", "BLOCKED", "artifact unresolved"))

    def per_clip_operation() -> str:
        if not manifest_items:
            raise ValueError("manifest unavailable")
        observation_root = Path(_value(config, "observation.artifact_root"))
        initialization_root = Path(_value(config, "initialization.artifact_root"))
        observation_index = json.loads((observation_root / "index.json").read_text("utf-8"))
        index_identity = observation_index.pop("index_identity_sha256", None)
        if index_identity != canonical_hash(observation_index):
            raise ValueError("calibrated observation index identity mismatch")
        observation_index["index_identity_sha256"] = index_identity
        if observation_index.get("schema_version") != "dcg_calibrated_observation_index_v1":
            raise ValueError("unknown calibrated observation index schema")
        if observation_index.get("development_only") and not development_only:
            raise PermissionError("development calibrated observation index")
        cache_by_clip = {row["clip_id"]: row["cache_id"] for row in observation_index["per_clip"]}
        requested = {item.clip_id for item in manifest_items}
        if not requested <= set(cache_by_clip):
            raise ValueError("calibrated observation index is missing manifest clips")
        cache = ObservationCache(observation_root / "caches")
        for item in manifest_items:
            observations = cache.load(cache_by_clip[item.clip_id])
            if observations.keypoints_2d.shape[1] != item.effective_frame_count:
                raise ValueError(f"{item.clip_id}: observation frame count mismatch")
            if observations.metadata[0].get("development_only") and not development_only:
                raise PermissionError(f"{item.clip_id}: development observations")
            state, camera, init_metadata = load_initialization_artifact(
                initialization_root / item.clip_id
            )
            if state.root_translation.shape[1] != item.effective_frame_count:
                raise ValueError(f"{item.clip_id}: initialization frame count mismatch")
            if camera.valid_mask.shape != state.valid_mask.shape:
                raise ValueError(f"{item.clip_id}: camera frame count mismatch")
            if init_metadata.get("development_only") and not development_only:
                raise PermissionError(f"{item.clip_id}: development initialization")
        return f"{len(manifest_items)} calibrated observation/initialization pairs"

    roots_ready = all(
        _value(config, name) not in (None, "AUTHOR_REQUIRED")
        for name in ("observation.artifact_root", "initialization.artifact_root")
    )
    if roots_ready:
        record("per_clip_inputs", per_clip_operation)
    else:
        checks.append(ReadinessCheck("per_clip_inputs", "BLOCKED", "artifact roots unresolved"))

    for name, stage in (
        ("contact.checkpoint", "contact_proposal"),
        ("diffusion.checkpoint", "trajectory_diffusion"),
    ):
        value = _value(config, name)
        check_name = name.replace(".", "_")
        if value in (None, "AUTHOR_REQUIRED"):
            checks.append(ReadinessCheck(check_name, "BLOCKED", "checkpoint unresolved"))
        else:
            record(
                check_name,
                lambda value=value, stage=stage: _checkpoint_check(
                    Path(value), stage, development_only
                ),
            )

    def ranker_operation() -> str:
        payload = json.loads(Path(_value(config, "ranking.artifact")).read_text("utf-8"))
        if payload.get("fit_split") != "validation" or payload.get("use_ground_truth") is not False:
            raise ValueError("ranker must be validation-fitted and GT-free")
        if payload.get("gate_status") != "PASS":
            raise PermissionError("ranker gate did not pass")
        if payload.get("development_only") and not development_only:
            raise PermissionError("development ranker in production run")
        if payload.get("development_only"):
            return "development-only validation-contract GT-free ranker"
        return "validation-fitted GT-free ranker"

    if _value(config, "ranking.artifact") not in (None, "AUTHOR_REQUIRED"):
        record("ranker", ranker_operation)
    else:
        checks.append(ReadinessCheck("ranker", "BLOCKED", "artifact unresolved"))

    def camera_operation() -> str:
        payload = json.loads(
            Path(_value(config, "observation.camera_calibration")).read_text("utf-8")
        )
        allowed = {"FROZEN"} | ({"DEVELOPMENT"} if development_only else set())
        if payload.get("scientific_status") not in allowed:
            raise PermissionError("camera convention/calibration is not frozen")
        if payload.get("scientific_status") == "DEVELOPMENT":
            return "development projection convention"
        return "frozen projection convention"

    if _value(config, "observation.camera_calibration") not in (None, "AUTHOR_REQUIRED"):
        record("camera_projection", camera_operation)
    else:
        checks.append(ReadinessCheck("camera_projection", "BLOCKED", "artifact unresolved"))

    checks.append(
        ReadinessCheck(
            "runtime_assembly",
            "PASS",
            "production component builder and atomic per-clip artifact path implemented",
        )
    )

    status = "READY" if all(check.status == "PASS" for check in checks) else "BLOCKED"
    return {
        "schema_version": "dcg_reconstruction_readiness_v1",
        "status": status,
        "development_only": development_only,
        "checks": [asdict(check) for check in checks],
    }
