"""Production DCG-Sign4D component assembly and per-clip reconstruction."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from dcg_sign4d.cli.train_contact import ContactTrainingConfig
from dcg_sign4d.cli.train_diffusion import DiffusionTrainingConfig
from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.contact.semi_markov import SemiMarkovDecoder
from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.dposer_bridge import OfficialDPoserXBridge
from dcg_sign4d.diffusion.sampler import GuidedTrajectorySampler
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.diffusion.trajectory_denoiser import DPoserXConditionedTrajectoryDenoiser
from dcg_sign4d.geometry.contact_geometry import ContactGeometry
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.geometry.penetration import OfficialSelfContactPenetration
from dcg_sign4d.geometry.smplx_adapter import SMPLXAdapter
from dcg_sign4d.geometry.state_geometry import StateContactGeometry
from dcg_sign4d.guidance.contact import ContactGuidance
from dcg_sign4d.guidance.depth import RelativeDepthGuidance
from dcg_sign4d.guidance.keypoint import KeypointGuidance
from dcg_sign4d.guidance.silhouette import SilhouetteGuidance
from dcg_sign4d.guidance.track import TrackGuidance
from dcg_sign4d.inference.alternating import AlternatingReconstructor
from dcg_sign4d.inference.hypothesis import Hypothesis, RoundResult
from dcg_sign4d.inference.ranker_fit import load_frozen_ranker
from dcg_sign4d.inference.ranking import HypothesisRanker
from dcg_sign4d.inference.windowing import (
    slice_camera,
    slice_observations,
    slice_trajectory,
    stitch_contact_graphs,
    stitch_trajectories,
    window_starts,
)
from dcg_sign4d.initialization.artifact import load_initialization_artifact
from dcg_sign4d.initialization.camera import (
    CameraTrajectory,
    StateJointDepthDifference,
    StateJointProjector,
    StatePartMaskRenderer,
)
from dcg_sign4d.observations.cache import ObservationCache
from dcg_sign4d.observations.schema import ObservationBatch
from dcg_sign4d.training.checkpoint import load_model_checkpoint
from dcg_sign4d.utils.hashing import file_sha256


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeExperiment(_Strict):
    name: str
    seed: int
    deterministic: bool
    development_only: bool = False


class RuntimeData(_Strict):
    window_length: int = Field(gt=0)
    window_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_is_smaller(self) -> RuntimeData:
        if self.window_overlap >= self.window_length:
            raise ValueError("window overlap must be smaller than length")
        return self


class RuntimeObservation(_Strict):
    artifact_root: Path
    calibration_artifact: Path
    camera_calibration: Path


class RuntimeInitialization(_Strict):
    backend: Literal["dexavatar", "artifact_replay"]
    artifact_root: Path
    commit: str = Field(pattern="^[0-9a-f]{40}$")


class RuntimeGeometry(_Strict):
    patch_map: Path
    smplx_model: Path
    smplx_model_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    trusted_smplx_model: bool
    selfcontact_source_root: Path
    selfcontact_commit: str = Field(pattern="^[0-9a-f]{40}$")
    selfcontact_essentials_root: Path
    selfcontact_essentials_registry: Path
    selfcontact_essentials_registry_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    trusted_selfcontact_assets: bool
    selfcontact_test_segments: bool
    winding_query_chunk_size: int = Field(default=1000, gt=0)
    distance_target_chunk_size: int = Field(default=1000, gt=0)
    separation_margin_m: float = Field(gt=0)
    sigma_distance_m: float = Field(gt=0)
    sigma_normal: float = Field(gt=0)
    sigma_velocity_m_per_s: float = Field(gt=0)
    normal_weight: float = Field(ge=0)
    hold_velocity_weight: float = Field(ge=0)
    penetration_area_weight: float = Field(ge=0)


class RuntimeSemiMarkov(_Strict):
    max_duration: int = Field(gt=0)


class RuntimeContact(_Strict):
    checkpoint: Path
    training_config: Path
    semi_markov: RuntimeSemiMarkov


class RuntimeDiffusion(_Strict):
    checkpoint: Path
    training_config: Path
    trajectory_normalizer: Path
    dposer_registry: Path
    train_steps: int = Field(gt=0)


class RuntimeGuidance(_Strict):
    enabled_terms: tuple[Literal["keypoint", "silhouette", "track", "depth", "contact"], ...]
    guidance_scale: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    trust_region_norm: float = Field(gt=0)
    keypoint_sigma_min_px: float = Field(gt=0)
    keypoint_sigma_occluded_px: float = Field(gt=0)
    depth_temperature_m: float = Field(default=0.02, gt=0)
    silhouette_sigma_px: float = Field(default=2.0, gt=0)


class RuntimeInference(_Strict):
    rounds: int = Field(gt=0)
    diffusion_steps: int = Field(gt=0)
    num_hypotheses: int = Field(gt=0)
    retry_guidance_factor: float | None = Field(default=None, ge=0, lt=1)
    fixed_rounds: bool
    alternating: bool

    @model_validator(mode="after")
    def fixed_protocol(self) -> RuntimeInference:
        if not self.fixed_rounds:
            raise ValueError("v1 runtime requires fixed rounds")
        if not self.alternating and self.rounds != 1:
            raise ValueError("single-pass runtime requires one round")
        return self


class RuntimeRanking(_Strict):
    artifact: Path
    fit_split: Literal["validation"]
    use_ground_truth: Literal[False]


class RuntimeEvaluation(_Strict):
    primary_endpoint: Literal["root_aligned_hand_pve"]
    bootstrap_unit: Literal["signer"]


class ReconstructionConfig(_Strict):
    experiment: RuntimeExperiment
    data: RuntimeData
    observation: RuntimeObservation
    initialization: RuntimeInitialization
    geometry: RuntimeGeometry
    contact: RuntimeContact
    diffusion: RuntimeDiffusion
    guidance: RuntimeGuidance
    inference: RuntimeInference
    ranking: RuntimeRanking
    evaluation: RuntimeEvaluation
    third_party_manifest: Path


def _move_state(state: TrajectoryState, device: torch.device) -> TrajectoryState:
    return replace(
        state,
        **{
            name: value.to(device)
            for name in state.__dataclass_fields__
            if isinstance((value := getattr(state, name)), torch.Tensor)
        },
    ).validate()


def _move_observations(observations: ObservationBatch, device: torch.device) -> ObservationBatch:
    return replace(
        observations,
        **{
            name: value.to(device)
            for name in observations.__dataclass_fields__
            if isinstance((value := getattr(observations, name)), torch.Tensor)
        },
    ).validate()


def _move_camera(camera: CameraTrajectory, device: torch.device) -> CameraTrajectory:
    return replace(
        camera,
        intrinsics=camera.intrinsics.to(device),
        world_to_camera=camera.world_to_camera.to(device),
        image_size_wh=camera.image_size_wh.to(device),
        valid_mask=camera.valid_mask.to(device),
    ).validate()


@dataclass(frozen=True)
class ClipInputs:
    trajectory: TrajectoryState
    camera: CameraTrajectory
    initialization_metadata: dict[str, object]
    observations: ObservationBatch
    observation_hashes: dict[str, str]


class ReconstructionRuntime:
    """Load all frozen assets once, then reconstruct independent clips."""

    def __init__(
        self,
        config: ReconstructionConfig,
        *,
        device: str,
        body_model_factory: Callable[[RuntimeGeometry, torch.device], nn.Module] | None = None,
        penetration_factory: Callable[[RuntimeGeometry, torch.device], nn.Module] | None = None,
        bridge_factory: Callable[[DiffusionTrainingConfig, torch.device], nn.Module] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if config.experiment.deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.experiment.seed)
        self.patch_map = PatchMap.load(config.geometry.patch_map)
        if self.patch_map.development_only and not config.experiment.development_only:
            raise PermissionError("development patch map cannot enter production runtime")
        if body_model_factory is None:
            self.body_model = SMPLXAdapter(
                config.geometry.smplx_model,
                expected_sha256=config.geometry.smplx_model_sha256,
                trusted_model=config.geometry.trusted_smplx_model,
            ).to(self.device)
            faces = self.body_model.model.faces_tensor.long().to(self.device)
        else:
            self.body_model = body_model_factory(config.geometry, self.device).to(self.device)
            faces = self.body_model.faces_tensor.long().to(self.device)
        if penetration_factory is None:
            penetration = OfficialSelfContactPenetration(
                config.geometry.selfcontact_essentials_root,
                source_root=config.geometry.selfcontact_source_root,
                expected_commit=config.geometry.selfcontact_commit,
                registry=config.geometry.selfcontact_essentials_registry,
                expected_registry_sha256=config.geometry.selfcontact_essentials_registry_sha256,
                trusted_licensed_assets=config.geometry.trusted_selfcontact_assets,
                test_segments=config.geometry.selfcontact_test_segments,
                winding_query_chunk_size=config.geometry.winding_query_chunk_size,
                distance_target_chunk_size=config.geometry.distance_target_chunk_size,
            ).to(self.device)
        else:
            penetration = penetration_factory(config.geometry, self.device).to(self.device)
        self.contact_geometry = ContactGeometry(
            self.patch_map,
            fps=1.0,
            separation_margin=config.geometry.separation_margin_m,
            sigma_distance=config.geometry.sigma_distance_m,
            sigma_normal=config.geometry.sigma_normal,
            sigma_velocity=config.geometry.sigma_velocity_m_per_s,
            normal_weight=config.geometry.normal_weight,
            hold_velocity_weight=config.geometry.hold_velocity_weight,
            penetration_area_weight=config.geometry.penetration_area_weight,
        ).to(self.device)
        self.state_geometry = StateContactGeometry(
            self.body_model, self.contact_geometry, faces, penetration
        ).to(self.device)
        edge_names = self.patch_map.admissible_edges
        contact_config = ContactTrainingConfig.model_validate(
            yaml.safe_load(config.contact.training_config.read_text("utf-8"))
        )
        if contact_config.model.max_duration != config.contact.semi_markov.max_duration:
            raise ValueError("training/inference contact duration mismatch")
        trajectory_width = sum(DPoserXConditionedTrajectoryDenoiser.PRODUCTION_PART_DIMS)
        self.proposal = ContactProposal(
            trajectory_width,
            len(edge_names),
            config.contact.semi_markov.max_duration,
            hidden_dim=contact_config.model.hidden_dim,
            heads=contact_config.model.heads,
            layers=contact_config.model.layers,
            dropout=contact_config.model.dropout,
            edge_names=edge_names,
        ).to(self.device)
        load_model_checkpoint(
            config.contact.checkpoint,
            self.proposal,
            expected_stage="contact_proposal",
            expected_config_sha256=file_sha256(config.contact.training_config),
            allow_development=config.experiment.development_only,
        )
        diffusion_config = DiffusionTrainingConfig.model_validate(
            yaml.safe_load(config.diffusion.training_config.read_text("utf-8"))
        )
        if diffusion_config.optimization.train_steps != config.diffusion.train_steps:
            raise ValueError("training/inference diffusion train-step identity mismatch")
        if file_sha256(config.diffusion.dposer_registry) != file_sha256(
            diffusion_config.dposer_x.registry
        ):
            raise ValueError("inference/training DPoser-X registry mismatch")
        if bridge_factory is None:
            bridge = OfficialDPoserXBridge(
                source_root=diffusion_config.dposer_x.source_root,
                runtime_root=diffusion_config.dposer_x.runtime_root,
                registry_path=config.diffusion.dposer_registry,
                expected_commit=diffusion_config.dposer_x.expected_commit,
                device=self.device,
            )
        else:
            bridge = bridge_factory(diffusion_config, self.device)
        denoiser = DPoserXConditionedTrajectoryDenoiser(
            bridge,
            trajectory_steps=diffusion_config.diffusion.steps,
            hidden_dim=diffusion_config.model.hidden_dim,
            heads=diffusion_config.model.heads,
            layers=diffusion_config.model.layers,
            dropout=diffusion_config.model.dropout,
        ).to(self.device)
        token_encoder = ContactTokenEncoder(
            len(edge_names), diffusion_config.model.hidden_dim, edge_names
        ).to(self.device)
        diffusion_model = nn.ModuleDict(
            {"denoiser": denoiser, "contact_token_encoder": token_encoder}
        )
        diffusion_metadata = load_model_checkpoint(
            config.diffusion.checkpoint,
            diffusion_model,
            expected_stage="trajectory_diffusion",
            expected_config_sha256=file_sha256(config.diffusion.training_config),
            allow_development=config.experiment.development_only,
        )
        if diffusion_metadata["asset_sha256"].get("trajectory_normalizer") != file_sha256(
            config.diffusion.trajectory_normalizer
        ):
            raise ValueError("trajectory normalizer/checkpoint mismatch")
        self.codec = StateCodec.from_payload(
            json.loads(config.diffusion.trajectory_normalizer.read_text("utf-8"))
        )
        self.denoiser = denoiser.eval()
        self.token_encoder = token_encoder.eval()
        self.schedule = DiffusionSchedule(
            diffusion_config.diffusion.steps,
            diffusion_config.diffusion.beta_start,
            diffusion_config.diffusion.beta_end,
        )
        self.conditioning_mode = diffusion_config.diffusion.conditioning_mode
        weights, self.ranker_payload = load_frozen_ranker(
            config.ranking.artifact,
            allow_development=config.experiment.development_only,
        )
        self.ranking_weights = weights
        camera_payload = json.loads(config.observation.camera_calibration.read_text("utf-8"))
        if camera_payload.get("schema_version") != "dcg_camera_projection_v1":
            raise ValueError("unknown camera projection schema")
        if (
            camera_payload.get("scientific_status") != "FROZEN"
            and not config.experiment.development_only
        ):
            raise PermissionError("camera projection asset is not frozen")
        self.camera_payload = camera_payload
        self.keypoint_joint_indices = torch.tensor(
            camera_payload["keypoint_joint_indices"], dtype=torch.long, device=self.device
        )
        self.keypoint_supported_mask = torch.tensor(
            camera_payload.get(
                "keypoint_supported_mask",
                [True] * len(camera_payload["keypoint_joint_indices"]),
            ),
            dtype=torch.bool,
            device=self.device,
        )
        if self.keypoint_supported_mask.shape != self.keypoint_joint_indices.shape:
            raise ValueError("camera keypoint support mask shape mismatch")
        self.observation_index = self._load_observation_index()

    def _load_observation_index(self) -> dict[str, dict[str, object]]:
        root = self.config.observation.artifact_root
        if not (root / "CALIBRATED_OBSERVATIONS_COMPLETE").is_file():
            raise ValueError("calibrated observation root is incomplete")
        payload = json.loads((root / "index.json").read_text("utf-8"))
        identity = payload.pop("index_identity_sha256", None)
        from dcg_sign4d.utils.hashing import canonical_hash

        if identity != canonical_hash(payload):
            raise ValueError("calibrated observation index identity mismatch")
        if payload.get("calibrator_sha256") != file_sha256(
            self.config.observation.calibration_artifact
        ):
            raise ValueError("observation index/calibrator mismatch")
        if payload.get("development_only") and not self.config.experiment.development_only:
            raise PermissionError("development observations cannot enter production runtime")
        return {row["clip_id"]: row for row in payload["per_clip"]}

    def load_clip(self, clip_id: str) -> ClipInputs:
        if clip_id not in self.observation_index:
            raise KeyError(f"clip missing from calibrated observation index: {clip_id}")
        row = self.observation_index[clip_id]
        cache_root = self.config.observation.artifact_root / "caches" / str(row["cache_id"])
        observations = ObservationCache(cache_root.parent).load(str(row["cache_id"]))
        support = self.keypoint_supported_mask.cpu()[None, None]
        observations = replace(
            observations,
            keypoint_valid=observations.keypoint_valid & support,
            keypoint_reliability=torch.where(
                support,
                observations.keypoint_reliability,
                torch.zeros_like(observations.keypoint_reliability),
            ),
        ).validate()
        trajectory, camera, metadata = load_initialization_artifact(
            self.config.initialization.artifact_root / clip_id
        )
        if metadata.get("dexavatar_commit") != self.config.initialization.commit:
            raise ValueError("initialization DexAvatar commit mismatch")
        if camera.coordinate_convention != self.camera_payload["coordinate_convention"]:
            raise ValueError("initialization/camera projection convention mismatch")
        if trajectory.valid_mask.shape != observations.frame_valid.shape:
            raise ValueError("initialization/observation frame mismatch")
        source_hashes = json.loads(
            (self.config.initialization.artifact_root / clip_id / "source_hashes.json").read_text(
                "utf-8"
            )
        )
        metadata = {**metadata, "source_hashes": source_hashes}
        observation_hashes = {
            "observations": file_sha256(cache_root / "observations.npz"),
            "observation_identity": file_sha256(cache_root / "identity.json"),
            "calibrator": file_sha256(self.config.observation.calibration_artifact),
        }
        return ClipInputs(
            _move_state(trajectory, self.device),
            _move_camera(camera, self.device),
            metadata,
            _move_observations(observations, self.device),
            observation_hashes,
        )

    def reconstruct_clip(self, clip_id: str) -> tuple[ClipInputs, list[Hypothesis]]:
        inputs = self.load_clip(clip_id)
        time = inputs.trajectory.valid_mask.shape[1]
        fps = float(inputs.observations.metadata[0]["preprocessing"]["fps_effective"])
        self.contact_geometry.fps = fps
        if time <= self.config.data.window_length:
            reconstructor = self._build_reconstructor(
                inputs, fps=fps, base_seed=self.config.experiment.seed
            )
            return inputs, reconstructor.reconstruct(inputs.trajectory, inputs.observations)
        return inputs, self._reconstruct_windows(inputs, fps=fps)

    def _build_reconstructor(
        self, inputs: ClipInputs, *, fps: float, base_seed: int
    ) -> AlternatingReconstructor:
        projector = StateJointProjector(
            self.body_model, inputs.camera, self.keypoint_joint_indices
        ).to(self.device)
        observation_terms = []
        guidance_terms = []
        optional_presence = {
            "silhouette": inputs.observations.part_masks is not None,
            "track": inputs.observations.tracks_2d is not None,
            "depth": inputs.observations.depth_order is not None,
        }
        ignored = [
            name
            for name, present in optional_presence.items()
            if present and name not in self.config.guidance.enabled_terms
        ]
        if ignored:
            raise ValueError(f"available observation cues are disabled: {ignored}")
        if "keypoint" in self.config.guidance.enabled_terms:
            keypoint = KeypointGuidance(
                projector,
                sigma_min=self.config.guidance.keypoint_sigma_min_px,
                sigma_occ=self.config.guidance.keypoint_sigma_occluded_px,
            )
            observation_terms.append(keypoint)
            guidance_terms.append(keypoint)
        if optional_presence["track"]:
            indices = self.camera_payload.get("track_joint_indices")
            if indices is None:
                raise ValueError("track observations require a frozen track_joint_indices map")
            track_projector = StateJointProjector(
                self.body_model,
                inputs.camera,
                torch.tensor(indices, dtype=torch.long, device=self.device),
            ).to(self.device)
            track = TrackGuidance(track_projector)
            observation_terms.append(track)
            guidance_terms.append(track)
        if optional_presence["depth"]:
            pairs = self.camera_payload.get("depth_joint_pairs")
            if pairs is None:
                raise ValueError("depth observations require a frozen depth_joint_pairs map")
            depth_projector = StateJointDepthDifference(
                self.body_model,
                inputs.camera,
                torch.tensor(pairs, dtype=torch.long, device=self.device),
            ).to(self.device)
            depth = RelativeDepthGuidance(
                depth_projector,
                temperature_m=self.config.guidance.depth_temperature_m,
            )
            observation_terms.append(depth)
            guidance_terms.append(depth)
        if optional_presence["silhouette"]:
            vertex_groups = self.camera_payload.get("silhouette_vertex_groups")
            if vertex_groups is None:
                raise ValueError("part-mask observations require frozen silhouette_vertex_groups")
            masks = inputs.observations.part_masks
            renderer = StatePartMaskRenderer(
                self.body_model,
                inputs.camera,
                tuple(
                    torch.tensor(group, dtype=torch.long, device=self.device)
                    for group in vertex_groups
                ),
                (masks.shape[-2], masks.shape[-1]),
                sigma_px=self.config.guidance.silhouette_sigma_px,
            ).to(self.device)
            silhouette = SilhouetteGuidance(renderer)
            observation_terms.append(silhouette)
            guidance_terms.append(silhouette)
        if "contact" in self.config.guidance.enabled_terms:
            guidance_terms.append(ContactGuidance(self.contact_geometry, self.state_geometry))
        if not observation_terms:
            raise ValueError("production ranking requires at least one observation guidance term")

        def observation_score(state, observations, graph):
            return -sum(term.loss(state, observations, graph) for term in observation_terms)

        ranker = HypothesisRanker(
            self.ranking_weights,
            observation_score=observation_score,
        )
        sampler = GuidedTrajectorySampler(
            self.denoiser,
            self.schedule,
            self.codec,
            self.token_encoder,
            tuple(guidance_terms),
            guidance_scale=self.config.guidance.guidance_scale,
            gradient_clip_norm=self.config.guidance.gradient_clip_norm,
            trust_region_norm=self.config.guidance.trust_region_norm,
            conditioning_mode=self.conditioning_mode,
        )
        reconstructor = AlternatingReconstructor(
            self.proposal,
            SemiMarkovDecoder(self.config.contact.semi_markov.max_duration, fps=fps),
            sampler,
            self.state_geometry,
            ranker,
            torch.ones(
                inputs.trajectory.valid_mask.shape[0],
                len(self.patch_map.admissible_edges),
                dtype=torch.bool,
                device=self.device,
            ),
            rounds=self.config.inference.rounds,
            diffusion_steps=self.config.inference.diffusion_steps,
            num_hypotheses=self.config.inference.num_hypotheses,
            base_seed=base_seed,
            retry_guidance_factor=self.config.inference.retry_guidance_factor,
            alternating=self.config.inference.alternating,
        )
        return reconstructor

    def _reconstruct_windows(self, inputs: ClipInputs, *, fps: float) -> list[Hypothesis]:
        """Infer overlapping windows, stitch on SO(3), then rerank full clips."""

        total_time = inputs.trajectory.valid_mask.shape[1]
        overlap = self.config.data.window_overlap
        starts = window_starts(total_time, self.config.data.window_length, overlap)
        window_inputs: list[ClipInputs] = []
        window_results: list[dict[int, Hypothesis]] = []
        for window_index, start in enumerate(starts):
            end = min(start + self.config.data.window_length, total_time)
            current = replace(
                inputs,
                trajectory=slice_trajectory(inputs.trajectory, start, end),
                camera=slice_camera(inputs.camera, start, end),
                observations=slice_observations(inputs.observations, start, end),
            )
            # A window-specific salt prevents repeated local noise while the
            # published clip-level hypothesis seed remains stable.
            local_seed = self.config.experiment.seed + (window_index + 1) * 104_729
            reconstructor = self._build_reconstructor(current, fps=fps, base_seed=local_seed)
            results = reconstructor.reconstruct(current.trajectory, current.observations)
            window_inputs.append(current)
            window_results.append({item.identifier: item for item in results})

        global_reconstructor = self._build_reconstructor(
            inputs, fps=fps, base_seed=self.config.experiment.seed
        )
        decoder = SemiMarkovDecoder(self.config.contact.semi_markov.max_duration, fps=fps)
        hypotheses: list[Hypothesis] = []
        for identifier in range(self.config.inference.num_hypotheses):
            local = [result[identifier] for result in window_results]
            seed = AlternatingReconstructor.derive_seed(self.config.experiment.seed, identifier)
            retry_count = sum(item.retry_count for item in local)
            window_diagnostics = [
                {
                    "start": start,
                    "end": start + window_inputs[index].trajectory.valid_mask.shape[1],
                    "local_seed": item.seed,
                    "status": item.status,
                    "diagnostics": item.diagnostics,
                }
                for index, (start, item) in enumerate(zip(starts, local, strict=True))
            ]
            if any(item.status != "ok" for item in local):
                # The artifact contract defines fallback as the exact complete
                # initializer, never a mixture of inferred and failed windows.
                state = inputs.trajectory
                graph, geometry = global_reconstructor._infer_graph(inputs.observations, state)
                terms = global_reconstructor.ranker.terms(
                    state, graph, inputs.observations, geometry
                )
                hypotheses.append(
                    Hypothesis(
                        identifier=identifier,
                        seed=seed,
                        trajectory=state,
                        graph=graph,
                        score=global_reconstructor.ranker.score(terms),
                        ranking_terms=terms,
                        diagnostics={"windows": window_diagnostics},
                        status="fallback_initialization",
                        retry_count=retry_count,
                    )
                )
                continue

            state = stitch_trajectories(
                [item.trajectory for item in local],
                starts,
                total_time=total_time,
                overlap=overlap,
            )
            graph = stitch_contact_graphs(
                [item.graph for item in local],
                starts,
                total_time=total_time,
                overlap=overlap,
                decoder=decoder,
                frame_valid=state.valid_mask,
            )
            geometry = self.state_geometry(state)
            terms = global_reconstructor.ranker.terms(state, graph, inputs.observations, geometry)
            rounds: list[RoundResult] = []
            for round_index in range(self.config.inference.rounds):
                round_state = stitch_trajectories(
                    [item.rounds[round_index].trajectory for item in local],
                    starts,
                    total_time=total_time,
                    overlap=overlap,
                )
                round_graph = stitch_contact_graphs(
                    [item.rounds[round_index].graph for item in local],
                    starts,
                    total_time=total_time,
                    overlap=overlap,
                    decoder=decoder,
                    frame_valid=round_state.valid_mask,
                )
                round_geometry = self.state_geometry(round_state)
                objective = global_reconstructor.ranker.terms(
                    round_state, round_graph, inputs.observations, round_geometry
                )
                rounds.append(
                    RoundResult(
                        round_index=round_index,
                        trajectory=round_state,
                        graph=round_graph,
                        diagnostics={
                            "windows": [item.rounds[round_index].diagnostics for item in local]
                        },
                        runtime_objective=objective,
                    )
                )
            hypotheses.append(
                Hypothesis(
                    identifier=identifier,
                    seed=seed,
                    trajectory=state,
                    graph=graph,
                    score=global_reconstructor.ranker.score(terms),
                    ranking_terms=terms,
                    diagnostics={"windows": window_diagnostics},
                    rounds=tuple(rounds),
                    retry_count=retry_count,
                )
            )
        return sorted(hypotheses, key=lambda item: (-item.score, item.identifier))
