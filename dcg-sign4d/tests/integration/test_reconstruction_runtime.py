from __future__ import annotations

import json

import torch
import yaml
from torch import nn

from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.dposer_normalizer import DPoserXWholeBodyNormalizer, ZScoreStats
from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.diffusion.trajectory_denoiser import DPoserXConditionedTrajectoryDenoiser
from dcg_sign4d.geometry.penetration import PenetrationOutput
from dcg_sign4d.geometry.smplx_adapter import SMPLXForwardOutput
from dcg_sign4d.inference.ranker_fit import load_frozen_ranker
from dcg_sign4d.inference.runtime import ReconstructionConfig, ReconstructionRuntime
from dcg_sign4d.initialization.artifact import save_initialization_artifact
from dcg_sign4d.initialization.camera import CameraTrajectory
from dcg_sign4d.observations.cache import ObservationCache
from dcg_sign4d.observations.schema import ObservationBatch
from dcg_sign4d.training.checkpoint import CheckpointMetadata, save_model_checkpoint
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


class _FakeBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "faces_tensor",
            torch.tensor(
                [[0, 1, 4], [1, 2, 4], [2, 3, 5], [4, 5, 6], [5, 6, 7]],
                dtype=torch.long,
            ),
        )

    def forward(self, state):
        batch, time = state.valid_mask.shape
        base = torch.zeros(8, 3, device=state.root_translation.device)
        base[:, 0] = torch.arange(8, device=base.device) * 0.01
        base[:, 2] = 5
        vertices = base[None, None].expand(batch, time, -1, -1).clone()
        vertices = vertices + state.root_translation[:, :, None]
        vertices[:, :, 0:2, 1] += state.left_hand_rot6d[:, :, :1, 0]
        vertices[:, :, 2:4, 1] += state.right_hand_rot6d[:, :, :1, 0]
        vertices[:, :, :, 0] += state.beta[:, None, :1]
        return SMPLXForwardOutput(vertices, vertices[:, :, :3])


class _FakePenetration(nn.Module):
    def forward(self, vertices, patch_map):
        shape = (*vertices.shape[:2], len(patch_map.admissible_edges))
        zeros = vertices.new_zeros(shape)
        return PenetrationOutput(zeros, zeros, zeros)


class _FakeBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("frozen", torch.tensor(1.0))
        self.normalizer = DPoserXWholeBodyNormalizer(
            {
                name: ZScoreStats(torch.zeros(size), torch.ones(size))
                for name, size in DPoserXWholeBodyNormalizer.PARTS
            }
        )

    def predict_noise(self, normalized, timesteps, *, trajectory_steps):
        del timesteps, trajectory_steps
        return normalized * self.frozen


def _state(time=3):
    generator = torch.Generator().manual_seed(44)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.05

    return TrajectoryState(
        root_rot6d=rand(1, time, 6),
        root_translation=rand(1, time, 3),
        root_velocity=rand(1, time, 3),
        body_rot6d=rand(1, time, 21, 6),
        left_hand_rot6d=rand(1, time, 15, 6),
        right_hand_rot6d=rand(1, time, 15, 6),
        face_state=rand(1, time, 19),
        beta=rand(1, 10),
        valid_mask=torch.ones(1, time, dtype=torch.bool),
    ).validate()


def _dependencies(path):
    payload = yaml.safe_load(path.read_text("utf-8"))
    return {row["name"]: row["commit"] for row in payload["repositories"]}


def test_frozen_component_assembly_runs_one_real_runtime_round(tmp_path):
    root = tmp_path
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    third_party_manifest = repository_root / "third_party/manifest.yaml"
    dependencies = _dependencies(third_party_manifest)
    patch_map = repository_root / "assets/patch_maps/synthetic_smoke.yaml"
    edge_names = (("left_tip", "right_tip"), ("left_tip", "face"))
    state = _state(time=5)
    digest = "1" * 64

    contact_config = {
        "experiment": {"seed": 2, "development_only": True},
        "data": {"train_bundle": "unused", "validation_bundle": "unused"},
        "model": {
            "hidden_dim": 16,
            "heads": 4,
            "layers": 1,
            "dropout": 0.0,
            "max_duration": 3,
        },
        "optimization": {
            "steps": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0,
            "gradient_clip_norm": 1,
            "validation_interval": 1,
        },
        "sampling": {"balanced": True, "effective_number_beta": 0.9},
        "objective": {
            "event_weight": 1,
            "duration_weight": 1,
            "transition_weight": 1,
            "calibration_weight": 1,
            "gold_label_weight": 1,
            "accepted_pseudo_label_weight": 0.5,
        },
        "third_party_manifest": str(third_party_manifest),
    }
    contact_config_path = root / "contact.yaml"
    contact_config_path.write_text(yaml.safe_dump(contact_config), "utf-8")
    contact_model = ContactProposal(
        337, 2, 3, hidden_dim=16, heads=4, layers=1, edge_names=edge_names
    )
    contact_checkpoint = root / "contact_checkpoint"
    save_model_checkpoint(
        contact_checkpoint,
        contact_model,
        CheckpointMetadata(
            stage="contact_proposal",
            model_class=f"{type(contact_model).__module__}.{type(contact_model).__qualname__}",
            step=1,
            epoch=0,
            seed=2,
            config_sha256=file_sha256(contact_config_path),
            manifest_sha256=digest,
            dependency_commits=dependencies,
            metrics={"validation": 1.0},
            development_only=True,
        ),
    )

    registry = root / "dposer_registry.json"
    registry.write_text(json.dumps({"fixture": True}), "utf-8")
    diffusion_config = {
        "experiment": {"seed": 2, "development_only": True},
        "data": {"train_bundle": "unused", "validation_bundle": "unused"},
        "dposer_x": {
            "source_root": "unused",
            "runtime_root": "unused",
            "registry": str(registry),
            "expected_commit": "2" * 40,
        },
        "model": {"hidden_dim": 16, "heads": 4, "layers": 1, "dropout": 0.0},
        "diffusion": {
            "steps": 3,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "conditioning_mode": "dynamic",
            "graph_dropout_probability": 0,
            "edge_dropout_probability": 0,
            "reliability_dropout_probability": 0,
            "root_weight": 1,
            "body_weight": 1,
            "left_hand_weight": 1,
            "right_hand_weight": 1,
            "face_weight": 1,
        },
        "optimization": {
            "train_steps": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0,
            "gradient_clip_norm": 1,
            "validation_interval": 1,
        },
        "third_party_manifest": str(third_party_manifest),
    }
    diffusion_config_path = root / "diffusion.yaml"
    diffusion_config_path.write_text(yaml.safe_dump(diffusion_config), "utf-8")
    codec = StateCodec.fit(state)
    normalizer = root / "normalizer.json"
    normalizer.write_text(json.dumps(codec.to_payload()), "utf-8")
    bridge = _FakeBridge()
    denoiser = DPoserXConditionedTrajectoryDenoiser(
        bridge, trajectory_steps=3, hidden_dim=16, heads=4, layers=1
    )
    token_encoder = ContactTokenEncoder(2, 16, edge_names)
    diffusion_model = nn.ModuleDict({"denoiser": denoiser, "contact_token_encoder": token_encoder})
    diffusion_checkpoint = root / "diffusion_checkpoint"
    save_model_checkpoint(
        diffusion_checkpoint,
        diffusion_model,
        CheckpointMetadata(
            stage="trajectory_diffusion",
            model_class=f"{type(diffusion_model).__module__}.{type(diffusion_model).__qualname__}",
            step=1,
            epoch=0,
            seed=2,
            config_sha256=file_sha256(diffusion_config_path),
            manifest_sha256=digest,
            dependency_commits=dependencies,
            metrics={"validation": 1.0},
            development_only=True,
            asset_sha256={
                "dposer_x_registry": file_sha256(registry),
                "trajectory_normalizer": file_sha256(normalizer),
            },
        ),
        state_scope="trainable",
    )

    ranker = root / "ranker.json"
    ranker_payload = {
        "schema_version": "dcg_ranker_v1",
        "development_only": True,
        "fit_split": "validation",
        "use_ground_truth": False,
        "gate_status": "PASS",
        "weights": {"observation": 1, "contact": 1, "event": 1, "motion": 1},
    }
    ranker_payload["artifact_identity_sha256"] = canonical_hash(ranker_payload)
    ranker.write_text(json.dumps(ranker_payload), "utf-8")
    load_frozen_ranker(ranker, allow_development=True)

    observation_root = root / "observations"
    cache = ObservationCache(observation_root / "caches")
    observations = ObservationBatch(
        keypoints_2d=torch.zeros(1, 5, 3, 2),
        keypoint_reliability=torch.ones(1, 5, 3),
        keypoint_valid=torch.ones(1, 5, 3, dtype=torch.bool),
        frame_valid=torch.ones(1, 5, dtype=torch.bool),
        metadata=(
            {
                "clip_id": "clip",
                "development_only": True,
                "frame_ids": [0, 1, 2, 3, 4],
                "timestamps_sec": [index / 30 for index in range(5)],
                "preprocessing": {"fps_effective": 30},
            },
        ),
    ).validate()
    cache.save("cache", observations)
    calibrator = root / "calibrator.json"
    calibrator.write_text(json.dumps({"fixture": True}), "utf-8")
    observation_index = {
        "schema_version": "dcg_calibrated_observation_index_v1",
        "development_only": True,
        "calibrator_sha256": file_sha256(calibrator),
        "per_clip": [{"clip_id": "clip", "cache_id": "cache"}],
    }
    observation_index["index_identity_sha256"] = canonical_hash(observation_index)
    (observation_root / "index.json").write_text(json.dumps(observation_index), "utf-8")
    (observation_root / "CALIBRATED_OBSERVATIONS_COMPLETE").write_text("complete\n", "utf-8")

    initialization_root = root / "initialization"
    intrinsics = torch.eye(3)[None, None].expand(1, 5, 3, 3).clone()
    intrinsics[..., 0, 0] = 100
    intrinsics[..., 1, 1] = 100
    camera = CameraTrajectory(
        intrinsics,
        torch.eye(4)[None, None].expand(1, 5, 4, 4).clone(),
        torch.tensor([640.0, 480.0])[None, None].expand(1, 5, 2).clone(),
        state.valid_mask,
        "fixture_camera",
    )
    save_initialization_artifact(
        initialization_root / "clip",
        state,
        camera,
        metadata={
            "clip_id": "clip",
            "dexavatar_commit": "3" * 40,
            "config_sha256": digest,
            "checkpoint_sha256": digest,
            "runtime": {},
            "development_only": True,
        },
        source_hashes={"video": "4" * 64},
    )
    camera_config = root / "camera.json"
    camera_config.write_text(
        json.dumps(
            {
                "schema_version": "dcg_camera_projection_v1",
                "scientific_status": "DEVELOPMENT",
                "coordinate_convention": "fixture_camera",
                "keypoint_joint_indices": [0, 1, 2],
            }
        ),
        "utf-8",
    )
    dummy = root / "dummy"
    dummy.write_bytes(b"dummy")
    config = ReconstructionConfig.model_validate(
        {
            "experiment": {
                "name": "fixture",
                "seed": 2,
                "deterministic": True,
                "development_only": True,
            },
            "data": {"window_length": 3, "window_overlap": 1},
            "observation": {
                "artifact_root": str(observation_root),
                "calibration_artifact": str(calibrator),
                "camera_calibration": str(camera_config),
            },
            "initialization": {
                "backend": "artifact_replay",
                "artifact_root": str(initialization_root),
                "commit": "3" * 40,
            },
            "geometry": {
                "patch_map": str(patch_map),
                "smplx_model": str(dummy),
                "smplx_model_sha256": file_sha256(dummy),
                "trusted_smplx_model": False,
                "selfcontact_source_root": str(root),
                "selfcontact_commit": "5" * 40,
                "selfcontact_essentials_root": str(root),
                "selfcontact_essentials_registry": str(dummy),
                "selfcontact_essentials_registry_sha256": file_sha256(dummy),
                "trusted_selfcontact_assets": False,
                "selfcontact_test_segments": False,
                "separation_margin_m": 0.03,
                "sigma_distance_m": 0.01,
                "sigma_normal": 0.25,
                "sigma_velocity_m_per_s": 0.1,
                "normal_weight": 1,
                "hold_velocity_weight": 1,
                "penetration_area_weight": 1,
            },
            "contact": {
                "checkpoint": str(contact_checkpoint),
                "training_config": str(contact_config_path),
                "semi_markov": {"max_duration": 3},
            },
            "diffusion": {
                "checkpoint": str(diffusion_checkpoint),
                "training_config": str(diffusion_config_path),
                "trajectory_normalizer": str(normalizer),
                "dposer_registry": str(registry),
                "train_steps": 1,
            },
            "guidance": {
                "enabled_terms": ["keypoint", "contact"],
                "guidance_scale": 0.01,
                "gradient_clip_norm": 1,
                "trust_region_norm": 10,
                "keypoint_sigma_min_px": 1,
                "keypoint_sigma_occluded_px": 20,
            },
            "inference": {
                "rounds": 1,
                "diffusion_steps": 2,
                "num_hypotheses": 1,
                "retry_guidance_factor": 0.5,
                "fixed_rounds": True,
                "alternating": True,
            },
            "ranking": {
                "artifact": str(ranker),
                "fit_split": "validation",
                "use_ground_truth": False,
            },
            "evaluation": {
                "primary_endpoint": "root_aligned_hand_pve",
                "bootstrap_unit": "signer",
            },
            "third_party_manifest": str(third_party_manifest),
        }
    )
    runtime = ReconstructionRuntime(
        config,
        device="cpu",
        body_model_factory=lambda *_: _FakeBody(),
        penetration_factory=lambda *_: _FakePenetration(),
        bridge_factory=lambda *_: _FakeBridge(),
    )
    inputs, hypotheses = runtime.reconstruct_clip("clip")
    assert inputs.trajectory.root_rot6d.shape == (1, 5, 6)
    assert len(hypotheses) == 1
    assert hypotheses[0].status == "ok"
    assert len(hypotheses[0].rounds) == 1
    assert len(hypotheses[0].diagnostics["windows"]) == 2
    assert torch.isfinite(hypotheses[0].trajectory.root_translation).all()
