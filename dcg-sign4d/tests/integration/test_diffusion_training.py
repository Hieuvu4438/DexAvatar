from pathlib import Path

import torch
import yaml

from dcg_sign4d.cli.train_diffusion import train
from dcg_sign4d.diffusion.dposer_normalizer import DPoserXWholeBodyNormalizer, ZScoreStats
from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.synthetic import make_graph, make_observations
from dcg_sign4d.training.batch import (
    SupervisedWindowBatch,
    SupervisedWindowMetadata,
    save_supervised_windows,
)
from dcg_sign4d.utils.hashing import canonical_hash

ROOT = Path(__file__).resolve().parents[2]


class FakeBridge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.normalizer = DPoserXWholeBodyNormalizer(
            {
                name: ZScoreStats(torch.zeros(size), torch.ones(size))
                for name, size in DPoserXWholeBodyNormalizer.PARTS
            }
        )

    def predict_noise(self, normalized, timesteps, *, trajectory_steps):
        return normalized * self.scale + timesteps[:, None] / trajectory_steps


def state(batch=2, time=4):
    generator = torch.Generator().manual_seed(9)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    return TrajectoryState(
        root_rot6d=rand(batch, time, 6),
        root_translation=rand(batch, time, 3),
        root_velocity=rand(batch, time, 3),
        body_rot6d=rand(batch, time, 21, 6),
        left_hand_rot6d=rand(batch, time, 15, 6),
        right_hand_rot6d=rand(batch, time, 15, 6),
        face_state=rand(batch, time, 19),
        beta=rand(batch, 10),
        valid_mask=torch.ones(batch, time, dtype=torch.bool),
    ).validate()


def bundle(path, split):
    trajectory = state()
    graph = make_graph(batch=2, time=4, edges=2)
    width = StateCodec().encode(trajectory)[0].shape[-1]
    digest = canonical_hash({"diffusion_fixture": True})
    return save_supervised_windows(
        path,
        SupervisedWindowBatch(
            trajectory=trajectory,
            observations=make_observations(batch=2, time=4),
            geometry_features=torch.randn(2, 4, 2, 5),
            graph=graph,
            duration_frames=torch.ones(2, 4, 2, dtype=torch.long),
            supervision_mask=torch.ones(2, 4, width, dtype=torch.bool),
            metadata=SupervisedWindowMetadata(
                split=split,
                manifest_sha256=digest,
                observation_cache_sha256=digest,
                patch_map_sha256=digest,
                contact_label_status="synthetic_fixture",
                sample_label_status=("synthetic_fixture",) * 2,
                sample_ids=tuple(f"{split}_{index}" for index in range(2)),
                edge_names=(("left", "right"), ("hand", "face")),
                development_only=True,
            ),
        ),
    )


def test_diffusion_trainer_uses_frozen_bridge_and_trainable_only_checkpoint(tmp_path):
    train_bundle = bundle(tmp_path / "train", "train")
    validation_bundle = bundle(tmp_path / "validation", "validation")
    config = {
        "experiment": {"seed": 12345, "development_only": True},
        "data": {
            "train_bundle": str(train_bundle),
            "validation_bundle": str(validation_bundle),
        },
        "dposer_x": {
            "source_root": str(ROOT / "third_party/DPoser-X"),
            "runtime_root": str(ROOT.parent / "DPoser-X"),
            "registry": str(ROOT / "configs/diffusion/dposer_x_registry.json"),
            "expected_commit": "c373fce3d364a4a0946e8445fdea5cbfd490e837",
        },
        "model": {"hidden_dim": 16, "heads": 4, "layers": 1, "dropout": 0.0},
        "diffusion": {
            "steps": 10,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "conditioning_mode": "dynamic",
            "graph_dropout_probability": 0.2,
            "edge_dropout_probability": 0.2,
            "reliability_dropout_probability": 0.2,
            "root_weight": 1.0,
            "body_weight": 1.0,
            "left_hand_weight": 2.0,
            "right_hand_weight": 2.0,
            "face_weight": 1.0,
        },
        "optimization": {
            "train_steps": 2,
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "validation_interval": 1,
        },
        "third_party_manifest": str(ROOT / "third_party/manifest.yaml"),
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "training"
    report = train(
        config_path,
        output,
        device="cpu",
        bridge_factory=lambda *_: FakeBridge(),
    )
    assert report["official_dposer_x_frozen"] is True
    assert report["selection_split"] == "validation"
    assert report["trainable_parameter_count"] > 0
    assert (output / "TRAINING_COMPLETE").is_file()
    metadata = yaml.safe_load((output / "checkpoint/metadata.json").read_text())
    assert metadata["state_scope"] == "trainable"
    assert metadata["asset_sha256"]["dposer_x_registry"]
