from pathlib import Path

import torch
import yaml

from dcg_sign4d.cli.train_contact import train
from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.synthetic import make_graph, make_observations, make_state
from dcg_sign4d.training.batch import (
    SupervisedWindowBatch,
    SupervisedWindowMetadata,
    save_supervised_windows,
)
from dcg_sign4d.training.checkpoint import load_model_checkpoint
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

ROOT = Path(__file__).resolve().parents[2]


def bundle(path, split):
    state = make_state(batch=3, time=4)
    graph = make_graph(batch=3, time=4, edges=2)
    graph.event_state[:, 1, 0] = 1
    graph.event_probability[:, 1, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    width = StateCodec().encode(state)[0].shape[-1]
    digest = canonical_hash({"shared": True})
    return save_supervised_windows(
        path,
        SupervisedWindowBatch(
            trajectory=state,
            observations=make_observations(batch=3, time=4),
            geometry_features=torch.randn(3, 4, 2, 5),
            graph=graph,
            duration_frames=torch.ones(3, 4, 2, dtype=torch.long),
            supervision_mask=torch.ones(3, 4, width, dtype=torch.bool),
            metadata=SupervisedWindowMetadata(
                split=split,
                manifest_sha256=digest,
                observation_cache_sha256=digest,
                patch_map_sha256=digest,
                contact_label_status="synthetic_fixture",
                sample_label_status=("synthetic_fixture",) * 3,
                sample_ids=tuple(f"{split}_{index}" for index in range(3)),
                edge_names=(("left", "right"), ("hand", "face")),
                development_only=True,
            ),
        ),
    )


def test_contact_trainer_selects_on_validation_and_writes_loadable_checkpoint(tmp_path):
    train_bundle = bundle(tmp_path / "train", "train")
    validation_bundle = bundle(tmp_path / "validation", "validation")
    config = {
        "experiment": {"seed": 12345, "development_only": True},
        "data": {
            "train_bundle": str(train_bundle),
            "validation_bundle": str(validation_bundle),
        },
        "model": {
            "hidden_dim": 16,
            "heads": 4,
            "layers": 1,
            "dropout": 0.0,
            "max_duration": 4,
        },
        "optimization": {
            "steps": 3,
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "validation_interval": 1,
        },
        "sampling": {"balanced": True, "effective_number_beta": 0.9},
        "objective": {
            "event_weight": 1.0,
            "duration_weight": 1.0,
            "transition_weight": 0.1,
            "calibration_weight": 0.1,
            "gold_label_weight": 1.0,
            "accepted_pseudo_label_weight": 0.5,
        },
        "third_party_manifest": str(ROOT / "third_party/manifest.yaml"),
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "training"
    report = train(config_path, output, device="cpu")
    assert report["selection_split"] == "validation"
    assert len(report["history"]) == 3
    assert (output / "TRAINING_COMPLETE").is_file()

    width = StateCodec().encode(make_state(batch=3, time=4))[0].shape[-1]
    restored = ContactProposal(
        width,
        2,
        4,
        hidden_dim=16,
        heads=4,
        layers=1,
        edge_names=(("left", "right"), ("hand", "face")),
    )
    metadata = load_model_checkpoint(
        output / "checkpoint",
        restored,
        expected_stage="contact_proposal",
        expected_config_sha256=file_sha256(config_path),
        allow_development=True,
    )
    assert metadata["step"] == report["best_step"]
