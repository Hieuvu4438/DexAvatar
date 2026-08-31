import pytest
import torch

from dcg_sign4d.contact.losses import masked_duration_loss
from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.diffusion.training import denoising_loss
from dcg_sign4d.diffusion.trajectory_denoiser import PartAwareTrajectoryDenoiser
from dcg_sign4d.synthetic import make_graph, make_observations, make_state
from dcg_sign4d.training.checkpoint import (
    CheckpointMetadata,
    load_model_checkpoint,
    save_model_checkpoint,
)
from dcg_sign4d.training.steps import contact_objective, diffusion_objective
from dcg_sign4d.utils.hashing import canonical_hash


def test_duration_loss_is_masked_and_rejects_out_of_range():
    logits = torch.randn(1, 2, 1, 3, requires_grad=True)
    target = torch.tensor([[[1], [3]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    uncertain = torch.tensor([[[False], [True]]])
    loss = masked_duration_loss(logits, target, valid, uncertain)
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() == 0
    with pytest.raises(ValueError, match=r"\[1,D\]"):
        masked_duration_loss(logits.detach(), torch.full_like(target, 4), valid, ~valid)


def test_contact_and_diffusion_training_objectives_are_finite():
    state = make_state(time=4)
    observations = make_observations(time=4)
    encoded, context = StateCodec().encode(state)
    proposal = ContactProposal(
        encoded.shape[-1], edge_count=2, max_duration=4, hidden_dim=16, heads=4, layers=1
    )
    proposal_output = proposal(observations, state, torch.randn(1, 4, 2, 5))
    labels = make_graph(time=4).event_state
    contact = contact_objective(
        proposal_output,
        event_state=labels,
        duration_frames=torch.ones_like(labels),
        edge_valid=torch.ones(1, 2, dtype=torch.bool),
        frame_valid=state.valid_mask,
        uncertain=torch.zeros_like(labels, dtype=torch.bool),
        class_counts=torch.tensor([8, 0, 0, 0]),
    )
    assert torch.isfinite(contact.total)
    contact.total.backward()

    part_dims = (
        context.widths[0] + context.widths[1] + context.widths[2],
        context.widths[3],
        context.widths[4],
        context.widths[5],
    )
    denoiser = PartAwareTrajectoryDenoiser(part_dims, hidden_dim=16, heads=4, layers=1)
    token_encoder = ContactTokenEncoder(2, 16)
    generator = torch.Generator().manual_seed(7)
    diffusion = diffusion_objective(
        denoiser,
        DiffusionSchedule(10),
        StateCodec(),
        token_encoder,
        state,
        make_graph(time=4),
        observations,
        conditioning_mode="dynamic",
        channel_weights=torch.ones(encoded.shape[-1]),
        generator=generator,
    )
    assert torch.isfinite(diffusion.total)
    diffusion.total.backward()


def test_diffusion_part_supervision_mask_removes_unsupervised_gradient():
    predicted = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.randn_like(predicted)
    mask = torch.tensor([[[True, False, False], [False, False, False]]])
    loss = denoising_loss(
        predicted,
        target,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(3),
        mask,
    )
    loss.backward()
    assert predicted.grad[0, 0, 0] != 0
    assert predicted.grad[..., 1:].abs().sum() == 0
    assert predicted.grad[0, 1].abs().sum() == 0


def test_checkpoint_roundtrip_is_hash_verified_and_development_gated(tmp_path):
    model = torch.nn.Linear(3, 2)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    digest = canonical_hash({"fixture": True})
    metadata = CheckpointMetadata(
        stage="contact_proposal",
        model_class=model_class,
        step=10,
        epoch=1,
        seed=123,
        config_sha256=digest,
        manifest_sha256=digest,
        dependency_commits={"fixture": "a" * 40},
        metrics={"loss": 0.5},
        development_only=True,
    )
    source = save_model_checkpoint(tmp_path / "checkpoint", model, metadata)
    restored = torch.nn.Linear(3, 2)
    with pytest.raises(PermissionError, match="development checkpoint"):
        load_model_checkpoint(source, restored, expected_stage="contact_proposal")
    payload = load_model_checkpoint(
        source,
        restored,
        expected_stage="contact_proposal",
        expected_config_sha256=digest,
        allow_development=True,
    )
    assert payload["weights_sha256"]
    assert all(
        torch.equal(first, second)
        for first, second in zip(model.parameters(), restored.parameters(), strict=True)
    )

    tampered = source / "weights.pt"
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="weights hash mismatch"):
        load_model_checkpoint(
            source,
            restored,
            expected_stage="contact_proposal",
            allow_development=True,
        )


def test_trainable_only_checkpoint_excludes_frozen_backbone(tmp_path):
    model = torch.nn.ModuleDict({"frozen": torch.nn.Linear(2, 2), "adapter": torch.nn.Linear(2, 2)})
    model["frozen"].requires_grad_(False)
    digest = canonical_hash({"trainable": True})
    metadata = CheckpointMetadata(
        stage="trajectory_diffusion",
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        step=1,
        epoch=0,
        seed=1,
        config_sha256=digest,
        manifest_sha256=digest,
        dependency_commits={"fixture": "a" * 40},
        metrics={"loss": 1.0},
        development_only=True,
    )
    frozen_before = model["frozen"].weight.detach().clone()
    adapter_before = model["adapter"].weight.detach().clone()
    source = save_model_checkpoint(tmp_path / "adapter", model, metadata, state_scope="trainable")
    with torch.no_grad():
        model["frozen"].weight.add_(10)
        model["adapter"].weight.add_(10)
    payload = load_model_checkpoint(
        source,
        model,
        expected_stage="trajectory_diffusion",
        allow_development=True,
    )
    assert payload["state_scope"] == "trainable"
    assert torch.equal(model["adapter"].weight, adapter_before)
    assert not torch.equal(model["frozen"].weight, frozen_before)
