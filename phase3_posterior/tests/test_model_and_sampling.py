from __future__ import annotations

import pytest
import torch

from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    OBSERVATION_EDGE_FEATURE_DIM,
    default_edge_index,
)
from phase3_posterior.losses.diffusion import SubVPSDE, region_balanced_score_loss
from phase3_posterior.losses.relation import (
    conditional_persistence_loss,
    stratified_sign_contact_loss,
)
from phase3_posterior.masked_spatial import (
    SpatialMask,
    fixed_condition_mask,
    inject_fixed_rotation_corruption,
)
from phase3_posterior.models.evidence_selector import EvidenceSelector
from phase3_posterior.models.relation_graph import RelationGraphEncoder
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.models.temporal_contact import TemporalContactRefiner
from phase3_posterior.sample import sample_candidates
from phase3_posterior.training import ExponentialMovingAverage
from phase3_posterior.train_diffusion import (
    _condition_masks,
    _conditioning_valid,
    _reset_dormant_conditioning_projections,
)


def _config() -> dict:
    return {
        "observation_dim": 45,
        "width": 32,
        "blocks": 1,
        "heads": 4,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "relation_width": 16,
        "relation_layers": 1,
    }


def _batch() -> dict[str, torch.Tensor]:
    edges = default_edge_index()
    return {
        "initial_state": torch.randn(1, 4, 51, 6),
        "features": torch.randn(1, 4, 51, 45),
        "frame_valid": torch.ones(1, 4, dtype=torch.bool),
        "edge_features": torch.randn(1, 4, edges.shape[1], EDGE_FEATURE_DIM),
        "edge_index": edges[None],
        "edge_valid": torch.ones(1, 4, edges.shape[1], dtype=torch.bool),
    }


def test_zero_initialized_residual_reproduces_prior() -> None:
    model = RelationalDiffusionPosterior(_config()).eval()
    batch = _batch()
    output = model(
        batch["initial_state"],
        torch.full((1,), 0.5),
        batch["features"],
        batch["frame_valid"],
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
    )
    assert torch.equal(output["score"], output["prior_score"])
    assert torch.count_nonzero(output["residual_score"]) == 0


def test_geometry_only_fallback_omits_contact_outputs() -> None:
    config = {
        **_config(),
        "predict_distance": True,
        "edge_identity": True,
        "contact_energy_enabled": False,
    }
    model = RelationalDiffusionPosterior(config).eval()
    batch = _batch()
    output = model(
        batch["initial_state"],
        torch.full((1,), 0.5),
        batch["features"],
        batch["frame_valid"],
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
    )
    assert "contact_logits" not in output
    assert "persistence_logits" not in output
    assert "depth_logits" in output
    assert "distance" in output
    assert "relation_token" in output


def test_score_loss_is_finite_with_partial_labels() -> None:
    score = torch.randn(2, 3, 51, 6)
    noise = torch.randn_like(score)
    valid = torch.ones(2, 3, 51, dtype=torch.bool)
    valid[:, :, 10:20] = False
    loss, regions = region_balanced_score_loss(score, noise, torch.ones(2), valid)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in regions.values())


def test_subvp_probability_flow_matches_public_coefficient() -> None:
    sde = SubVPSDE()
    time = torch.tensor([0.1, 0.5, 0.9])
    state = torch.randn(3, 2, 51, 6)
    score = torch.randn_like(state)
    beta = sde.beta(time)[:, None, None, None]
    discount = (
        1.0
        - torch.exp(
            -2.0 * sde.beta_min * time - (sde.beta_max - sde.beta_min) * time.square()
        )
    )[:, None, None, None]
    expected = -0.5 * beta * state - 0.5 * beta * discount * score
    assert torch.allclose(sde.probability_flow_drift(state, score, time), expected)


def test_auxiliary_snr_weight_suppresses_high_noise() -> None:
    sde = SubVPSDE()
    weight = sde.clipped_auxiliary_weight(torch.tensor([0.001, 0.5, 0.999]), gamma=5.0)
    assert torch.all((0.0 <= weight) & (weight <= 1.0))
    assert weight[0] == 1.0
    assert weight[0] > weight[1] > weight[2]


def test_training_step_reaches_zero_initialized_residual_head() -> None:
    model = RelationalDiffusionPosterior(_config()).train()
    batch = _batch()
    noise = torch.randn_like(batch["initial_state"])
    output = model(
        batch["initial_state"],
        torch.full((1,), 0.5),
        batch["features"],
        batch["frame_valid"],
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
    )
    loss, _ = region_balanced_score_loss(
        output["score"], noise, torch.ones(1), torch.ones(1, 4, 51, dtype=torch.bool)
    )
    loss.backward()
    gradient = model.residual.output.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_candidates_replay_and_candidate_zero_identity() -> None:
    model = RelationalDiffusionPosterior(_config()).eval()
    batch = _batch()
    first = sample_candidates(model, batch, SubVPSDE(), candidates=4, steps=2, seed=7)
    second = sample_candidates(model, batch, SubVPSDE(), candidates=4, steps=2, seed=7)
    assert torch.equal(first, second)
    assert torch.equal(first[:, 0], batch["initial_state"])
    assert not torch.equal(first[:, 1], first[:, 2])


def test_conditional_sampler_clamps_observed_and_generates_hidden_joints() -> None:
    model = RelationalDiffusionPosterior(_config()).eval()
    batch = _batch()
    condition = torch.ones(1, 4, 51, dtype=torch.bool)
    condition[..., 21:36] = False
    candidates = sample_candidates(
        model,
        batch,
        SubVPSDE(),
        candidates=2,
        steps=2,
        seed=11,
        condition_mask=condition,
    )
    prediction = candidates[:, 1]
    assert torch.equal(prediction[condition], batch["initial_state"][condition])
    assert torch.isfinite(prediction).all()
    assert not torch.equal(
        prediction[..., 21:36, :], batch["initial_state"][..., 21:36, :]
    )


def test_conditional_sampler_accepts_expanded_frame_mask() -> None:
    model = RelationalDiffusionPosterior(_config()).eval()
    batch = _batch()
    condition = batch["frame_valid"][..., None].expand(-1, -1, 51)
    prediction = sample_candidates(
        model,
        batch,
        SubVPSDE(),
        candidates=2,
        steps=1,
        condition_mask=condition,
    )[:, 1]
    assert torch.equal(prediction, batch["initial_state"])


def test_fixed_spatial_mask_only_removes_declared_joints() -> None:
    valid = torch.ones(2, 4, 51, dtype=torch.bool)
    result = fixed_condition_mask(valid, SpatialMask("test", "lhand", (21, 23)))
    assert not result[..., 21].any()
    assert not result[..., 23].any()
    assert result.sum() == valid.numel() - 2 * 4 * 2


def test_fixed_rotation_corruption_is_replayable_and_scoped() -> None:
    matrix = torch.eye(3).expand(1, 4, 51, 3, 3).clone()
    valid = torch.ones(1, 4, 51, dtype=torch.bool)
    frames = torch.ones(1, 4, dtype=torch.bool)
    mask = SpatialMask("test", "lhand", (21, 23))
    first, affected = inject_fixed_rotation_corruption(
        matrix, valid, frames, mask, seed=7
    )
    second, _ = inject_fixed_rotation_corruption(matrix, valid, frames, mask, seed=7)
    assert torch.equal(first, second)
    assert affected[..., 21].all() and affected[..., 23].all()
    assert torch.equal(first[..., 22, :, :], matrix[..., 22, :, :])
    assert not torch.equal(first[..., 21, :, :], matrix[..., 21, :, :])


def test_ema_validation_context_restores_training_parameters() -> None:
    model = torch.nn.Linear(2, 2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(1.0)
    changed = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with ema.average_parameters(model):
        assert all(
            torch.equal(model.state_dict()[key], original[key]) for key in original
        )
    assert all(torch.equal(model.state_dict()[key], changed[key]) for key in changed)


def test_conditioning_validity_uses_full_initializer_not_partial_targets() -> None:
    batch = {
        "target_rotation_valid": torch.zeros(1, 2, 51, dtype=torch.bool),
        "joint_valid": torch.zeros(1, 2, 51, dtype=torch.bool),
        "frame_valid": torch.tensor([[True, False]]),
    }
    valid = _conditioning_valid(batch)
    assert valid[:, 0].all()
    assert not valid[:, 1].any()


def test_unconditional_warm_start_resets_only_conditioning_weights() -> None:
    model = RelationalDiffusionPosterior(_config())
    observation_bias = model.residual.observation.bias.detach().clone()
    relation_bias = model.residual.relation.bias.detach().clone()
    _reset_dormant_conditioning_projections(model)
    assert torch.count_nonzero(model.residual.observation.weight) == 0
    assert torch.count_nonzero(model.residual.relation.weight) == 0
    assert torch.equal(model.residual.observation.bias, observation_bias)
    assert torch.equal(model.residual.relation.bias, relation_bias)


def test_condition_and_corruption_masks_are_disjoint() -> None:
    valid = torch.ones(2, 8, 51, dtype=torch.bool)
    condition, corruption = _condition_masks(valid, seed=3, step=7, dropout=0.0)
    assert not torch.any(condition & corruption)
    assert torch.equal(condition | corruption, valid)
    dropped, dropped_corruption = _condition_masks(
        valid, seed=3, step=7, dropout=1.0
    )
    assert not dropped.any()
    assert not dropped_corruption.any()


def test_masked_rotation_hint_projection_is_zero_initialized() -> None:
    config = {**_config(), "masked_rotation_hints": True}
    model = RelationalDiffusionPosterior(config).eval()
    assert model.residual.corruption_observation is not None
    assert torch.count_nonzero(model.residual.corruption_observation.weight) == 0
    batch = _batch()
    condition = torch.ones(1, 4, 51, dtype=torch.bool)
    hint = torch.zeros_like(condition)
    condition[..., 21:36] = False
    hint[..., 21:36] = True
    output = model(
        batch["initial_state"],
        torch.full((1,), 0.5),
        batch["features"],
        batch["frame_valid"],
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
        condition,
        rotation_hint_mask=hint,
    )
    assert torch.isfinite(output["score"]).all()


def test_masked_rotation_hint_projection_receives_gradient() -> None:
    config = {**_config(), "masked_rotation_hints": True}
    model = RelationalDiffusionPosterior(config).train()
    with torch.no_grad():
        model.residual.output.weight.normal_(std=0.01)
    batch = _batch()
    condition = torch.ones(1, 4, 51, dtype=torch.bool)
    hint = torch.zeros_like(condition)
    condition[..., 21:36] = False
    hint[..., 21:36] = True
    output = model(
        batch["initial_state"],
        torch.full((1,), 0.5),
        batch["features"],
        batch["frame_valid"],
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
        condition,
        rotation_hint_mask=hint,
    )
    output["score"].square().mean().backward()
    gradient = model.residual.corruption_observation.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_selector_inference_contract_has_no_gt_argument() -> None:
    selector = EvidenceSelector(8)
    evidence = torch.randn(2, 4, 8)
    assert selector.select(evidence).shape == (2,)


def test_relation_graph_cpu_bfloat16_autocast_has_consistent_accumulator_dtype() -> (
    None
):
    batch = _batch()
    model = RelationGraphEncoder(width=16, layers=1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = model(batch["edge_features"], batch["edge_index"], batch["edge_valid"])
    assert torch.isfinite(result["contact_logits"]).all()


def test_corrected_relation_graph_predicts_finite_distance() -> None:
    batch = _batch()
    model = RelationGraphEncoder(
        width=16, layers=1, predict_distance=True, edge_identity=True
    )
    result = model(batch["edge_features"], batch["edge_index"], batch["edge_valid"])
    assert result["distance"].shape == batch["edge_valid"].shape
    assert torch.isfinite(result["distance"]).all()


def test_temporal_contact_refiner_is_identity_initialized() -> None:
    batch = _batch()
    backbone = RelationGraphEncoder(
        width=16, layers=1, predict_distance=True, edge_identity=True
    ).eval()
    expected = backbone(
        batch["edge_features"], batch["edge_index"], batch["edge_valid"]
    )
    model = TemporalContactRefiner(
        backbone, width=16, temporal_hidden=16, persistence_fusion_weight=2.0
    ).eval()
    result = model(
        batch["edge_features"], batch["edge_index"], batch["edge_valid"]
    )
    assert torch.equal(result["contact_logits"], expected["contact_logits"])
    assert torch.equal(result["persistence_logits"], expected["persistence_logits"])
    assert torch.equal(result["distance"], expected["distance"])
    assert torch.equal(result["depth_logits"], expected["depth_logits"])


def test_temporal_contact_refiner_separates_frozen_and_trainable_encoders() -> None:
    backbone = RelationGraphEncoder(
        width=16, layers=1, predict_distance=True, edge_identity=True
    )
    model = TemporalContactRefiner(
        backbone,
        width=16,
        temporal_hidden=16,
        persistence_fusion_weight=2.0,
        train_contact_encoder=True,
    )
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.contact_encoder is not None
    assert any(
        parameter.requires_grad for parameter in model.contact_encoder.edge_input.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.contact_encoder.head.parameters()
    )


def test_observation_contact_refiner_is_identity_initialized_and_trainable() -> None:
    batch = _batch()
    backbone = RelationGraphEncoder(
        width=16, layers=1, predict_distance=True, edge_identity=True
    ).eval()
    expected = backbone(
        batch["edge_features"], batch["edge_index"], batch["edge_valid"]
    )
    model = TemporalContactRefiner(
        backbone,
        width=16,
        temporal_hidden=16,
        observation_features=True,
    ).eval()
    observation = torch.randn(
        1,
        4,
        batch["edge_index"].shape[-1],
        OBSERVATION_EDGE_FEATURE_DIM,
    )
    result = model(
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
        observation,
        batch["edge_valid"],
    )
    assert torch.equal(result["contact_logits"], expected["contact_logits"])
    result["contact_logits"].sum().backward()
    gradient = model.observation_projection[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_contextual_observation_graph_is_identity_initialized() -> None:
    batch = _batch()
    backbone = RelationGraphEncoder(
        width=16, layers=1, predict_distance=True, edge_identity=True
    ).eval()
    expected = backbone(
        batch["edge_features"], batch["edge_index"], batch["edge_valid"]
    )
    model = TemporalContactRefiner(
        backbone,
        width=16,
        temporal_hidden=16,
        observation_features=True,
        observation_graph_layers=2,
    ).eval()
    observation = torch.randn(
        1,
        4,
        batch["edge_index"].shape[-1],
        OBSERVATION_EDGE_FEATURE_DIM,
    )
    result = model(
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
        observation,
        batch["edge_valid"],
    )
    assert model.observation_encoder is not None
    assert model.observation_projection is None
    assert torch.equal(result["contact_logits"], expected["contact_logits"])


def test_contained_observation_delta_preserves_zero_residual_logits() -> None:
    batch = _batch()
    backbone = RelationGraphEncoder(
        width=16, layers=1, predict_distance=True, edge_identity=True
    ).eval()
    expected = backbone(
        batch["edge_features"], batch["edge_index"], batch["edge_valid"]
    )
    model = TemporalContactRefiner(
        backbone,
        width=16,
        temporal_hidden=16,
        observation_features=True,
        observation_graph_layers=2,
        observation_logit_residual=True,
        observation_only_training=True,
    ).eval()
    observation = torch.randn(
        1,
        4,
        batch["edge_index"].shape[-1],
        OBSERVATION_EDGE_FEATURE_DIM,
    )
    observation[..., -6:] = 0
    with torch.no_grad():
        model.observation_contact_delta.weight.fill_(1.0)
    result = model(
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
        observation,
        batch["edge_valid"],
    )
    assert torch.equal(result["contact_logits"], expected["contact_logits"])
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(
        name.startswith(("observation_encoder.", "observation_contact_delta."))
        for name in trainable
    )


def test_hand_body_containment_preserves_other_contact_logits() -> None:
    batch = _batch()
    backbone = RelationGraphEncoder(
        width=16, layers=1, predict_distance=True, edge_identity=True
    ).eval()
    expected = backbone(
        batch["edge_features"], batch["edge_index"], batch["edge_valid"]
    )
    model = TemporalContactRefiner(
        backbone,
        width=16,
        temporal_hidden=16,
        observation_features=True,
        observation_graph_layers=2,
        observation_logit_residual=True,
        observation_only_training=True,
        observation_hand_body_only=True,
    ).eval()
    observation = torch.randn(
        1,
        4,
        batch["edge_index"].shape[-1],
        OBSERVATION_EDGE_FEATURE_DIM,
    )
    observation[..., -6:] = 1
    with torch.no_grad():
        model.observation_contact_delta.weight.fill_(1.0)
    result = model(
        batch["edge_features"],
        batch["edge_index"],
        batch["edge_valid"],
        observation,
        batch["edge_valid"],
    )
    source, target = batch["edge_index"][0]
    hand_body = (source >= 10) ^ (target >= 10)
    assert torch.equal(
        result["contact_logits"][..., ~hand_body],
        expected["contact_logits"][..., ~hand_body],
    )
    assert not torch.equal(
        result["contact_logits"][..., hand_body],
        expected["contact_logits"][..., hand_body],
    )


def test_stratified_contact_and_conditional_persistence_losses_are_finite() -> None:
    logits = torch.tensor([[[-2.0, 1.0, 0.5, -0.5]]], requires_grad=True)
    target = torch.tensor([[[True, False, True, False]]])
    sign = torch.ones_like(target)
    contact = stratified_sign_contact_loss(
        logits, target, sign, hard_negative_ratio=2
    )
    persistence = conditional_persistence_loss(
        logits,
        torch.tensor([[[True, False, False, False]]]),
        target,
        torch.ones_like(target),
    )
    (contact + persistence).backward()
    assert torch.isfinite(contact) and torch.isfinite(persistence)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast unavailable")
def test_relation_graph_cuda_bfloat16_autocast_has_consistent_accumulator_dtype() -> (
    None
):
    batch = {key: value.cuda() for key, value in _batch().items()}
    model = RelationGraphEncoder(width=16, layers=1).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model(batch["edge_features"], batch["edge_index"], batch["edge_valid"])
    assert torch.isfinite(result["contact_logits"]).all()
