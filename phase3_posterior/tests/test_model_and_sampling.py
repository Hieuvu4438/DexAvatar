from __future__ import annotations

import torch

from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    default_edge_index,
)
from phase3_posterior.losses.diffusion import SubVPSDE, region_balanced_score_loss
from phase3_posterior.models.evidence_selector import EvidenceSelector
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.sample import sample_candidates


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


def test_score_loss_is_finite_with_partial_labels() -> None:
    score = torch.randn(2, 3, 51, 6)
    noise = torch.randn_like(score)
    valid = torch.ones(2, 3, 51, dtype=torch.bool)
    valid[:, :, 10:20] = False
    loss, regions = region_balanced_score_loss(score, noise, torch.ones(2), valid)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in regions.values())


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


def test_selector_inference_contract_has_no_gt_argument() -> None:
    selector = EvidenceSelector(8)
    evidence = torch.randn(2, 4, 8)
    assert selector.select(evidence).shape == (2,)
