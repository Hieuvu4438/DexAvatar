from __future__ import annotations

import torch

from phase2_refiner.geometry.rotations import axis_angle_to_matrix
from phase3_posterior.geometry.contact import (
    contact_hysteresis,
    contact_persistence_target,
)
from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    NUM_RELATION_NODES,
    build_edge_features,
    default_edge_index,
)
from phase3_posterior.geometry.state_adapter import matrices_to_state, state_to_matrices


def test_phase3_rotation_round_trip() -> None:
    axis_angle = torch.randn(2, 7, 51, 3) * 0.2
    matrix = axis_angle_to_matrix(axis_angle)
    reconstructed = state_to_matrices(matrices_to_state(matrix))
    assert torch.allclose(matrix, reconstructed, atol=1e-5)


def test_relation_contract_and_finite_features() -> None:
    nodes = torch.randn(5, NUM_RELATION_NODES, 3)
    valid = torch.ones(5, NUM_RELATION_NODES, dtype=torch.bool)
    edges = default_edge_index()
    features, edge_valid = build_edge_features(nodes, valid, edges)
    assert features.shape == (5, edges.shape[1], EDGE_FEATURE_DIM)
    assert edge_valid.all()
    assert torch.isfinite(features).all()


def test_contact_hysteresis_and_persistence() -> None:
    distances = torch.tensor([[0.03], [0.01], [0.015], [0.025], [0.01]])
    valid = torch.ones_like(distances, dtype=torch.bool)
    contact = contact_hysteresis(distances, valid, onset=0.012, release=0.020)
    assert contact[:, 0].tolist() == [False, True, True, False, True]
    persistence = contact_persistence_target(contact)
    assert persistence[:, 0].tolist() == [False, False, True, False, False]
