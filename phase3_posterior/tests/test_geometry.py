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
    OBSERVATION_EDGE_FEATURE_DIM,
    build_edge_features,
    build_observation_edge_features,
    default_edge_index,
    mask_relation_inputs,
    relation_node_conditioning_mask,
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


def test_observation_relation_features_are_masked_and_target_independent() -> None:
    keypoints = torch.rand(2, 5, 51, 2) * 2.0 - 1.0
    valid = torch.ones(2, 5, 51, dtype=torch.bool)
    reliability = torch.rand(2, 5, 51)
    residual = torch.rand(2, 5, 51, 2)
    valid[:, :, 21:36] = False
    edges = default_edge_index()
    features, edge_valid = build_observation_edge_features(
        keypoints, valid, reliability, edges, residual
    )
    assert features.shape == (2, 5, edges.shape[1], OBSERVATION_EDGE_FEATURE_DIM)
    source, target = edges
    left_nodes = set(range(10, 21))
    left_edges = torch.tensor(
        [int(a) in left_nodes or int(b) in left_nodes for a, b in zip(source, target)]
    )
    assert not edge_valid[..., left_edges].any()
    assert torch.count_nonzero(features[..., left_edges, :]) == 0
    assert torch.isfinite(features).all()


def test_contact_hysteresis_and_persistence() -> None:
    distances = torch.tensor([[0.03], [0.01], [0.015], [0.025], [0.01]])
    valid = torch.ones_like(distances, dtype=torch.bool)
    contact = contact_hysteresis(distances, valid, onset=0.012, release=0.020)
    assert contact[:, 0].tolist() == [False, True, True, False, True]
    persistence = contact_persistence_target(contact)
    assert persistence[:, 0].tolist() == [False, False, True, False, False]


def test_masked_hand_cannot_leak_through_relation_edges() -> None:
    edges = default_edge_index()
    conditioning = torch.ones(2, 4, 51, dtype=torch.bool)
    conditioning[:, 1:3, 21:36] = False
    node_mask = relation_node_conditioning_mask(conditioning)
    assert not node_mask[:, 1:3, 10:21].any()
    features = torch.randn(2, 4, edges.shape[1], EDGE_FEATURE_DIM)
    valid = torch.ones(2, 4, edges.shape[1], dtype=torch.bool)
    masked_features, masked_valid = mask_relation_inputs(
        features, valid, edges[None].expand(2, -1, -1), conditioning
    )
    source, target = edges
    left_nodes = set(range(10, 21))
    left_edges = torch.tensor(
        [int(a) in left_nodes or int(b) in left_nodes for a, b in zip(source, target)]
    )
    assert not masked_valid[:, 1:3, left_edges].any()
    assert torch.count_nonzero(masked_features[:, 1:3, left_edges]) == 0
