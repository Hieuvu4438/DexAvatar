"""Fixed relational node/edge construction for body, wrists, palms, and fingers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phase2_refiner.geometry.palm import FINGERTIP_INDICES, MCP_INDICES, palm_center


BODY_ANCHORS = {
    "head": 14,
    "neck": 11,
    "chest": 8,
    "spine2": 5,
    "left_shoulder": 15,
    "right_shoulder": 16,
    "left_elbow": 17,
    "right_elbow": 18,
    "left_wrist": 19,
    "right_wrist": 20,
}
NUM_BODY_NODES = len(BODY_ANCHORS)
HAND_NODE_COUNT = 11
NUM_RELATION_NODES = NUM_BODY_NODES + 2 * HAND_NODE_COUNT
EDGE_FEATURE_DIM = 16
OBSERVATION_EDGE_FEATURE_DIM = 16

# Fixed anatomical proxy radii used to convert decoded joint-centre distances
# into an approximate surface gap.  Keeping this contract next to the fixed
# node ordering makes the feature reproducible at training and inference time.
CONTACT_PROXY_RADII_M = torch.tensor(
    (
        0.10,
        0.075,
        0.16,
        0.14,
        0.06,
        0.06,
        0.055,
        0.055,
        0.04,
        0.04,
        0.055,
        *([0.022] * 5),
        *([0.015] * 5),
        0.055,
        *([0.022] * 5),
        *([0.015] * 5),
    ),
    dtype=torch.float32,
)


@dataclass(frozen=True)
class RelationGeometry:
    nodes: torch.Tensor
    node_valid: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_valid: torch.Tensor


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError("Expected points (...,N,3) and transforms (...,4,4)")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return points @ rotation.transpose(-1, -2) + translation[..., None, :]


def _hand_nodes(hand: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            palm_center(hand)[..., None, :],
            hand[..., list(MCP_INDICES), :],
            hand[..., list(FINGERTIP_INDICES), :],
        ),
        dim=-2,
    )


def build_relation_nodes(
    torso_positions: torch.Tensor,
    torso_valid: torch.Tensor,
    wrist_local_positions: torch.Tensor,
    wrist_local_valid: torch.Tensor,
    wrist_to_torso: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torso_positions.shape[-2:] != (51, 3):
        raise ValueError("torso_positions must end in (51,3)")
    body_indices = list(BODY_ANCHORS.values())
    body = torso_positions[..., body_indices, :]
    body_valid = torso_valid[..., body_indices]
    left = transform_points(
        wrist_local_positions[..., 21:36, :], wrist_to_torso[..., 0, :, :]
    )
    right = transform_points(
        wrist_local_positions[..., 36:51, :], wrist_to_torso[..., 1, :, :]
    )
    left_nodes = _hand_nodes(left)
    right_nodes = _hand_nodes(right)
    left_source_valid = wrist_local_valid[..., 21:36]
    right_source_valid = wrist_local_valid[..., 36:51]
    left_valid = torch.cat(
        (
            left_source_valid[..., list(MCP_INDICES)].all(dim=-1, keepdim=True),
            left_source_valid[..., list(MCP_INDICES)],
            left_source_valid[..., list(FINGERTIP_INDICES)],
        ),
        dim=-1,
    )
    right_valid = torch.cat(
        (
            right_source_valid[..., list(MCP_INDICES)].all(dim=-1, keepdim=True),
            right_source_valid[..., list(MCP_INDICES)],
            right_source_valid[..., list(FINGERTIP_INDICES)],
        ),
        dim=-1,
    )
    return torch.cat((body, left_nodes, right_nodes), dim=-2), torch.cat(
        (body_valid, left_valid, right_valid), dim=-1
    )


def default_edge_index(device: torch.device | str | None = None) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    left_palm = NUM_BODY_NODES
    left_tips = range(NUM_BODY_NODES + 6, NUM_BODY_NODES + 11)
    right_palm = NUM_BODY_NODES + HAND_NODE_COUNT
    right_tips = range(right_palm + 6, right_palm + 11)
    edges.append((8, 9))
    edges.append((left_palm, right_palm))
    edges.extend(zip(left_tips, right_tips, strict=True))
    edges.extend((tip, right_palm) for tip in left_tips)
    edges.extend((tip, left_palm) for tip in right_tips)
    body_targets = (0, 1, 2, 4, 5)
    hand_effectors = (left_palm, *left_tips, right_palm, *right_tips)
    edges.extend((hand, body) for hand in hand_effectors for body in body_targets)
    edges.extend(((8, 6), (9, 7), (left_palm, 8), (right_palm, 9)))
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def relation_node_conditioning_mask(conditioning: torch.Tensor) -> torch.Tensor:
    """Map a ``(...,51)`` joint mask to the fixed 32 relation nodes."""
    if conditioning.shape[-1] != 51:
        raise ValueError("conditioning must end in 51 joints")
    body = conditioning[..., list(BODY_ANCHORS.values())]
    left = conditioning[..., 21:36]
    right = conditioning[..., 36:51]

    def hand_nodes(hand: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                hand[..., list(MCP_INDICES)].all(dim=-1, keepdim=True),
                hand[..., list(MCP_INDICES)],
                hand[..., list(FINGERTIP_INDICES)],
            ),
            dim=-1,
        )

    result = torch.cat((body, hand_nodes(left), hand_nodes(right)), dim=-1)
    if result.shape[-1] != NUM_RELATION_NODES:
        raise AssertionError(result.shape)
    return result


def mask_relation_inputs(
    edge_features: torch.Tensor,
    edge_valid: torch.Tensor,
    edge_index: torch.Tensor,
    conditioning: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove every relation whose endpoint is absent from conditioning."""
    if edge_index.ndim == 3:
        if not torch.equal(edge_index, edge_index[:1].expand_as(edge_index)):
            raise ValueError("batched edge indexes must use identical ordering")
        edge_index = edge_index[0]
    nodes = relation_node_conditioning_mask(conditioning)
    source, target = edge_index
    conditioned_valid = edge_valid & nodes[..., source] & nodes[..., target]
    conditioned_features = torch.where(
        conditioned_valid[..., None], edge_features, torch.zeros_like(edge_features)
    )
    return conditioned_features, conditioned_valid


def build_edge_features(
    nodes: torch.Tensor,
    node_valid: torch.Tensor,
    edge_index: torch.Tensor,
    node_reliability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    source, target = edge_index
    relative = nodes[..., target, :] - nodes[..., source, :]
    valid = node_valid[..., source] & node_valid[..., target]
    distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    velocity = torch.zeros_like(relative)
    acceleration = torch.zeros_like(relative)
    if nodes.shape[-3] > 1:
        velocity[..., 1:, :, :] = relative[..., 1:, :, :] - relative[..., :-1, :, :]
    if nodes.shape[-3] > 2:
        acceleration[..., 2:, :, :] = (
            velocity[..., 2:, :, :] - velocity[..., 1:-1, :, :]
        )
    depth = relative[..., 2:3]
    radii = CONTACT_PROXY_RADII_M.to(device=nodes.device, dtype=nodes.dtype)
    surface_gap = distance - (radii[source] + radii[target])[..., None]
    if node_reliability is None:
        rel_source = torch.ones_like(distance)
        rel_target = torch.ones_like(distance)
    else:
        rel_source = node_reliability[..., source, None]
        rel_target = node_reliability[..., target, None]
    previous_contact = torch.zeros_like(distance)
    features = torch.cat(
        (
            relative,
            distance,
            velocity,
            acceleration,
            depth,
            surface_gap,
            rel_source,
            rel_target,
            valid[..., None].float(),
            previous_contact,
        ),
        dim=-1,
    )
    if features.shape[-1] != EDGE_FEATURE_DIM:
        raise AssertionError(features.shape)
    return torch.where(valid[..., None], features, torch.zeros_like(features)), valid


def build_observation_edge_features(
    keypoints_2d: torch.Tensor,
    joint_valid: torch.Tensor,
    joint_reliability: torch.Tensor,
    edge_index: torch.Tensor,
    reprojection_residual_2d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build target-independent 2D relation evidence from detector tracks.

    The node ordering exactly mirrors :func:`build_relation_nodes`. These
    features expose the observation that drives the independent How2Sign target
    refinement; they never read target rotations or target geometry.
    """
    if keypoints_2d.shape[-2:] != (51, 2):
        raise ValueError("keypoints_2d must end in (51,2)")
    if joint_valid.shape != keypoints_2d.shape[:-1]:
        raise ValueError("joint_valid shape must match keypoints_2d")
    if joint_reliability.shape != joint_valid.shape:
        raise ValueError("joint_reliability shape must match joint_valid")
    if reprojection_residual_2d is None:
        reprojection_residual_2d = torch.zeros_like(keypoints_2d)
    if reprojection_residual_2d.shape != keypoints_2d.shape:
        raise ValueError("reprojection_residual_2d shape must match keypoints_2d")

    body_indices = list(BODY_ANCHORS.values())

    def hand_nodes(hand: torch.Tensor) -> torch.Tensor:
        palm = hand[..., list(MCP_INDICES), :].mean(dim=-2, keepdim=True)
        return torch.cat(
            (
                palm,
                hand[..., list(MCP_INDICES), :],
                hand[..., list(FINGERTIP_INDICES), :],
            ),
            dim=-2,
        )

    def hand_valid(hand: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                hand[..., list(MCP_INDICES)].all(dim=-1, keepdim=True),
                hand[..., list(MCP_INDICES)],
                hand[..., list(FINGERTIP_INDICES)],
            ),
            dim=-1,
        )

    def hand_reliability(hand: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                hand[..., list(MCP_INDICES)].mean(dim=-1, keepdim=True),
                hand[..., list(MCP_INDICES)],
                hand[..., list(FINGERTIP_INDICES)],
            ),
            dim=-1,
        )

    nodes = torch.cat(
        (
            keypoints_2d[..., body_indices, :],
            hand_nodes(keypoints_2d[..., 21:36, :]),
            hand_nodes(keypoints_2d[..., 36:51, :]),
        ),
        dim=-2,
    )
    residual_nodes = torch.cat(
        (
            reprojection_residual_2d[..., body_indices, :],
            hand_nodes(reprojection_residual_2d[..., 21:36, :]),
            hand_nodes(reprojection_residual_2d[..., 36:51, :]),
        ),
        dim=-2,
    )
    node_valid = torch.cat(
        (
            joint_valid[..., body_indices],
            hand_valid(joint_valid[..., 21:36]),
            hand_valid(joint_valid[..., 36:51]),
        ),
        dim=-1,
    )
    reliability = torch.cat(
        (
            joint_reliability[..., body_indices],
            hand_reliability(joint_reliability[..., 21:36]),
            hand_reliability(joint_reliability[..., 36:51]),
        ),
        dim=-1,
    )
    source, target = edge_index
    relative = nodes[..., target, :] - nodes[..., source, :]
    distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    velocity = torch.zeros_like(relative)
    acceleration = torch.zeros_like(relative)
    if nodes.shape[-3] > 1:
        velocity[..., 1:, :, :] = relative[..., 1:, :, :] - relative[..., :-1, :, :]
    if nodes.shape[-3] > 2:
        acceleration[..., 2:, :, :] = (
            velocity[..., 2:, :, :] - velocity[..., 1:-1, :, :]
        )
    valid = node_valid[..., source] & node_valid[..., target]
    features = torch.cat(
        (
            relative,
            distance,
            velocity,
            acceleration,
            reliability[..., source, None],
            reliability[..., target, None],
            valid[..., None].float(),
            residual_nodes[..., source, :],
            residual_nodes[..., target, :],
            residual_nodes[..., target, :] - residual_nodes[..., source, :],
        ),
        dim=-1,
    )
    if features.shape[-1] != OBSERVATION_EDGE_FEATURE_DIM:
        raise AssertionError(features.shape)
    return torch.where(valid[..., None], features, torch.zeros_like(features)), valid


def build_relation_geometry(
    torso_positions: torch.Tensor,
    torso_valid: torch.Tensor,
    wrist_local_positions: torch.Tensor,
    wrist_local_valid: torch.Tensor,
    wrist_to_torso: torch.Tensor,
) -> RelationGeometry:
    nodes, node_valid = build_relation_nodes(
        torso_positions,
        torso_valid,
        wrist_local_positions,
        wrist_local_valid,
        wrist_to_torso,
    )
    edge_index = default_edge_index(nodes.device)
    edge_features, edge_valid = build_edge_features(nodes, node_valid, edge_index)
    return RelationGeometry(nodes, node_valid, edge_index, edge_features, edge_valid)
