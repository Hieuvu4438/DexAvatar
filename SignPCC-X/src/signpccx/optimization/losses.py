from __future__ import annotations

import torch


def safe_project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project while preserving negative-z camera conventions."""
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    homogeneous = torch.matmul(intrinsics, points.transpose(1, 2)).transpose(1, 2)
    denominator = homogeneous[..., 2:3]
    safe = torch.where(
        denominator.abs() < 1e-8,
        torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator)) * 1e-8,
        denominator,
    )
    return homogeneous[..., :2] / safe


def affine_homogeneous(affine: torch.Tensor) -> torch.Tensor:
    if affine.ndim == 2:
        affine = affine.unsqueeze(0)
    bottom = torch.zeros((*affine.shape[:-2], 1, 3), dtype=affine.dtype, device=affine.device)
    bottom[..., 0, 2] = 1.0
    return torch.cat((affine, bottom), dim=-2)


def geman_mcclure(residual: torch.Tensor, sigma: float) -> torch.Tensor:
    squared = residual.square()
    sigma2 = float(sigma) ** 2
    return sigma2 * squared / (sigma2 + squared)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.sum().clamp_min(eps)


def keypoint_loss(
    predicted_xy: torch.Tensor,
    observed_xy: torch.Tensor,
    confidence: torch.Tensor,
    image_hw: tuple[int, int],
    part_weight: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    h, w = image_hw
    residual = (predicted_xy - observed_xy) / (float(h * h + w * w) ** 0.5)
    robust = geman_mcclure(residual, sigma=0.02).sum(dim=-1)
    weight = confidence.clamp(0, 1).square() * part_weight
    return weighted_mean(robust, weight)


def centered_point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    residual = torch.linalg.vector_norm(prediction - target, dim=-1)
    if weight is None:
        return residual.mean()
    return weighted_mean(residual, weight)


def unit_vector(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def arm_chain_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    indices: tuple[int, int, int],
    confidence: torch.Tensor,
    length_coefficient: float = 5.0,
) -> torch.Tensor:
    shoulder, elbow, wrist = indices
    pred_upper = predicted[:, elbow] - predicted[:, shoulder]
    pred_fore = predicted[:, wrist] - predicted[:, elbow]
    target_upper = target[:, elbow] - target[:, shoulder]
    target_fore = target[:, wrist] - target[:, elbow]
    direction = (
        1.0 - (unit_vector(pred_upper) * unit_vector(target_upper)).sum(-1)
        + 1.0 - (unit_vector(pred_fore) * unit_vector(target_fore)).sum(-1)
    )
    length = (
        (pred_upper.norm(dim=-1) - target_upper.norm(dim=-1)).abs()
        + (pred_fore.norm(dim=-1) - target_fore.norm(dim=-1)).abs()
    )
    return (confidence * (direction + float(length_coefficient) * length)).mean()


def pose_anatomy_loss(body_pose: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # Soft trust region: discourages pathological rotations without imposing a
    # learned pose prior or any temporal evidence.
    body_excess = torch.relu(body_pose.norm(dim=-1) - 2.8).square().mean()
    hand_excess = 0.5 * (
        torch.relu(left.norm(dim=-1) - 2.6).square().mean()
        + torch.relu(right.norm(dim=-1) - 2.6).square().mean()
    )
    return body_excess + hand_excess


def point_segment_closest(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    vector = end - start
    t = ((point - start) * vector).sum(-1, keepdim=True) / vector.square().sum(-1, keepdim=True).clamp_min(1e-12)
    return start + t.clamp(0.0, 1.0) * vector


def signed_point_to_triangle(point: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
    """Differentiable signed distance to selected oriented triangles."""
    a, b, c = triangle.unbind(dim=-2)
    normal_raw = torch.cross(b - a, c - a, dim=-1)
    normal = normal_raw / normal_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    plane_distance = ((point - a) * normal).sum(-1, keepdim=True)
    projected = point - plane_distance * normal
    v0, v1, v2 = b - a, c - a, projected - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denominator = (d00 * d11 - d01.square()).clamp_min(1e-12)
    bary_v = (d11 * d20 - d01 * d21) / denominator
    bary_w = (d00 * d21 - d01 * d20) / denominator
    bary_u = 1.0 - bary_v - bary_w
    inside = (bary_u >= 0) & (bary_v >= 0) & (bary_w >= 0)
    edge_points = torch.stack((
        point_segment_closest(point, a, b),
        point_segment_closest(point, b, c),
        point_segment_closest(point, c, a),
    ), dim=-2)
    edge_distance = torch.linalg.vector_norm(edge_points - point.unsqueeze(-2), dim=-1)
    nearest_edge = edge_distance.argmin(dim=-1)
    closest_edge = torch.gather(
        edge_points, -2, nearest_edge[..., None, None].expand(*nearest_edge.shape, 1, 3)
    ).squeeze(-2)
    closest = torch.where(inside[..., None], projected, closest_edge)
    unsigned = torch.linalg.vector_norm(point - closest, dim=-1)
    sign = torch.sign(((point - closest) * normal).sum(-1))
    # Exactly-on-surface samples have zero distance regardless of sign.
    return unsigned * sign


def selected_surface_signed_distances(
    query: torch.Tensor,
    vertices: torch.Tensor,
    surface_faces: torch.Tensor,
) -> torch.Tensor:
    """Select closest triangles without gradient, then retain distance gradient."""
    triangles = vertices[:, surface_faces]
    with torch.no_grad():
        centroids = triangles.mean(dim=-2)
        selected = torch.cdist(query, centroids).argmin(dim=-1)
    batch = torch.arange(vertices.shape[0], device=vertices.device)[:, None]
    nearest = triangles[batch, selected]
    return signed_point_to_triangle(query, nearest)


def penetration_from_surfaces(
    vertices: torch.Tensor,
    query_ids: torch.Tensor,
    surface_faces: torch.Tensor,
    margin_m: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signed = selected_surface_signed_distances(vertices[:, query_ids], vertices, surface_faces)
    depth = torch.relu(float(margin_m) - signed)
    return depth.square().mean(), depth.max(), (depth > 0).sum()

