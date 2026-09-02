import numpy as np

from signdart.geometry.ray_sphere import (
    enumerate_three_link_branches,
    pixel_ray,
    positive_sphere_ray_roots,
    project,
)


def test_solver_recovers_known_positive_depth_child():
    K = np.asarray([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    parent = np.asarray([0.1, -0.1, 2.0])
    child = np.asarray([0.2, 0.05, 2.3])
    ray = pixel_ray(K, project(K, child))
    roots = positive_sphere_ray_roots(parent, ray, np.linalg.norm(child - parent))
    assert any(np.linalg.norm(root - child) < 1e-8 for root in roots)


def test_tangent_is_unique_and_negative_discriminant_is_empty():
    roots = positive_sphere_ray_roots(
        np.asarray([1.0, 0.0, 2.0]), np.asarray([0.0, 0.0, 1.0]), 1.0
    )
    assert len(roots) == 1
    assert np.allclose(roots[0], [0.0, 0.0, 2.0])
    assert positive_sphere_ray_roots(
        np.asarray([2.0, 0.0, 2.0]), np.asarray([0.0, 0.0, 1.0]), 1.0
    ) == []


def test_negative_depth_roots_are_removed():
    roots = positive_sphere_ray_roots(
        np.asarray([0.0, 0.0, -2.0]), np.asarray([0.0, 0.0, 1.0]), 0.5
    )
    assert roots == []


def test_three_link_tree_contains_known_chain():
    K = np.asarray([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
    collar = np.asarray([-0.1, 0.2, 2.0])
    shoulder = np.asarray([-0.25, 0.18, 2.1])
    elbow = np.asarray([-0.42, 0.05, 2.3])
    wrist = np.asarray([-0.55, -0.08, 2.45])
    branches = enumerate_three_link_branches(
        collar,
        project(K, shoulder),
        project(K, elbow),
        project(K, wrist),
        np.linalg.norm(shoulder - collar),
        np.linalg.norm(elbow - shoulder),
        np.linalg.norm(wrist - elbow),
        K,
    )
    assert any(
        np.linalg.norm(item["shoulder"] - shoulder) < 1e-8
        and np.linalg.norm(item["elbow"] - elbow) < 1e-8
        and np.linalg.norm(item["wrist"] - wrist) < 1e-8
        for item in branches
    )
