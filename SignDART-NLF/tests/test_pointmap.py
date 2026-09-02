import numpy as np

from signdart.pointmap import (
    block_bootstrap_axes,
    mask_bone_endpoints,
    pointmap_bootstrap_decision,
    robust_axis,
)


def test_robust_axis_and_bootstrap_recover_oriented_line():
    rng = np.random.default_rng(7)
    progress = np.linspace(0.0, 1.0, 256)
    points = np.stack((progress, np.zeros_like(progress), progress), axis=1)
    points += rng.normal(scale=0.002, size=points.shape)
    pixels = np.stack((20.0 + 100.0 * progress, 40.0 + 20.0 * progress), axis=1)
    axis, quality = robust_axis(points, pixels, pixels[0], pixels[-1])
    expected = np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    assert float(axis @ expected) > 0.999
    assert quality["eigen_gap"] > 0.99
    bootstrap = block_bootstrap_axes(
        points, pixels, pixels[0], pixels[-1], axis, "synthetic", repetitions=64
    )
    assert bootstrap.shape[0] >= 32
    assert np.min(bootstrap @ expected) > 0.99


def test_pointmap_bootstrap_selector_accepts_consistent_alternative():
    incumbent = np.zeros((55, 3), dtype=np.float64)
    alternative = incumbent.copy()
    ids = (16, 18, 20)
    incumbent[list(ids)] = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]
    ])
    alternative[list(ids)] = np.asarray([
        [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]
    ])
    axes = np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    boot = np.repeat(axes[:, None, :], 64, axis=1)
    selected, diagnostics = pointmap_bootstrap_decision(
        np.stack((incumbent, alternative)), axes, boot,
        np.ones(2), "left", 0,
    )
    assert selected == 1
    assert diagnostics["reason"] == "pointmap_branch_accepted"


def test_mask_bone_endpoints_follow_upper_to_forearm_chain():
    upper = np.zeros((100, 100), dtype=bool)
    forearm = np.zeros_like(upper)
    upper[20:55, 40:48] = True
    forearm[52:60, 43:80] = True
    result = mask_bone_endpoints(upper, forearm)
    shoulder, elbow_upper = result["upper"]
    elbow_forearm, wrist = result["forearm"]
    assert shoulder[1] < elbow_upper[1]
    assert wrist[0] > elbow_forearm[0]
