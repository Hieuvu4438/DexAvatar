import torch

from signal4d.factors.observation_2d import observation_2d_factor
from signal4d.factors.observation_3d import observation_3d_factor
from signal4d.factors.prior import pose_prior_factor
from signal4d.factors.temporal import adaptive_weights, temporal_position_factor


def test_observation_mask_and_sigma() -> None:
    observations = torch.zeros(5, 2, 3, 3, dtype=torch.float64)
    valid = torch.ones(5, 2, 3, dtype=torch.bool)
    state = torch.zeros(5, 3, 3, dtype=torch.float64, requires_grad=True)
    sigma_small = torch.ones_like(observations) * 0.01
    baseline = observation_3d_factor(state, observations, valid, sigma_small)
    assert baseline.loss == 0
    observations[0, 0, 0] = 1
    high_force = observation_3d_factor(state, observations, valid, sigma_small).loss
    low_force = observation_3d_factor(state, observations, valid, sigma_small * 10).loss
    assert high_force > low_force
    valid[0, 0, 0] = False
    assert observation_3d_factor(state, observations, valid, sigma_small).loss == 0


def test_constant_velocity_temporal_zero() -> None:
    time = torch.arange(8, dtype=torch.float64)[:, None, None]
    joints = time * torch.ones(8, 3, 3, dtype=torch.float64)
    weights = torch.ones(8, 3, dtype=torch.float64)
    assert temporal_position_factor(joints, 25.0, weights).loss < 1e-12


def test_change_point_reduces_temporal_weight() -> None:
    uncertainty = torch.ones(8, 3)
    low = adaptive_weights(uncertainty, torch.zeros(8))
    change = torch.zeros(8)
    change[4] = 1
    high = adaptive_weights(uncertainty, change)
    assert high[4].mean() < low[4].mean()


def test_observation_2d_and_pose_prior_are_zero_at_target() -> None:
    joints = torch.tensor([[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]])
    camera = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    observed = torch.tensor([[[[50.0, 50.0], [100.0, 50.0]]]])
    reprojection = observation_2d_factor(
        joints,
        observed,
        torch.ones((1, 1, 2), dtype=torch.bool),
        camera,
        torch.tensor([100.0, 100.0]),
    )
    assert float(reprojection.loss) < 1e-8
    rotations = torch.eye(3).expand(2, 4, 3, 3).clone()
    assert float(pose_prior_factor(rotations, rotations).loss) < 1e-8
