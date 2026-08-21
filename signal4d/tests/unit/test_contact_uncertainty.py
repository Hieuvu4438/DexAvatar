import numpy as np
import torch

from signal4d.evaluation.bootstrap import paired_hierarchical_bootstrap
from signal4d.evaluation.contact import binary_contact_metrics, event_metrics
from signal4d.evaluation.uncertainty import risk_coverage_curve
from signal4d.factors.collision import collision_factor
from signal4d.factors.contact import contact_factor
from signal4d.models.contact_proposer import (
    ContactEdgeSpec,
    decode_hysteresis,
    propose_contacts,
)
from signal4d.models.uncertainty import GroupCalibration


def test_contact_hysteresis_and_metrics() -> None:
    probability = torch.tensor([[0.1], [0.7], [0.6], [0.3], [0.2]])
    distance = torch.tensor([[0.1], [0.02], [0.03], [0.03], [0.06]])
    decoded = decode_hysteresis(probability, distance, 0.65, 0.35, 0.025, 0.04)
    target = torch.tensor([[False], [True], [True], [False], [False]])
    assert torch.equal(decoded, target)
    assert binary_contact_metrics(decoded, target)["f1"] == 1
    assert event_metrics(decoded, target)["event_macro_f1"] == 1


def test_group_calibration_and_risk_curve() -> None:
    residual = torch.tensor([1.0, 2.0, 1.0, 4.0])
    sigma = torch.ones(4)
    calibration = GroupCalibration.fit(residual, sigma, ["a", "a", "b", "b"], 0.5)
    assert set(calibration.scales) == {"a", "b"}
    curve = risk_coverage_curve(torch.tensor([0.1, 0.2, 2.0]), torch.tensor([0.1, 0.2, 0.9]))
    assert curve["aurc"] < 1


def test_paired_hierarchical_bootstrap() -> None:
    candidate = np.array([1.0, 2.0, 1.5, 2.5])
    baseline = candidate + 0.5
    result = paired_hierarchical_bootstrap(
        candidate,
        baseline,
        np.array(["a", "a", "b", "b"]),
        np.array(["a1", "a2", "b1", "b2"]),
        replicates=200,
        seed=3,
    )
    assert result.point_estimate == -0.5
    assert result.ci_high < 0


def test_contact_attraction_and_collision_have_pose_gradients() -> None:
    joints = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.002, 0.0, 0.0]]],
        requires_grad=True,
    )
    edge = ContactEdgeSpec("synthetic", 0, 1, 0.006, 0.025, 0.04)
    candidates = propose_contacts(joints.detach(), (edge,), proposal_radius_m=0.08)
    contact = contact_factor(
        joints,
        torch.full((1, 1), 5.0),
        candidates,
        torch.zeros(1),
    )
    collision = collision_factor(joints, ((0, 2),), minimum_distance_m=0.008)
    total = contact.loss + collision.loss
    total.backward()
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
    assert float(joints.grad.abs().sum()) > 0
