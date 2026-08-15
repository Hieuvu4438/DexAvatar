import torch

from phase2_refiner.geometry.rotations import axis_angle_to_matrix
from phase2_refiner.evaluate_t1_vertices import _vertex_error
from phase2_refiner.losses.geometry import (
    balanced_region_vertex_loss,
    regional_vertex_errors,
)
from phase2_refiner.losses.sequence import RefinerLoss


REGION_MASKS = {
    "ubody": torch.tensor([0]),
    "lhand": torch.tensor([1]),
    "rhand": torch.tensor([2]),
}


def test_regional_vertex_errors_remove_per_region_translation() -> None:
    target = torch.zeros(1, 1, 6, 3)
    prediction = target.clone()
    prediction[..., :2, 0] += 3.0
    prediction[..., 2:4, 1] -= 2.0
    prediction[..., 4:, 2] += 5.0
    masks = {
        "ubody": torch.tensor([0, 1]),
        "lhand": torch.tensor([2, 3]),
        "rhand": torch.tensor([4, 5]),
    }

    centered = regional_vertex_errors(
        prediction, target, masks, translation_centered=True
    )
    absolute = regional_vertex_errors(
        prediction, target, masks, translation_centered=False
    )

    assert all(torch.count_nonzero(value) == 0 for value in centered.values())
    assert all(torch.all(value > 0) for value in absolute.values())


def test_t1_vertex_error_uses_the_same_regional_centering() -> None:
    target = torch.zeros(1, 1, 2, 3)
    prediction = target + torch.tensor([3.0, -2.0, 1.0])

    centered = _vertex_error(
        prediction,
        target,
        torch.tensor([0, 1]),
        translation_centered=True,
    )

    assert torch.count_nonzero(centered) == 0


def test_balanced_vertex_loss_honors_target_quality_weights() -> None:
    target = torch.zeros(1, 2, 3, 3)
    prediction = target.clone()
    prediction[:, 0, :, 0] = 100.0
    prediction[:, 1, :, 0] = 1.0
    weights = {name: torch.tensor([[0.0, 1.0]]) for name in REGION_MASKS}

    loss = balanced_region_vertex_loss(
        prediction,
        target,
        REGION_MASKS,
        frame_valid=torch.ones(1, 2, dtype=torch.bool),
        region_frame_weight=weights,
    )

    assert loss == 1.0


def test_vertex_benefit_labels_follow_mesh_improvement() -> None:
    initial_matrix = axis_angle_to_matrix(torch.zeros(1, 1, 51, 3))
    target_matrix = initial_matrix.clone()
    initial_vertices = torch.zeros(1, 1, 3, 3)
    initial_vertices[..., 0, 0] = 1.0
    predicted_vertices = torch.zeros_like(initial_vertices)
    predicted_vertices[..., 1:, 0] = 1.0
    target_vertices = torch.zeros_like(initial_vertices)
    prediction = {
        "matrix": initial_matrix.clone(),
        "raw_delta": torch.zeros(1, 1, 51, 3),
        "benefit_logit": torch.tensor([[[10.0, -10.0, -10.0]]]),
    }
    loss_fn = RefinerLoss(
        benefit_target="vertex",
        benefit_weight=1.0,
        benefit_margin_mm=1.0,
        vertex_translation_centered=False,
    )

    losses = loss_fn(
        prediction,
        initial_matrix,
        target_matrix,
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 51, dtype=torch.bool),
        torch.ones(1, 1, 51),
        predicted_vertices=predicted_vertices,
        initial_vertices=initial_vertices,
        target_vertices=target_vertices,
        vertex_region_masks=REGION_MASKS,
    )

    assert losses["benefit"] < 1e-3
