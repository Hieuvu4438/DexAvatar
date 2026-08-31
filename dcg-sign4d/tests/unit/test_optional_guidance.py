from dataclasses import replace

import torch

from dcg_sign4d.guidance.depth import RelativeDepthGuidance
from dcg_sign4d.guidance.silhouette import SilhouetteGuidance
from dcg_sign4d.guidance.track import TrackGuidance
from dcg_sign4d.synthetic import make_graph, make_observations, make_state


def test_missing_optional_cues_have_exact_zero_gradient():
    state = make_state()
    observations = make_observations()
    graph = make_graph()
    parameter = state.root_translation.clone().requires_grad_()
    terms = (
        SilhouetteGuidance(
            lambda _: parameter[..., 0][:, :, None, None, None].expand(-1, -1, 2, 4, 4)
        ),
        TrackGuidance(lambda _: parameter[:, :, None, :2].expand(-1, -1, 3, -1)),
        RelativeDepthGuidance(lambda _: parameter[..., :1].expand(-1, -1, 2)),
    )
    total = sum(term.loss(state, observations, graph) for term in terms)
    total.backward()
    assert parameter.grad is not None
    assert parameter.grad.abs().sum() == 0


def test_optional_cues_produce_finite_nonzero_gradients():
    state = make_state()
    base = make_observations()
    observations = replace(
        base,
        part_masks=torch.ones(1, 4, 1, 2, 2),
        mask_reliability=torch.ones(1, 4, 1),
        tracks_2d=torch.zeros(1, 4, 1, 2),
        track_reliability=torch.ones(1, 4, 1),
        depth_order=torch.ones(1, 4, 1),
        depth_reliability=torch.ones(1, 4, 1),
    ).validate()
    graph = make_graph()
    parameter = torch.full((1, 4, 1), 0.25, requires_grad=True)
    mask = torch.sigmoid(parameter[:, :, :, None, None]).expand(-1, -1, -1, 2, 2)
    track = torch.stack((parameter, parameter.square()), dim=-1)
    depth = parameter
    total = SilhouetteGuidance(lambda _: mask).loss(state, observations, graph)
    total += TrackGuidance(lambda _: track).loss(state, observations, graph)
    total += RelativeDepthGuidance(lambda _: depth).loss(state, observations, graph)
    total.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0
