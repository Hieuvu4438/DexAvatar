import pytest
import torch

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.geometry.contact_geometry import ContactGeometry
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.inference.ranking import HypothesisRanker, RankingWeights
from dcg_sign4d.synthetic import make_observations, make_state
from dcg_sign4d.synthetic_pipeline import SyntheticGeometryAdapter


def _fixture():
    state = make_state(time=3)
    observations = make_observations(time=3)
    patch_map = PatchMap.load("assets/patch_maps/synthetic_smoke.yaml")
    geometry = SyntheticGeometryAdapter(
        ContactGeometry(patch_map, fps=30, allow_missing_penetration=True)
    )(state)
    shape = (1, 3, len(patch_map.admissible_edges))
    probability = torch.zeros(*shape, 4)
    probability[..., 0] = 1
    graph = ContactGraphBatch(
        event_state=torch.zeros(shape, dtype=torch.long),
        event_probability=probability,
        edge_valid=torch.ones(shape[0], shape[2], dtype=torch.bool),
        uncertain_mask=torch.zeros(shape, dtype=torch.bool),
        segment_id=torch.zeros(shape, dtype=torch.long),
        segment_duration=torch.zeros(shape),
    )
    return state, observations, graph, geometry


def test_ranker_requires_hypothesis_dependent_observation_score():
    state, observations, graph, geometry = _fixture()
    ranker = HypothesisRanker(RankingWeights(1, 1, 1, 1), allow_missing_penetration=True)
    with pytest.raises(RuntimeError, match="audited observation residual"):
        ranker.terms(state, graph, observations, geometry)


def test_development_ranker_uses_explicit_zero_for_missing_observation_score():
    state, observations, graph, geometry = _fixture()
    ranker = HypothesisRanker(
        RankingWeights(1, 1, 1, 1),
        allow_missing_observation_score=True,
        allow_missing_penetration=True,
    )
    assert ranker.terms(state, graph, observations, geometry)["observation"] == 0.0


def test_ranker_requires_signed_penetration_by_default():
    state, observations, graph, geometry = _fixture()
    ranker = HypothesisRanker(RankingWeights(1, 1, 1, 1), observation_score=lambda *_: -1.0)
    with pytest.raises(RuntimeError, match="requires signed penetration geometry"):
        ranker.terms(state, graph, observations, geometry)
