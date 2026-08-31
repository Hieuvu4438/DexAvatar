import pytest

from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.synthetic import make_observations, make_state
from dcg_sign4d.synthetic_pipeline import build_smoke_reconstructor


class _FailingSampler:
    def __init__(self, wrapped, *, failures, exception_type=FloatingPointError):
        self.wrapped = wrapped
        self.guidance_scale = 2.0
        self.failures = failures
        self.exception_type = exception_type
        self.calls = []

    def sample(self, *args, **kwargs):
        self.calls.append((kwargs["seed"], kwargs.get("guidance_scale_override")))
        if self.failures:
            self.failures -= 1
            raise self.exception_type("fixture failure")
        return self.wrapped.sample(*args, **kwargs)


class _CountingProposal:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.wrapped(*args, **kwargs)


def _reconstructor(failures, exception_type=FloatingPointError):
    state = make_state(time=6)
    patch_map = PatchMap.load("assets/patch_maps/synthetic_smoke.yaml")
    reconstructor = build_smoke_reconstructor(
        state,
        patch_map,
        rounds=1,
        diffusion_steps=4,
        retry_guidance_factor=0.25,
    )
    failing = _FailingSampler(
        reconstructor.sampler,
        failures=failures,
        exception_type=exception_type,
    )
    reconstructor.sampler = failing
    return state, reconstructor, failing


def test_numerical_failure_retries_once_with_same_seed_and_lower_guidance():
    state, reconstructor, sampler = _reconstructor(1)
    hypothesis = reconstructor.reconstruct(state, make_observations(time=6))[0]
    assert hypothesis.status == "ok"
    assert hypothesis.retry_count == 1
    assert len(sampler.calls) == 2
    assert sampler.calls[0][0] == sampler.calls[1][0]
    assert sampler.calls[0][1] is None
    assert sampler.calls[1][1] == 0.5


def test_second_numerical_failure_returns_initializer_with_status():
    state, reconstructor, sampler = _reconstructor(2)
    hypothesis = reconstructor.reconstruct(state, make_observations(time=6))[0]
    assert hypothesis.status == "fallback_initialization"
    assert hypothesis.retry_count == 1
    assert len(sampler.calls) == 2
    assert hypothesis.trajectory is state


def test_configuration_error_is_not_hidden_as_numerical_fallback():
    state, reconstructor, _ = _reconstructor(1, ValueError)
    with pytest.raises(ValueError, match="fixture failure"):
        reconstructor.reconstruct(state, make_observations(time=6))


def test_single_pass_control_does_not_reinfer_graph_after_sampling():
    state = make_state(time=6)
    reconstructor = build_smoke_reconstructor(
        state,
        PatchMap.load("assets/patch_maps/synthetic_smoke.yaml"),
        rounds=1,
        diffusion_steps=2,
        alternating=False,
    )
    proposal = _CountingProposal(reconstructor.proposal)
    reconstructor.proposal = proposal
    reconstructor.reconstruct(state, make_observations(time=6))
    assert proposal.calls == 1


def test_single_pass_control_rejects_multiple_rounds():
    with pytest.raises(ValueError, match="exactly one"):
        build_smoke_reconstructor(
            make_state(time=6),
            PatchMap.load("assets/patch_maps/synthetic_smoke.yaml"),
            rounds=2,
            alternating=False,
        )
