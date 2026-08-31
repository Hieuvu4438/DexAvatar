import torch
from torch import nn

from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.sampler import GuidedTrajectorySampler
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.synthetic import make_graph, make_observations, make_state


class ZeroDenoiser(nn.Module):
    def forward(self, noisy, timesteps, contact_tokens, valid_mask, reliability, shape=None):
        del timesteps, contact_tokens, valid_mask, reliability, shape
        return torch.zeros_like(noisy)


class TranslationTerm:
    name = "translation"

    def loss(self, clean_state, observations, graph):
        del observations, graph
        return clean_state.root_translation.square().mean()


class MissingTerm:
    name = "missing"

    def loss(self, clean_state, observations, graph):
        del observations, graph
        return clean_state.root_translation.sum() * 0


def sampler(terms=(), scale=0.0):
    return GuidedTrajectorySampler(
        ZeroDenoiser(),
        DiffusionSchedule(6),
        StateCodec(),
        ContactTokenEncoder(2, 8),
        terms,
        guidance_scale=scale,
        gradient_clip_norm=0.01,
    )


def test_zero_guidance_matches_base_and_seed_is_deterministic():
    state, graph, observations = make_state(), make_graph(), make_observations()
    first, _ = sampler().sample(state, graph, observations, seed=7, num_steps=4)
    second, _ = sampler((TranslationTerm(),), 0.0).sample(
        state, graph, observations, seed=7, num_steps=4
    )
    third, _ = sampler().sample(state, graph, observations, seed=7, num_steps=4)
    assert torch.equal(first.root_translation, second.root_translation)
    assert torch.equal(first.root_translation, third.root_translation)


def test_different_seeds_are_not_identical():
    state, graph, observations = make_state(), make_graph(), make_observations()
    first, _ = sampler().sample(state, graph, observations, seed=1, num_steps=4)
    second, _ = sampler().sample(state, graph, observations, seed=2, num_steps=4)
    assert not torch.equal(first.root_translation, second.root_translation)


def test_guidance_clips_and_missing_term_has_zero_gradient():
    state, graph, observations = make_state(), make_graph(), make_observations()
    output, diagnostics = sampler((TranslationTerm(), MissingTerm()), 1.0).sample(
        state, graph, observations, seed=3, num_steps=4
    )
    assert torch.isfinite(output.root_translation).all()
    assert diagnostics.clip_count > 0
    assert all(value == 0 for value in diagnostics.gradient_norms["missing"])
