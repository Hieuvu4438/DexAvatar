import torch

from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.dposer_normalizer import DPoserXWholeBodyNormalizer, ZScoreStats
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.diffusion.trajectory_denoiser import (
    DPoserXConditionedTrajectoryDenoiser,
    PartAwareTrajectoryDenoiser,
)
from dcg_sign4d.synthetic import make_graph, make_state


def components():
    state = make_state()
    encoded, context = StateCodec().encode(state)
    part_dims = (
        context.widths[0] + context.widths[1] + context.widths[2],
        context.widths[3],
        context.widths[4],
        context.widths[5],
    )
    denoiser = PartAwareTrajectoryDenoiser(part_dims, hidden_dim=16, heads=4, layers=1)
    encoder = ContactTokenEncoder(edge_count=2, hidden_dim=16)
    return state, encoded, denoiser, encoder


def test_schedule_forward_and_clean_recovery():
    schedule = DiffusionSchedule(10)
    clean = torch.randn(2, 4, 9)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([0, 9])
    noisy = schedule.q_sample(clean, timestep, noise)
    recovered = schedule.predict_clean(noisy, timestep, noise)
    assert torch.allclose(clean, recovered, atol=1e-5)


def test_matched_contact_modes_and_denoiser_shapes():
    state, encoded, denoiser, encoder = components()
    graph = make_graph()
    counts = {sum(parameter.numel() for parameter in denoiser.parameters())}
    outputs = []
    for mode in ("null", "static", "dynamic"):
        token = encoder(graph, mode)
        outputs.append(
            denoiser(encoded, torch.tensor([3]), token, state.valid_mask, torch.ones(1, 4))
        )
        counts.add(sum(parameter.numel() for parameter in denoiser.parameters()))
    assert len(counts) == 1
    assert all(output.shape == encoded.shape for output in outputs)
    assert torch.isfinite(torch.stack(outputs)).all()


def test_padding_is_zeroed():
    state, encoded, denoiser, encoder = components()
    state.valid_mask[:, -1] = False
    output = denoiser(
        encoded,
        torch.tensor([3]),
        encoder(make_graph(), "dynamic"),
        state.valid_mask,
        torch.ones(1, 4),
    )
    assert output[:, -1].abs().sum() == 0


def test_invalid_edge_is_encoded_as_null_independent_of_event():
    _, _, _, encoder = components()
    first = make_graph()
    first.edge_valid[:, 0] = False
    second = make_graph()
    second.edge_valid[:, 0] = False
    second.event_state[:, :, 0] = 2
    second.event_probability[:, :, 0, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    first_token = encoder(first, "dynamic")
    second_token = encoder(second, "dynamic")
    assert torch.allclose(first_token[:, :, 0], second_token[:, :, 0])


class _FakeOfficialBridge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.normalizer = DPoserXWholeBodyNormalizer(
            {
                name: ZScoreStats(torch.zeros(size), torch.ones(size))
                for name, size in DPoserXWholeBodyNormalizer.PARTS
            }
        )

    def predict_noise(self, normalized, timesteps, *, trajectory_steps):
        assert trajectory_steps == 10
        return normalized * self.scale + timesteps[:, None] / 10


def test_dposer_conditioned_holistic_denoiser_shape_and_input_gradient():
    model = DPoserXConditionedTrajectoryDenoiser(
        _FakeOfficialBridge(), trajectory_steps=10, hidden_dim=16, heads=4, layers=1
    )
    noisy = torch.randn(2, 3, 337, requires_grad=True)
    output = model(
        noisy,
        torch.tensor([2, 7]),
        torch.randn(2, 3, 5, 16),
        torch.ones(2, 3, dtype=torch.bool),
        shape=torch.randn(2, 10),
    )
    assert output.shape == noisy.shape
    output.square().mean().backward()
    assert noisy.grad is not None and torch.isfinite(noisy.grad).all()


def test_dposer_holistic_denoiser_is_shape_conditioned():
    torch.manual_seed(91)
    model = DPoserXConditionedTrajectoryDenoiser(
        _FakeOfficialBridge(), trajectory_steps=10, hidden_dim=16, heads=4, layers=1
    ).eval()
    noisy = torch.randn(1, 3, 337)
    arguments = (
        noisy,
        torch.tensor([2]),
        torch.randn(1, 3, 5, 16),
        torch.ones(1, 3, dtype=torch.bool),
    )
    first = model(*arguments, shape=torch.zeros(1, 10))
    second = model(*arguments, shape=torch.ones(1, 10))
    assert not torch.allclose(first, second)
