import torch

from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.synthetic import make_observations, make_state


def test_contact_proposal_shapes_and_padding():
    state = make_state(batch=2, time=5)
    width = StateCodec().encode(state)[0].shape[-1]
    model = ContactProposal(width, edge_count=3, max_duration=5, hidden_dim=16, heads=4, layers=1)
    result = model(make_observations(2, 5), state, torch.randn(2, 5, 3, 5))
    assert result.event_logits.shape == (2, 5, 3, 4)
    assert result.duration_logits.shape == (2, 5, 3, 5)
    assert result.edge_embedding.shape == (2, 5, 3, 16)
    assert torch.isfinite(result.event_logits).all()
