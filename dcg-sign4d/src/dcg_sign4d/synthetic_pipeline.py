"""Authorized development-only end-to-end wiring fixture."""

from __future__ import annotations

import torch

from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.contact.semi_markov import SemiMarkovDecoder
from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.sampler import GuidedTrajectorySampler
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.diffusion.trajectory_denoiser import PartAwareTrajectoryDenoiser
from dcg_sign4d.geometry.contact_geometry import ContactGeometry, GeometryOutput
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.inference.alternating import AlternatingReconstructor
from dcg_sign4d.inference.ranking import HypothesisRanker, RankingWeights


class SyntheticGeometryAdapter:
    def __init__(self, geometry: ContactGeometry):
        self.geometry = geometry
        offsets = torch.zeros(8, 3)
        offsets[:, 0] = torch.linspace(0, 0.07, 8)
        self.offsets = offsets

    def __call__(self, state: TrajectoryState) -> GeometryOutput:
        batch, time = state.root_translation.shape[:2]
        vertices = self.offsets.to(state.root_translation)[None, None].expand(batch, time, -1, -1)
        vertices = vertices + state.root_translation[:, :, None]
        vertices = vertices.clone()
        vertices[:, :, 0:2, 1] += state.left_hand_rot6d[:, :, :1, 0]
        vertices[:, :, 2:4, 1] += state.right_hand_rot6d[:, :, :1, 0]
        return self.geometry.features(vertices)


def build_smoke_reconstructor(
    state: TrajectoryState,
    patch_map: PatchMap,
    *,
    rounds: int = 1,
    diffusion_steps: int = 4,
    num_hypotheses: int = 1,
    seed: int = 12345,
    retry_guidance_factor: float | None = None,
    alternating: bool = True,
) -> AlternatingReconstructor:
    torch.manual_seed(seed)
    encoded, context = StateCodec().encode(state)
    trajectory_dim = encoded.shape[-1]
    part_dims = (
        context.widths[0] + context.widths[1] + context.widths[2],
        context.widths[3],
        context.widths[4],
        context.widths[5],
    )
    edge_count = len(patch_map.admissible_edges)
    hidden = 16
    geometry = ContactGeometry(patch_map, fps=30, allow_missing_penetration=True)
    geometry_adapter = SyntheticGeometryAdapter(geometry)
    proposal = ContactProposal(
        trajectory_dim, edge_count, max_duration=32, hidden_dim=hidden, heads=4, layers=1
    )
    token_encoder = ContactTokenEncoder(edge_count, hidden)
    denoiser = PartAwareTrajectoryDenoiser(part_dims, hidden_dim=hidden, heads=4, layers=1)
    sampler = GuidedTrajectorySampler(
        denoiser,
        DiffusionSchedule(10),
        StateCodec(),
        token_encoder,
        guidance_scale=0,
        gradient_clip_norm=1,
        trust_region_norm=10,
    )
    return AlternatingReconstructor(
        proposal,
        SemiMarkovDecoder(32, fps=30),
        sampler,
        geometry_adapter,
        HypothesisRanker(
            RankingWeights(1, 1, 1, 1),
            allow_missing_observation_score=True,
            allow_missing_penetration=True,
        ),
        torch.ones(1, edge_count, dtype=torch.bool),
        rounds=rounds,
        diffusion_steps=diffusion_steps,
        num_hypotheses=num_hypotheses,
        base_seed=seed,
        retry_guidance_factor=retry_guidance_factor,
        alternating=alternating,
    )
