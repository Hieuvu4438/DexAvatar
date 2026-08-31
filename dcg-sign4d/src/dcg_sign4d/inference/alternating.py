"""Independent-hypothesis alternating geometry-contact inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import torch

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.contact.proposal import ContactProposal
from dcg_sign4d.contact.semi_markov import SemiMarkovDecoder
from dcg_sign4d.diffusion.sampler import GuidedTrajectorySampler
from dcg_sign4d.diffusion.state_codec import TrajectoryState, rotation_6d_to_matrix
from dcg_sign4d.geometry.contact_geometry import GeometryOutput
from dcg_sign4d.inference.hypothesis import Hypothesis, RoundResult
from dcg_sign4d.inference.ranking import HypothesisRanker
from dcg_sign4d.observations.schema import ObservationBatch


class AlternatingReconstructor:
    def __init__(
        self,
        proposal: ContactProposal,
        decoder: SemiMarkovDecoder,
        sampler: GuidedTrajectorySampler,
        geometry_from_state: Callable[[TrajectoryState], GeometryOutput],
        ranker: HypothesisRanker,
        edge_valid: torch.Tensor,
        *,
        rounds: int,
        diffusion_steps: int,
        num_hypotheses: int,
        base_seed: int,
        retry_guidance_factor: float | None = None,
        alternating: bool = True,
    ) -> None:
        if rounds < 1 or diffusion_steps < 1 or num_hypotheses < 1:
            raise ValueError("rounds, steps and hypotheses must be positive")
        self.proposal = proposal
        self.decoder = decoder
        self.sampler = sampler
        self.geometry_from_state = geometry_from_state
        self.ranker = ranker
        self.edge_valid = edge_valid
        self.rounds = rounds
        self.diffusion_steps = diffusion_steps
        self.num_hypotheses = num_hypotheses
        self.base_seed = base_seed
        if retry_guidance_factor is not None and not 0 <= retry_guidance_factor < 1:
            raise ValueError("retry guidance factor must lie in [0,1)")
        self.retry_guidance_factor = retry_guidance_factor
        if not alternating and rounds != 1:
            raise ValueError("single-pass inference requires exactly one sampling round")
        self.alternating = alternating

    @staticmethod
    def derive_seed(base_seed: int, hypothesis: int, round_index: int = 0) -> int:
        return (base_seed * 1_000_003 + hypothesis * 10_007 + round_index * 101) % (2**31)

    @staticmethod
    def derive_round_seed(hypothesis_seed: int, round_index: int) -> int:
        return (hypothesis_seed * 1_000_003 + round_index * 101) % (2**31)

    @staticmethod
    def validate_trajectory(state: TrajectoryState) -> None:
        state.validate()
        for rotations in (
            state.root_rot6d,
            state.body_rot6d,
            state.left_hand_rot6d,
            state.right_hand_rot6d,
        ):
            matrices = rotation_6d_to_matrix(rotations)
            if not torch.isfinite(matrices).all():
                raise FloatingPointError("invalid rotation matrices")
            if not torch.allclose(
                torch.linalg.det(matrices), torch.ones_like(torch.linalg.det(matrices)), atol=1e-4
            ):
                raise FloatingPointError("improper rotation matrix")

    def _infer_graph(
        self, observations: ObservationBatch, state: TrajectoryState
    ) -> tuple[ContactGraphBatch, GeometryOutput]:
        geometry = self.geometry_from_state(state)
        proposal = self.proposal(observations, state, geometry.features)
        edge_valid = self.edge_valid.to(state.root_rot6d.device)
        if edge_valid.shape[0] == 1 and state.root_rot6d.shape[0] > 1:
            edge_valid = edge_valid.expand(state.root_rot6d.shape[0], -1)
        graph = self.decoder.decode(
            proposal.event_logits,
            proposal.duration_logits,
            edge_valid,
            state.valid_mask,
        )
        return graph, geometry

    def reconstruct(
        self, initial: TrajectoryState, observations: ObservationBatch
    ) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        for hypothesis_id in range(self.num_hypotheses):
            seed = self.derive_seed(self.base_seed, hypothesis_id)
            state = initial
            graph, geometry = self._infer_graph(observations, state)
            round_diagnostics: list[dict[str, object]] = []
            round_results: list[RoundResult] = []
            status = "ok"
            retry_count = 0
            try:
                for round_index in range(self.rounds):
                    round_seed = self.derive_round_seed(seed, round_index)
                    try:
                        state, diagnostics = self.sampler.sample(
                            state,
                            graph,
                            observations,
                            seed=round_seed,
                            num_steps=self.diffusion_steps,
                        )
                        self.validate_trajectory(state)
                    except FloatingPointError as first_error:
                        if self.retry_guidance_factor is None:
                            raise
                        retry_count += 1
                        round_diagnostics.append(
                            {
                                "round": round_index,
                                "attempt": "initial",
                                "failure": type(first_error).__name__,
                                "message": str(first_error),
                                "seed": round_seed,
                            }
                        )
                        state, diagnostics = self.sampler.sample(
                            state,
                            graph,
                            observations,
                            seed=round_seed,
                            num_steps=self.diffusion_steps,
                            guidance_scale_override=(
                                self.sampler.guidance_scale * self.retry_guidance_factor
                            ),
                        )
                        self.validate_trajectory(state)
                    if self.alternating:
                        graph, geometry = self._infer_graph(observations, state)
                    else:
                        geometry = self.geometry_from_state(state)
                    runtime_objective = self.ranker.terms(state, graph, observations, geometry)
                    diagnostic_payload = {"round": round_index, **asdict(diagnostics)}
                    round_diagnostics.append(diagnostic_payload)
                    round_results.append(
                        RoundResult(
                            round_index=round_index,
                            trajectory=state,
                            graph=graph,
                            diagnostics=diagnostic_payload,
                            runtime_objective=runtime_objective,
                        )
                    )
            except FloatingPointError as exc:
                # A failed hypothesis returns the explicit initializer fallback.
                state = initial
                graph, geometry = self._infer_graph(observations, state)
                status = "fallback_initialization"
                round_diagnostics.append({"failure": type(exc).__name__, "message": str(exc)})
            terms = self.ranker.terms(state, graph, observations, geometry)
            hypotheses.append(
                Hypothesis(
                    identifier=hypothesis_id,
                    seed=seed,
                    trajectory=state,
                    graph=graph,
                    score=self.ranker.score(terms),
                    ranking_terms=terms,
                    diagnostics={"rounds": round_diagnostics},
                    status=status,
                    rounds=tuple(round_results),
                    retry_count=retry_count,
                )
            )
        return sorted(hypotheses, key=lambda item: (-item.score, item.identifier))
