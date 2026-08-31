"""Dynamic graph tokens shared by all matched conditioning variants."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from dcg_sign4d.contact.ontology import ContactGraphBatch, EventState

ConditioningMode = Literal["null", "static", "dynamic"]


class ContactTokenEncoder(nn.Module):
    def __init__(
        self,
        edge_count: int,
        hidden_dim: int,
        edge_names: tuple[tuple[str, str], ...] | None = None,
    ):
        super().__init__()
        self.edge_count = edge_count
        if edge_names is None:
            edge_names = tuple(
                (f"source_{index}", f"target_{index}") for index in range(edge_count)
            )
        if len(edge_names) != edge_count:
            raise ValueError("edge-name topology does not match edge_count")
        patch_names = sorted({name for edge in edge_names for name in edge})
        patch_index = {name: index for index, name in enumerate(patch_names)}
        self.register_buffer(
            "source_patch_index",
            torch.tensor([patch_index[source] for source, _ in edge_names], dtype=torch.long),
        )
        self.register_buffer(
            "target_patch_index",
            torch.tensor([patch_index[target] for _, target in edge_names], dtype=torch.long),
        )
        self.patch = nn.Embedding(len(patch_names) + 1, hidden_dim)
        self.event = nn.Embedding(5, hidden_dim)
        self.duration = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.reliability = nn.Linear(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _time_embedding(time: int, width: int, reference: Tensor) -> Tensor:
        position = torch.arange(time, device=reference.device, dtype=reference.dtype)[:, None]
        half = width // 2
        frequency = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=reference.device, dtype=reference.dtype)
            / max(half - 1, 1)
        )[None]
        angle = position * frequency
        result = torch.cat((angle.sin(), angle.cos()), -1)
        if width % 2:
            result = torch.cat((result, torch.zeros_like(result[:, :1])), -1)
        return result

    def forward(self, graph: ContactGraphBatch, mode: ConditioningMode) -> Tensor:
        graph.validate()
        batch, time, edges = graph.event_state.shape
        if edges != self.edge_count:
            raise ValueError("edge count mismatch")
        source_ids = self.source_patch_index.to(graph.event_state.device)[None, None, :]
        target_ids = self.target_patch_index.to(graph.event_state.device)[None, None, :]
        edge_tokens = self.patch(source_ids) + self.patch(target_ids)
        edge_tokens = edge_tokens.expand(batch, time, edges, -1)
        if mode == "null":
            null_ids = torch.full_like(source_ids, self.patch.num_embeddings - 1)
            edge_tokens = (2 * self.patch(null_ids)).expand(batch, time, edges, -1)
            event_ids = torch.full_like(graph.event_state, 4)
            duration = torch.zeros_like(graph.segment_duration)
            reliability = torch.zeros_like(graph.event_probability[..., 0])
        elif mode == "static":
            contact = graph.event_state != EventState.OFF
            event_ids = torch.where(contact, EventState.HOLD, EventState.OFF).long()
            duration = torch.zeros_like(graph.segment_duration)
            reliability = graph.event_probability.amax(-1)
        elif mode == "dynamic":
            event_ids = graph.event_state
            duration = graph.segment_duration
            reliability = graph.event_probability.amax(-1)
        else:
            raise ValueError(f"unknown conditioning mode: {mode}")
        invalid = ~graph.edge_valid[:, None, :]
        if bool(invalid.any()):
            null_ids = torch.full_like(source_ids, self.patch.num_embeddings - 1)
            null_token = (2 * self.patch(null_ids)).expand(batch, time, edges, -1)
            edge_tokens = torch.where(invalid[..., None], null_token, edge_tokens)
            event_ids = torch.where(invalid, torch.full_like(event_ids, 4), event_ids)
            duration = torch.where(invalid, torch.zeros_like(duration), duration)
            reliability = torch.where(invalid, torch.zeros_like(reliability), reliability)
        token = edge_tokens + self.event(event_ids)
        token = token + self.duration(duration[..., None])
        token = token + self.reliability(reliability[..., None])
        token = token + self._time_embedding(time, token.shape[-1], token)[None, :, None, :]
        return self.norm(token)
