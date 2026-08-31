"""Compile clip artifacts into immutable Stage 2/3 supervised windows."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch, EventState
from dcg_sign4d.data.manifest import load_manifest
from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.inference.windowing import slice_observations, slice_trajectory, window_starts
from dcg_sign4d.initialization.artifact import load_initialization_artifact
from dcg_sign4d.observations.cache import ObservationCache
from dcg_sign4d.observations.schema import ObservationBatch
from dcg_sign4d.training.batch import (
    SupervisedWindowBatch,
    SupervisedWindowMetadata,
    save_supervised_windows,
)
from dcg_sign4d.utils.hashing import file_sha256


def _pad_tensor(value: Tensor, length: int, *, fill: float | bool = 0) -> Tensor:
    missing = length - value.shape[1]
    if missing < 0:
        raise ValueError("cannot pad a tensor beyond its window")
    if missing == 0:
        return value
    padding = torch.full(
        (value.shape[0], missing, *value.shape[2:]),
        fill,
        dtype=value.dtype,
        device=value.device,
    )
    return torch.cat((value, padding), 1)


def _pad_state(state: TrajectoryState, length: int) -> TrajectoryState:
    updates = {}
    for name in state.__dataclass_fields__:
        value = getattr(state, name)
        if isinstance(value, Tensor) and name != "beta":
            updates[name] = _pad_tensor(value, length, fill=False if name == "valid_mask" else 0)
    return replace(state, **updates).validate()


def _pad_observations(observations: ObservationBatch, length: int) -> ObservationBatch:
    updates = {}
    for name in observations.__dataclass_fields__:
        value = getattr(observations, name)
        if isinstance(value, Tensor):
            updates[name] = _pad_tensor(
                value, length, fill=False if value.dtype == torch.bool else 0
            )
    updates["metadata"] = ()
    return replace(observations, **updates).validate()


def _segments(states: Tensor, fps: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return segment ID, seconds and positive duration target for [T,E]."""

    time, edges = states.shape
    identifiers = torch.zeros_like(states)
    seconds = torch.zeros_like(states, dtype=torch.float32)
    duration = torch.ones_like(states)
    next_identifier = 1
    for edge in range(edges):
        frame = 0
        while frame < time:
            if int(states[frame, edge]) == int(EventState.OFF):
                frame += 1
                continue
            end = frame + 1
            while end < time and int(states[end, edge]) != int(EventState.OFF):
                end += 1
            identifiers[frame:end, edge] = next_identifier
            seconds[frame:end, edge] = (end - frame) / fps
            duration[frame:end, edge] = end - frame
            next_identifier += 1
            frame = end
    return identifiers, seconds, duration


def _concatenate_states(states: list[TrajectoryState]) -> TrajectoryState:
    return TrajectoryState(
        **{
            name: torch.cat([getattr(state, name) for state in states], 0)
            for name in states[0].__dataclass_fields__
            if getattr(states[0], name) is not None
        }
    ).validate()


def _concatenate_observations(values: list[ObservationBatch]) -> ObservationBatch:
    fields = {}
    for name in values[0].__dataclass_fields__:
        value = getattr(values[0], name)
        if isinstance(value, Tensor):
            fields[name] = torch.cat([getattr(item, name) for item in values], 0)
    fields["metadata"] = ()
    return ObservationBatch(**fields).validate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--observation-root", required=True)
    parser.add_argument("--initialization-root", required=True)
    parser.add_argument("--contact-root", required=True)
    parser.add_argument("--patch-map", required=True)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--window-overlap", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.window_length < 1 or not 0 <= args.window_overlap < args.window_length:
        raise ValueError("invalid window geometry")
    items = load_manifest(args.manifest, require_existing_video=True)
    if any(item.split != args.split for item in items):
        raise ValueError("manifest contains clips outside requested split")
    patch = PatchMap.load(args.patch_map)
    if not patch.development_only:
        raise ValueError("this compiler currently accepts development pseudo-labels only")
    observation_root = Path(args.observation_root)
    index_path = observation_root / "index.json"
    index = json.loads(index_path.read_text("utf-8"))
    cache_ids = {row["clip_id"]: row["cache_id"] for row in index["per_clip"]}
    cache = ObservationCache(observation_root / "caches")
    states: list[TrajectoryState] = []
    observations: list[ObservationBatch] = []
    features: list[Tensor] = []
    graph_fields: dict[str, list[Tensor]] = {
        name: [] for name in ContactGraphBatch.__dataclass_fields__
    }
    durations: list[Tensor] = []
    sample_ids: list[str] = []
    expected_edge_names = tuple(patch.admissible_edges)
    for item in items:
        state, _, _ = load_initialization_artifact(Path(args.initialization_root) / item.clip_id)
        observation = cache.load(cache_ids[item.clip_id])
        contact_path = Path(args.contact_root) / item.clip_id / "contact_geometry.npz"
        with np.load(contact_path, allow_pickle=False) as arrays:
            edge_names = tuple(
                tuple(str(value).split("::")) for value in arrays["edge_names"].tolist()
            )
            contact = {
                name: torch.from_numpy(arrays[name])
                for name in arrays.files
                if name != "edge_names"
            }
        if edge_names != expected_edge_names:
            raise ValueError(f"contact edge topology mismatch: {item.clip_id}")
        if not (
            state.valid_mask.shape[1]
            == observation.frame_valid.shape[1]
            == contact["features"].shape[0]
        ):
            raise ValueError(f"clip artifact frame mismatch: {item.clip_id}")
        fps = item.fps_effective or item.fps_native
        starts = window_starts(state.valid_mask.shape[1], args.window_length, args.window_overlap)
        for start in starts:
            end = min(start + args.window_length, state.valid_mask.shape[1])
            local_state = _pad_state(slice_trajectory(state, start, end), args.window_length)
            local_observation = _pad_observations(
                slice_observations(observation, start, end), args.window_length
            )
            event = contact["pseudo_event_state"][start:end].long()
            uncertain = contact["pseudo_uncertain_mask"][start:end].bool()
            segment_id, segment_seconds, duration = _segments(event, fps)
            event = _pad_tensor(event[None], args.window_length)[0]
            uncertain = _pad_tensor(uncertain[None], args.window_length, fill=True)[0]
            segment_id = _pad_tensor(segment_id[None], args.window_length)[0]
            segment_seconds = _pad_tensor(segment_seconds[None], args.window_length)[0]
            duration = _pad_tensor(duration[None], args.window_length, fill=1)[0]
            probability = torch.nn.functional.one_hot(event, num_classes=4).float()
            edge_valid = torch.ones(len(expected_edge_names), dtype=torch.bool)
            graph = ContactGraphBatch(
                event_state=event[None],
                event_probability=probability[None],
                edge_valid=edge_valid[None],
                uncertain_mask=uncertain[None],
                segment_id=segment_id[None],
                segment_duration=segment_seconds[None],
            ).validate()
            states.append(local_state)
            observations.append(local_observation)
            features.append(_pad_tensor(contact["features"][None, start:end], args.window_length))
            for name in graph_fields:
                graph_fields[name].append(getattr(graph, name))
            durations.append(duration[None])
            sample_ids.append(f"{item.clip_id}:{start}:{end}")
    trajectory = _concatenate_states(states)
    observation_batch = _concatenate_observations(observations)
    graph = ContactGraphBatch(
        **{name: torch.cat(values, 0) for name, values in graph_fields.items()}
    ).validate()
    encoded, _ = StateCodec().encode(trajectory)
    supervision_mask = trajectory.valid_mask[:, :, None].expand_as(encoded).clone()
    bundle = SupervisedWindowBatch(
        trajectory=trajectory,
        observations=observation_batch,
        geometry_features=torch.cat(features, 0),
        graph=graph,
        duration_frames=torch.cat(durations, 0),
        supervision_mask=supervision_mask,
        metadata=SupervisedWindowMetadata(
            split=args.split,
            manifest_sha256=file_sha256(args.manifest),
            observation_cache_sha256=file_sha256(index_path),
            patch_map_sha256=patch.content_hash,
            contact_label_status="accepted_pseudo",
            sample_label_status=tuple("accepted_pseudo" for _ in sample_ids),
            sample_ids=tuple(sample_ids),
            edge_names=expected_edge_names,
            development_only=True,
        ),
    ).validate()
    destination = save_supervised_windows(args.output, bundle)
    print(
        json.dumps(
            {
                "output": str(destination.resolve()),
                "split": args.split,
                "windows": len(sample_ids),
                "frames": int(trajectory.valid_mask.sum()),
                "edges": len(expected_edge_names),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
