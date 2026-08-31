"""Rotation-aware trajectory windows and global contact reconciliation."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.contact.semi_markov import SemiMarkovDecoder
from dcg_sign4d.diffusion.state_codec import TrajectoryState, rotation_6d_to_matrix
from dcg_sign4d.geometry.so3 import log_map
from dcg_sign4d.initialization.camera import CameraTrajectory
from dcg_sign4d.initialization.dexavatar_adapter import axis_angle_to_matrix, matrix_to_rotation_6d
from dcg_sign4d.observations.schema import ObservationBatch


def window_starts(time: int, length: int, overlap: int) -> tuple[int, ...]:
    if time < 1 or length < 1 or overlap < 0 or overlap >= length:
        raise ValueError("invalid time/window length/overlap")
    if time <= length:
        return (0,)
    stride = length - overlap
    starts = list(range(0, time - length + 1, stride))
    final = time - length
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def slice_trajectory(state: TrajectoryState, start: int, end: int) -> TrajectoryState:
    state.validate()
    if not 0 <= start < end <= state.valid_mask.shape[1]:
        raise ValueError("invalid trajectory window")
    updates = {}
    for name in state.__dataclass_fields__:
        value = getattr(state, name)
        if isinstance(value, Tensor) and name not in {"beta"}:
            updates[name] = value[:, start:end]
    return replace(state, **updates).validate()


def slice_camera(camera: CameraTrajectory, start: int, end: int) -> CameraTrajectory:
    """Slice every time-varying camera field with the trajectory window."""

    camera.validate()
    if not 0 <= start < end <= camera.valid_mask.shape[1]:
        raise ValueError("invalid camera window")
    return replace(
        camera,
        intrinsics=camera.intrinsics[:, start:end],
        world_to_camera=camera.world_to_camera[:, start:end],
        image_size_wh=camera.image_size_wh[:, start:end],
        valid_mask=camera.valid_mask[:, start:end],
    ).validate()


def slice_observations(observations: ObservationBatch, start: int, end: int) -> ObservationBatch:
    """Slice calibrated cues and their frame identities without losing provenance."""

    observations.validate()
    if not 0 <= start < end <= observations.frame_valid.shape[1]:
        raise ValueError("invalid observation window")
    updates: dict[str, object] = {}
    for name in observations.__dataclass_fields__:
        value = getattr(observations, name)
        if isinstance(value, Tensor):
            updates[name] = value[:, start:end]
    metadata = []
    for item in observations.metadata:
        sliced = dict(item)
        if "frame_ids" in sliced:
            sliced["frame_ids"] = list(sliced["frame_ids"][start:end])
            sliced["timestamps_sec"] = list(sliced["timestamps_sec"][start:end])
        sliced["window_start"] = start
        sliced["window_end"] = end
        metadata.append(sliced)
    updates["metadata"] = tuple(metadata)
    return replace(observations, **updates).validate()


def _weights(start: int, length: int, total_time: int, overlap: int, like: Tensor) -> Tensor:
    weight = like.new_ones(length)
    ramp_count = min(overlap, length)
    if ramp_count:
        ramp = torch.arange(1, ramp_count + 1, device=like.device, dtype=like.dtype)
        ramp = ramp / (ramp_count + 1)
        if start > 0:
            weight[:ramp_count] = ramp
        if start + length < total_time:
            weight[-ramp_count:] = ramp.flip(0)
    return weight


def _rotation_blend(first: Tensor, second: Tensor, alpha: Tensor) -> Tensor:
    relative = first.transpose(-1, -2) @ second
    tangent = log_map(relative)
    while alpha.ndim < tangent.ndim:
        alpha = alpha.unsqueeze(-1)
    return first @ axis_angle_to_matrix(tangent * alpha)


def _stitch_rot6d(
    windows: list[TrajectoryState],
    starts: tuple[int, ...],
    field: str,
    weights: list[Tensor],
    time: int,
) -> Tensor:
    example = getattr(windows[0], field)
    matrices = rotation_6d_to_matrix(example)
    output_shape = (example.shape[0], time, *matrices.shape[2:])
    output = torch.eye(3, device=example.device, dtype=example.dtype).expand(output_shape).clone()
    accumulated = example.new_zeros(example.shape[0], time)
    for state, start, weight in zip(windows, starts, weights, strict=True):
        incoming = rotation_6d_to_matrix(getattr(state, field))
        end = start + incoming.shape[1]
        for local, global_index in enumerate(range(start, end)):
            valid = state.valid_mask[:, local]
            incoming_weight = weight[local] * valid
            total = accumulated[:, global_index] + incoming_weight
            alpha = torch.where(total > 0, incoming_weight / total.clamp_min(1e-12), total)
            blended = _rotation_blend(output[:, global_index], incoming[:, local], alpha)
            selector = valid.reshape(valid.shape[0], *((1,) * (blended.ndim - 1)))
            output[:, global_index] = torch.where(selector, blended, output[:, global_index])
            accumulated[:, global_index] = total
    return matrix_to_rotation_6d(output)


def stitch_trajectories(
    windows: list[TrajectoryState],
    starts: tuple[int, ...],
    *,
    total_time: int,
    overlap: int,
) -> TrajectoryState:
    if not windows or len(windows) != len(starts):
        raise ValueError("windows and starts must be non-empty and aligned")
    if tuple(sorted(starts)) != starts:
        raise ValueError("window starts must be sorted")
    for state in windows:
        state.validate()
        if not torch.allclose(state.beta, windows[0].beta, atol=1e-6, rtol=1e-6):
            raise ValueError("window beta values violate clip-shared shape")
    weights = [
        _weights(start, state.valid_mask.shape[1], total_time, overlap, state.root_translation)
        for state, start in zip(windows, starts, strict=True)
    ]
    batch = windows[0].valid_mask.shape[0]
    accumulated = windows[0].root_translation.new_zeros(batch, total_time)
    linear_names = ["root_translation", "root_velocity"]
    if windows[0].face_state is not None:
        linear_names.append("face_state")
    linear: dict[str, Tensor] = {
        name: getattr(windows[0], name).new_zeros(
            batch, total_time, *getattr(windows[0], name).shape[2:]
        )
        for name in linear_names
    }
    for state, start, weight in zip(windows, starts, weights, strict=True):
        length = state.valid_mask.shape[1]
        end = start + length
        active_weight = weight[None] * state.valid_mask
        accumulated[:, start:end] += active_weight
        for name in linear_names:
            value = getattr(state, name)
            expanded = active_weight.reshape(batch, length, *((1,) * (value.ndim - 2)))
            linear[name][:, start:end] += value * expanded
    if bool((accumulated <= 0).any()):
        raise ValueError("window plan leaves uncovered frames")
    for name in linear_names:
        denominator = accumulated.reshape(batch, total_time, *((1,) * (linear[name].ndim - 2)))
        linear[name] = linear[name] / denominator

    # Jaw and eye components are rotations even though face_state also contains
    # linear expression coefficients. Re-stitch those three axis-angle groups.
    face = linear.get("face_state")
    if face is not None:
        for offset in (0, 3, 6):
            face_windows = [
                replace(
                    state,
                    root_rot6d=matrix_to_rotation_6d(
                        axis_angle_to_matrix(state.face_state[..., offset : offset + 3])
                    ),
                )
                for state in windows
            ]
            blended = _stitch_rot6d(face_windows, starts, "root_rot6d", weights, total_time)
            face[..., offset : offset + 3] = log_map(rotation_6d_to_matrix(blended))

    result = TrajectoryState(
        root_rot6d=_stitch_rot6d(windows, starts, "root_rot6d", weights, total_time),
        root_translation=linear["root_translation"],
        root_velocity=linear["root_velocity"],
        body_rot6d=_stitch_rot6d(windows, starts, "body_rot6d", weights, total_time),
        left_hand_rot6d=_stitch_rot6d(windows, starts, "left_hand_rot6d", weights, total_time),
        right_hand_rot6d=_stitch_rot6d(windows, starts, "right_hand_rot6d", weights, total_time),
        face_state=face,
        beta=windows[0].beta,
        valid_mask=accumulated > 0,
    )
    return result.validate()


def stitch_contact_graphs(
    graphs: list[ContactGraphBatch],
    starts: tuple[int, ...],
    *,
    total_time: int,
    overlap: int,
    decoder: SemiMarkovDecoder,
    frame_valid: Tensor,
) -> ContactGraphBatch:
    if not graphs or len(graphs) != len(starts):
        raise ValueError("graphs and starts must be non-empty and aligned")
    for graph in graphs:
        graph.validate()
    batch, _, edges = graphs[0].event_state.shape
    probability = graphs[0].event_probability.new_zeros(batch, total_time, edges, 4)
    accumulated = probability.new_zeros(batch, total_time, 1, 1)
    edge_valid = graphs[0].edge_valid.clone()
    for graph, start in zip(graphs, starts, strict=True):
        length = graph.event_state.shape[1]
        weight = _weights(start, length, total_time, overlap, probability)[None, :, None, None]
        probability[:, start : start + length] += graph.event_probability * weight
        accumulated[:, start : start + length] += weight
        edge_valid &= graph.edge_valid
    probability = probability / accumulated.clamp_min(1e-12)
    logits = probability.clamp_min(1e-8).log()
    duration_logits = logits.new_zeros(batch, total_time, edges, decoder.max_duration)
    return decoder.decode(logits, duration_logits, edge_valid, frame_valid)
