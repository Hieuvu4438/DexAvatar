from __future__ import annotations

import torch
from torch import nn

from dcg_sign4d.geometry.smplx_adapter import SMPLXForwardOutput
from dcg_sign4d.initialization.artifact import (
    load_initialization_artifact,
    save_initialization_artifact,
)
from dcg_sign4d.initialization.camera import (
    CameraTrajectory,
    StateJointDepthDifference,
    StatePartMaskRenderer,
)
from dcg_sign4d.synthetic import make_state


def camera(time=3):
    intrinsics = torch.eye(3)[None, None].expand(1, time, 3, 3).clone()
    intrinsics[..., 0, 0] = 100
    intrinsics[..., 1, 1] = 100
    world_to_camera = torch.eye(4)[None, None].expand(1, time, 4, 4).clone()
    return CameraTrajectory(
        intrinsics,
        world_to_camera,
        torch.tensor([[[640.0, 480.0]]] * time).reshape(1, time, 2),
        torch.ones(1, time, dtype=torch.bool),
        "fixture_camera",
    ).validate()


def test_camera_projection_and_initialization_roundtrip(tmp_path):
    state = make_state(time=3)
    target = save_initialization_artifact(
        tmp_path / "clip",
        state,
        camera(),
        metadata={
            "clip_id": "clip",
            "dexavatar_commit": "a" * 40,
            "config_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
            "runtime": {"seconds": 1.0},
            "development_only": True,
        },
        source_hashes={"video": "d" * 64},
    )
    restored, restored_camera, metadata = load_initialization_artifact(target)
    assert torch.equal(restored.root_translation, state.root_translation)
    assert metadata["clip_id"] == "clip"
    points = torch.tensor([[[[1.0, 2.0, 2.0]]]]).expand(1, 3, 1, 3)
    projected = restored_camera.project(points)
    assert torch.allclose(projected[0, 0, 0], torch.tensor([50.0, 100.0]))


def test_initialization_tamper_is_detected(tmp_path):
    target = save_initialization_artifact(
        tmp_path / "clip",
        make_state(time=3),
        camera(),
        metadata={
            "clip_id": "clip",
            "dexavatar_commit": "a" * 40,
            "config_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
            "runtime": {},
            "development_only": True,
        },
        source_hashes={"video": "d" * 64},
    )
    camera_path = target / "camera.npz"
    camera_path.write_bytes(camera_path.read_bytes() + b"tamper")
    try:
        load_initialization_artifact(target)
    except ValueError as exc:
        assert "camera.npz hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered camera artifact was accepted")


class _Body(nn.Module):
    def forward(self, state):
        points = torch.tensor(
            [[[[-0.1, 0.0, 2.0], [0.1, 0.0, 3.0], [0.0, 0.1, 4.0]]]],
            dtype=state.root_translation.dtype,
            device=state.root_translation.device,
        ).expand(1, state.valid_mask.shape[1], 3, 3)
        points = points + state.root_translation[:, :, None]
        return SMPLXForwardOutput(points, points)


def test_depth_pairs_and_soft_part_masks_are_differentiable():
    state = make_state(time=3)
    state = __import__("dataclasses").replace(
        state, root_translation=state.root_translation.clone().requires_grad_()
    )
    depth = StateJointDepthDifference(
        _Body(), camera(), torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    )(state)
    renderer = StatePartMaskRenderer(
        _Body(),
        camera(),
        (torch.tensor([0, 1]), torch.tensor([2])),
        (8, 8),
        sigma_px=1.5,
    )
    masks = renderer(state)
    assert depth.shape == (1, 3, 2)
    assert torch.allclose(depth[0, 0], torch.tensor([-1.0, -1.0]))
    assert masks.shape == (1, 3, 2, 8, 8)
    assert bool(((masks >= 0) & (masks <= 1)).all())
    (depth.square().mean() + masks.mean()).backward()
    assert state.root_translation.grad is not None
    assert torch.isfinite(state.root_translation.grad).all()
