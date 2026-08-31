import pytest
import torch

from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.synthetic import make_graph, make_observations, make_state
from dcg_sign4d.training.batch import (
    SupervisedWindowBatch,
    SupervisedWindowMetadata,
    load_supervised_windows,
    save_supervised_windows,
)
from dcg_sign4d.utils.hashing import canonical_hash


def fixture():
    state = make_state(batch=3, time=4)
    graph = make_graph(batch=3, time=4, edges=2)
    width = StateCodec().encode(state)[0].shape[-1]
    digest = canonical_hash({"fixture": True})
    return SupervisedWindowBatch(
        trajectory=state,
        observations=make_observations(batch=3, time=4),
        geometry_features=torch.randn(3, 4, 2, 5),
        graph=graph,
        duration_frames=torch.ones(3, 4, 2, dtype=torch.long),
        supervision_mask=torch.ones(3, 4, width, dtype=torch.bool),
        metadata=SupervisedWindowMetadata(
            split="train",
            manifest_sha256=digest,
            observation_cache_sha256=digest,
            patch_map_sha256=digest,
            contact_label_status="synthetic_fixture",
            sample_label_status=("synthetic_fixture",) * 3,
            sample_ids=("a", "b", "c"),
            edge_names=(("left", "right"), ("hand", "face")),
            development_only=True,
        ),
    ).validate()


def test_supervised_bundle_roundtrip_selection_and_development_gate(tmp_path):
    source = save_supervised_windows(tmp_path / "bundle", fixture())
    with pytest.raises(PermissionError, match="development bundle"):
        load_supervised_windows(source, expected_split="train")
    restored = load_supervised_windows(source, expected_split="train", allow_development=True)
    assert restored.trajectory.root_rot6d.shape == (3, 4, 6)
    selected = restored.select(torch.tensor([2, 0]))
    assert selected.metadata.sample_ids == ("c", "a")
    assert selected.geometry_features.shape == (2, 4, 2, 5)
    assert selected.to("cpu").graph.event_state.device.type == "cpu"
    with pytest.raises(ValueError, match="split mismatch"):
        load_supervised_windows(source, expected_split="validation", allow_development=True)


def test_supervised_bundle_hash_tamper_is_detected(tmp_path):
    source = save_supervised_windows(tmp_path / "bundle", fixture())
    data = source / "windows.npz"
    data.write_bytes(data.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_supervised_windows(source, expected_split="train", allow_development=True)
