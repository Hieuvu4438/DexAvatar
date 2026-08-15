import argparse
import json
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import save_cache_clip
from phase2_refiner.data.run_a1r_fitting import run
from phase2_refiner.tests.test_cache import make_clip


def test_a1r_runner_binds_clip_folder_and_requires_complete_results(
    tmp_path: Path, monkeypatch
) -> None:
    clip = make_clip(3)
    clip.clip_id = "how2sign_train_portable_clip"
    clip.frame_names = np.asarray(["portable_000", "portable_002", "portable_004"])
    clip.track_valid = np.ones((3, 51), dtype=bool)
    clip.keypoint_valid = clip.track_valid.copy()
    clip.keypoints_2d[1:, :, 0] = np.asarray([0.01, 0.02])[:, None]
    cache = tmp_path / "clip.npz"
    save_cache_clip(cache, clip)

    images = tmp_path / "images"
    images.mkdir()
    for name in clip.frame_names:
        (images / f"{name}.png").write_bytes(b"image")
    expert_output = tmp_path / "expert_output"
    results = expert_output / "smplifyx" / "results"
    results.mkdir(parents=True)
    contract = tmp_path / "contract"
    observed: dict[str, object] = {}

    def fake_subprocess(command, *, cwd, env, check):
        observed.update(command=command, cwd=cwd, env=env, check=check)
        image_folder = Path(command[command.index("--img_folder") + 1])
        assert image_folder.name == clip.clip_id
        assert image_folder.is_symlink()
        assert image_folder.resolve() == images.resolve()
        signs = Path(command[command.index("--sign_class") + 1]).read_text()
        segments = json.loads(
            Path(command[command.index("--sign_segment") + 1]).read_text()
        )
        assert signs.startswith(f"{clip.clip_id} ")
        assert segments == {clip.clip_id: [0, 4]}
        for name in clip.frame_names:
            (results / f"{name}.pkl").write_bytes(b"result")

    monkeypatch.setattr(
        "phase2_refiner.data.run_a1r_fitting.subprocess.run", fake_subprocess
    )
    run(
        argparse.Namespace(
            cache=cache,
            image_root=images,
            output_root=expert_output,
            contract_root=contract,
            config=Path("cfg_files/fit_smplx_vposer_x_ensemble.yaml"),
            smplx_init_dir="ensemble_smplx",
            gpu=0,
            image_suffix=".png",
        )
    )

    assert observed["check"] is True
    report = json.loads((contract / "result.json").read_text())
    assert report["passed"] is True
    assert report["frames"] == 3
    assert report["clip_id"] == clip.clip_id
