from __future__ import annotations

from ..data.split import freeze_clip_splits


def run(
    source_manifest: str,
    output_dir: str,
    calibration_clips: int,
    development_clips: int,
    seed: int,
    extra_development: tuple[str, ...] = (),
) -> dict[str, object]:
    return freeze_clip_splits(
        source_manifest,
        output_dir,
        calibration_clips=calibration_clips,
        development_clips=development_clips,
        seed=seed,
        extra_development=extra_development,
    )
