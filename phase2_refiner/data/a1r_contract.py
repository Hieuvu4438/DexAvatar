"""Portable, dataset-independent fitting contract for the A1R initializer.

The frozen A1 fitter indexes a German-sign lookup table by image-directory
name.  A1R replaces that benchmark label dependency with a contract inferred
only from the observations belonging to the current clip.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import CacheClip


@dataclass(frozen=True)
class FittingContract:
    clip_name: str
    sign_class: str
    active_side: str
    segment_start: int
    segment_end: int
    left_coverage: float
    right_coverage: float
    left_motion: float
    right_motion: float


def _frame_number(name: str) -> int:
    match = re.search(r"(\d+)$", Path(name).stem)
    if match is None:
        raise ValueError(
            f"A1R fitting requires frame names ending in an integer: {name!r}"
        )
    return int(match.group(1))


def _side_statistics(clip: CacheClip, joint_slice: slice) -> tuple[float, float]:
    valid = clip.track_valid[:, joint_slice]
    coverage = float(valid.mean())
    if len(clip.frame_names) < 2:
        return coverage, 0.0
    pair_valid = valid[1:] & valid[:-1]
    displacement = np.linalg.norm(
        clip.keypoints_2d[1:, joint_slice] - clip.keypoints_2d[:-1, joint_slice],
        axis=-1,
    )
    values = displacement[pair_valid]
    return coverage, float(np.median(values)) if values.size else 0.0


def infer_fitting_contract(
    clip: CacheClip,
    image_directory_name: str,
    *,
    minimum_coverage: float = 0.35,
    minimum_motion: float = 0.002,
) -> FittingContract:
    """Infer one/two-hand fitting mode without using dataset class labels."""
    clip.validate()
    if not image_directory_name or any(char.isspace() for char in image_directory_name):
        raise ValueError("image_directory_name must be a non-empty token")
    left_coverage, left_motion = _side_statistics(clip, slice(21, 36))
    right_coverage, right_motion = _side_statistics(clip, slice(36, 51))
    left_active = left_coverage >= minimum_coverage and left_motion >= minimum_motion
    right_active = right_coverage >= minimum_coverage and right_motion >= minimum_motion
    if not left_active and not right_active:
        # Static signs still need a hand.  Coverage is a safer fallback than a
        # language/dataset label, and the decision remains recorded below.
        left_active = left_coverage > right_coverage
        right_active = not left_active
    two_handed = left_active and right_active
    if two_handed:
        active_side = "both"
    else:
        active_side = "left" if left_active else "right"
    frame_numbers = [_frame_number(str(name)) for name in clip.frame_names]
    return FittingContract(
        clip_name=image_directory_name,
        sign_class="1" if two_handed else "0",
        active_side=active_side,
        segment_start=min(frame_numbers),
        segment_end=max(frame_numbers),
        left_coverage=left_coverage,
        right_coverage=right_coverage,
        left_motion=left_motion,
        right_motion=right_motion,
    )


def write_fitting_contract(directory: Path, contract: FittingContract) -> dict[str, Path]:
    """Write isolated fitter inputs and an auditable decision record."""
    directory.mkdir(parents=True, exist_ok=False)
    signs = directory / "signs.txt"
    segments = directory / "segments.json"
    decision = directory / "decision.json"
    signs.write_text(
        f"{contract.clip_name} {contract.sign_class}\n", encoding="utf-8"
    )
    segments.write_text(
        json.dumps(
            {contract.clip_name: [contract.segment_start, contract.segment_end]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    decision.write_text(
        json.dumps(contract.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"sign_class": signs, "sign_segment": segments, "decision": decision}
