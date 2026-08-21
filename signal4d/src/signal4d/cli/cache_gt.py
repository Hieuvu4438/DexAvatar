from __future__ import annotations

from ..evaluation.sgnify import cache_sgnify_ground_truth


def run(manifest: str, gt_root: str, output_root: str) -> dict[str, int]:
    return cache_sgnify_ground_truth(manifest, gt_root, output_root)
