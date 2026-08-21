from __future__ import annotations

from ..evaluation.author_sgnify import evaluate_author_sgnify


def run(
    manifest: str,
    method_specs: list[str],
    baseline: str,
    gt_root: str,
    author_source: str,
    author_asset_root: str,
    author_sign_file: str,
    author_segment_file: str,
    frame_policy: str,
    prediction_format: str,
    output: str,
) -> dict[str, object]:
    methods: dict[str, str] = {}
    for spec in method_specs:
        if "=" not in spec:
            raise ValueError(f"method must use LABEL=PREDICTION_ROOT syntax: {spec}")
        label, root = spec.split("=", 1)
        if not label or not root or label in methods:
            raise ValueError(f"invalid or duplicate method specification: {spec}")
        methods[label] = root
    return evaluate_author_sgnify(
        manifest_path=manifest,
        methods=methods,
        baseline=baseline,
        gt_root=gt_root,
        author_source=author_source,
        author_asset_root=author_asset_root,
        author_sign_file=author_sign_file,
        author_segment_file=author_segment_file,
        frame_policy=frame_policy,
        prediction_format=prediction_format,
        output_root=output,
    )
