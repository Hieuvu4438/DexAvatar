from __future__ import annotations

import argparse
import json

from ..scripts.synthetic import create_synthetic_artifact
from . import (
    apply_gate,
    apply_multigate,
    assess_confirmatory,
    build_extended_manifest,
    build_manifest,
    build_multigate,
    cache_gt,
    calibrate,
    compare,
    compose_legacy,
    confirmatory,
    evaluate,
    evaluate_author_sgnify,
    evaluate_sgnify,
    export_dexavatar_obj,
    extrapolate,
    fit_smplx,
    freeze_release,
    freeze_split,
    materialize_legacy,
    preprocess,
    render_confirmatory_report,
    render_dexavatar_reconstruction,
    report_final,
    run_pipeline,
    train_gate,
    verify_repro,
    verify_tree,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal4d")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", required=True)

    extended_manifest = subparsers.add_parser("build-extended-manifest")
    extended_manifest.add_argument("--segments", required=True)
    extended_manifest.add_argument("--frames-root", required=True)
    extended_manifest.add_argument("--body-root", required=True)
    extended_manifest.add_argument("--wilor-root", required=True)
    extended_manifest.add_argument("--gt-root", required=True)
    extended_manifest.add_argument("--output", required=True)
    extended_manifest.add_argument("--split", default="test")

    ground_truth_cache = subparsers.add_parser("cache-gt")
    ground_truth_cache.add_argument("--manifest", required=True)
    ground_truth_cache.add_argument("--gt-root", required=True)
    ground_truth_cache.add_argument("--output-root", required=True)

    split = subparsers.add_parser("freeze-split")
    split.add_argument("--source-manifest", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--calibration-clips", type=int, default=12)
    split.add_argument("--development-clips", type=int, default=20)
    split.add_argument("--seed", type=int, default=20260819)
    split.add_argument("--extra-development", action="append", default=[])

    release = subparsers.add_parser("freeze-release")
    release.add_argument("--output", required=True)
    release.add_argument("--config", action="append", required=True)
    release.add_argument("--manifest", action="append", required=True)
    release.add_argument("--artifact", action="append", default=[])

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--output-root", required=True)
    synthetic.add_argument("--num-clips", type=int, default=3)
    synthetic.add_argument("--frames", type=int, default=24)
    synthetic.add_argument("--seed", type=int, default=12345)

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--manifest", required=True)
    calibration.add_argument("--cache-root", required=True)
    calibration.add_argument("--gt-root", required=True)
    calibration.add_argument("--model-path", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--epochs", type=int, default=100)
    calibration.add_argument("--learning-rate", type=float, default=1e-2)
    calibration.add_argument("--seed", type=int, default=12345)
    calibration.add_argument("--device", default="cpu")
    calibration.add_argument("--conformal-clips", type=int, default=4)
    calibration.add_argument("--sigma-min", type=float, default=0.002)
    calibration.add_argument("--sigma-max", type=float, default=0.5)

    preprocessing = subparsers.add_parser("preprocess")
    preprocessing.add_argument("--manifest", required=True)
    preprocessing.add_argument("--output-root", required=True)
    preprocessing.add_argument("--body-root", required=True)
    preprocessing.add_argument("--wilor-root", required=True)
    preprocessing.add_argument("--model-path", required=True)
    preprocessing.add_argument("--body-subpath", default="smplerx/smplx")
    preprocessing.add_argument("--body-source-name", default="smplerx")
    preprocessing.add_argument("--legacy-root")
    preprocessing.add_argument("--legacy-subpath", default="smplifyx/results")
    preprocessing.add_argument("--legacy-source-name", default="legacy_dexavatar")
    preprocessing.add_argument("--device", default="cpu")

    pipeline = subparsers.add_parser("run-pipeline")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--manifest", required=True)
    pipeline.add_argument("--cache-root", required=True)
    pipeline.add_argument("--output-root", required=True)

    gate_training = subparsers.add_parser("train-gate")
    gate_training.add_argument("--config", required=True)
    gate_training.add_argument("--output", required=True)

    gate_application = subparsers.add_parser("apply-gate")
    gate_application.add_argument("--manifest", required=True)
    gate_application.add_argument("--candidate-root", required=True)
    gate_application.add_argument("--baseline-root", required=True)
    gate_application.add_argument("--cache-root", required=True)
    gate_application.add_argument("--artifact", required=True)
    gate_application.add_argument("--output-root", required=True)

    multi_gate_build = subparsers.add_parser("build-multigate")
    multi_gate_build.add_argument("--config", required=True)
    multi_gate_build.add_argument("--output", required=True)

    multi_gate_apply = subparsers.add_parser("apply-multigate")
    multi_gate_apply.add_argument("--manifest", required=True)
    multi_gate_apply.add_argument("--baseline-root", required=True)
    multi_gate_apply.add_argument("--cache-root", required=True)
    multi_gate_apply.add_argument("--bundle", required=True)
    multi_gate_apply.add_argument("--hypothesis", action="append", required=True)
    multi_gate_apply.add_argument("--output-root", required=True)

    geodesic = subparsers.add_parser("extrapolate")
    geodesic.add_argument("--manifest", required=True)
    geodesic.add_argument("--candidate-root", required=True)
    geodesic.add_argument("--baseline-root", required=True)
    geodesic.add_argument("--cache-root", required=True)
    geodesic.add_argument("--model-path", required=True)
    geodesic.add_argument("--alpha", required=True, type=float)
    geodesic.add_argument("--output-root", required=True)
    geodesic.add_argument("--device", default="cuda")

    legacy_materialization = subparsers.add_parser("materialize-legacy")
    legacy_materialization.add_argument("--manifest", required=True)
    legacy_materialization.add_argument("--primary-root", required=True)
    legacy_materialization.add_argument("--primary-subpath", default="smplifyx/results")
    legacy_materialization.add_argument("--fallback-root")
    legacy_materialization.add_argument("--fallback-subpath", default="smplifyx/results")
    legacy_materialization.add_argument("--model-path", required=True)
    legacy_materialization.add_argument("--output-root", required=True)
    legacy_materialization.add_argument("--method-name", required=True)
    legacy_materialization.add_argument("--device", default="cuda")

    legacy_composition = subparsers.add_parser("compose-legacy")
    legacy_composition.add_argument("--manifest", required=True)
    legacy_composition.add_argument("--primary-root", required=True)
    legacy_composition.add_argument("--primary-subpath", default="smplifyx/results")
    legacy_composition.add_argument("--fallback-root", required=True)
    legacy_composition.add_argument("--fallback-subpath", default="smplifyx/results")
    legacy_composition.add_argument("--output-root", required=True)
    legacy_composition.add_argument(
        "--method-name", default="legacy_primary_fallback_composition"
    )

    pose_fit = subparsers.add_parser("fit-smplx")
    pose_fit.add_argument("--config", required=True)
    pose_fit.add_argument("--manifest", required=True)
    pose_fit.add_argument("--cache-root", required=True)
    pose_fit.add_argument("--output-root", required=True)
    pose_fit.add_argument("--model-path", required=True)
    pose_fit.add_argument("--device", default="cuda")
    pose_fit.add_argument("--warm-start-root")

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--output", required=True)

    sgnify_eval = subparsers.add_parser("evaluate-sgnify")
    sgnify_eval.add_argument("--manifest", required=True)
    sgnify_eval.add_argument("--predictions", required=True)
    sgnify_eval.add_argument("--gt-root", required=True)
    sgnify_eval.add_argument("--model-path", required=True)
    sgnify_eval.add_argument("--upper-indices", required=True)
    sgnify_eval.add_argument("--left-indices", required=True)
    sgnify_eval.add_argument("--right-indices", required=True)
    sgnify_eval.add_argument("--output", required=True)
    sgnify_eval.add_argument("--gt-cache-root")

    author_sgnify_eval = subparsers.add_parser("evaluate-author-sgnify")
    author_sgnify_eval.add_argument("--manifest", required=True)
    author_sgnify_eval.add_argument("--method", action="append", required=True)
    author_sgnify_eval.add_argument("--baseline", required=True)
    author_sgnify_eval.add_argument("--gt-root", required=True)
    author_sgnify_eval.add_argument("--author-source", required=True)
    author_sgnify_eval.add_argument("--author-asset-root", required=True)
    author_sgnify_eval.add_argument("--author-sign-file", required=True)
    author_sgnify_eval.add_argument("--author-segment-file", required=True)
    author_sgnify_eval.add_argument(
        "--frame-policy", choices=("author-central", "manifest"), default="author-central"
    )
    author_sgnify_eval.add_argument(
        "--prediction-format",
        choices=("safetensors", "dexavatar-obj"),
        default="safetensors",
    )
    author_sgnify_eval.add_argument("--output", required=True)

    obj_export = subparsers.add_parser("export-dexavatar-obj")
    obj_export.add_argument("--manifest", required=True)
    obj_export.add_argument("--predictions", required=True)
    obj_export.add_argument("--model-path", required=True)
    obj_export.add_argument("--output", required=True)
    obj_export.add_argument("--method-name", required=True)
    obj_export.add_argument("--decimals", type=int, default=8)

    reconstruction = subparsers.add_parser("render-dexavatar-reconstruction")
    reconstruction.add_argument("--manifest", required=True)
    reconstruction.add_argument("--mesh-root", required=True)
    reconstruction.add_argument("--image-root", required=True)
    reconstruction.add_argument("--camera-root", required=True)
    reconstruction.add_argument("--output", required=True)
    reconstruction.add_argument("--method-name", required=True)
    reconstruction.add_argument("--workers", type=int, default=4)
    reconstruction.add_argument("--mesh-opacity", type=float, default=0.9)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--candidate-csv", required=True)
    comparison.add_argument("--baseline-csv", required=True)
    comparison.add_argument("--metric", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--replicates", type=int, default=10000)

    tree_verification = subparsers.add_parser("verify-tree")
    tree_verification.add_argument("--first", required=True)
    tree_verification.add_argument("--second", required=True)
    tree_verification.add_argument("--output", required=True)

    assessment = subparsers.add_parser("assess-confirmatory")
    assessment.add_argument("--candidate-summary", required=True)
    assessment.add_argument("--baseline-summary", required=True)
    assessment.add_argument("--comparison-root", required=True)
    assessment.add_argument("--reproducibility", required=True)
    assessment.add_argument("--output", required=True)
    assessment.add_argument("--expected-clips", required=True, type=int)
    assessment.add_argument("--expected-frames", required=True, type=int)

    confirmatory_report = subparsers.add_parser("render-confirmatory-report")
    confirmatory_report.add_argument("--decision", required=True)
    confirmatory_report.add_argument("--baseline-summary", required=True)
    confirmatory_report.add_argument("--candidate-summary", required=True)
    confirmatory_report.add_argument("--gate-metadata", required=True)
    confirmatory_report.add_argument("--release", required=True)
    confirmatory_report.add_argument("--output", required=True)

    confirmation = subparsers.add_parser("confirmatory")
    confirmation.add_argument("--freeze-file", required=True)
    confirmation.add_argument("--config", required=True)
    confirmation.add_argument("--manifest", required=True)
    confirmation.add_argument("--cache-root", required=True)
    confirmation.add_argument("--output-root", required=True)
    confirmation.add_argument("--gt-root", required=True)
    confirmation.add_argument("--gt-cache-root", required=True)
    confirmation.add_argument("--model-path", required=True)
    confirmation.add_argument("--upper-indices", required=True)
    confirmation.add_argument("--left-indices", required=True)
    confirmation.add_argument("--right-indices", required=True)
    confirmation.add_argument("--device", default="cuda")

    final_report = subparsers.add_parser("report-final")
    final_report.add_argument("--method", action="append", required=True)
    final_report.add_argument("--comparison", action="append", default=[])
    final_report.add_argument("--output-root", required=True)
    final_report.add_argument("--stress-slices")

    reproducibility = subparsers.add_parser("verify-repro")
    reproducibility.add_argument("--manifest", required=True)
    reproducibility.add_argument("--first-predictions", required=True)
    reproducibility.add_argument("--second-predictions", required=True)
    reproducibility.add_argument("--first-summary", required=True)
    reproducibility.add_argument("--second-summary", required=True)
    reproducibility.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build-manifest":
        build_manifest.run(args.config, args.output)
    elif args.command == "build-extended-manifest":
        report = build_extended_manifest.run(
            args.segments,
            args.frames_root,
            args.body_root,
            args.wilor_root,
            args.gt_root,
            args.output,
            args.split,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "cache-gt":
        report = cache_gt.run(args.manifest, args.gt_root, args.output_root)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "freeze-split":
        report = freeze_split.run(
            args.source_manifest,
            args.output_dir,
            args.calibration_clips,
            args.development_clips,
            args.seed,
            tuple(args.extra_development),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "freeze-release":
        report = freeze_release.run(args.output, args.config, args.manifest, args.artifact)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "synthetic":
        path = create_synthetic_artifact(
            args.output_root, args.num_clips, args.frames, seed=args.seed
        )
        print(path)
    elif args.command == "calibrate":
        metrics = calibrate.run(
            args.manifest,
            args.cache_root,
            args.gt_root,
            args.model_path,
            args.output,
            args.epochs,
            args.learning_rate,
            args.seed,
            args.device,
            args.conformal_clips,
            args.sigma_min,
            args.sigma_max,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
    elif args.command == "preprocess":
        preprocess.run(
            args.manifest,
            args.output_root,
            args.body_root,
            args.wilor_root,
            args.model_path,
            args.body_subpath,
            args.body_source_name,
            args.device,
            args.legacy_root,
            args.legacy_subpath,
            args.legacy_source_name,
        )
    elif args.command == "run-pipeline":
        run_pipeline.run(args.config, args.manifest, args.cache_root, args.output_root)
    elif args.command == "train-gate":
        report = train_gate.run(args.config, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "apply-gate":
        report = apply_gate.run(
            args.manifest,
            args.candidate_root,
            args.baseline_root,
            args.cache_root,
            args.artifact,
            args.output_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "build-multigate":
        report = build_multigate.run(args.config, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "apply-multigate":
        report = apply_multigate.run(
            args.manifest,
            args.baseline_root,
            args.cache_root,
            args.bundle,
            args.hypothesis,
            args.output_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "extrapolate":
        report = extrapolate.run(
            args.manifest,
            args.candidate_root,
            args.baseline_root,
            args.cache_root,
            args.model_path,
            args.alpha,
            args.output_root,
            args.device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "materialize-legacy":
        report = materialize_legacy.run(
            args.manifest,
            args.primary_root,
            args.primary_subpath,
            args.model_path,
            args.output_root,
            args.method_name,
            args.device,
            args.fallback_root,
            args.fallback_subpath,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "compose-legacy":
        report = compose_legacy.run(
            args.manifest,
            args.primary_root,
            args.fallback_root,
            args.output_root,
            args.primary_subpath,
            args.fallback_subpath,
            args.method_name,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "fit-smplx":
        fit_smplx.run(
            args.config,
            args.manifest,
            args.cache_root,
            args.output_root,
            args.model_path,
            args.device,
            args.warm_start_root,
        )
    elif args.command == "evaluate":
        evaluate.run(args.manifest, args.predictions, args.output)
    elif args.command == "evaluate-sgnify":
        report = evaluate_sgnify.run(
            manifest_path=args.manifest,
            prediction_root=args.predictions,
            gt_root=args.gt_root,
            model_path=args.model_path,
            upper_indices=args.upper_indices,
            left_indices=args.left_indices,
            right_indices=args.right_indices,
            output_root=args.output,
            gt_cache_root=args.gt_cache_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "evaluate-author-sgnify":
        report = evaluate_author_sgnify.run(
            args.manifest,
            args.method,
            args.baseline,
            args.gt_root,
            args.author_source,
            args.author_asset_root,
            args.author_sign_file,
            args.author_segment_file,
            args.frame_policy,
            args.prediction_format,
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "export-dexavatar-obj":
        report = export_dexavatar_obj.run(
            args.manifest,
            args.predictions,
            args.model_path,
            args.output,
            args.method_name,
            args.decimals,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "render-dexavatar-reconstruction":
        report = render_dexavatar_reconstruction.run(
            args.manifest,
            args.mesh_root,
            args.image_root,
            args.camera_root,
            args.output,
            args.method_name,
            args.workers,
            args.mesh_opacity,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "compare":
        report = compare.run(
            args.candidate_csv,
            args.baseline_csv,
            args.metric,
            args.output,
            args.replicates,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "verify-tree":
        report = verify_tree.run(args.first, args.second, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "assess-confirmatory":
        report = assess_confirmatory.run(
            args.candidate_summary,
            args.baseline_summary,
            args.comparison_root,
            args.reproducibility,
            args.output,
            args.expected_clips,
            args.expected_frames,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "render-confirmatory-report":
        report = render_confirmatory_report.run(
            args.decision,
            args.baseline_summary,
            args.candidate_summary,
            args.gate_metadata,
            args.release,
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "confirmatory":
        report = confirmatory.run(
            args.freeze_file,
            args.config,
            args.manifest,
            args.cache_root,
            args.output_root,
            args.gt_root,
            args.gt_cache_root,
            args.model_path,
            args.upper_indices,
            args.left_indices,
            args.right_indices,
            args.device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "report-final":
        report = report_final.run(
            args.method, args.comparison, args.output_root, args.stress_slices
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "verify-repro":
        report = verify_repro.run(
            args.manifest,
            args.first_predictions,
            args.second_predictions,
            args.first_summary,
            args.second_summary,
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
