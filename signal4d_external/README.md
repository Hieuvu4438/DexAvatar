# SIGNAL4D external-only lane

This directory is an append-only replacement for the learned/calibrated part
of SIGNAL4D V5/V6. It does not import V5 predictions, V5's 12-clip calibrator,
or V6 predictions. The old `signal4d/` and `phase2_refiner/` implementations
remain unchanged.

## Final outcome (2026-08-24)

The frozen V1 output is the final external-only result:
`outputs/signal4d_external/full1493_wilor_clipnorm_v1`.  Its strict
57-clip/1,493-frame author-evaluator result is 42.2423 mm All, 26.2236 mm
upper body, 29.6196 mm UBody-F, 12.8102 mm left hand, and 12.1148 mm right
hand.  Hand V2, NLF V2, sign DPoser-X, sign VQ-VAE, and arm bundle-adjustment
V4 were all evaluated under frozen external gates and were not promoted.  See
`reports/EXTERNAL_ONLY_FINAL_20260824.md` for the comparison, hashes, leakage
audit, and rejection reasons.

## Frozen protocol

- Training: 10,822 How2Sign clips / 346,304 frames.
- Checkpoint selection: 498 source-group-disjoint How2Sign validation clips /
  15,936 frames.
- Abstention calibration: 497 additional source-group-disjoint How2Sign clips /
  15,904 frames.
- SGNify: inference images only until the checkpoint, thresholds, and 1,493
  predictions are frozen. Ground truth is opened only by the final evaluator.
- Initialization: frozen DexAvatar WiLoR view with deterministic HaMeR coverage
  fallback. No V5 or V6 prediction is an input.

The model appends clip-relative normalized reprojection XY and its log median
scale to the existing 45-D observation token. This normalization is computed
independently for the current clip, so no target-population statistic is
fitted. A factorized spatial/temporal/group attention network predicts bounded
SO(3) residuals and a region benefit probability. Region thresholds are chosen
only on the held-out How2Sign calibration split; rejected regions fall back
exactly to the clean initializer.

ARCTIC and InterHand2.6M were audited but not mixed into the primary run:
ARCTIC is generic hand-object motion and InterHand has only partial hand
targets. SignAvatars is the preferred future exact-3D source, but the local
download was incomplete when this protocol was frozen.

## Commands

All commands run from `/home/haipd/DexAvatar`.

```bash
python -m signal4d_external.leakage \
  --train cache/phase2/t2_how2sign_2d_temporal_reprojection_v2/splits/train.json \
  --validation cache/phase2/t2_how2sign_2d_temporal_reprojection_v2/splits/val.json \
  --calibration cache/phase2/t2_how2sign_2d_temporal_reprojection_v2/splits/calibration.json \
  --output outputs/signal4d_external/how2sign_clipnorm_benefit_v1_lineage.json

python -m signal4d_external.train \
  --config signal4d_external/configs/how2sign_clipnorm_benefit_v1.yaml \
  --device cuda

python -m signal4d_external.calibrate \
  --config signal4d_external/configs/how2sign_clipnorm_benefit_v1.yaml \
  --checkpoint outputs/signal4d_external/how2sign_clipnorm_benefit_v1_seed42/best.pt \
  --output outputs/signal4d_external/how2sign_clipnorm_benefit_v1_seed42/calibration.json \
  --device cuda

python -m signal4d_external.materialize_initializer \
  --template-root cache/phase2/lane_l_a1_ensemble_v1 \
  --initializer-root outputs/phase2_gates/g1_views/output_wilor \
  --source-manifest outputs/phase2_gates/g1_views/output_wilor/locked_view_manifest.json \
  --output cache/signal4d_external/wilor_clean_v1

python -m phase2_refiner.data.add_reprojection_residuals \
  --input-root cache/signal4d_external/wilor_clean_v1 \
  --output cache/signal4d_external/wilor_clean_reprojection_v1 \
  --mode lane \
  --device cuda

python -m signal4d_external.infer \
  --config signal4d_external/configs/how2sign_clipnorm_benefit_v1.yaml \
  --checkpoint outputs/signal4d_external/how2sign_clipnorm_benefit_v1_seed42/best.pt \
  --calibration outputs/signal4d_external/how2sign_clipnorm_benefit_v1_seed42/calibration.json \
  --cache-root cache/signal4d_external/wilor_clean_reprojection_v1 \
  --output outputs/signal4d_external/full1493_wilor_clipnorm_v1 \
  --render \
  --device cuda
```

Only after those outputs are frozen:

```bash
PYTHONPATH=signal4d/src python -m signal4d_external.register_obj \
  --manifest signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl \
  --source-root outputs/signal4d_external/full1493_wilor_clipnorm_v1 \
  --model-path data/ARCTIC/body_models/smplx/SMPLX_NEUTRAL.npz \
  --output-root signal4d/outputs/strict_dexavatar_obj_external_20260824/full1493/SIGNAL4D_EXT \
  --method-name SIGNAL4D_EXTERNAL_HOW2SIGN_CLIPNORM_V1

PYTHONPATH=signal4d/src python -m signal4d.cli.main evaluate-author-sgnify \
  --manifest signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl \
  --method DexAvatar=signal4d/outputs/strict_dexavatar_obj_20260820/full_1493/DexAvatar_HaMeR \
  --method SIGNAL4D_EXT=signal4d/outputs/strict_dexavatar_obj_external_20260824/full1493/SIGNAL4D_EXT \
  --baseline DexAvatar \
  --gt-root data/smplx_gt \
  --author-source data/evaluation_from_author/evaluate_new_fitting.py \
  --author-asset-root data/evaluation_from_author/data/data \
  --author-sign-file data/evaluation_from_author/data/data/signs.txt \
  --author-segment-file data/evaluation_from_author/data/data/segment.json \
  --frame-policy manifest \
  --prediction-format dexavatar-obj \
  --output signal4d/reports/external_only_full1493_v1_reveal
```

The final result and limitations are recorded in
`signal4d_external/reports/HOW2SIGN_CLIPNORM_BENEFIT_V1.md`.

## Selectable SGNify initializer for the sign-domain V1R checkpoint

`materialize_initializer` retains the External-V1 fitted initializer as its
default.  Omitting `--initializer-mode` is therefore backward compatible with
the `wilor_clean_v1` command above.  The opt-in `raw-smplerx-wilor` mode builds
the frontend used during sign-domain V1R training: SMPLer-X H32 supplies the
21 body rotations and WiLoR supplies each detected 15-joint hand.  A missing
WiLoR side falls back independently to that side's SMPLer-X hand pose.

The complete target-free SGNify raw-frontend path is:

```bash
PYTHONPATH=signal4d/src:. python -m signal4d_external.materialize_initializer \
  --template-root cache/phase2/lane_l_a1_ensemble_v1 \
  --initializer-mode raw-smplerx-wilor \
  --raw-observation-root signal4d/artifacts/cache/sgnify_smplerx_wilor_a1_leftmirror_v5_all \
  --raw-smplerx-root outputs/output_baseline \
  --raw-smplerx-subpath smplerx/smplx \
  --output cache/signal4d_external/sgnify_raw_smplerx_wilor_v2

python -m phase2_refiner.data.add_reprojection_residuals \
  --input-root cache/signal4d_external/sgnify_raw_smplerx_wilor_v2 \
  --output cache/signal4d_external/sgnify_raw_smplerx_wilor_reprojection_v2 \
  --mode lane --device cuda

python -m phase2_refiner.data.audit_reprojection_domain_shift \
  --source-manifest cache/signal4d_external/sign_domain_raw_fusion_reprojection_train_v1/splits/train.json \
  --target-root cache/signal4d_external/sgnify_raw_smplerx_wilor_reprojection_v2 \
  --output outputs/phase2r/sign_domain_raw_fusion_v1_seed42/sgnify_raw_reprojection_domain_shift_v2.json

PYTHONPATH=signal4d/src:. python \
  outputs/phase2r/sign_domain_raw_fusion_v1_seed42/sgnify_targetless_infer.py \
  --config phase2_refiner/configs/sign_domain_raw_fusion_v1.yaml \
  --checkpoint outputs/phase2r/sign_domain_raw_fusion_v1_seed42/best.pt \
  --calibration outputs/phase2r/sign_domain_raw_fusion_v1_seed42/calibration.json \
  --cache-root cache/signal4d_external/sgnify_raw_smplerx_wilor_reprojection_v2 \
  --domain-shift-report outputs/phase2r/sign_domain_raw_fusion_v1_seed42/sgnify_raw_reprojection_domain_shift_v2.json \
  --expected-initializer-mode raw-smplerx-wilor \
  --render-mode direct-smplx \
  --method-name SIGNAL4D_SIGN_DOMAIN_V1R_RAW_INIT \
  --scope "Targetless SGNify evaluation using the frontend-matched raw SMPLer-X H32 body plus WiLoR hands initializer." \
  --output outputs/phase2r/sign_domain_raw_fusion_v1_seed42/sgnify_raw_frontend_full1493_v2 \
  --render --device cuda
```

The strict 57-clip/1,493-frame result and the zero-acceptance gate diagnosis
are recorded in
`outputs/phase2r/sign_domain_raw_fusion_v1_seed42/SGNIFY_RAW_INITIALIZER_EVALUATION_20260826.md`.

## External-only NLF V2 (external gate failed; not materialized)

V2 is append-only and leaves every V1 artifact unchanged. It uses NLF as an
uncertainty-bearing upper-body candidate, trains a benefit router on
signer-disjoint How2Sign subsets, preserves WiLoR hand poses and global wrist
rotations, and selects all hyperparameters externally. The locked protocol,
decision gates, literature basis, and non-confirmatory claim boundary are in
`reports/EXTERNAL_ONLY_NLF_V2_PREREGISTRATION.md`.

The deterministic manifests are under
`cache/signal4d_external/nlf_v2_protocol_seed42`. The execution order is:

```bash
conda run -n nlf python -m signal4d_external.extract_external_nlf_v2 \
  --manifest cache/signal4d_external/nlf_v2_protocol_seed42/train.json \
  --manifest cache/signal4d_external/nlf_v2_protocol_seed42/validation.json \
  --manifest cache/signal4d_external/nlf_v2_protocol_seed42/calibration.json \
  --model /home/haipd/nlf_data/models/nlf_l_multi_0.3.2.torchscript \
  --nlf-root nlf \
  --output-root outputs/signal4d_external/nlf_v2_how2sign_observations_seed42 \
  --device cuda:0 --batch-size 4 --num-aug 1

python -m signal4d_external.train_nlf_router_v2 \
  --train-manifest cache/signal4d_external/nlf_v2_protocol_seed42/train.json \
  --validation-manifest cache/signal4d_external/nlf_v2_protocol_seed42/validation.json \
  --calibration-manifest cache/signal4d_external/nlf_v2_protocol_seed42/calibration.json \
  --observation-root outputs/signal4d_external/nlf_v2_how2sign_observations_seed42 \
  --model-path data/ARCTIC/body_models/smplx/SMPLX_NEUTRAL.npz \
  --output-root outputs/signal4d_external/nlf_body_router_v2_seed42
```

The completed router calibration emitted `decision=FAIL`, so target inference
was not run. Had it passed, `materialize_nlf_v2` would consume the frozen
target-image NLF observations and the V1 baseline without reading SGNify
targets.

## External-only hand V2 (evaluated; not promoted)

V2H addresses the V1 hand residuals that were almost completely suppressed by
absolute benefit thresholds after the domain shift. It selects SO(3) residual
scale, intervention coverage, and temporal score smoothing on signer-disjoint
How2Sign validation; a third signer set is gate-only. At target inference it
uses ranks of unlabeled benefit scores to enforce the externally frozen
coverage. This transductive covariate use is disclosed in both the
preregistration and run manifest; SGNify labels and evaluator assets remain
unread.

```bash
python -m signal4d_external.extract_hand_v2_predictions \
  --config signal4d_external/configs/how2sign_clipnorm_benefit_v1.yaml \
  --checkpoint outputs/signal4d_external/how2sign_clipnorm_benefit_v1_seed42/best.pt \
  --validation-manifest cache/signal4d_external/nlf_v2_protocol_seed42/validation.json \
  --calibration-manifest cache/signal4d_external/nlf_v2_protocol_seed42/calibration.json \
  --output outputs/signal4d_external/hand_v2_external_predictions_seed42 \
  --device cpu

python -m signal4d_external.calibrate_hand_v2 \
  --prediction-cache outputs/signal4d_external/hand_v2_external_predictions_seed42 \
  --output outputs/signal4d_external/hand_v2_external_predictions_seed42/calibration.json
```

Only after calibration emits `decision=PASS`:

```bash
python -m signal4d_external.materialize_hand_v2 \
  --config signal4d_external/configs/how2sign_clipnorm_benefit_v1.yaml \
  --checkpoint outputs/signal4d_external/how2sign_clipnorm_benefit_v1_seed42/best.pt \
  --calibration outputs/signal4d_external/hand_v2_external_predictions_seed42/calibration.json \
  --cache-root cache/signal4d_external/wilor_clean_reprojection_v1 \
  --baseline-root outputs/signal4d_external/full1493_wilor_clipnorm_v1 \
  --output outputs/signal4d_external/full1493_hand_v2_seed42 \
  --device cpu
```

The frozen protocol is in
`reports/EXTERNAL_ONLY_HAND_V2_PREREGISTRATION.md`. Its frozen target reveal
was slightly worse than V1, so it was not promoted.

## Combining the passed body and hand lanes

If and only if both external calibration reports emit `decision=PASS`, run NLF
body materialization with V2H as its baseline. The NLF materializer replaces
only SMPL-X body pose on selected frames and copies V2H left/right hand poses
exactly. It also verifies that the external and target NLF observations share
the same model hash, source commit, augmentation count, detector threshold,
and person-selection rule.

```bash
python -m signal4d_external.materialize_nlf_v2 \
  --router-root outputs/signal4d_external/nlf_body_router_v2_seed42 \
  --observation-root signal4d_v7_nlf_fusion/outputs/nlf_v032_full1493 \
  --cache-root cache/signal4d_external/wilor_clean_reprojection_v1 \
  --baseline-root outputs/signal4d_external/full1493_hand_v2_seed42 \
  --model-path data/ARCTIC/body_models/smplx/SMPLX_NEUTRAL.npz \
  --output-root outputs/signal4d_external/full1493_nlf_body_hand_v2_seed42 \
  --device cpu
```

No target metric may be opened between V2H and combined materialization. Freeze
both run manifests and predictions first, then register OBJ files and perform
the one preregistered author-protocol evaluation.
