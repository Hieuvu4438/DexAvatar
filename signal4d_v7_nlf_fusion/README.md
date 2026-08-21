# SIGNAL-4D V7 NLF fusion (isolated research lane)

This directory is an additive research lane. It does not modify or overwrite
the frozen SIGNAL-4D V5/V6 code, runs, predictions, or release artifacts.

The first stage exports the information that the historical
`scripts/S1_nlf_adapter.py` discarded: NLF non-parametric surface queries,
2D/3D locations, and per-query uncertainties. These observations are intended
for uncertainty-weighted residual factors; they are not treated as a blind
replacement for SMPLer-X parameters.

The pinned model used in experiments is the official NLF v0.3.2 TorchScript
release. Every export records the model SHA-256, NLF source commit, input
manifest hash, inference settings, and selected detection.

Smoke example:

```bash
conda run -n nlf python signal4d_v7_nlf_fusion/extract_nlf_observations.py \
  --manifest signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_calibration.jsonl \
  --data-root data \
  --model /home/haipd/nlf_data/models/nlf_l_multi_0.3.2.torchscript \
  --output-root signal4d_v7_nlf_fusion/outputs/nlf_v032_calibration \
  --limit 3
```

The exporter intentionally keeps the native NLF camera-space convention. A
later fusion stage must estimate an explicit robust alignment before comparing
NLF observations with DexAvatar/SMPL-X coordinates.

## Current empirical status

NLF v0.3.2 is rejected as a wholesale SMPLer-X replacement. Direct NLF on the
locked 1,493-frame author protocol obtains 30.022 mm UBody-F and severely
regresses both hands. The exploratory body router instead treats NLF as a
complementary articulation expert, preserves the frozen V6 identity and hand
articulation, and compensates the local wrist rotations so that the global
wrist orientations remain those of V6.

The calibration-selected alpha is 0.75. On the full author-protocol diagnostic
it changes V6 UBody-F from 29.519 to 26.886 mm and UBody from 26.139 to 23.829
mm, while LHand changes by +0.006 mm and RHand by -0.053 mm. See
`artifacts/results_summary.json` for the exact values and claim boundary.

These are exploratory results, not a pristine held-out SOTA claim: development
labels train the router, calibration labels select its threshold and alpha, and
the test result has now been inspected. Test target values are excluded from
the router feature artifact and are not used for fitting or selection. A paper
claim still requires an externally trained/GT-free gate or sign-level
out-of-fold evaluation as specified in the research plan under `docs/proposal3`.

## Preregistered zero-training evaluation

`nlf_gtfree_2d_temporal_gate.py` is a separate clean-claim lane. It has no
trained parameters and selects the fixed SO(3) midpoint candidate using only
independent image-space arm keypoints, camera reprojection, NLF uncertainty,
and temporal displacement. Its exact configuration and pre-evaluation protocol
are locked in `configs/v7_gtfree_2d_temporal_gate_v1.json` and
`docs/proposal3/SIGNAL-4D_V7_zero_SGNify_training_preregistered_protocol_2026-08-21.md`.

The preregistered full evaluation selected 1/1,493 frames and produced 29.524
mm UBody-F versus 29.519 mm for V6. It is retained as a protocol-clean negative
result and must not be presented as an improvement. See
`docs/proposal3/SIGNAL-4D_V7_clean_evaluation_report_2026-08-21.md`.

`external_how2sign_residual_transport.py` is the external-supervision lane. It
transports a checkpoint residual trained and selected only on source-disjoint
How2Sign onto V6, while preserving global wrists and all local hand rotations.
Its configuration and pre-evaluation contract are locked separately from both
the exploratory SGNify-trained router and the zero-training 2D gate.
