# Pre-registered Experiment Plan

**Version:** M1, locked 2026-08-21 before any local SGNify metric was observed.  
**Co-primary metrics:** `TR-V2V UBody(-F)`, `TR-V2V LHand`, `TR-V2V RHand`, all lower-is-better in mm.  
**Evaluator hash:** `2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300`.

## Protocol lock

1. Build an explicit `(sign_id, frame_id)` manifest from GT and predictions; assert exact equality, uniqueness, topology, finite vertices, and expected region counts.
2. Keep original evaluator untouched. A wrapper may only parameterise paths, enforce completeness, and log counts. It must agree with the original on complete valid data to floating-point tolerance (`1e-9 m` per stored mean before mm conversion).
3. Reproduction match tolerance: each reported DexAvatar regional mean must differ by at most `max(0.5 mm, 2% of the paper value)` using the official checkpoint and matching frame manifest. Otherwise label `not reproduced` and investigate before method comparison.
4. No SGNify test result is used for keep/kill, architecture, hyperparameters, checkpoint selection, or early stopping. If SGNify has no valid development split, reserve it for one final evaluation and use a predeclared validation set from training data.
5. Co-primary metrics remain a vector. Report Pareto trade-offs; do not create a weighted scalar after seeing results.

## Experiment ladder and matrix

| Run ID | Hypothesis / purpose | Change vs baseline | Dataset/split | Seed(s) | Budget | Keep/kill rule | Status |
|---|---|---|---|---|---|---|---|
| EVAL-001 | Metric math is translation-only and numerically sane | None; toy arrays | Synthetic | N/A | <1 min CPU | PASS all known cases | [VERIFIED] PASS |
| REPO-STATIC-001 | Released source is syntactically inspectable | None | Source only | N/A | <1 min CPU | 0 AST/shell failures | [VERIFIED] PASS |
| BASE-CLI-001 | Entry point can expose CLI without model imports | None | Source only | N/A | <1 min CPU | exit 0 | [VERIFIED] PASS |
| BASE-OFFICIAL-001 | Official checkpoint matches paper | None | SGNify central frames | deterministic + repeat | ≤1 full eval | All regions within reproduction tolerance | [BLOCKED] assets/GPU |
| BASE-RETRAIN-001 | Released training/prior procedure is reproducible | Retrain priors | Author training data or declared substitute | 3 | To be budgeted | Curves finite; validation within expected variance | [BLOCKED] data/compute |
| HAND-001 | WiLoR proposal improves hands under same fitter | Hand proposal only | Validation | fixed | 1 short run | Improvement exceeds noise in ≥1 hand; no other region regression > noise | [PLANNED] |
| IK-001 | Kinematic alignment fixes wrist/forearm inconsistency | Deterministic IK only | Validation | fixed | 1 short run | Predicted wrist/forearm slice improves and no completeness loss | [PLANNED] |
| TEMP-001 | Visibility-weighted acceleration helps low-confidence frames | Temporal loss only | Validation | fixed | 1 short run | Low-confidence/blur slice improves beyond noise; high-confidence slice stable | [PLANNED] |
| GATE-001 | Reliability gate selects the better proposal | Deterministic gate, no learned residual | Validation | fixed | 2 threshold settings fixed in advance | Oracle gap narrows; shuffled reliability does not match | [PLANNED] |
| GATE-002 | Learned gate adds value beyond parameters | Learned gate and parameter-matched control | Train/validation only | 3 | Small model budget | Mean gain > noise; variance acceptable; control weaker | [PLANNED] |
| RGKRF-001 | Coupled gate + kinematic residual is synergistic | Full method | Validation | 3 | Full short schedule | All co-primary metrics non-worse than noise and ≥1 meaningful gain | [PLANNED] |
| ABL-001 | Isolate components/interactions | Factorial subset A0–A9 | Validation | 3 for learned variants | Budget after RGKRF keep | Interaction evidence supports claim | [PLANNED] |
| FINAL-001 | Clean same-protocol comparison | Baseline vs selected method | Held-out test | 5 if learned; deterministic repeat otherwise | One locked opening | No seed exclusion; report mean±SD/CI | [BLOCKED] prior gates |

## Noise floor and decision rules

Before candidate testing, repeat the complete baseline with identical inputs and, where stochastic kernels exist, at least three fixed seeds/runs. For region `r`, define `noise_r = max(0.25 mm, 2 × baseline_SD_r)` on validation.

- **KEEP:** at least one prespecified target region improves by more than `noise_r`, no co-primary region regresses by more than its noise, completeness is identical, and added cost/licence is acceptable.
- **REVISE:** mechanism slice improves but aggregate does not, or instability/compute rises; run only the prespecified diagnostic ablation.
- **KILL:** no target slice improves beyond noise after the planned trials, reliability shuffling performs similarly, or gain disappears under completeness/compute control.
- **PIVOT:** baseline error map contradicts the assumed body–hand reliability bottleneck, or the required licence/data cannot support publication.

## Statistics

- Learned variants: three seeds for screening; five for final claims when gain is small or variance is material.
- Report unrounded per-seed values, mean, standard deviation, 95% bootstrap CI over sign sequences, and paired per-frame/per-sign deltas where dependence assumptions permit.
- Use a paired sign-level permutation/bootstrap test for the final selected comparison; disclose the unit of resampling and correct for multiple comparisons across the three co-primary regions.
- Deterministic fitting still receives a repeatability check and full per-sign distribution, not only one rounded mean.

## Required slices and efficiency fields

One-handed vs two-handed, left vs right dominant, high/low keypoint confidence, blur/crop/occlusion proxy, contact vs non-contact, fast vs slow motion, early/middle/late frames, and prediction-completeness status. Log parameter count, runtime/frame and/video, peak GPU memory, input resolution, checkpoint/data provenance, and preprocessing cost.

