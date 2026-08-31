# Signal4D External — How2Sign Synth3D exact V1

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-24 (Asia/Ho_Chi_Minh)
- Verification Status: VERIFIED for deterministic author evaluation; ANALYZED for the single-seed training run
- Version Label: signal4d_external_how2sign_synth3d_exact_v1
- Overall Confidence: CAUTION
- Git revision recorded by the checkpoint: `8401491c9eb4dea8db8a228187d867b23ceee159`
- Seed: `42`
- Runtime: PyTorch `2.11.0+cu128`, NumPy `2.2.6`, SciPy `1.17.1`, SMPL-X `0.1.28`
- SGNify reads used for training, validation, or calibration selection: `0`

## Outcome

The exact SMPL-H supervision run completed successfully, and the model learned a large rotation-error reduction on held-out How2Sign. That improvement did not transfer through the benefit router to SGNify: the SGNify benefit-score distribution collapsed far below the How2Sign calibration distribution, so only 5 of 4,479 region-frames were allowed to change.

On the official author evaluator, the new run is better than DexAvatar but statistically and practically indistinguishable from External V1. It must not be claimed as an improvement over External V1 from this single run.

## Method

### Data and target construction

- Released source: How2Sign Synth3D green-screen SMPL-H fits.
- Available fits: 2,266 sequences and 6,165,742 frames in total.
- Split inventory:
  - train: 1,983 fits / 5,405,652 frames
  - validation: 119 fits / 303,184 frames
  - test: 164 fits / 456,906 frames
- The release is missing one upstream sequence listed by the alignment table: `train/0H60j0VxTaE-5.npz`.
- Exact target rotation for the refiner is `thetas[:, 1:52]`, covering 21 upper-body joints and 15 joints per hand.
- Camera-aligned sentence start time is converted to a source-fit index as `floor(START_aligned * clip_fps + 0.5) + frame_number`.
- Existing 45-D observation features and the SMPLer-X/WiLoR initializer are preserved. Only the older pseudo target is replaced by the released exact SMPL-H rotations.
- Fail-closed materialization excludes missing, unaligned, and out-of-bounds examples.

Materialized learning splits:

| Split | Clips | Frames | Source groups | Fits | Main exclusions |
|---|---:|---:|---:|---:|---|
| Train | 9,637 | 308,384 | 1,904 | 1,923 | 1,139 unaligned; 5 missing; 41 out of bounds |
| Validation | 477 | 15,264 | 56 | 62 | 17 unaligned; 4 out of bounds |
| Calibration | 408 | 13,056 | 49 | 55 | 88 unaligned; 1 out of bounds |

There are no source-group overlaps across train, validation, and calibration. The lineage audit reports zero SGNify reads.

### Model and objective

- Non-causal spatial-temporal Transformer: 6 layers, hidden size 256, 8 heads, MLP ratio 4, dropout 0.1, sequence length up to 64.
- Input is 45-D per joint plus augmented signals and a 2-D reprojection residual skip.
- Output is a bounded residual rotation composed with the initializer; limits are 18 degrees for upper body and 25 degrees for hands.
- The uncertainty head is disabled. Therefore this run does **not** train or estimate sigma.
- A three-region benefit head predicts whether applying the residual should improve upper body, left hand, or right hand.
- Benefit labels are supervised binary labels: refined geodesic rotation error must beat initializer error by more than 0.05 degrees in that region.

The optimized objective is:

`L = L_rotation + 0.25 L_velocity + 0.10 L_acceleration + 0.08 L_anchor + 0.01 L_biomechanical + 0.15 L_benefit`.

Vertex, joint-position, fingertip, palm, observation, and uncertainty losses are disabled for this run. Training fine-tunes External V1 for 2,000 optimizer steps with BF16, batch size 6, gradient accumulation 8, learning rate `4e-5`, weight decay 0.05, EMA 0.999, and a 60/20/20 real/synthetic/clean residual mixture.

### Calibration and inference

- Threshold selection uses only the held-out How2Sign calibration split.
- A region is refined only when `sigmoid(benefit_logit) >= threshold`; otherwise its initializer pose is copied exactly.
- Selected thresholds: upper body 0.95, left hand 0.75, right hand 0.80.
- SGNify inference is target-free. Author ground truth is read only after predictions are frozen and registered.

## Training and calibration findings

The best checkpoint is the EMA state at step 2,000. On validation, prediction-over-initializer geodesic-error ratios are:

| Region | Ratio | Relative reduction |
|---|---:|---:|
| Upper body | 0.7194 | 28.1% |
| Left hand | 0.6260 | 37.4% |
| Right hand | 0.6483 | 35.2% |

Calibration PASS results are:

| Region | Initializer rad | Selected candidate rad | Ratio |
|---|---:|---:|---:|
| Upper body | 0.3682 | 0.2562 | 0.6959 |
| Left hand | 0.7262 | 0.4459 | 0.6140 |
| Right hand | 0.7750 | 0.4952 | 0.6389 |

All 13,056 How2Sign calibration frames exceed every selected regional threshold. On SGNify, the score distributions shift sharply:

| Region | How2Sign calibration median | SGNify median | SGNify maximum | Selected SGNify region-frames |
|---|---:|---:|---:|---:|
| Upper body | 0.9922 | 0.0064 | 0.6765 | 0 / 1,493 |
| Left hand | 0.9883 | 0.0100 | 0.8172 | 3 / 1,493 |
| Right hand | 0.9883 | 0.0079 | 0.9127 | 2 / 1,493 |

This is direct evidence of benefit-head domain shift. The Transformer learned a useful How2Sign pose correction, but the learned router treats almost all SGNify frames as out-of-domain/harmful and suppresses that correction.

## Author evaluation

All values are millimetres, lower is better. The five displayed columns match the user's External V1 row: all, upper body, upper body minus face, left hand, right hand.

| Method | All | Upper body | Upper body − face | Left hand | Right hand |
|---|---:|---:|---:|---:|---:|
| DexAvatar | 42.5867 | 26.4560 | 29.9074 | 13.5735 | 12.9271 |
| External V1 | **42.2423** | **26.2236** | **29.6196** | **12.8102** | 12.1148 |
| Synth3D exact V1 | 42.2427 | 26.2239 | 29.6201 | 12.8114 | **12.1125** |

Synth3D exact V1 versus DexAvatar:

- all: −0.3441 mm (−0.81%)
- upper body: −0.2321 mm (−0.88%)
- upper body minus face: −0.2874 mm (−0.96%)
- left hand: −0.7620 mm (−5.61%)
- right hand: −0.8146 mm (−6.30%)

Synth3D exact V1 versus External V1:

- all: +0.00036 mm
- upper body: +0.00031 mm
- upper body minus face: +0.00046 mm
- left hand: +0.00121 mm
- right hand: −0.00234 mm

These differences are orders of magnitude below a practically meaningful mesh error and are based on one training seed. There are no confidence intervals or hypothesis tests supporting superiority over External V1.

## Coverage diagnosis

| Run | Changed region-frames | Total region-frames | Changed clips |
|---|---:|---:|---:|
| External V1 | 6 | 4,479 | Frisch, Jahr, Schnee, Schwer |
| Synth3D exact V1 | 5 | 4,479 | Frisch, Jahr, Regen, Schnee, Schwer |

The large improvement of both External variants over DexAvatar is therefore primarily inherited from the cleaner frozen SMPLer-X/WiLoR initializer and its export path, not from broad Transformer refinement on SGNify. The new exact-target Transformer changes only 0.112% of regional decisions.

## Validation Report

### Statistical findings

No p-values, confidence intervals, or repeated-seed dispersion are available. The correct interpretation is descriptive only. Confidence is CAUTION because the model was trained once and the target-domain gate has severe score shift.

### Fallacy scan

Coverage: 11/11 checked.

| Fallacy | Status | Detail |
|---|---|---|
| Simpson's paradox | NOTE | No subgroup reversal analysis is available; only micro and clip-macro aggregates are reported. |
| Ecological fallacy | NOTE | Claims are restricted to the evaluated clips/frames; no individual-signer inference is made. |
| Berkson's paradox | CAUTION | SGNify development is a selected benchmark and may not represent arbitrary sign videos. |
| Collider bias | NOTE | No regression controls or conditioned causal model are used. |
| Base-rate neglect | NOTE | No diagnostic sensitivity/specificity claim is made. |
| Regression to the mean | NOTE | The comparison is not a pre/post extreme-group design. |
| Survivorship bias | CAUTION | Materialization excludes unaligned/out-of-bounds samples; exclusions are disclosed and fail-closed. |
| Look-elsewhere effect | CAUTION | Multiple regions and checkpoints were observed; no multiplicity-corrected significance claim is made. |
| Garden of forking paths | CAUTION | One seed/config is reported; the calibration grid and checkpoint rule are fixed, but no preregistration exists. |
| Correlation is not causation | NOTE | The report describes measured benchmark differences and does not claim dataset causality. |
| Reverse causality | NOTE | Not applicable to the benchmark design. |

### Reproducibility

- Author evaluator method: deterministic re-run.
- Verdict: REPRODUCIBLE.
- `comparison.json`, `comparison.csv`, and `comparison.md` are byte-identical across two independent evaluator invocations.
- Maximum numeric difference across all metrics: exactly `0.0`.
- Training verdict: CANNOT VERIFY across runs because only one GPU training seed was executed. The checkpoint, resolved config, data manifests, dependencies, and seed are recorded for later reproduction.

## Artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| Exact materialization report | `05c727b3c9c055ae075d95a9a7bdcf29af70e00340ea160c73b6331b91c0f68b` |
| Lineage audit | `36dd2c0732935909735f78047da2816fb24e1c026b9bbf55b87de73b1d745c5a` |
| Best checkpoint | `348c4037a26b11a814b28fb07085335e269b095328909abeab090eedecb05f5a` |
| Calibration | `0c7d6715638ac98f064756b89a2ed2b55707eef6ff7c54ad6e5dfab328edc486` |
| Inference manifest | `3a00a307bbd73651f7c59cbf4398f121858306e2bae4b0b542837d68d67bf019` |
| Author comparison | `79167a23ad747762593258232c4d563129804d00bf371cb2c3396a2717a38042` |

## Limitations and next experiment

1. The exact-target pose refiner transfers poorly through its benefit router because How2Sign and SGNify feature distributions differ drastically.
2. The result is single-seed; no variance estimate exists.
3. The cache/config method label inherited `SIGNAL4D_EXTERNAL_HOW2SIGN_CLIPNORM_V1`; the registered evaluation method is correctly overridden to `SIGNAL4D_EXTERNAL_HOW2SIGN_SYNTH3D_EXACT_V1`. The post-run config is intentionally not edited because that would invalidate checkpoint provenance.
4. A defensible next experiment is not “more How2Sign epochs.” It is target-domain adaptation of the gate without SGNify ground-truth leakage: score-distribution alignment or self-supervised reliability calibration on unlabeled SGNify inputs, frozen before author evaluation. A labeled same-domain non-SGNify sign dataset is stronger if its camera, crop, initializer, and feature pipeline match SGNify.
