# Phase 2R Recovery Branch: Domain-Aligned, Selective Whole-Sequence Refinement

- **Project:** DexAvatar / SignPosterior4D de-risking program
- **Branch:** Phase 2R, derived from Phase 2 UAWSR
- **Date:** 3 August 2026
- **Status:** proposed recovery branch; Phase 3 remains blocked
- **Purpose:** explain the executed Phase 2 NO-GO results, identify their code-level root causes, and define the shortest scientifically valid route to a spatially useful and uncertainty-aware sequence refiner

---

## 1. Executive decision

Do **not** continue from the current Phase 2 checkpoint to relational diffusion,
phonology conditioning, or any other Phase 3 mechanism.

The Phase 2 implementation is numerically stable and has demonstrated that its
architecture can recover synthetic corruption. It has not demonstrated that it
can safely improve the selected strong initializer in the target domain. The
locked result is decisive:

| Method | Equal-region Lane-L gain | Hard-subset gain | Right-hand change | Fallback | Decision |
|---|---:|---:|---:|---:|---|
| direct T2 v6 | `+0.0666%` | left `-6.15%`, right `-3.76%` | `+0.1648 mm` worse | `0%` | G6 NO-GO |
| T2 v6 + T5 | `-0.5934%` | `-5.2511%` | `+0.2465 mm` worse | `30.5425%` | G6 NO-GO |

The failure is not explained by insufficient transformer size or too few
training steps. It is caused by a broken experimental chain between the
initializer used for supervision, the semantics of cached observations, the
target construction objective, the model's shortcut path, the release safety
criterion, and the final mesh metric.

The replacement branch is **Phase 2R: Domain-Aligned Selective Sequence
Refinement**. Its central change is conceptual:

> A sequence model may change a body or hand group only when it has evidence
> that the exact deployed initializer is wrong and evidence that the proposed
> correction is safer than identity.

Phase 2R must first create a deployable, dataset-neutral initializer contract;
then train and validate on outputs from that same contract; then prove that
temporal context adds spatial value beyond a framewise residual model; and only
then add calibrated selective prediction. A new untouched confirmatory dataset
is required before Phase 3 because Lane-L has already been observed during
several remediation cycles.

---

## 2. What was audited

This diagnosis follows the executed artifacts back through the code that
produced them. The main evidence sources are:

- the superseding Phase 2 results in Sections 29--31 of
  `DEXAVATAR_PHASE2_UNCERTAINTY_AWARE_WHOLE_SEQUENCE_REFINER.md`;
- G0/G1 locked evaluator outputs under `outputs/phase2_gates/g0_a0` and
  `outputs/phase2_gates/g1_eval`;
- formal G4, G5, and G6 JSON reports under `outputs/phase2_gates`;
- the exact-A1 Stage-1 failure report
  `outputs/phase2_gates/g4/how2sign_exact_a1_stage1_failure.json`;
- the frozen fitter in `dexavatar_fitting/smplifyx` and its shell pipeline;
- How2Sign extraction, pseudo-target, reprojection, cache, corruption, training,
  inference, T5, calibration, and evaluation code under `phase2_refiner`;
- the saved How2Sign and Lane-L caches; and
- the exported per-frame, per-sign, and diagnostic outputs for direct T2 and
  T5.

No SGNify evaluation mesh was read as a training target during this audit. No
existing model, cache, result, or evaluator was modified.

---

## 3. Current gate truth

The following is the appropriate interpretation of all executed evidence as of
this branch point.

| Gate | Current status | What is actually established |
|---|:---:|---|
| G0 evaluator/coverage | **GO** | the author-released 57-sign, 1,493-frame population is reproducible with strict coverage |
| G1 initializer | **GO on Lane-L only** | the locked `method_ensemble` view improves A0; it is 1,450 primary frames plus 43 HaMeR fallback frames |
| G2 data volume | **conditional GO** | 10,822 accepted sign clips and 346,304 frames exist, but target validity is only weakly audited and the initializer is H32 |
| G3 synthetic recoverability | **GO** | complete-region ARCTIC corruption recovery passes; this establishes model capacity, not target-domain value |
| G4 real validation value | **proxy GO / formal NO-GO** | H32-to-2D-bundle-adjusted targets improve in their own proxy domain; exact deployed-initializer provenance is absent |
| G5 uncertainty | **ranking proxy GO / formal and causal NO-GO** | v7 predicts proxy-domain error well, but exact-A1 provenance is absent and uncertainty is not used to alter refinement |
| G6 locked spatial benchmark | **NO-GO** | direct T2 and T5 fail effect size, hard-subset safety, and right-hand safety |
| G7 evaluation scope | **GO by project scope** | the author-released 1,493-frame population is canonical for this project; it is not the unavailable 2,872-frame paper population |

The decisive blocker is G6. Even if the formal provenance bits in G4/G5 were
manually changed, the spatial result would still fail.

---

## 4. Evidence that localizes the failure

### 4.1 The model works on controlled corruption

The complete-region ARCTIC T1 checkpoint recovered `66.34--79.36%` of decoded
regional vertex error over 4/8/16-frame bursts while keeping clean drift below
`0.003 mm`. This rules out a basic inability to learn temporal interpolation,
an SO(3) composition failure, and a general optimization failure.

### 4.2 The proxy task is learnable in its own domain

On 498 source-disjoint How2Sign validation clips, T2 v6 reduced mean joint
rotation error by:

- `11.76%` for upper body;
- `13.56%` for left hand;
- `12.80%` for right hand; and
- `12.71%` with equal regional weighting.

T5 increased the equal-region proxy gain to `33.14%`. These measurements prove
that the model and optimizer can reproduce the 2D-derived target builder. They
do not prove better 3D reconstruction.

### 4.3 The same correction policy does not transfer to A1

On Lane-L, direct T2 significantly improves upper body and left hand but
significantly worsens the right hand:

| Region | A1 | T2 v6 | Delta | Sign-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| upper body | 29.5347 | 29.3801 | `-0.1546 mm` | `[-0.2281, -0.0236]` |
| left hand | 12.8249 | 12.6919 | `-0.1330 mm` | `[-0.2695, -0.0024]` |
| right hand | 12.1129 | 12.2777 | `+0.1648 mm` | `[+0.0852, +0.2975]` |

The hard subsets become worse, not better. This is the opposite of the Phase 2
hypothesis that the refiner should intervene primarily on temporary expert
failures.

### 4.4 The observation-only safety objective is not a spatial safety signal

T5 accepts all left- and right-hand clips on How2Sign, but only 7/57 left-hand
and 9/57 right-hand clips on Lane-L. It accepts no Lane-L body clip. Despite
falling back on `30.54%` of all Lane group-frames, it still worsens the right
hand by `0.2465 mm` and the aggregate hard subset by `5.25%`.

This is direct evidence that lower single-view reprojection error is not a
reliable surrogate for lower regional TR-V2V in this setup.

### 4.5 The source and target observation distributions are not close

The saved domain-shift audit reports Lane-A1 / How2Sign-H32 median
reprojection-residual ratios of:

- `2.35172` for upper body;
- `0.32744` for left hand; and
- `0.34763` for right hand.

The shift is not a uniform scale error. Body residuals grow while hand residuals
shrink, so one global token scale cannot repair it.

### 4.6 Cadence is also shifted

A direct cache audit found:

| Sequence source | Frames per clip | Median source-frame gap | Mean time gap | Maximum time gap |
|---|---:|---:|---:|---:|
| How2Sign validation | always 32 | 4 frames | `0.229 s` | `2.292 s` |
| Lane-L | 12--48 | 2 frames | `0.080 s` | `0.080 s` |

How2Sign uses 32 uniformly sampled frames over each source clip. The training
velocity and acceleration features/losses use consecutive array entries
without dividing rotations by elapsed time. Therefore "one temporal step" has
different physical meaning in training and deployment.

---

## 5. Root-cause tree

```text
G6 spatial NO-GO
  |
  +-- deployed initializer is not the training initializer
  |     +-- H32 training vs ensemble+fallback deployment
  |     +-- frozen fitter is coupled to 57 sign names/classes/segments
  |     +-- fallback/expert identity is absent from per-frame model tokens
  |
  +-- observation tokens do not have provider-invariant semantics
  |     +-- confidence and presence are calibrated differently
  |     +-- feature 5 means 2D displacement in How2Sign but pose innovation in Lane
  |     +-- crop-camera reprojection vs saved perspective-K reprojection
  |     +-- sparse irregular training cadence vs fixed Lane cadence
  |
  +-- proxy supervision is misaligned with the final claim
  |     +-- targets and reprojection input come from the same 2D tracks
  |     +-- monocular 2D bundle adjustment is depth ambiguous
  |     +-- checkpoint selection uses rotation degrees
  |     +-- release uses regional mesh millimetres
  |
  +-- model can solve the proxy through a framewise shortcut
  |     +-- dense reprojection skip bypasses temporal attention
  |     +-- no required framewise-vs-temporal causal ablation
  |     +-- useful advertised 3D observation channels are empty in T2
  |
  +-- intervention and safety are not trained for selective benefit
        +-- T2 fallback catches only numerical/angle failures
        +-- T5 safety checks reprojection, not spatial benefit
        +-- U1 v7 predicts error but does not control refinement
```

The branches reinforce one another. Fixing only one is unlikely to pass G6.

---

## 6. Detailed root causes

### RC1 — The selected A1 is not a portable frozen expert

**Severity:** critical; blocks formal G4/G5 data construction.

The selected Lane-L initializer is a locked view, not one self-contained
dataset-neutral network:

- 1,450/1,493 frames come from `outputs/method_ensemble`;
- 43/1,493 frames, or `2.8801%`, fall back to `outputs/method_hamer`; and
- the fitter changes behavior according to the sign's class in
  `data/signs.txt`.

The frozen fitter reads the input directory basename and executes
`class_sign[folder_name]` in `dexavatar_fitting/smplifyx/main.py`. It then reads
fixed frame bounds from `data/segment.json`. The sign class controls one-hand
versus two-hand branches throughout `data_parser.py`, `fit_single_frame.py`, and
`fitting.py`.

The exact-A1 How2Sign preflight therefore reached the frozen SMPLify-X stage and
failed with `KeyError: 'images'`. Earlier experts produced 32 Sapiens entries,
32 SMPLer-X initializations, and 32 WiLoR/HaMeR entries, but the fitter emitted
zero result PKLs. The wrapper correctly stopped Stage 1.

There is also an orchestration defect: `methods/Full_running_command_wilor_ensemble.sh`
does not use fail-fast shell settings. Its nested fitter failure was masked by a
later success message. The new wrapper detected the failure only because it
required complete schema-valid outputs.

**Why this causes the observed result:** H32 residuals teach how to repair H32.
The deployed A1 contains a stronger fitted ensemble, class-conditioned fitting,
and a small fallback population. Its error distribution is different by
construction.

**What will not fix it:** adding How2Sign clip IDs to `signs.txt`, assigning an
arbitrary German sign class, or assigning artificial segment bounds. Each
changes the fitter's observation population and hand optimization policy and
would falsely label a different estimator as exact A1.

**Required repair:** define a portable initializer `A1R` whose complete input
contract is available at inference on arbitrary clips. There are two valid
options:

1. Extract the estimator and fitting logic from benchmark metadata. Replace
   sign-name lookup with explicit per-frame observations and a frozen,
   observation-derived hand-activity policy. Prove functional parity on all
   primary Lane frames before using it externally.
2. If parity is impossible because the old sign-class label is essential,
   declare the old A1 non-deployable, freeze a new dataset-neutral A1R, rerun
   G1 against A0, and use A1R consistently for all Phase 2R training,
   validation, and evaluation.

Option 2 is the recommended honest path. "Exact initializer" should mean the
same deployable function and fallback policy, not merely the same checkpoint
hashes.

### RC2 — Cache channels have different meanings across providers

**Severity:** critical; corrupts transfer even after coordinate ranges match.

Several token channels are numerically compatible but semantically different:

| Channel | How2Sign construction | Lane-L construction | Consequence |
|---|---|---|---|
| confidence | sigmoid of supplied track logits shifted by 3 | raw Sapiens confidence | scores are not calibrated to the same probability |
| hand presence | keypoint validity | existence of a HaMeR/WiLoR candidate | missingness means different events |
| observation feature 5 | 2D displacement divided by `0.15` | SMPL-X pose innovation divided by pi | the same index encodes different physical quantities |
| U0 reliability | confidence × validity | confidence × presence × missing/truncation/duplicate penalties × pose-innovation penalty | reliability has different formulas |
| reprojection | SMPLer-X crop-camera approximation | perspective `K` saved in each result PKL | residual direction/magnitude is camera-domain dependent |
| sequence step | irregular uniform samples | fixed 2-frame cadence | velocity and acceleration magnitudes are incomparable |

`data.reprojection_residual_scale: 10.0` repairs neither meaning nor
calibration. The existing domain-shift audit checks residual magnitude,
reliability mean, and validity fraction, but it cannot detect semantic aliasing
of feature 5 or cadence.

**Required repair:** introduce cache schema v4 with provider-invariant fields:

- separate `pose_innovation_rad_per_second` and
  `keypoint_velocity_normalized_per_second` fields;
- store raw provider confidence and a separately calibrated probability;
- split `detector_present`, `track_valid`, `in_frame`, `copied`, and
  `initializer_source` rather than compressing them into one presence value;
- store camera type and normalized rays, not only projected image residuals;
- store the exact per-frame initializer component and fallback reason;
- resample or window contiguous sequences at a frozen physical cadence; and
- require a schema conformance test proving equal units and meanings for every
  provider adapter.

All rotation velocity and acceleration features and losses must divide by the
actual `delta_t`. Acceleration must account for unequal adjacent intervals.

### RC3 — The proxy target and final metric optimize different claims

**Severity:** critical; explains strong proxy gains with negligible/worse mesh
gains.

`refine_how2sign_targets.py` creates targets by optimizing the H32 pose against
the same ordered 2D tracks later used to create the model's reprojection
residual input. The target builder accepts a clip when overall reprojection
improves by `0.5%`, no region worsens by more than `2%`, and the pose changes by
a nonzero amount. It does not require independent 3D improvement.

The resulting validation targets differ from H32 by approximately:

- `2.561 deg` mean upper-body rotation error;
- `7.478 deg` mean left-hand rotation error; and
- `7.366 deg` mean right-hand rotation error.

Target construction improves hand reprojection by roughly `54--58%` but body
reprojection by only about `4%`. These are valid optimization targets for
distilling that bundle-adjustment procedure, but they are not verified clean
3D targets.

The G4 evaluator and training checkpoint selector then report mean joint
rotation error in degrees. G6 reports regional translation-centered vertex
error in millimetres. Kinematic amplification, wrist attachment, and monocular
depth ambiguity make these objectives non-equivalent.

T5 is the clearest falsification: it dramatically improves the same 2D-derived
proxy and still worsens Lane-L 3D geometry.

**Required repair:**

- use a target signal independent of the observation channels presented to the
  model: multi-view triangulation, mocap/SMPL-X, or at minimum a held-out camera
  view not used in model input;
- require decoded upper-body/left/right vertex metrics on the external
  validation set;
- choose checkpoints using the same equal-region spatial score used for release;
- treat single-view bundle adjustment only as an auxiliary consistency loss;
- add palm, fingertip, wrist-attachment, and bone/kinematic constraints with
  nonzero weights only when their coordinate-valid targets exist; and
- never use lower reprojection error alone as a fallback acceptance condition.

### RC4 — The current success can bypass sequence reasoning

**Severity:** high; the central temporal hypothesis remains unproven.

The v6 model adds a zero-initialized dense `reprojection_skip` mapping all 102
per-frame 2D residual components directly to 153 rotation-residual components.
This path is applied independently at every frame after the transformer. It can
learn the same inverse relation used by the 2D target builder without using
past or future context.

No executed accepted experiment requires the full model to beat:

- the same network with temporal attention disabled;
- the dense reprojection skip alone;
- a per-frame MLP with the same parameter budget; or
- a short deterministic observation-only optimizer under the final spatial
  metric.

In addition, the T2 configs set joint, fingertip, palm, and observation losses
to zero. The cached torso, wrist-local 3D, palm, and 3D keypoint channels are
unavailable in both relevant caches. The executed T2 is therefore principally
a rotation-plus-2D refiner, not the complete geometric observation system
described by the original proposal.

**Required repair:** make temporal causality a gate, not an architectural
assumption.

- Train a framewise residual baseline and a temporal model with identical
  inputs, target, loss, and parameter budget.
- Remove the dense global reprojection skip in the first Phase 2R model, or
  restrict it to a small per-joint proposal that the temporal model must accept.
- Evaluate center-frame performance while masking future/past context to measure
  the marginal value of each direction.
- Report improvement by burst length and by distance to the nearest clean
  observation.
- Progress only if temporal context adds spatial improvement beyond the best
  framewise model, especially on the predeclared hard subset.

### RC5 — The refiner is not trained to abstain when A1 is already good

**Severity:** high; directly causes clean and hard-subset regressions.

The direct T2 safety path catches NaN/Inf, uncertainty-range failure, and
corrections above 25/35 degrees. It does not estimate whether a valid bounded
correction improves geometry. Consequently direct T2 reports `0%` fallback even
though the right hand is significantly worse.

T5 adds an observation-only check, but that check asks whether reprojection
improves. It is unable to reject corrections that improve a 2D track while
worsening 3D shape.

The actual Lane output changes are small enough to pass all angle checks:

| Region | Mean geodesic change from A1 |
|---|---:|
| upper body | `0.218 deg` |
| left hand | `2.594 deg` |
| right hand | `0.884 deg` |

The right-hand regression shows that even a sub-degree mean intervention can be
harmful when the initializer is already strong.

**Required repair:** separate correction from selection.

1. A **benefit head** predicts the expected spatial improvement of applying a
   candidate correction versus identity, by body/left/right group and frame.
2. A **correction head** proposes the SO(3) residual.
3. A group is changed only when a lower confidence bound on expected benefit is
   positive and the correction passes kinematic checks.
4. Identity selection is an ordinary model decision; numerical fallback is
   counted separately.

Train the benefit head on paired initializer/candidate/target spatial deltas.
Calibrate it on an exact-A1R held-out split. Optimize precision of beneficial
interventions and tail risk, not correction coverage.

### RC6 — Handedness and activity policies are mismatched

**Severity:** high for the right-hand failure.

Lane-L evaluates the left hand on 42 two-handed signs and the right hand on all
57 signs. The old fitter uses sign class to change which hands are fitted. The
How2Sign cache supplies complete bilateral targets for every accepted clip and
does not expose the Lane fitter's class policy to the refiner.

The right-hand A1 baseline (`12.1129 mm`) is already better than the left-hand
baseline (`12.8249 mm`), leaving less safe correction margin. Direct T2 worsens
the right hand in both observed class populations:

- class `0`: `+0.2528 mm`, or `1.84%` worse; and
- class `~0`: `+0.1399 mm`, or `1.20%` worse.

For class `0`, upper body also worsens by `0.1378 mm`, while it improves on the
two-handed population. A single correction policy is crossing distinct
activity/fitting regimes.

**Required repair:**

- add observation-derived left/right activity state and its confidence;
- stratify sampling and metrics by one-hand/two-hand regime and dominant side;
- use separate group-specific benefit and residual heads after a shared torso
  encoder;
- use mirror augmentation with verified SMPL-X joint remapping;
- calibrate left and right independently; and
- require each activity regime to pass clean and hard-subset safety.

Ground-truth gloss or benchmark sign class must not be required at inference.

### RC7 — U1 v7 establishes ranking, not uncertainty-aware improvement

**Severity:** high for the claimed Phase 2 method; formal G5 remains blocked.

U1 v7 has good proxy-domain calibration statistics: Spearman `0.8171`, overall
AUC `0.8023`, left/right AUC `0.7537/0.7703`, monotonic risk, and better NLL.
However its config sets `uncertainty_feedback: false`. In
`WholeSequenceRefiner.forward`, this means learned variance is not passed to
`effective_reliability`; the deterministic attention path continues to use U0.

The U1 checkpoint also receives an uncertainty-only warm-up followed by joint
fine-tuning. Its lower reconstruction error versus the earlier U0 checkpoint
can therefore come from additional deterministic fine-tuning. It does not show
that using uncertainty improved a decision.

There is a second split issue: validation and calibration are separated by
`source_group`, which is a video identifier. The cache does not demonstrate an
explicit signer-disjoint calibration split.

**Required repair:** evaluate uncertainty as an intervention:

- freeze one deterministic candidate generator;
- compare the identical candidates under U0 selection, U1 selection, and an
  oracle benefit selector;
- prevent additional backbone training from confounding the comparison;
- use learned uncertainty for groupwise abstention or correction scaling only
  after calibration;
- require U1 to improve spatial risk-coverage and hard-subset TR-V2V over U0 at
  matched correction coverage; and
- record signer IDs and enforce signer-disjoint calibration.

If U1 predicts error but cannot improve selective decisions, retain it as a
diagnostic only and do not call the refiner uncertainty-aware.

### RC8 — G2 target-quality checks are too weak for a spatial claim

**Severity:** medium to high.

The automatic cache quality check primarily rejects non-finite poses and large
axis-angle outliers. The 100-sequence delegated visual audit reported zero
catastrophic failures, but it samples less than one percent of accepted clips
and cannot measure monocular depth correctness. The target builder's acceptance
criterion is again based on the same 2D tracks.

**Required repair:** add quantitative target checks before retraining:

- held-out-view reprojection where available;
- temporal bone-length and wrist-attachment stability;
- left/right consistency and mirror tests;
- manual review stratified by high target correction, occlusion, handedness,
  and low confidence rather than uniform random sampling; and
- a provider-quality score carried into the training mask instead of treating
  every accepted target joint as equally valid.

---

## 7. Phase 2R scientific hypothesis

> Given observations from one deployable frozen initializer distribution,
> normalized physical-time features, and independent spatial supervision, a
> bidirectional sequence model can identify temporary initializer failures and
> selectively apply corrections that reduce decoded regional vertex error more
> than a matched framewise model, while abstaining on already-correct frames.

This hypothesis has three separable claims:

1. **recoverability:** temporal context contains information beyond the current
   frame;
2. **transfer:** the learned correction applies to the deployed initializer and
   camera/observation contract; and
3. **selection:** uncertainty or benefit prediction can decide when correction
   is safer than identity.

Phase 2R must test the claims separately. A pass on synthetic recoverability
cannot substitute for transfer, and an error-ranking AUC cannot substitute for
selective spatial improvement.

---

## 8. Revised system design

### 8.1 Deployable initializer contract

Freeze `A1R` with the following explicit interface:

```text
ordered RGB frames + timestamps
  -> frozen Sapiens/WiLoR/SMPLer-X observations
  -> observation-derived hand-activity state
  -> dataset-neutral SMPL-X fitting
  -> deterministic per-frame fallback policy
  -> result PKL + per-frame component/fallback provenance
```

The provider may not read a benchmark sign name, benchmark segment JSON, GT
mesh, or evaluation-only hand-class label. It must fail nonzero on any missing
stage and publish only complete atomic clips.

Before external extraction, run it over Lane inputs and compare with the locked
A1 view:

- identical scheduled coverage;
- explicit accounting for the old 43 fallback frames;
- parameter and mesh delta distributions by component;
- regional metrics versus A0; and
- deterministic rerun hashes or bounded numerical tolerances.

If it is not functionally equivalent, name it A1R, rerun G1, and never claim it
is the old exact A1.

### 8.2 Canonical cache v4

Use a new append-only cache root. Do not reinterpret schema-v3 arrays in place.

Required additions:

```text
cache/phase2r/v4/
  manifest.json
  providers/<provider_id>.json
  splits/{train,val,calibration,test}.json
  clips/<source>/<clip_id>.npz
  audits/
```

Each clip must include:

- raw and calibrated confidence by provider;
- explicit detector/track/in-frame/copied/interpolated masks;
- per-frame initializer component and fallback reason;
- camera model, intrinsics, crop transform, and normalized camera rays;
- pose and keypoint velocities in physical units per second;
- target provider, target-view IDs, and target quality masks;
- signer/source identities used by split auditing; and
- immutable hashes for every frame, expert, and target artifact.

The loader must reject any field whose declared units or semantics differ from
the selected model contract.

### 8.3 Independent spatial targets

Preferred target tiers are:

1. synchronized multi-view/mocap SMPL-X with held-out signers;
2. multi-view triangulated upper body and hands with a held-out camera;
3. a separately trained 3D provider whose input/checkpoint is independent of
   A1R and whose errors are audited; or
4. single-view 2D bundle adjustment only as auxiliary supervision, never as the
   formal G4 target.

Every target must be paired with an A1R output on the same exact frame. The
formal validation metric is decoded, equal-region spatial error. Rotation,
reprojection, velocity, and acceleration remain diagnostics.

### 8.4 Selective candidate generator

The first accepted model should be smaller and more falsifiable than v6:

- shared per-joint embedding with provider-invariant features;
- bidirectional temporal attention using continuous relative time;
- separate upper-body, left-hand, and right-hand candidate heads;
- no dense all-joint framewise reprojection skip;
- zero-initialized bounded SO(3) residuals;
- an explicit identity candidate for every group/frame; and
- a benefit/abstention head trained on spatial improvement versus identity.

Use reliability to mask or bias observations, but do not allow an uncalibrated
head to alter reconstruction during warm-up.

### 8.5 Safety contract

Keep numerical and rotation-bound fallback, but add a learned selective layer.
Report three different quantities:

- **identity selection:** the model intentionally makes no change;
- **safety fallback:** a proposed output is rejected for numerical or geometric
  invalidity; and
- **correction coverage:** the fraction of group-frames actually changed.

The original `<1%` limit should apply to unexpected safety fallback, not to
intentional identity selection. A good selective refiner may change only a
small hard subset.

---

## 9. Required experiments

### R0 — Provider portability and fail-fast execution

Build A1R and run a one-clip external preflight through every expert and fitter.
Require complete PKLs, meshes, per-stage exit codes, and hashes. Then run at
least 100 source-disjoint clips before authorizing full extraction.

**GO:** no benchmark-name dependency, no masked shell failure, complete atomic
coverage, and frozen provider provenance.

### R1 — Cache semantic equivalence

Compare train/validation/deployment distributions for every token channel, not
only reprojection magnitude. Include units, calibration curves, missingness,
camera type, cadence, and activity regime.

**GO:** no aliased feature meaning; all physical quantities share units;
calibrated confidence ECE is below `0.05` per provider/group; cadence support
overlaps or is explicitly resampled.

### R2 — Independent-target spatial preflight

Before full training, optimize or train on a small exact-A1R paired set and
evaluate decoded spatial error on held-out clips and views.

**GO:** target quality passes the stratified audit, and a simple supervised
oracle correction demonstrably improves all three regions. If even the oracle
target delta cannot improve decoded metrics, stop and repair coordinates or
targets.

### R3 — Framewise versus temporal causality

Train with the same data and losses:

- identity A1R;
- per-frame MLP;
- per-frame reprojection proposal;
- causal temporal model;
- bidirectional temporal model; and
- bidirectional model with shuffled temporal order.

**GO:** bidirectional temporal context beats the best framewise baseline by at
least `1%` equal-region spatial error and by at least `3%` on the hard subset;
shuffling time removes a material part of the gain.

### R4 — Selective deterministic U0

Train the candidate and benefit heads. Freeze thresholds on external validation.

**GO:**

- equal-region spatial gain at least `3%` over A1R;
- no region regresses more than `1%`;
- hard-subset gain at least `8%`;
- clean low-error regression below `1%` per region;
- beneficial-intervention precision at least `80%` per group;
- unexpected safety fallback below `1%`; and
- all thresholds pass across three fixed seeds.

### R5 — Interventional uncertainty U1

Freeze the U0 candidate generator. Train/calibrate uncertainty on a signer- and
source-disjoint calibration split. Compare matched candidate sets.

**GO:** U1 selection beats U0 selection in spatial risk-coverage and hard-subset
TR-V2V at matched correction coverage; all original ranking/NLL conditions pass;
and no clean region regresses more than `1%`.

### R6 — Existing Lane-L regression check

Run Lane-L only after R0--R5 configurations are frozen. Because Lane-L has
already influenced this recovery design, label it **diagnostic replication**,
not untouched confirmation.

Use it to detect catastrophic incompatibility and compare with the historical
G6 result. Do not tune thresholds or architecture from it.

### R7 — New confirmatory progression gate

Acquire or reserve a target-domain set not used in Phase 2 or Phase 2R design.
Run the frozen three seeds once.

**GO to Phase 3 only when:** R4 and R5 pass on external validation, Lane-L does
not show a material safety failure, and the untouched confirmatory set passes
the full spatial, hard-subset, coverage, fallback, and three-seed criteria.

---

## 10. Implementation work packages

### WP0 — Preserve the negative result

- freeze hashes for v6, v7, T5, current caches, and G6 artifacts;
- mark direct T2 and T5 as non-release methods; and
- keep the original Phase 2 report unchanged as the historical record.

### WP1 — Portable A1R

- create a provider wrapper with explicit frame schedule, timestamps, and
  hand-activity input;
- remove benchmark-name/segment lookup from the deployable path;
- make every shell stage fail-fast and validate output schema before continuing;
- implement the actual per-frame fallback policy; and
- record component identity for every result.

The old frozen scripts remain unchanged for reproduction. New portable code is
versioned and parity-tested beside them.

### WP2 — Cache v4 and adapter conformance

- implement new fields without overloading schema-v3 indices;
- provide H2S/A1R and Lane/A1R adapters with identical field semantics;
- add physical-time derivatives and camera rays;
- add signer-disjoint split validation; and
- add distribution reports with fail-closed thresholds.

### WP3 — Target provider and audit

- ingest independent spatial or held-out-view targets;
- compute target-valid masks per joint and region;
- create stratified audit queues; and
- add decoded external spatial evaluation used during checkpoint selection.

### WP4 — Baselines and candidate model

- implement matched framewise, causal, bidirectional, and shuffled-time models;
- remove or isolate the dense reprojection shortcut;
- add group-specific candidate and benefit heads; and
- train U0 only after R0--R2 pass.

### WP5 — Selective safety and uncertainty

- calibrate benefit thresholds per group/activity regime;
- separate identity decisions from unexpected fallback;
- train U1 on frozen candidates;
- add matched-coverage risk-coverage evaluation; and
- prohibit U1 release unless its intervention improves spatial error.

### WP6 — Frozen evaluation

- run three external-validation seeds;
- freeze all settings;
- run Lane-L diagnostic replication once; and
- run the new untouched confirmatory set once for the Phase 3 progression
  decision.

---

## 11. Experiments that should not be repeated

Do not spend additional compute on the following branches:

- two more v6 seeds after the decisive seed-42 G6 failure;
- more T5 steps or Lane-tuned reprojection thresholds;
- a larger transformer trained on the same H32/2D-derived cache;
- U1 feedback tuning on Lane-L;
- relabeling H32 outputs as exact A1;
- treating the old 57-sign class lookup as a dataset-neutral inference input;
- using a lower-jerk result as evidence of spatial improvement; or
- proceeding to diffusion in the hope that sampling repairs a biased residual
  domain.

Diffusion would model uncertainty around the wrong correction distribution and
could make selection harder, not safer.

---

## 12. Priority and stop conditions

| Priority | Action | Stop condition |
|---:|---|---|
| P0 | make A1R portable and fail-fast | external one-clip full pipeline cannot emit valid PKLs without benchmark metadata |
| P0 | define invariant cache v4 semantics and cadence | any token field retains provider-specific meaning under one name |
| P0 | obtain independent spatial/held-out-view targets | only the same single-view 2D tracks are available for both input and target |
| P1 | prove temporal gain over framewise | bidirectional model does not materially beat the matched framewise baseline |
| P1 | train selective deterministic U0 | benefit precision or clean safety fails on external exact-A1R validation |
| P2 | add interventional U1 | uncertainty ranks error but does not improve matched-coverage spatial decisions |
| P2 | run confirmatory evaluation | no untouched target-domain evaluation set can be reserved |

Any P0 stop condition keeps Phase 3 blocked and preserves A1/A1R as the final
geometric method.

---

## 13. Required reporting table for the next decision

Every Phase 2R result must publish one table with at least:

| Category | Required fields |
|---|---|
| provider | A1R version, component hashes, fallback policy, per-component coverage |
| data | datasets, signers, sources, cadence, cameras, target providers, overlap audit |
| target quality | held-out-view/spatial checks, stratified manual audit, invalid masks |
| baselines | A1R identity, framewise MLP, causal, bidirectional, shuffled time |
| spatial | upper body, left hand, right hand, equal-region gain, CIs, worst decile |
| subsets | failure/occlusion/activity regimes, clean frames, correction coverage |
| selection | beneficial precision/recall, identity rate, unexpected fallback rate |
| uncertainty | calibration, AUC, NLL, matched-coverage risk, U1-minus-U0 spatial gain |
| reproducibility | three seeds, regional SD, checkpoint/config/cache hashes |

Rotation and reprojection metrics may be included, but they cannot replace
decoded spatial measurements.

---

## 14. Final recommendation

Phase 2 should be recorded as a valuable negative result:

- the sequence architecture is capable of synthetic temporal recovery;
- the H32 proxy target is learnable;
- learned uncertainty can rank proxy error;
- but neither direct refinement nor observation-only T5 safely improves the
  deployed strong initializer.

The dominant root cause is broader than "initializer mismatch." The exact
training initializer cannot currently be run outside the benchmark contract,
and the supposedly common observation cache changes semantics, camera model,
cadence, and reliability formula between training and deployment. The proxy
target then rewards the same 2D signal exposed through a framewise shortcut,
while release is judged in 3D mesh space. Finally, neither T2, T5, nor U1 makes
a calibrated spatial-benefit decision before changing a strong hand estimate.

The correct new branch is therefore:

> **portable A1R + invariant physical-time observation schema + independent
> spatial targets + causal temporal ablation + selective benefit-gated SO(3)
> correction + interventional uncertainty evaluation.**

Only this branch can answer the original scientific question cleanly. Until it
passes an untouched confirmatory gate, Phase 3 remains NO-GO.

---

## 15. Implementation status (2026-08-03)

The first Phase 2R implementation slice now exists under `phase2_refiner/`.
It is opt-in and preserves the legacy Phase 2 execution path:

- cache schema v4 stores separate detector presence, track validity,
  in-frame, copied/interpolated, raw/calibrated confidence, initializer and
  fallback provenance, target quality, camera/crop, and hand-activity fields;
- strict Phase 2R cache validation rejects legacy or semantically contradictory
  caches before training or inference;
- feature derivatives and rotation motion losses can be normalized by physical
  elapsed seconds, with `0.04 s` only as a reference scale;
- A1R fitting metadata is inferred per clip from observed hand coverage and
  motion, then passed through isolated contract files instead of the frozen
  German-sign lookup table;
- the A1R expert shell stops on the first failing command, the fitter launcher
  propagates nonzero exit status, and incomplete frame coverage is rejected;
- temporal attention can be disabled without changing the remaining model,
  providing the matched framewise baseline required by G4/G5;
- an optional regional benefit head learns whether body, left-hand, and
  right-hand candidates beat the initializer, and inference can retain the
  initializer below a calibrated benefit threshold;
- target-quality weighting masks low-quality supervision; uncertainty feedback
  is enabled; and the dense reprojection skip is disabled in the Phase 2R v1
  configuration; and
- a static boundary test rejects any import from `phase2_refiner` into
  `phase3_posterior`.

The executable starting configuration is
`phase2_refiner/configs/phase2r_domain_aligned_v1.yaml`. It intentionally uses
new cache and output roots. The current automated Phase 2 suite passes 66/66
tests. This is an implementation milestone, not a gate result: full A1R cache
materialization, independent target generation, training, calibration, and the
three-seed decoded-mesh evaluation remain required before Phase 2 can be
declared GO.
