# Phase 2 Build Plan: Uncertainty-Aware Whole-Sequence Refiner

- **Project:** DexAvatar / SignPosterior4D de-risking program
- **Phase:** Phase 2 only
- **Method name:** `UAWSR` (Uncertainty-Aware Whole-Sequence Refiner)
- **Date:** 22 July 2026
- **Primary objective:** determine whether a deterministic, bidirectional sequence model can improve a strong frozen DexAvatar initializer by correcting temporary body, wrist, and hand failures without oversmoothing valid signing motion.

---

## 1. Executive decision

Build Phase 2 as a **standalone deterministic residual refiner over cached frozen observations**. It should consume a complete isolated-sign clip, reason jointly over upper body, both wrists, and both hands, estimate how much each observation should be trusted, and emit a complete SMPL-X parameter and mesh sequence.

Do **not** add diffusion, multi-hypothesis sampling, phonology, learned contact, or phase prediction in this phase. Those mechanisms belong after the deterministic refiner proves that useful temporal signal exists beyond the stronger framewise experts.

The required Phase 2 chain is:

```text
RGB clip
  -> frozen SMPLer-X/NLF, WiLoR, and Sapiens observations
  -> versioned observation cache with missing-data masks
  -> coordinate and rotation canonicalization
  -> deterministic bidirectional joint body-wrist-hand refiner
  -> uncertainty-gated residual composition on SO(3)
  -> differentiable SMPL-X decoding
  -> optional short observation-only sequence refinement
  -> result PKLs + meshes
  -> locked author-style evaluation manifest and diagnostic tests
```

The phase is a success only if the improvement is spatial, statistically supported, robust to observation failure, and achieved on the exact same frame population as its initializer. Lower velocity or jerk alone is not success.

---

## 2. Repository facts that constrain the design

### 2.1 Published reference and local measurements are not interchangeable

The published DexAvatar reference is:

| Method | UBody(-F) | LHand | RHand |
|---|---:|---:|---:|
| DexAvatar (published) | 30.13 | 13.53 | 13.08 |

Keep this as an external reference. Do not claim parity or improvement from a local aggregate until the alignment and frame protocol have been reconciled.

The repository audit in `docs/dexavatar_diagnosis/E0_PHASE_REPORT.md` found that the author-style local path produces:

- 57 signs;
- 1,493 ordinal GT/prediction pairs, not the paper-stated 2,872 frames;
- 1,163 LHand pairs over 42 signs because class-0 signs skip the left hand;
- 1,493 RHand pairs over all 57 signs; and
- exact author masks of 7,279 / 778 / 778 vertices for UBody(-F), LHand, and RHand.

The immutable local comparison manifest is `probes/results/phase0/frame_manifest.csv`. Phase 2 must use this same population for the initializer and refined result. A method with missing frames fails the coverage gate; it may not be compared by silently truncating both lists with `min(...)`.

The local evaluator independently translation-centers each evaluated region. Report this as **repository-local author-style regional TR-V2V** until equivalence to the intended official protocol is demonstrated.

### 2.2 Useful existing artifacts

The current tree already provides:

- source frames in `data/frames/<sign>/`;
- mesh-only SGNify ground truth in `data/smplx_gt/<sign>/`;
- author evaluation assets in `data/evaluation_from_author/data/data/`;
- the original/HaMeR-compatible reconstruction in `outputs/method_hamer/`;
- stronger WiLoR and NLF experiments in `outputs/method_nlf_wilor/`;
- Sapiens whole-body 2D detections in per-sign `sapiens.pkl` files;
- SMPLer-X or NLF SMPL-X initializer PKLs;
- raw WiLoR output in `wilor/wilor.pkl` and a HaMeR-compatible export in `hamer/hamer.pkl`;
- fitted result PKLs containing 63-D body and two 45-D hand axis-angle poses; and
- a single 48 GB NVIDIA RTX 5880 Ada GPU, sufficient for the proposed deterministic model.

The primary input contract should be the **Phase 1 stronger frozen initializer**, not `method_hamer` by assumption. `method_hamer` remains A0, the historical DexAvatar-compatible reference. Select the Phase 1 initializer only after a common-manifest evaluation confirms its geometry and coverage.

### 2.3 The current temporal prototype must not be extended into the main method

`dexavatar_fitting/smplifyx/fit_temporal_window.py` is useful as a failed/simple-smoothing baseline, but it is not Phase 2 because it:

- optimizes separate per-frame VAE latents rather than learning a sequence correction;
- applies velocity, acceleration, and jerk penalties to body pose only;
- does not temporally refine either hand;
- initializes every latent at zero instead of the fitted sequence;
- directly averages overlapping axis-angle vectors, which is not valid rotation averaging;
- does not construct observation-specific uncertainty;
- inherits frame filtering that removes frames with missing hand detections;
- uses a simplified subset of the original fitting terms;
- writes result dictionaries but does not complete the standard mesh-rendering contract; and
- has produced zero result PKLs and zero meshes under `outputs/output_wilor_temporal/` in the current checkout.

Keep it as baseline `B2: existing temporal-window prototype`. Implement UAWSR in a new module so its data, training, and inference behavior are testable independently.

### 2.4 Current observation handling destroys information needed for uncertainty

The existing `data_parser.py` behavior is unsuitable for the new cache:

- frames without a required hand detection are removed;
- for a two-handed sign, both hands must be detected;
- WiLoR hand keypoint confidence is replaced with `1` after insertion;
- a missing one-handed observation can be copied from the previous frame;
- expert absence and copied observations are not exposed to the loss; and
- the existing “uncertainty-aware” fitting option scales hand terms using Sapiens wrist confidence only.

Phase 2 must retain every scheduled frame and represent absence explicitly. A missing hand is exactly the case the whole-sequence model is meant to solve.

### 2.5 The currently staged sign training set is not a sequence corpus

`data/body_data/sign_v1` contains only 1,449 / 181 / 182 framewise 63-D body poses. The local How2Sign staging contains roughly 100 short clips of about ten frames. The PHOENIX staging currently has only the train split and short extracted groups. These assets are useful for smoke tests and spatial initialization, but are not sufficient evidence for a 32–64-frame joint body-hand refiner.

Acquiring or constructing a source-disjoint sequence corpus is therefore an explicit Phase 2 data gate, not an optional improvement.

---

## 3. Phase 2 scope and non-goals

### 3.1 In scope

- frozen, cached framewise experts;
- all-frame sequence ingestion with explicit missingness;
- unified body, wrist, and two-hand state;
- bidirectional complete-clip or padded-window inference;
- deterministic residual rotation correction;
- fixed heuristic reliability baseline;
- learned heteroscedastic reliability after calibration gates pass;
- realistic estimator-error and burst-occlusion corruption;
- joint/vertex/fingertip/palm and motion-preservation losses;
- valid SMPL-X PKL and mesh export;
- common-manifest TR-V2V, temporal, corruption, and calibration evaluation; and
- automatic fallback to the initializer when the refiner fails a test-time safety check.

### 3.2 Explicitly out of scope

- diffusion or score-based sequence sampling;
- best-of-K or oracle candidate selection;
- gloss, HamNoSys, or claimed phonology conditioning;
- learned sign-phase classes;
- dense contact prediction;
- changing or fine-tuning WiLoR/SMPLer-X/NLF for the main result;
- training on the SGNify evaluation meshes;
- choosing hyperparameters from the 57-sign benchmark;
- claiming that the current local metric reproduces the published `30.13 / 13.53 / 13.08`; and
- modifying `evaluation/evaluate_new_fitting_local.py` to make a result look better.

### 3.3 Scientific hypothesis

> Given the same frozen observations as a strong framewise initializer, a body-wrist-two-hand bidirectional refiner trained on realistic estimator failures will reduce spatial reconstruction error because clear observations before and after a failure constrain the ambiguous interval. Explicit reliability should improve this correction by preventing confident but inconsistent observations from dominating the sequence.

The null hypothesis is that the initializer already contains all recoverable information and temporal modeling only changes smoothness. The experiment plan is designed to accept that null and stop before Phase 3 if necessary.

---

## 4. System contract

### 4.1 Inputs

For each clip and scheduled frame `t`, cache:

1. **Frame identity**
   - `clip_id`, source frame number, path, timestamp, sampling rate;
   - original image width and height;
   - SHA-256 of the frame and each expert checkpoint/config.

2. **Initializer SMPL-X**
   - `global_orient [3]`, `body_pose [63]`;
   - `left_hand_pose [45]`, `right_hand_pose [45]`;
   - `betas [10]`, `transl [3]`, camera intrinsics;
   - source expert name and whether a field was substituted from another expert.

3. **Sapiens observations**
   - 133 2D joints and original confidences;
   - validity and in-frame masks.

4. **WiLoR observations for every detected hand**
   - 2D and 3D joints, MANO articulation, global/crop camera values;
   - handedness, box center/size, detector confidence, and crop truncation;
   - duplicate-candidate information rather than only the first candidate per side.

5. **Derived observations**
   - SMPL-X joints/vertices from the frozen initializer;
   - wrist-relative 3D hand joints in a documented frame;
   - fingertips, MCP centers, palm center, palm normal;
   - 2D reprojection residuals;
   - frame-to-frame rotation and joint-position innovations; and
   - expert-present, copied, interpolated, and missing masks.

RGB or crop features are an optional Phase 2.5 experiment. The first accepted Phase 2 model should prove value using cached geometry, because this isolates sequence inference from a newly trained image encoder.

### 4.2 Outputs

For each input frame, write:

- the standard DexAvatar result PKL fields and shapes;
- `refiner_delta_rotvec` for the 51 refined joints;
- `uncertainty` and `fallback_mask` by body/left-hand/right-hand group;
- a valid 10,475-vertex SMPL-X OBJ;
- per-frame diagnostic JSON; and
- a clip-level inference report containing coverage, runtime, NaNs, safety fallbacks, and cache/model hashes.

The ordinary result PKL must remain consumable by the existing renderer:

| Field | Required shape | Phase 2 behavior |
|---|---:|---|
| `body_pose` | `(1, 63)` | refined for upper-body joints; unchanged for lower-body joints |
| `left_hand_pose` | `(1, 45)` | refined or safely retained from initializer |
| `right_hand_pose` | `(1, 45)` | refined or safely retained from initializer |
| `betas` | `(1, 10)` | one robust shared clip value |
| `global_orient` | `(1, 3)` | frozen in the first model; optional low-frequency correction ablation |
| `transl` | `(1, 3)` | frozen in the first model |
| face fields | existing shapes | copied unchanged |

Do not let a translation update exploit the evaluator's regional centering. Pose improvement must stand on its own.

### 4.3 Cache layout

Use a versioned, append-only layout:

```text
cache/phase2/v1/
  manifest.json
  splits/{train,val,test}.json
  clips/<source>/<clip_id>.npz
  diagnostics/<source>/<clip_id>.json
```

Every cache schema change increments the version. Never overwrite a cache in place. The manifest records source license, source split, signer, language, expert commit/checkpoint hashes, units, coordinates, FPS, and preprocessing command.

---

## 5. Canonical state and coordinate policy

### 5.1 Rotation representation

SMPL-X files use axis-angle, but the network should not regress or average raw axis-angle values.

For each of the 21 body joints and 30 hand joints:

1. convert initializer axis-angle to a rotation matrix;
2. encode the matrix as continuous 6D rotation input;
3. predict a local residual `delta_omega in R^3`;
4. clamp the residual magnitude with a differentiable bound; and
5. compose rotations on the manifold:

$$
R^{out}_{t,j} = \exp(\alpha_{t,j}\,\Delta\omega_{t,j})R^{init}_{t,j}.
$$

`alpha` is an observation/reconstruction trust gate in `[0,1]`. Convert to axis-angle only when exporting or calling SMPL-X.

For overlapping windows, blend rotation matrices using quaternion hemisphere alignment followed by normalized weighted averaging, or geodesic averaging. Never average axis-angle vectors directly.

### 5.2 Refined joint set

The network receives all 51 pose joints but changes only:

- spine1, spine2, spine3;
- neck and both collars;
- both shoulders, elbows, and wrists;
- all 15 joints of each hand.

Head may be input context but is frozen initially. Hips, knees, ankles, and feet are copied from the initializer. This focuses capacity on the three evaluated regions and prevents unrelated lower-body drift.

### 5.3 Coordinate frames

Maintain three explicit frames:

- **camera frame:** 2D projection and expert camera outputs;
- **root/torso frame:** body joint positions and cross-arm context;
- **wrist-local frame:** hand articulation and palm geometry.

Store a homogeneous transform for every conversion and add round-trip tests. WiLoR 3D is not assumed metric-compatible with SMPL-X merely because both use XYZ. Estimate wrist-local scale from MANO bone lengths and use wrist-relative geometry unless an audited camera-frame alignment is available.

The model should see both wrist-local hand geometry and torso-frame palm/wrist features. This allows finger articulation to remain scale stable while arm and palm orientation stay coordinated.

### 5.4 Shared shape

Compute one clip shape from the robust median of frozen initializer betas, reject per-frame outliers with a predeclared threshold, and freeze it. A learned shape correction is out of scope because the benchmark contains one signer and shape can confound pose gains.

---

## 6. Observation uncertainty

### 6.1 Reliability features

For every frame and body/hand joint, construct reliability features from information available without benchmark GT:

- detector confidence and presence;
- 2D keypoint confidence;
- bounding-box scale and fraction outside the image;
- duplicate or conflicting handedness detections;
- WiLoR-to-Sapiens wrist/fingertip 2D disagreement;
- initializer-to-keypoint reprojection residual;
- bone-length inconsistency;
- palm-normal and rotation geodesic innovation from neighbors;
- acceleration relative to a robust local median;
- blur score and crop resolution, if computed deterministically; and
- whether the value is original, copied, interpolated, or missing.

The WiLoR exporter currently does not retain YOLO detection confidence in the raw hand record. Add it to the new cache/exporter before training the learned reliability head. Do not reconstruct it later from an unavailable value.

### 6.2 Two uncertainty modes

Implement uncertainty in two ordered modes.

#### U0: fixed reliability

Use a documented deterministic mapping from the features above to weights. It is a baseline and fail-safe, not a calibrated uncertainty claim. U0 allows the temporal model to be tested before a trustworthy calibration set exists.

#### U1: learned heteroscedastic reliability

A small MLP predicts group-specific log variance for:

- body/arm rotation;
- left-hand articulation;
- right-hand articulation;
- 2D observation; and
- wrist-local 3D observation.

Bound log variance to a numerically safe range and use it in both attention and observation losses. For residual `r`:

$$
L_{NLL} = \frac{\rho(r)}{2\sigma^2} + \frac{1}{2}\log\sigma^2.
$$

Stop gradients from the refiner into a reliability feature when that feature is computed from the refiner's own prediction; otherwise the system can lower its loss by declaring its errors uncertain.

### 6.3 Calibration gate

U1 replaces U0 only if, on a source- and signer-disjoint calibration set:

- uncertainty/error Spearman correlation is at least `0.35` for body and each hand;
- AUROC for detecting the worst 10% errors is at least `0.75` for both hands and `0.70` for body;
- risk decreases monotonically as the most uncertain 10%, 20%, and 30% observations are rejected;
- calibrated NLL improves over detector-confidence-only U0; and
- U1 improves reconstruction on corrupted validation clips without a clean-set regression over 1% in any region.

If any of these conditions fail, ship U0 and report learned uncertainty as a negative result. Do not call U0 calibrated.

---

## 7. Deterministic refiner architecture

### 7.1 Recommended model

Use an alternating spatial-temporal Transformer over `T x 51` joint tokens:

- maximum training window: 64 frames;
- hidden size: 256;
- 6 alternating blocks;
- 8 attention heads;
- MLP ratio: 4;
- relative temporal position encoding;
- learned joint/side/type embeddings;
- explicit validity and uncertainty attention bias; and
- approximately 15–30 million trainable parameters.

Each block contains:

1. spatial attention among the 51 joints within a frame;
2. temporal attention for each joint across all valid frames;
3. cross-part summary tokens for torso, left arm, right arm, left hand, and right hand; and
4. a feed-forward residual update.

This factorization is tractable on the available 48 GB GPU and still allows wrist/forearm, left/right hand, and torso coordination. Use padding masks so clips of 12–48 evaluated frames can be processed as complete sequences.

### 7.2 Inputs per joint token

- initializer rotation in 6D;
- initializer torso- or wrist-relative 3D joint position;
- matching 2D position normalized by image size;
- expert 3D joint when available;
- finite-difference velocity and acceleration;
- observation confidence, learned/fixed log variance;
- visibility, in-frame, and missing masks;
- expert identity and handedness; and
- group context such as palm normal, crop size, and wrist disagreement.

Never encode a missing value as zero without a separate validity mask.

### 7.3 Outputs

The model predicts:

- 3D residual rotation vectors for the selected joints;
- a residual-application gate `alpha` for each joint/frame;
- corrected torso/wrist/hand joint positions as auxiliary outputs; and
- U1 log variances when learned uncertainty is enabled.

Initialize the last residual layer to zero, making the initial network an identity function. This sharply reduces the risk that early training destroys a strong initializer.

### 7.4 Why residual refinement is preferred here

Absolute sequence regression would need to relearn valid poses, identity, camera conventions, and expert strengths. Residual composition makes the hypothesis narrower: correct the frames and joints that are inconsistent with the complete clip while retaining clear observations.

---

## 8. Training data strategy

### 8.1 Data tiers

Use three tiers, each with a distinct purpose.

| Tier | Purpose | Suitable sources | Required property |
|---|---|---|---|
| A: clean motion | learn body-hand temporal structure | licensed SignAvatars/other SMPL-X sign sequences; high-quality whole-body sequences | complete ordered SMPL-X with provenance |
| B: high-quality hands | learn articulation and interacting-hand dynamics | InterHand2.6M, ARCTIC, WHIM or other licensed MANO data | reliable hand geometry and temporal identity |
| C: real expert residuals | learn the actual correction problem | videos with GT/pseudo-clean targets after running the exact frozen experts | paired observation/target sequences |

Generic hand-object motion may pretrain hand dynamics but must be downweighted during sign adaptation. It does not replace sign sequence data.

### 8.2 Data readiness gate

Do not begin the main training run until all of the following are true:

- at least 10,000 non-overlapping or source-distinct sign clips are available, or at least 250,000 valid sign frames retain clip boundaries;
- at least 80% of selected training clips contain 16 or more consecutive frames;
- body and both 45-D hand pose fields exist for at least 70% of training frames, with missingness retained for the rest;
- train/validation/test are source-, video-, and signer-disjoint where signer IDs exist;
- no SGNify evaluation frame or derived target appears in training or validation;
- data and checkpoint licenses are recorded; and
- a manual audit of 100 randomly sampled sequences finds fewer than 10% catastrophic pseudo-target failures.

If this gate fails, restrict work to synthetic-corruption feasibility experiments and do not interpret the result as a trained whole-sign refiner.

### 8.3 Target construction

Use the strongest target available for each tier:

- true SMPL-X/MANO parameters where provided;
- multi-view or mocap-derived parameters where licensed;
- otherwise, a frozen offline teacher refined with multi-frame evidence and quality filtering.

Do not train a model to map a framewise estimator back to an identical copy of itself. Such pairs teach identity and cannot establish correction ability. When pseudo targets come from the same estimator family, require an independent refinement signal such as multi-view, stronger hand annotations, or consensus with a second expert.

### 8.4 Exact-expert cache

Run the same frozen expert versions used at inference over Tier C training videos. Cache their real failures. Random Gaussian noise is supplementary; it is not a substitute for estimator residuals.

### 8.5 Corruption curriculum

Apply corruptions to observations, never to clean targets:

- mask one finger chain for 2–8 frames;
- mask a complete hand for 4, 8, and 16 frames;
- mask both hands briefly during interaction;
- introduce wrist-rotation errors of 10–45 degrees;
- insert empirically sampled WiLoR and body-estimator residuals;
- swap a hand hypothesis or flip handedness with low probability;
- perturb palm orientation independently of articulation;
- drop 2D confidences or corrupt a crop near the image boundary;
- introduce burst rather than only independent frame noise; and
- leave 25–35% of batches completely clean.

Clean batches are necessary to teach the identity behavior expected when the initializer is already correct.

---

## 9. Losses

Use a balanced loss that measures both pose fidelity and semantic hand geometry:

$$
L =
\lambda_R L_{rot}
+ \lambda_J L_{joint}
+ \lambda_V L_{region-vertex}
+ \lambda_F L_{fingertip}
+ \lambda_P L_{palm}
+ \lambda_O L_{obs-NLL}
+ \lambda_M L_{motion}
+ \lambda_A L_{anchor}
+ \lambda_B L_{biomech}.
$$

### 9.1 Rotation and geometry

- `L_rot`: geodesic rotation loss on selected body and hand joints;
- `L_joint`: torso-relative body and wrist-local hand joint error;
- `L_region-vertex`: balanced UBody(-F), LHand, and RHand vertex loss through SMPL-X;
- `L_fingertip`: extra weight on ten fingertips; and
- `L_palm`: palm-normal geodesic error plus wrist-to-MCP orientation.

Normalize region losses independently so 7,279 upper-body vertices do not overwhelm each 778-vertex hand.

### 9.2 Observation likelihood

`L_obs-NLL` compares the refined sequence with valid cached 2D/3D observations using U0 or U1 reliability. Missing observations contribute no direct data term but remain eligible for sequence inference.

### 9.3 Motion preservation

Do not minimize velocity toward zero. Match the clean target's velocity and acceleration:

$$
L_{motion} =
\|\Delta J^{out}-\Delta J^{target}\|_1
+ 0.5\|\Delta^2 J^{out}-\Delta^2 J^{target}\|_1.
$$

Upweight transition frames using target motion magnitude during training. This penalizes jitter and oversmoothing symmetrically.

### 9.4 Reliable-observation anchor

For low-uncertainty observations, penalize unnecessary corrections. The anchor weight decreases smoothly with uncertainty. This prevents the model from changing clear WiLoR hands merely to satisfy a motion prior.

### 9.5 Biomechanical safety

Use soft joint-limit and mesh-validity terms only as safety regularizers. They must not dominate the hand target loss. Hard clamping occurs only in the final safety layer and must be logged.

---

## 10. Training curriculum

### 10.1 Starting optimization configuration

Use this as the first reproducible configuration, then change one factor at a time on external validation:

| Item | Starting value |
|---|---|
| optimizer | AdamW |
| learning rate | `2e-4` for UAWSR; `1e-4` for U1 reliability |
| weight decay | `0.05`, excluding norm and bias parameters |
| schedule | 5% linear warm-up, then cosine decay |
| precision | BF16 when supported; FP32 for SMPL-X geometry losses if needed |
| physical batch | 8 windows of up to 64 frames |
| gradient accumulation | 4, giving effective batch 32 |
| gradient clipping | global norm `1.0` |
| dropout | `0.1` |
| training length | maximum 100,000 updates with early stopping |
| model averaging | EMA `0.999`, evaluated alongside raw weights |
| seeds | three fixed seeds for accepted experiments |

Bucket clips by length to reduce padding. Sample at the clip level, then balance sources and one-/two-handed content; never sample uniformly from frames, which would let long videos dominate. Save `last`, `best`, and periodic recovery checkpoints with optimizer, scheduler, scaler, RNG, cache hash, and resolved config.

Choose `best` using one predeclared external-validation score:

$$
S_{val} = \frac{1}{3}\sum_{r \in \{U,L,R\}}
\frac{E_r^{model}}{E_r^{initializer}}
+ 0.5\sum_r \max\left(0,
\frac{E_r^{model}}{E_r^{initializer}}-1.01\right).
$$

This treats the three regions equally and penalizes a checkpoint that sacrifices one region. Do not select checkpoints from SGNify Lane L.

### Stage T0: contracts and identity

- overfit 4–8 short clips;
- verify zero-initialized output exactly reproduces the initializer before training;
- verify loss reaches near-zero when input equals target;
- verify padding does not change valid-frame outputs; and
- verify SO(3) conversion and export round-trip.

**Go:** exported meshes reproduce input meshes within `0.01 mm` mean vertex error with residuals disabled.

**No-go:** stop and fix coordinate, shape, joint-order, or renderer mismatch.

### Stage T1: synthetic corruption recovery

- train on clean motion sequences with burst corruption;
- use U0 only;
- no RGB features and no final optimization.

**Go:** recover at least 30% of injected vertex error for 4/8/16-frame hand dropout, while clean inputs regress by less than 2% in every region.

**No-go:** simplify the model, inspect target quality and corruption scale, and do not start real-residual training.

### Stage T2: real residual learning

- run exact frozen experts on Tier C videos;
- train observation-to-clean residual correction;
- mix 50% real residual, 25% synthetic burst, and 25% clean batches initially;
- tune the mixture only on external validation.

**Go:** improve the external clean validation weighted regional error by at least 3%, with no region worse by more than 1%, and improve the predefined failure subset by at least 8%.

**No-go:** determine whether failures come from coordinate conversion, target noise, or absence of recoverable future context. Do not add learned uncertainty yet.

### Stage T3: sign adaptation

- fine-tune on source-disjoint sign sequences;
- retain 20–30% generic/high-quality hand batches to avoid catastrophic hand drift;
- use a lower learning rate and early stopping on external validation.

**Go:** sign validation improves over T2 and transition error does not regress.

**No-go:** keep T2 as Phase 2 and report that the available sign pseudo targets do not help.

### Stage T4: learned uncertainty

- train the reliability head on a disjoint calibration partition;
- freeze the refiner first, then jointly fine-tune briefly if calibration remains valid;
- temperature/variance-scale calibration is fitted on calibration only.

**Go:** pass every calibration criterion in Section 6.3 and beat U0 on the failure subset.

**No-go:** ship U0.

### Stage T5: optional short sequence optimization

Starting from direct UAWSR output, run at most 10–20 Adam steps over the complete clip using valid observation likelihood, reliable-output anchor, and soft biomechanics. Shape, global translation, and face remain frozen.

**Go:** common-manifest validation improves by at least 0.2 mm in one region with no region regression above 0.1 mm and runtime remains acceptable.

**No-go:** use direct UAWSR output as the final Phase 2 method.

---

## 11. Inference algorithm

For one isolated sign:

1. enumerate all scheduled frames from the clip, not only frames with successful hands;
2. load or compute the frozen observation cache;
3. validate checkpoint/config hashes and coordinate metadata;
4. select the Phase 1 initializer and construct the shared clip shape;
5. convert all rotations to matrices/6D and derive joint, palm, and temporal features;
6. compute U0 reliability or calibrated U1 variance;
7. pad the complete clip to 64 frames and run bidirectional inference once;
8. for clips longer than 64, use 50% overlap and geodesic overlap blending;
9. compose bounded residual rotations with the initializer;
10. run optional T5 refinement only if enabled by a frozen config;
11. apply test-time safety checks and groupwise fallback;
12. decode SMPL-X, save standard PKLs, render meshes, and write diagnostics; and
13. assert that output frame IDs exactly match the locked input manifest.

For the audited benchmark, clip lengths are 12–48 paired frames with median 25, so a 64-frame model can process every sign as a whole sequence.

### 11.1 Safety and fallback

Fall back to the frozen initializer for a body or hand group if any of the following occurs:

- NaN/Inf in parameters, vertices, or uncertainty;
- residual angle exceeds the frozen validation bound;
- projected reliable-keypoint error worsens beyond a predeclared tolerance;
- bone length or palm geometry becomes invalid;
- the output mesh changes topology; or
- uncertainty is outside the calibrated training range.

Fallback is groupwise where possible, so a failed left-hand correction does not discard a useful right-hand or body correction. Every fallback is counted and reported. More than 1% group-frame fallback on clean validation is a no-go for release.

---

## 12. Evaluation design

### 12.1 Evaluation lanes

Maintain two clearly named lanes.

#### Lane L: locked local development

- use `probes/results/phase0/frame_manifest.csv`;
- require 1,493 UBody/RHand pairs and 1,163 LHand pairs;
- use exact author masks and class-0 behavior;
- report repository-local author-style regional TR-V2V;
- use the same prediction frame ID for initializer and refiner; and
- never use `min(GT, prediction)` as a missing-output policy.

#### Lane O: official/published comparison

Open only after the 2,872-versus-1,493 discrepancy and alignment definition are resolved. The `30.13 / 13.53 / 13.08` line belongs here. Until then, do not mix L and O values in one ranking table.

### 12.2 Required baselines

| ID | Configuration | Question |
|---|---|---|
| A0 | `outputs/method_hamer` | historical local DexAvatar-compatible reference |
| A1 | selected frozen Phase 1 stronger initializer | how much comes from expert replacement/fusion? |
| B1 | initializer + Gaussian/Savitzky/velocity smoothing | does simple smoothing explain the gain? |
| B2 | existing temporal-window prototype, if made runnable without changing its method | does parameter optimization alone help? |
| P2.0 | UAWSR, no uncertainty features | does whole-sequence residual inference help? |
| P2.1 | UAWSR + U0 fixed reliability | does explicit missing/reliability handling help? |
| P2.2 | UAWSR + calibrated U1 | does learned uncertainty add value? |
| P2.3 | P2.2 + optional T5 optimization | does final observation refinement add value? |

### 12.3 Required ablations

- causal/past-only versus bidirectional context;
- 8, 16, 32, and 64 frames;
- body-only, hands-only, and joint body-hand refinement;
- no wrist/palm features;
- no burst corruptions;
- zero-filled missing observations versus explicit missing masks;
- U0 versus detector confidence only versus U1;
- target motion matching versus zero-velocity smoothing;
- initializer-only versus residual correction magnitude bands;
- full sequence versus sliding-window blending; and
- direct output versus T5 optimization.

### 12.4 Metrics

Primary:

- UBody(-F), LHand, and RHand repository-local author-style regional TR-V2V;
- per-sign paired difference;
- sign-clustered bootstrap 95% confidence interval; and
- exact coverage/failure count.

Diagnostic:

- wrist, fingertip, and palm-normal error on external datasets with compatible GT;
- MPJVE, acceleration error, and jerk;
- clean, blur, occlusion, dropout, and disagreement subsets;
- transition/high-velocity versus low-velocity subsets;
- uncertainty NLL, coverage, risk-coverage curve, Spearman correlation, and worst-decile AUROC;
- percentage and cause of test-time fallback; and
- runtime, peak memory, parameter count, and cache size.

Temporal metrics never replace spatial TR-V2V in a go/no-go decision.

### 12.5 Statistical unit

Bootstrap by sign/clip, not by individual vertex or frame. Frames within a sign are correlated. Report mean, median, and worst-decile per-sign change in addition to the pooled vertex-frame aggregate.

---

## 13. Master go/no-go strategy

### Gate G0: evaluator and coverage lock

**Go when:** A0 and A1 are rendered and evaluated on one immutable manifest; frame IDs, hashes, topology, units, masks, and alignment are recorded; A1 has no missing output.

**No-go when:** aggregate values change with file ordering, truncation, frame availability, or region alignment. Stop modeling and repair the evaluator contract.

### Gate G1: Phase 1 initializer quality

**Go when:** the selected stronger initializer has valid wrist/forearm attachment and improves or preserves the common-manifest baseline without a material regional regression.

**No-go when:** WiLoR/MANO conversion, handedness, camera, scale, or wrist orientation is inconsistent. Fix Phase 1 before temporal training.

### Gate G2: data readiness

**Go when:** Section 8.2 passes and observation caches reproduce deterministically.

**No-go when:** only the current framewise `sign_v1` set and short local snippets are available. Run feasibility experiments only; do not launch the main model.

### Gate G3: synthetic recoverability

**Go when:** T1 recovers at least 30% of injected error and preserves clean clips within 2% per region.

**No-go when:** the model cannot solve controlled missing bursts. Debug representation and targets; adding complexity is prohibited.

### Gate G4: real validation value

**Go when:** on external source-disjoint validation, weighted regional error improves at least 3%, no region regresses more than 1%, and the predefined hard subset improves at least 8%.

**No-go when:** only smoothness metrics improve. Revisit temporal alignment, pseudo-target quality, and target-motion losses. Do not add diffusion.

### Gate G5: uncertainty validity

**Go when:** U1 passes every calibration condition and improves hard-subset reconstruction over U0.

**No-go when:** confidence is uncalibrated or merely tracks detector confidence. Retain U0.

### Gate G6: locked local benchmark

Phase 2 is accepted for progression only if, relative to the selected A1 initializer on Lane L:

- all three regions have full, identical coverage;
- no region regresses by more than `0.20 mm` pooled error;
- at least two of the three regions improve with a sign-clustered 95% CI excluding zero;
- the equally weighted relative improvement across the three reported regions is at least `3%`;
- the predefined occlusion/dropout/disagreement subset improves at least `8%`;
- clean low-uncertainty frames regress by less than `1%` in every region;
- fewer than 1% of group-frames trigger safety fallback; and
- the result is reproducible across three seeds, with regional standard deviation below `0.20 mm`.

If G6 fails, Phase 2 does not justify relational diffusion or phonology. Preserve the best geometric Phase 1 method and report the temporal result honestly.

### Gate G7: official comparison

**Go when:** an independently verified protocol reproduces or explains the published baseline and uses a common official frame population. Only then compare Phase 2 against `30.13 / 13.53 / 13.08`.

**No-go when:** published and local protocols remain irreconcilable. Report Lane L only and avoid SOTA language.

---

## 14. Failure-directed pivot table

| Observed failure | Most likely cause | Required pivot |
|---|---|---|
| synthetic dropout is not recovered | representation/model bug or weak temporal context | overfit one clip, inspect masks and rotation composition |
| clean frames worsen | residual head not identity-preserving | zero-init output, strengthen reliable anchor, add clean batches |
| hands improve but upper body worsens | wrist/forearm coupling or loss imbalance | freeze torso, reduce cross-part update, rebalance region losses |
| upper body improves but hands worsen | body vertex dominance | normalize per region, raise fingertip/palm loss |
| low jerk but unchanged/worse TR-V2V | oversmoothing | target velocity/acceleration, reduce context prior, compare B1 |
| missing frames disappear | inherited `data_parser.py` filtering | rebuild cache loader; missingness must be explicit |
| left/right behavior is asymmetric | handedness conversion or evaluation population | audit side routing and report class-0 population separately |
| uncertainty rises everywhere | variance inflation shortcut | stop-gradient feedback, variance bounds, calibration split |
| uncertainty has no error correlation | insufficient reliability labels/features | keep U0; add detector score/disagreement data |
| validation improves but SGNify does not | domain shift or protocol mismatch | inspect per-sign subsets; do not tune on SGNify |
| results vary across runs | nondeterministic cache/order/training | freeze manifest, seed workers, deterministic export |
| T5 optimization erases motion | observation energy dominates temporal target | disable T5 or reduce steps; direct model remains primary |

---

## 15. Implementation layout

Create a new isolated package instead of adding branches throughout `fitting.py`:

```text
phase2_refiner/
  README.md
  configs/
    uawsr_u0.yaml
    uawsr_u1.yaml
  data/
    cache_schema.py
    build_observation_cache.py
    build_sequence_index.py
    corruptions.py
    dataset.py
  geometry/
    rotations.py
    coordinates.py
    palm.py
    smplx_decode.py
  models/
    embeddings.py
    reliability.py
    spatial_temporal_refiner.py
    heads.py
  losses/
    geometry.py
    motion.py
    uncertainty.py
  train.py
  calibrate.py
  infer.py
  render.py
  evaluate.py
  tests/
```

Suggested generated artifacts:

```text
outputs/phase2_<experiment>/
  <sign>/smplifyx/results/*.pkl
  <sign>/smplifyx/meshes/*.obj
  <sign>/phase2_diagnostics/*.json
  run_manifest.json
  per_frame.csv
  per_sign.csv
  summary.csv
```

Do not overwrite `outputs/method_hamer`, `outputs/method_nlf_wilor`, or any Phase 1 output.

### 15.1 Minimal command contract

```bash
python -m phase2_refiner.data.build_observation_cache \
  --frames data/frames \
  --initializer outputs/<phase1_method> \
  --out cache/phase2/v1

python -m phase2_refiner.train \
  --config phase2_refiner/configs/uawsr_u0.yaml

python -m phase2_refiner.infer \
  --config phase2_refiner/configs/uawsr_u0.yaml \
  --cache cache/phase2/v1 \
  --output outputs/phase2_uawsr_u0

python -m phase2_refiner.evaluate \
  --manifest probes/results/phase0/frame_manifest.csv \
  --baseline outputs/<phase1_method> \
  --prediction outputs/phase2_uawsr_u0
```

Every command writes its resolved config, git SHA, dependency versions, checkpoint hashes, input manifest hash, and random seeds.

---

## 16. Required tests before a full run

### Unit tests

- axis-angle -> matrix -> 6D -> matrix -> axis-angle round-trip;
- left/right MANO convention on a known asymmetric pose;
- wrist-local -> torso -> camera -> torso round-trip;
- palm-normal sign and fingertip joint order;
- missing observation produces finite tokens and zero direct data loss;
- padding invariance;
- causal masking versus bidirectional masking;
- overlap rotation blending across the `pi` boundary;
- shared betas across every output frame;
- result PKL schema and mesh topology; and
- evaluator rejects a missing or duplicate frame.

### Integration tests

- identity pass on one sign reproduces all meshes within `0.01 mm`;
- a deliberately masked 8-frame hand burst is filled without changing unmasked opposite-hand frames excessively;
- one-handed class-0 signs retain the passive side in reconstruction even though LHand is skipped in the author metric;
- inference produces the same output hash when repeated with the same seed/config; and
- a 64-frame batch fits in GPU memory with at least 20% headroom.

### Red-team tests

- wrong handedness candidate;
- two detections for the same side;
- no hands in the first frames;
- complete hand absence for the whole clip;
- frame-number gaps;
- crop truncation;
- extreme axis-angle near `pi`;
- NaN expert input;
- inconsistent image resolution; and
- an output directory containing stale extra meshes.

---

## 17. Milestones and estimated order

| Milestone | Work | Exit artifact |
|---|---|---|
| M0 | freeze Lane L manifest and select A1 | signed evaluation and baseline manifest |
| M1 | observation cache with all-frame missing masks | cache v1 plus audit report |
| M2 | geometry/SO(3)/SMPL-X identity path | identity integration test |
| M3 | UAWSR skeleton and synthetic corruption | T1 report and checkpoint |
| M4 | exact-expert real residual dataset | Tier C manifest and quality audit |
| M5 | real residual + sign adaptation | P2.0/P2.1 models and validation report |
| M6 | calibration study | U0/U1 decision report |
| M7 | complete benchmark and ablations | Lane L table, CIs, diagnostics |
| M8 | optional T5 and release hardening | final Phase 2 package |

A realistic single-GPU order is 10–14 weeks if the sequence data are already licensed and accessible. Data acquisition, pseudo-target generation, or license clarification is outside that estimate and is a hard dependency rather than hidden schedule slack.

---

## 18. Definition of done

Phase 2 is complete when:

1. the selected stronger initializer and UAWSR share one immutable evaluation manifest;
2. all scheduled frames, including expert failures, pass through the model;
3. the model jointly refines upper body, wrists, and both 45-D hands using valid SO(3) composition;
4. U0 is available and U1 is used only if calibrated;
5. result PKLs and 10,475-vertex meshes are emitted for every frame;
6. identity, missingness, rotation, padding, and deterministic-output tests pass;
7. the full baseline/ablation table is evaluated with sign-clustered confidence intervals;
8. the master G6 criteria pass, or a documented no-go decision is issued;
9. the published DexAvatar scores are not mixed with unresolved local-protocol scores; and
10. every data source, expert, cache, config, checkpoint, and output is traceable by hash.

---

## 19. Final recommendation

The highest-probability Phase 2 is deliberately narrower than the complete SignPosterior4D proposal:

> **Frozen strong observations + explicit missingness + coordinate-correct unified body/wrist/two-hand state + deterministic bidirectional residual refinement + a calibration-gated reliability model.**

This phase should answer one decisive question before more complex research is attempted: **can complete-sign context correct spatial failures that a strong framewise initializer cannot?**

If the answer passes G6, proceed next to a small relational/contact study and only later to diffusion or structured linguistic conditioning. If it does not, stop escalating model complexity and return to observation geometry, target quality, and evaluation protocol.

---

## 20. Implementation status

- **Implementation date:** 22 July 2026
- **Code status:** first executable Phase 2 vertical slice completed
- **Research status:** untrained for the final task; no accuracy or SOTA claim

The implementation was intentionally added as the isolated `phase2_refiner/` package. No file under `dexavatar_fitting/`, `methods/`, `runners/`, `evaluation/`, or an existing output directory was changed. Existing methods remain reusable and serve as read-only initializers/baselines.

### 20.1 Implemented files

| File | Completed responsibility |
|---|---|
| `phase2_refiner/__init__.py` | package/version entry point |
| `phase2_refiner/config.py` | YAML configuration loading |
| `phase2_refiner/data/cache_schema.py` | validated, versioned, non-pickle NPZ clip contract |
| `phase2_refiner/data/build_observation_cache.py` | read-only conversion of existing result/Sapiens/HaMeR-WiLoR artifacts into per-sign caches |
| `phase2_refiner/data/dataset.py` | padded/windowed whole-sequence loading, feature construction, and batching |
| `phase2_refiner/data/corruptions.py` | contiguous body/left-hand/right-hand dropout and SO(3) perturbation |
| `phase2_refiner/geometry/rotations.py` | differentiable axis-angle, matrix, 6D, quaternion, geodesic, bounded residual, and composition operations |
| `phase2_refiner/models/spatial_temporal_refiner.py` | 51-joint factorized spatial/temporal/group Transformer with bidirectional or causal mode |
| `phase2_refiner/losses/sequence.py` | rotation, target-velocity, target-acceleration, reliable anchor, and heteroscedastic losses |
| `phase2_refiner/train.py` | AdamW training, burst curriculum, gradient accumulation/clipping, validation, and checkpointing |
| `phase2_refiner/infer.py` | complete-clip and overlapping-window inference, geodesic/quaternion blending, safety fallback, PKL export, and diagnostics |
| `phase2_refiner/render.py` | standard and source-anchored SMPL-X mesh rendering |
| `phase2_refiner/evaluate.py` | strict common-manifest regional TR-V2V, coverage/topology checks, per-frame/per-sign output, and sign-bootstrap CIs |
| `phase2_refiner/calibrate.py` | variance scaling, NLL, Spearman, worst-decile AUROC, and risk-coverage audit |
| `phase2_refiner/configs/uawsr_u0.yaml` | fixed-reliability starting configuration |
| `phase2_refiner/configs/uawsr_u1.yaml` | learned-uncertainty starting configuration |
| `phase2_refiner/README.md` | runnable cache/train/infer/evaluate/calibrate commands and non-destructive contract |
| `phase2_refiner/tests/` | rotation, cache, model, corruption, training-gradient, fallback, calibration, and strict-evaluator tests |

### 20.2 Implemented behavior

- Existing initializers are only read. Cache and inference commands refuse to overwrite by default.
- The cache retains explicit confidence, expert presence, missingness, normalized crop size, crop truncation, temporal innovation, duplicate-side detection, normalized 2D joints, and the source result path for every scheduled initializer frame.
- The state contains all 21 body rotations and both 15-joint hands. The default output mask refines 12 upper-body/arm joints plus all 30 hand joints and freezes lower-body joints.
- Residuals are zero initialized, bounded to 25 degrees for body and 35 degrees for hands, gated, and left-composed on SO(3).
- Clips up to 64 frames run in one bidirectional pass. Longer clips use 50% overlap, Hann weights, quaternion hemisphere alignment, and normalized quaternion blending rather than axis-angle averaging.
- Non-finite or over-limit outputs fall back independently for body, left hand, or right hand and are counted in diagnostics.
- Output PKLs retain the existing DexAvatar shapes. Diagnostics are stored separately so they cannot break legacy renderers.
- Source-anchored rendering compensates for the observed mismatch between some saved legacy PKLs and their saved meshes: it applies the same-model refined-minus-initializer vertex displacement to the original mesh.
- The new evaluator rejects incomplete coverage and stale extra meshes rather than using `min(GT, prediction)` truncation.

### 20.3 Validation completed

| Check | Result |
|---|---|
| Python compilation | all `phase2_refiner` modules compile |
| Automated tests | **13 passed** |
| Rotation matrix/6D/axis-angle numerical round trip | maximum matrix error below `5e-7` in the standalone audit |
| One-sign cache | Ablehnen: 14 frames, `(14, 51, 3)` pose state, `(14, 51, 8)` observation state |
| Full local cache | **57 signs and 1,493 frames**, matching the audited `method_hamer` manifest population |
| Full identity inference | **1,493/1,493** result PKLs emitted |
| Identity parameter preservation | maximum difference across body, both hands, shape, root, translation, jaw, and expression: **0.0** |
| Identity safety behavior | **0** group-frame fallbacks over 57 signs |
| Strict full-manifest identity evaluation | prediction equals baseline at **29.907 / 13.573 / 12.927 mm** (upper body / left / right); paired deltas are **0.0 mm** with `[0.0, 0.0] mm` sign-bootstrap intervals for every region |
| One-sign mesh export | 14/14 meshes, 10,475 vertices, unchanged face topology |
| Source-anchored identity mesh error | about `4.1e-6 mm` mean and `8.7e-6 mm` maximum on checked frames |
| Training smoke test | one full UAWSR optimizer update completed and wrote a 31 MB checkpoint |
| Checkpoint inference smoke test | 14/14 PKLs emitted; learned residual path was nonzero and bounded |

All smoke caches, checkpoints, and predictions were written under temporary `/tmp/dexavatar_phase2_*` directories. No experimental output was mixed with an existing method.

### 20.4 Important compatibility finding

A fresh SMPL-X forward pass from an existing `method_hamer` result PKL differed from its already saved mesh by roughly `0.33–0.47 mm` mean on two inspected frames, with larger local maxima, despite identical exported parameter arrays. Possible causes include historical renderer/model state or saved-PKL/mesh drift.

Directly re-rendering every initializer would therefore change the baseline before Phase 2 made a prediction. The implemented source-anchored renderer avoids that confound:

$$
V^{phase2} = V^{saved-init} +
\left(F_{SMPLX}(X^{phase2}) - F_{SMPLX}(X^{init})\right).
$$

This makes a zero residual reproduce the saved baseline mesh while still applying learned pose-dependent vertex displacement. It should remain an explicit ablation against direct fresh rendering.

### 20.5 Gates that remain open

This implementation does **not** mean Phase 2 has passed its research gates.

- **G0 remains partially open:** code and coverage checks exist, but the 1,493-versus-2,872 official protocol discrepancy is unresolved.
- **G1 remains open:** the final stronger Phase 1 initializer has not been selected on one complete common manifest. The current `method_nlf_wilor` output does not yet have the audited 1,493-frame coverage.
- **G2 is a no-go for main training:** the required licensed, source-/signer-disjoint whole-sequence target corpus has not been prepared. The existing staged `sign_v1` data remain framewise and body-only.
- **G3–G6 remain open:** the one-step smoke checkpoint is not a trained model and must never be evaluated or reported as Phase 2 accuracy.
- U1 calibration requires a disjoint residual dataset. The implemented calibration utility is ready, but no calibrated U1 model exists.
- The first cache version supplies rotation, 2D, and reliability features. Audited metric-compatible full 3D body/hand observations, palm geometry, and RGB/crop features remain follow-up cache revisions.
- Learned groupwise reprojection-based fallback and biomechanical checks remain to be added after coordinate-valid 3D observations are available; current safety covers numerical and rotation-bound failures.

### 20.6 Next authorized execution order

1. complete G0 and select a full-coverage A1 initializer;
2. build `cache/phase2/<a1>_v1` without overwriting A1;
3. acquire/prepare a licensed sequence target corpus and pass G2;
4. run T0 identity and T1 synthetic-recoverability training with three seeds;
5. stop if G3 fails;
6. build exact-expert real residual pairs and run T2/P2.0;
7. compare P2.0, U0/P2.1, and simple smoothing under the strict evaluator;
8. train/calibrate U1 only after deterministic refinement passes G4; and
9. render and evaluate Lane L only after complete identical coverage is verified.

---

## 21. Completion report — 22 July 2026

Phase 2 implementation is complete as an **isolated, non-destructive executable foundation**. The newly added `phase2_refiner/` package contains cache construction, sequence refinement, training, inference, rendering, strict evaluation, calibration, configurations, documentation, and tests. Existing DexAvatar methods, their code, their output directories, and the supplied evaluator remain unchanged and can be reused as before.

### 21.1 Files added

- `phase2_refiner/data/`: versioned observation cache, sequence dataset, and synthetic corruption curriculum;
- `phase2_refiner/geometry/rotations.py`: valid SO(3) conversions, residual bounds, and composition;
- `phase2_refiner/models/spatial_temporal_refiner.py`: 51-joint whole-sequence body/wrist/two-hand refiner;
- `phase2_refiner/losses/sequence.py` and `phase2_refiner/train.py`: objective functions and training loop;
- `phase2_refiner/infer.py` and `phase2_refiner/render.py`: compatible PKL output, overlapping-window inference, safety fallback, and source-anchored mesh rendering;
- `phase2_refiner/evaluate.py` and `phase2_refiner/calibrate.py`: strict locked-manifest TR-V2V evaluation and uncertainty audit; and
- `phase2_refiner/tests/`, `phase2_refiner/configs/`, and `phase2_refiner/README.md`: test coverage, U0/U1 starting configurations, and commands.

### 21.2 Final implementation validation

| Validation item | Final result |
|---|---|
| Static checks | `ruff check phase2_refiner`: passed; Python compilation: passed |
| Automated tests | **13 passed** |
| Locked cache coverage | **57 signs, 1,493 frames** |
| Identity output | **1,493/1,493** PKLs; maximum parameter difference **0.0** |
| Identity safety | **0** body/left/right group fallbacks |
| Identity mesh compatibility | 10,475 vertices and unchanged topology; source-anchored error about `4.1e-6 mm` mean on checked frames |
| Strict full-manifest evaluation | Phase 2 identity equals `method_hamer`: **29.907 / 13.573 / 12.927 mm** for upper body / left hand / right hand |
| Paired comparison | **0.0 mm** delta in every region; sign-bootstrap 95% intervals `[0.0, 0.0] mm` |
| Training path | one complete optimizer update and checkpoint-inference smoke test passed |

The strict evaluation uses the audited local 1,493-frame manifest. Its score must remain separate from the previously quoted `30.13 / 13.53 / 13.08` result until the 1,493-versus-2,872 protocol discrepancy is resolved.

### 21.3 Final Go/No-Go status

- **Implementation integrity: GO.** The identity path reproduces the baseline exactly and legacy artifacts were not overwritten.
- **Training/research claim: NO-GO.** Do not report a Phase 2 quality result yet: G0 protocol reconciliation, G1 stronger-initializer selection, and G2 sequence-supervision data are still required.
- **U1 uncertainty: NO-GO.** Calibration is implemented but requires a held-out real residual dataset before it can be enabled.
