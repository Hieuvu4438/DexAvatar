# Phase 3 Build Plan: Relational Diffusion Posterior

- **Project:** DexAvatar / SignPosterior4D de-risking program
- **Phase:** Phase 3 only
- **Method name:** `RDP` (Relational Diffusion Posterior)
- **Proposal date:** 3 August 2026
- **Primary objective:** infer several coordinated whole-sequence SMPL-X hypotheses from uncertain frozen observations, explicitly model hand–hand and hand–body relations, and select a hypothesis using video evidence rather than benchmark ground truth.
- **Primary evaluation contract:** the author-released 57-sign / 1,493-frame SGNify population selected by the project owner
- **Design sources:** [method-centered SignPosterior4D proposal](DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md) and [Phase 2 UAWSR plan and execution report](DEXAVATAR_PHASE2_UNCERTAINTY_AWARE_WHOLE_SEQUENCE_REFINER.md)

---

## 1. Executive decision

Proceed to Phase 3 as a **new, isolated research branch**, while freezing Phase 2 for possible later reuse.

Phase 2 produced useful infrastructure and strong controlled synthetic-recovery results, but it did not pass the complete formal progression contract:

| Phase 2 gate | Current decision | Consequence for Phase 3 |
|---|:---:|---|
| G0 evaluator/coverage | GO | reuse the immutable author-evaluation contract |
| G1 strongest initializer | GO | use the locked A1 ensemble/fallback as the final benchmark initializer |
| G2 data readiness | owner-delegated GO | reuse caches only with their truthful target/provenance labels |
| G3 synthetic recovery | GO | reuse geometry, SO(3), corruption, and decoded-vertex tests |
| G4 real residual value | proxy GO / formal NO-GO | do not treat T2 as a validated posterior backbone |
| G5 uncertainty | proxy GO / formal NO-GO | do not enable U1 in the main Phase 3 configuration |
| G6 locked transfer | NO-GO | do not initialize the main model from the failed Lane-L T2/T5 result |
| G7 author evaluation scope | GO | 1,493 frames are the canonical final population |

This pivot is intentionally different from pretending that Phase 2 succeeded. The main Phase 3 route must be able to train and run **without** a Phase 2 quality checkpoint:

```text
frozen observations + fixed U0 reliability
  -> canonical body/wrist/two-hand sequence
  -> frozen DPoser-X whole-body spatial score
  -> trainable temporal-relational score adapters
  -> probabilistic relation/contact graph
  -> masked conditional diffusion posterior
  -> K complete-sequence candidates
  -> evidence-only selector
  -> optional safe observation refinement
  -> standard PKLs + meshes + diagnostics
```

Phase 2 remains reusable in three ways:

1. immediately reuse its cache schema, rotation/coordinate utilities, SMPL-X decoder, strict evaluator, renderer, provenance helpers, and corruption tests;
2. treat the generic ARCTIC T1 checkpoint as an **optional initialization ablation**, because it passed generic G3 but not sign-domain G6; and
3. add a later `Phase2-GO -> Phase3` initialization route if exact-A1 Phase 2 training eventually passes G4–G6.

The central scientific question for this phase is:

> Can a relational, multimodal whole-sequence posterior recover ambiguous and interacting hand motion that neither the frozen initializer nor a deterministic temporal correction can recover?

Phase 3 is successful only if diffusion and relational reasoning improve spatial accuracy on predeclared difficult subsets. Candidate diversity, visual smoothness, or lower jerk alone is not a GO.

---

## 2. Exact Phase 3 scope

### 2.1 In scope

- a unified sequence distribution over upper body, wrists, and both hands;
- masked conditional diffusion over complete signs or 64-frame windows;
- a compact hand–hand and hand–body relational graph;
- probabilistic contact, depth ordering, and contact persistence;
- full-observation, part-masked, and burst-masked score training;
- frozen DPoser-X whole-body spatial prior initialization or distillation;
- source-specific supervision masks for whole-body and hand-only datasets;
- `K = 1` deterministic-like and `K = 4` multi-hypothesis inference;
- evidence-only sample ranking with held-out observation prediction;
- optional short differentiable refinement after candidate selection;
- exact coverage, safety fallback, provenance, and three-seed evaluation; and
- additive code and artifacts that do not modify previous methods.

### 2.2 Explicitly deferred to Phase 4

- SHuBERT or another sign-video semantic encoder;
- HamNoSys conditioning;
- handshape/orientation/location/movement semantic tokens;
- learned hold/transition/repetition/contact-phase classes;
- semantic–kinematic cycle consistency;
- probabilistic dominance or symmetry as linguistic variables; and
- any claim that the model is already phonology conditioned.

Relations in Phase 3 are **geometric and interaction based**, not phonological. This boundary is important for the required ablation ladder in the main proposal:

```text
A4 uncertainty-aware deterministic model
  -> A5 relational graph/contact
  -> A6 masked diffusion, K=1
  -> A7 K-hypothesis evidence selection
  -> Phase 4 A8/A9 phonology and phase
```

### 2.3 Other non-goals

- changing the author evaluator or frame population;
- training on `data/smplx_gt` or `data/evaluation_from_author`;
- tuning on the 57 SGNify signs;
- fine-tuning WiLoR, SMPLer-X, Sapiens, or another observation expert;
- presenting DPoser-X, FUSION, or an expert ensemble as the paper novelty;
- best-of-`K` selection using GT;
- forcing contact whenever two surfaces are close;
- diffusing camera, translation, body shape, or face parameters in the first model; and
- overwriting Phase 1, Phase 2, DexAvatar, or legacy output directories.

---

## 3. Repository and data facts that constrain the plan

### 3.1 Assets ready now

The following measurements come from the current repository and existing Phase 2 audits.

| Asset | Local state | Phase 3 role | Limitation |
|---|---:|---|---|
| How2Sign raw data | approximately 86 GB | RGB/2D sign-domain observations and source-disjoint adaptation | no uniformly clean metric SMPL-X GT |
| `cache/phase2/how2sign_t1_v1` | 11,000 train clips / 352,000 frames; 1,200 val clips / 38,400 frames | sign-motion and masking pilot | H32 pseudo-SMPL-X teacher, not exact A1 |
| How2Sign reprojection cache | 10,822 train / 498 val / 497 calibration clips | observation-conditioning pilot | residual domain differs strongly from Lane A1 |
| ARCTIC local data | approximately 20 GB; 301 sequences / 218,273 raw frames | complete SMPL-X pretraining and relation/contact geometry | hand–object rather than sign contact |
| `cache/phase2/arctic_t1_v1` | 2,351 train clips / 146,781 frames; 511 val clips / 31,822 frames | clean whole-body masked diffusion and temporal validation | generic domain |
| InterHand2.6M local annotations | 1,148 train clips / 16,096 frames; 47 val clips / 2,736 frames | left/right articulation and hand–hand relation pretraining | partial hand supervision, limited local subset |
| PHOENIX-2014T local release | 4,121 locally present extracted PNG frames plus evaluation assets | optional cross-language qualitative/2D-consistency pilot | incomplete for large-scale training and has no compatible metric 3D target |
| DPoser-X local sign checkpoint | `checkpoints/dposer/sign/sign_body_ft/last.ckpt` | optional torso-only adapter ablation | body-only; it is not a whole-body body-plus-hands model |
| Phase 2 ARCTIC T1 checkpoint | generic G3 GO | optional deterministic warm-start ablation | not a sign-domain or Lane-G6-passing checkpoint |
| Phase 2 U1 v7 checkpoint | strong H32-domain numerical calibration | diagnostic ablation only | formal G5 NO-GO and wrong final initializer domain |

### 3.2 Assets not ready and therefore forbidden as hidden dependencies

| Asset | Current local finding | Required action |
|---|---|---|
| public DPoser-X whole-body weights | `DPoser-X/pretrained_models/` contains no usable weights | download and hash `wholebody/mixed/last.ckpt` plus required body/hand submodels |
| SignAvatars | not present locally | acquire under its research terms and build a signer/source-disjoint cache |
| Motion-X | local 498 MB archive fails ZIP central-directory validation | redownload and verify, or omit because DPoser-X already supplies generic pretraining |
| WHIM | local multipart training archive is incomplete | complete and verify only if a hand-data ablation needs it |
| DexYCB | local directory is effectively empty | acquire only if required; it is not a Phase 3 blocker |
| FUSION | no approved local dependency or redistribution permission recorded | keep optional behind a written license/derived-weight gate |
| exact-A1 How2Sign outputs | frozen stack fails at its hard-coded German sign/segment contract | not required by the standalone Phase 3 route; never fake sign labels to bypass it |

The official [DPoser-X release](https://dposer.github.io/) describes a masked whole-body diffusion prior and publishes a `wholebody/mixed/last.ckpt` model through its [official weight repository](https://huggingface.co/Moon-bow/DPoser-X). The current repository does not yet contain that file. Download completion and checkpoint compatibility are therefore formal gates, not assumptions.

### 3.3 Dataset interpretation

- [SignAvatars](https://signavatars.github.io/) provides the largest directly relevant sign-specific SMPL-X corpus and is the preferred final adaptation source. Its SMPL-X annotations are reconstructed/pseudo annotations, so they require quality filtering and must not be called mocap GT.
- [How2Sign](https://how2sign.github.io/index.html) provides more than 80 hours of continuous ASL and official train/validation/test splits. In this project, its H32 SMPL-X sequences are useful as sign-motion pseudo targets and its RGB/2D observations are useful for evidence conditioning.
- [ARCTIC](https://arctic.is.tue.mpg.de/) provides synchronized SMPL-X/MANO geometry and dynamic hand–object contact. It is valuable for learning precise temporal geometry, but its object contacts cannot be relabeled as hand–face or hand–torso sign contacts.
- [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) provides accurate single/interacting-hand annotations. It is valuable for hand identity and hand–hand geometry, but does not supervise full upper-body signing.
- [Motion-X](https://motion-x-dataset.github.io/) is an optional large generic whole-body source after the local archive is repaired and its license is recorded. Generic data must be downweighted during sign adaptation.
- SGNify evaluation data stay evaluation-only under all configurations.

---

## 4. Scientific hypotheses

### H3.1 Relational conditioning

Explicit relative geometry among wrists, palms, fingertips, face, chest, shoulders, and upper arms will improve occluded/interacting frames because these relations remain informative when one local hand observation fails.

### H3.2 Masked whole-sequence diffusion

A conditional sequence distribution can represent several plausible completions for long hand dropouts, depth ambiguity, palm flips, and crossed hands, where a deterministic residual model tends to average or copy the teacher.

### H3.3 Evidence-selected hypotheses

When candidate selection uses observations withheld from candidate generation, selected `K = 4` hypotheses will outperform a one-path posterior and close a measurable fraction of the oracle candidate gap without using GT.

### H3.4 Standalone Phase 3

A pretrained spatial score plus new temporal/relation adapters can provide value even when the Phase 2 deterministic checkpoint does not transfer. If this hypothesis fails, posterior complexity is not justified and the project should return to target/observation geometry rather than proceed to Phase 4.

---

## 5. Formal posterior and state

Let the fixed observations be `O`, fixed reliability be `U`, and predicted geometric relations be `R`. Phase 3 models:

$$
p_{\theta}(X_{1:T}\mid O_{1:T}, U_{1:T}, R_{1:T}).
$$

No phonology variable appears in Phase 3.

For each frame:

$$
X_t = [
R_t^{body,21},
R_t^{left,15},
R_t^{right,15}
],
$$

with 51 local joint rotations. The model receives all 51 joints for context, but the default mutable set is the same identity-safe set used by Phase 2:

- spine1, spine2, spine3;
- neck and both collars;
- both shoulders, elbows, and wrists; and
- all 15 joints of each hand.

Lower-body rotations, head, jaw, eyes, expression, global orientation, translation, camera, and shared shape are copied from the initializer in the main configuration.

### 5.1 Diffusion representation

Use normalized continuous 6D rotation coordinates for the forward SDE:

$$
z_0 = \operatorname{rot6d}(X_{1:T}),
$$

then project every predicted clean rotation back to `SO(3)` by Gram–Schmidt before geometry losses or SMPL-X decoding. Axis-angle is used only for cache compatibility and export.

This is an Euclidean diffusion representation with an explicit manifold projection. It must pass round-trip and gradient tests before training. Raw axis-angle values must never be averaged across candidates or windows.

### 5.2 Canonical frames

- body joints and cross-part relations: torso-centered frame;
- hand articulation: wrist-local frame;
- palm orientation: both wrist-local and torso-relative;
- hand–body distances: torso frame using decoded SMPL-X anchors;
- 2D evidence: camera frame with the cached intrinsic matrix;
- shape: one robust median `betas` vector per sequence; and
- sequence time: seconds plus normalized frame index, not frame index alone.

Phase 2 coordinate transforms and round-trip tests are reused read-only. A Phase 3 cache may reference a Phase 2 clip only if its schema version and transform hashes are accepted.

---

## 6. Observation and reliability contract

### 6.1 Main observation route

The main reproducible route freezes the already selected A1 observation stack for final evaluation and consumes:

- initializer SMPL-X rotations and decoded joints;
- WiLoR/HaMeR-compatible hand observations where available;
- Sapiens 2D whole-body joints and original confidence;
- camera intrinsics;
- torso/wrist-local 3D geometry;
- palm centers/normals, fingertips, MCP centers, and wrist attachment;
- missingness, crop truncation, temporal innovation, and expert-disagreement features; and
- source/provenance masks.

### 6.2 Reliability decision

Use deterministic `U0` reliability in the main Phase 3 experiment.

Reasons:

1. it is available for all caches;
2. it does not inherit the H32-to-A1 formal mismatch;
3. it isolates the Phase 3 contribution from an unaccepted uncertainty model; and
4. it provides a stable fallback when observations are absent.

U1 v7 may appear only as an H32-domain ablation. It can become part of the main configuration later only if it is recalibrated on a source-disjoint cache whose initializer provenance matches the Phase 3 observation domain and passes all formal Phase 2 G5 criteria.

### 6.3 Conditioning/evidence split

Evidence selection must not score a candidate with exactly the same observations that forced its generation.

For every training and validation sequence:

1. stratify high-confidence 2D/3D observations by region and time;
2. hold out 20% as evidence tokens, ensuring every region retains conditioning observations;
3. generate candidates from the remaining 80%;
4. rank candidates by their prediction of the held-out evidence; and
5. optionally refine the selected candidate with all observations.

At evaluation, use four deterministic evidence folds and average their candidate scores. The split seed is derived from the clip hash, not chosen per result.

---

## 7. Relational hand–body graph

### 7.1 Nodes

Use a small, fixed graph rather than all 10,475 vertices:

- left/right wrists;
- left/right palm centers and palm normals;
- 10 MCP anchors;
- 10 fingertips;
- head, chin, upper chest, sternum, and pelvis/root;
- left/right shoulders and upper arms; and
- optional nearest SMPL-X surface anchors for predicted contact.

### 7.2 Candidate edges

- wrist-to-wrist;
- palm-to-palm;
- fingertip-to-opposite-fingertip;
- fingertip-to-opposite-palm;
- each palm/fingertip to face/chin/chest/shoulder/upper-arm anchors;
- wrist-to-own-elbow and palm-to-own-forearm; and
- temporal self-edges for contact persistence.

Each edge token contains:

- relative 3D vector and distance;
- relative velocity and acceleration;
- relative palm orientation;
- signed front/back depth ordering;
- 2D overlap and visibility;
- both endpoint reliabilities;
- edge validity; and
- previous-frame contact probability.

### 7.3 Relation encoder

Starting configuration:

| Item | Value |
|---|---:|
| node hidden size | 256 |
| edge hidden size | 128 |
| graph layers | 4 |
| attention heads | 8 |
| dropout | 0.1 |
| temporal edge context | 5 frames centered on the current frame |
| outputs | relation token, contact logit, depth-order logit, persistence logit |

The graph emits frame-level relation tokens and per-edge probabilities. These condition cross-part attention in the diffusion denoiser.

### 7.4 Contact labels and fail-safe behavior

Geometry-derived pseudo contact is positive only when all available conditions agree:

- decoded surface distance is below 12 mm for contact onset;
- contact remains active until distance exceeds 20 mm;
- relative tangential speed is below 0.15 m/s for a persistent contact;
- both endpoints are valid; and
- penetration is below the allowed safety tolerance.

These are starting thresholds. They may be changed only using the manually verified external relation set, never Lane-L.

Contact is always probabilistic. The contact energy is zeroed when:

- probability is below 0.6;
- endpoint uncertainty is outside the training range;
- the candidate edge is invalid or out of frame;
- the contact-location entropy is above its validation threshold; or
- enforcing contact increases reliable-observation error beyond the frozen tolerance.

ARCTIC object contact may pretrain the generic notions of approach, persistence, and slip, but it cannot directly label sign hand–body edges. InterHand supplies hand–hand geometry. Sign-specific hand–body contacts are derived from filtered SignAvatars/How2Sign SMPL-X and checked on a manually verified subset.

---

## 8. Relational diffusion architecture

### 8.1 Backbone

Use a factorized spatio-temporal score network over `B x T x 51` joint tokens:

1. frozen per-frame DPoser-X whole-body spatial score;
2. trainable projection from the DPoser-X representation to the 51-joint Phase 3 state;
3. intra-part attention for torso/arms, left hand, and right hand;
4. cross-part attention through wrist and relation tokens;
5. bidirectional temporal attention for each joint and relation edge; and
6. cross-attention to observation, validity, U0, and diffusion-time tokens.

The predicted score is residual around the frozen spatial prior:

$$
s_{RDP}(z_t,t,c) =
s_{DPoserX}(z_t,t)
+
\Delta s_{temporal-rel}(z_t,t,c).
$$

The residual score projection is zero initialized. Before training, RDP must reproduce the frozen DPoser-X score to numerical tolerance when the residual branch is disabled.

### 8.2 Starting network dimensions

| Component | Starting value |
|---|---:|
| maximum window | 64 frames |
| joint tokens | 51 |
| model width | 384 |
| alternating blocks | 8 |
| attention heads | 8 |
| MLP ratio | 4 |
| dropout | 0.1 |
| relative temporal bias | clipped at 64 frames |
| group tokens | torso, left arm, right arm, left hand, right hand, relation |
| trainable parameters target | approximately 45–70 million, excluding frozen DPoser-X |
| activation checkpointing | enabled for temporal and cross-part blocks |

Factorized attention is mandatory. Full attention over all `64 x 51` tokens is an ablation only if memory and runtime profiling justify it.

### 8.3 Diffusion process

Match the public DPoser-X continuous sub-VP formulation in the primary route:

| Item | Value |
|---|---:|
| SDE | continuous sub-VP |
| `beta_min` | 0.1 |
| `beta_max` | 20.0 |
| nominal scales | 1,000 |
| training time range | `[1e-3, 1.0]` |
| objective | denoising score matching, region balanced |
| likelihood weighting | off initially, matching DPoser-X |
| score scaling | by marginal standard deviation |
| EMA | 0.9999 |

Keeping the SDE and normalization compatible allows the frozen DPoser-X score to be used directly. A different cosine/DDPM/EDM route must be a separate ablation and cannot reuse the same pretrained-score claim without a validated conversion.

### 8.4 Masked conditioning

For each sample, independently choose a supervision-valid mask:

| Mask/corruption | Starting probability |
|---|---:|
| full observations / ordinary diffusion | 20% |
| left-hand burst | 12% |
| right-hand burst | 12% |
| both-hand burst | 10% |
| one finger chain | 10% |
| wrist/forearm attachment | 10% |
| upper-body/arm burst | 8% |
| hand swap or mirror ambiguity | 6% |
| palm/depth ambiguity | 6% |
| 2D dropout/crop truncation | 6% |

Burst lengths are sampled from 4, 8, and 16 frames with equal probability during formal recovery training. Additional 2–12-frame bursts may be used for ordinary augmentation.

Classifier-free condition dropout starts at 10%. Guidance starts at `1.0` and may be compared against `1.2` and `1.5` on external validation. Higher guidance is prohibited if it reduces candidate diversity or worsens any clean region.

---

## 9. Pretrained-model strategy

### 9.1 Primary: public DPoser-X whole-body mixed checkpoint

Download and pin:

```text
DPoser-X/pretrained_models/body/BaseMLP/last.ckpt
DPoser-X/pretrained_models/hand/BaseMLP/last.ckpt
DPoser-X/pretrained_models/wholebody/mixed/last.ckpt
```

The face model is unnecessary for Phase 3 because face state is frozen. If the official whole-body loader requires its face branch, load it frozen but exclude face from the Phase 3 state and losses.

Checkpoint intake must record:

- source URL and retrieval date;
- SHA-256 of every file;
- source-code commit;
- source-code license and model-weight license separately;
- normalization statistics;
- expected rotation representation and joint order; and
- exact tensor-key coverage during load.

### 9.2 Compatibility gate

Direct spatial-score reuse is accepted only if:

- 100% of required body and hand prior tensors load without shape coercion;
- the DPoser-X joint order is mapped explicitly to the Phase 3 51-joint order;
- normalization round-trip error is below `1e-6` in normalized coordinates;
- the adapter reproduces the official per-frame score within `1e-5` maximum absolute error on 100 fixed poses; and
- completion/generation sanity tests remain finite.

If direct transfer fails, do **not** partially load arbitrary matching names. Use DPoser-X as a frozen teacher:

1. sample corrupted clean poses;
2. obtain the DPoser-X denoised pose/score;
3. distill it into the Phase 3 spatial branch; and
4. compare distillation against random spatial initialization.

### 9.3 Local body-only sign checkpoint

`DPoser-X/checkpoints/dposer/sign/sign_body_ft/last.ckpt` may initialize only the 21-body/torso path. It must never initialize or be described as the hand prior. Include three controlled rows:

- public DPoser-X body prior;
- public body prior plus local sign-body adapter; and
- local sign-body adapter without hand transfer.

Retain the sign-body adapter only if it improves source-disjoint sign validation without degrading either hand.

### 9.4 Optional Phase 2 initialization

The ARCTIC T1 checkpoint can initialize temporal embeddings or selected geometry projections only after a tensor-level compatibility report. It remains an ablation because its accepted result is generic synthetic G3, not sign-domain G6.

No T2/T5/U1 Lane result is used in the main Phase 3 route.

### 9.5 Optional FUSION route

[FUSION](https://arxiv.org/abs/2601.03959) is technically relevant as a body–hand temporal diffusion prior. It is not part of the main reproducible configuration unless the project records written permission to:

- modify/fine-tune the code and checkpoint;
- redistribute derived weights or adapters; and
- publish the intended research result.

Without this permission, FUSION may be a frozen comparison only if its terms allow it. The release must not depend on late license approval.

### 9.6 Models intentionally not used in Phase 3

- SHuBERT: reserved for Phase 4 phonology/phase;
- a new trainable RGB backbone: unnecessary for the first relational-posterior test;
- fine-tuned WiLoR/SMPLer-X/Sapiens: observation experts stay frozen; and
- U1 v7: diagnostic H32-domain ablation until exact-domain formal calibration exists.

---

## 10. Dataset strategy

### 10.1 Three supervision tiers

| Tier | Purpose | Main sources | Supervision |
|---|---|---|---|
| A: clean/generic geometry | learn valid whole-body/hand score and temporal geometry | DPoser-X domains, ARCTIC, repaired Motion-X | SMPL-X/MANO parameters, joints, vertices, valid masks |
| B: relation/contact | learn hand identity, relative geometry, contact onset/persistence | InterHand2.6M, ARCTIC, verified sign subset | edges, distances, ordering, contact, slip |
| C: sign adaptation | learn signing motion distribution and observation failures | SignAvatars, How2Sign, PHOENIX unlabeled | filtered SMPL-X pseudo targets, RGB/2D evidence, missingness |

### 10.2 Paper-grade data mixture

Acquire SignAvatars before the final model. Its scale and sign-domain diversity make it the primary Phase 3 adaptation source. The final sequence sampler should use these clip-level proportions:

| Source | Fraction during sign adaptation | Purpose |
|---|---:|---|
| SignAvatars | 50% | main sign-specific whole-body distribution |
| How2Sign | 25% | continuous ASL, realistic observation failures |
| ARCTIC | 15% | retain accurate complete body/two-hand geometry |
| InterHand2.6M | 10% | retain interacting-hand articulation and identity |

Motion-X can replace at most half of the ARCTIC share during generic pretraining. Generic motion must not exceed 30% during final sign adaptation.

### 10.3 Ready-now pilot mixture

Before SignAvatars is available, an engineering/scientific pilot may use:

| Source | Pilot fraction |
|---|---:|
| How2Sign H32 pseudo sequences | 45% |
| ARCTIC | 30% |
| InterHand2.6M | 15% |
| PHOENIX self-supervised observation clips | optional 0–10%, taken from the How2Sign share only after a sequence/coverage audit |

Without PHOENIX, use `55% How2Sign / 30% ARCTIC / 15% InterHand`. This pilot can pass implementation, relation, masking, and sampling gates. It cannot establish the final paper-grade sign-prior claim because How2Sign H32 is a single-teacher pseudo target and PHOENIX has no compatible 3D target.

### 10.4 Source-specific loss masks

- ARCTIC: full 51-joint rotation and SMPL-X geometry; relation/contact only where the corresponding label is valid.
- InterHand: left/right hand articulation, hand joints, and hand–hand relations; body losses are zero.
- SignAvatars: full target where quality masks pass; confidence-weight pseudo geometry and exclude catastrophic frames.
- How2Sign: sign motion and observation consistency; pseudo-target geometry weight starts at 0.25 relative to clean data.
- PHOENIX: 2D/temporal observation consistency only; no fabricated 3D loss.
- Motion-X: full geometry after archive, license, coordinate, and quality audits pass.

Missing labels must be represented by masks. They are never replaced with a zero pose.

### 10.5 Split and leakage policy

Use source, video, signer, and sequence groups—not windows—as the split unit.

- preserve official dataset test splits untouched;
- use signer-disjoint validation wherever signer IDs exist;
- ensure adjacent windows from one source video stay in one split;
- deduplicate RGB frames by perceptual hash and exact SHA-256;
- deduplicate SMPL-X sequences by normalized pose hash and source metadata;
- reject any path or hash under `data/smplx_gt` or `data/evaluation_from_author`;
- keep all 57 SGNify signs outside train/validation/calibration; and
- publish split manifests and hashes.

### 10.6 Target quality filtering

For pseudo-SMPL-X sources, compute a sequence quality score from:

- 2D reprojection error;
- left/right identity consistency;
- shape stability;
- rotation validity and mesh validity;
- hand bone-length consistency;
- palm-normal jumps;
- wrist/forearm attachment;
- acceleration/jerk outliers;
- interpenetration; and
- agreement with a second frozen expert where available.

Use three bands:

| Band | Rule | Use |
|---|---|---|
| Q0 clean | no catastrophic flags; top 60% quality | all geometry and diffusion losses |
| Q1 usable | no catastrophic flags; middle 30% | confidence-weighted geometry and observation consistency |
| Q2 weak | bottom 10% or severe disagreement | corrupted input/unlabeled evidence only; never a clean target |

Manually review at least 300 source-disjoint sign clips, oversampling two-hand/contact and high-motion clips. At least 10% must be double reviewed. Catastrophic target error must be below 10%, and contact labels must report inter-reviewer agreement before the full run.

---

## 11. Corruption and reconstruction curriculum

Corrupt observations, not clean targets.

### 11.1 Synthetic corruptions

- 4/8/16-frame complete hand dropout;
- one-finger-chain dropout for 2–8 frames;
- wrist errors of 10–45 degrees;
- independent palm-orientation flips;
- body/hand coupled wrist-break errors;
- left/right identity swaps;
- interacting-hand depth swaps;
- 2D noise scaled to crop resolution;
- crop truncation at 10%, 25%, and 40%;
- motion blur and downsampling where RGB is available;
- body/hand expert disagreement sampled from external validation residuals; and
- frame gaps and variable frame rate.

### 11.2 Real residuals

Use real frozen-expert residual pairs only when the target is independent of the input expert. H32-to-H32 identity pairs are not real correction supervision.

Eligible examples include:

- ARCTIC/InterHand GT versus frozen expert outputs;
- SignAvatars high-quality consensus target versus frozen observations;
- multi-view or stronger offline refined targets versus single-view expert outputs; and
- manually accepted sequences with independent geometry evidence.

The How2Sign H32 cache can teach the sign distribution, but its Phase 2 domain-shift result prohibits relabeling it as exact-A1 residual supervision.

### 11.3 Clean conditioning

At least 20% of batches keep all observations unmasked. This teaches posterior concentration around a correct observation and is required for clean-set safety.

---

## 12. Losses

The starting objective is:

$$
\begin{aligned}
\mathcal{L} ={}&
\lambda_s L_{score}
+ \lambda_R L_{rot}
+ \lambda_V L_{region-vertex}
+ \lambda_J L_{joint}
+ \lambda_F L_{fingertip}
+ \lambda_P L_{palm} \\
&+ \lambda_{rel} L_{relation}
+ \lambda_C L_{contact}
+ \lambda_{slip} L_{slip}
+ \lambda_O L_{observation}
+ \lambda_M L_{motion}
+ \lambda_A L_{anchor}
+ \lambda_D L_{DPoser-distill}.
\end{aligned}
$$

### 12.1 Starting weights

| Loss | Weight |
|---|---:|
| score matching | 1.0 |
| geodesic rotation | 0.5 |
| equal-region vertex | 1.0 |
| joint position | 0.5 |
| fingertips | 2.0 |
| palm normal/orientation | 0.5 |
| relative geometry | 0.5 |
| contact classification/distance | 0.25 |
| contact slip/persistence | 0.10 |
| U0-weighted observation likelihood | 0.50 |
| target velocity/acceleration | 0.25 |
| reliable-observation anchor | 0.10 |
| DPoser-X distillation | 0.25, only when direct score reuse is unavailable |

These are starting values, not benchmark-tuned constants. Change one loss family at a time using external validation and record every experiment.

### 12.2 Loss rules

- score loss is normalized separately for upper body, left hand, and right hand;
- geometry losses use the model's projected `x0` estimate and source-valid masks;
- decoded geometry losses run in FP32 even when the network uses BF16;
- fingertip and palm losses are not diluted by the upper-body vertex count;
- motion loss matches target velocity/acceleration instead of minimizing them toward zero;
- reliable observations receive a stronger identity anchor;
- contact loss uses focal BCE for sparse edges plus a soft distance target;
- penetration and biomechanics remain safety terms, not the primary objective; and
- shape, translation, camera, and face receive no trainable update in the first configuration.

Use SNR-clipped auxiliary geometry weighting with `gamma = 5`. Do not apply large decoded-vertex gradients at the highest-noise timesteps.

---

## 13. Ordered training program

Each stage depends on the previous one. Stop when its gate fails.

### Stage R0: contracts, data, and leakage

Build `cache/phase3/v1` as append-only manifests and relation sidecars referencing immutable Phase 2 clips where possible.

Required outputs:

- dataset/license table;
- source/signer/video-disjoint manifests;
- forbidden-source scan;
- coordinate/joint-order report;
- DPoser-X checkpoint hashes;
- 300-clip manual quality/contact report; and
- deterministic cache rebuild hashes.

**GO:** all splits are disjoint; no SGNify/evaluation leakage; all required checkpoints and data licenses are recorded; catastrophic target failure is below 10%.

**NO-GO:** stop before training. Do not compensate for missing data with benchmark targets.

### Stage R1: DPoser-X spatial prior intake

1. download the official body, hand, and whole-body mixed weights;
2. reproduce an official completion sanity case;
3. implement the explicit 51-joint mapping and normalization adapter;
4. verify frozen-score equivalence; and
5. benchmark direct reuse versus teacher distillation on ARCTIC validation.

**GO:** compatibility criteria in Section 9.2 pass, and DPoser-X initialization improves masked ARCTIC validation by at least 5% over random initialization after an equal 10,000-step budget.

**NO-GO:** use teacher distillation. If neither direct transfer nor distillation helps, train the spatial branch independently and remove the pretrained-prior claim.

### Stage R2: relation/contact graph pretraining

Train first on InterHand and ARCTIC, then adapt on the verified sign subset.

**GO:** on source-disjoint validation:

- hand–hand distance MAE improves at least 10% over a geometry-only MLP;
- depth-order accuracy is at least 80% on non-ambiguous labels;
- contact F1 is at least 0.65 overall and at least 0.60 on sign hand–body edges;
- contact slip decreases at least 15% versus no persistence loss; and
- adding the graph to a frozen reconstruction does not worsen any region over 1%.

**NO-GO:** keep relative geometry tokens but disable the contact energy. Do not force low-quality contact labels.

### Stage R3: masked spatial diffusion

Train on individual frames and short 8-frame clips from clean Tier A/B sources. No temporal adapter beyond the short context, no multi-hypothesis selector.

**GO:** decoded vertex recovery exceeds 30% for every available region under fixed hand/finger/wrist masks, with clean regression below 1% per region.

**NO-GO:** debug state normalization, score mapping, masks, and geometry losses. Do not add long temporal modeling.

### Stage R4: generic temporal-relational diffusion

Train 64-frame windows on ARCTIC plus repaired Motion-X if available. Initialize from R3 and add bidirectional temporal/relation blocks.

**GO:** on untouched ARCTIC validation:

- 4/8/16-frame regional recovery is at least 35% in every available region;
- relational conditioning improves the predefined interaction subset at least 5%;
- clean regional regression is below 1%;
- palm-normal and fingertip trajectory errors improve; and
- generated candidates are finite and non-collapsed.

**NO-GO:** reduce sequence complexity or relation coupling. Diffusion that only lowers jerk is rejected.

### Stage R5: sign-domain adaptation

Fine-tune on the paper-grade mixture. Freeze DPoser-X part priors for the first 20,000 steps, then optionally unfreeze only the fused whole-body module at `2e-5` if external validation improves.

Retain at least 25% clean generic hand/whole-body batches to prevent drift.

**GO:** on source- and signer-disjoint sign validation:

- equal-region error improves at least 3% over the frozen observation initializer;
- no region regresses over 1%;
- the occlusion/interaction subset improves at least 8%;
- contact-conditioned clips improve without non-contact regression over 1%; and
- transition/high-velocity error does not regress.

**NO-GO:** keep R4 as the research result and report sign pseudo-target adaptation as negative. Do not tune on Lane-L.

### Stage R6: reconstruction-residual conditioning

Mix 40% real independent residual pairs, 30% synthetic bursts, 20% clean full observations, and 10% unlabeled observation consistency. Train the observation cross-attention and score residual jointly.

**GO:** external validation meets the R5 thresholds and improves the hard subset by an additional 2% relative to R5.

**NO-GO:** revert to R5. A generic prior that cannot use observations is not a successful reconstruction posterior.

### Stage R7: K-hypothesis evidence selector

Generate `K = 4` candidates from the frozen R6 posterior and train a separate selector. Candidate generation is frozen while training the selector.

Selector inputs:

- held-out 2D/3D observation likelihood;
- U0-weighted reprojection error;
- DPoser-X and RDP score/energy;
- relation/contact consistency;
- fingertip/palm trajectory consistency;
- motion and biomechanical validity; and
- candidate diversity relative to the initializer and other candidates.

Use a listwise softmax ranking loss against equal-region target error on training/validation data. GT is the training label only; it is never an inference feature.

**GO:** selected `K = 4`:

- improves the hard subset at least 2% beyond `K = 1`, or closes at least 25% of the oracle-`K=4` gap;
- does not regress any clean region over 0.5%;
- beats random candidate and minimum-prior-energy selection;
- uses full identical coverage; and
- preserves a nonzero oracle gap, proving that candidates are diverse rather than duplicates.

**NO-GO:** release `K = 1`. Do not report oracle selection as a method result.

### Stage R8: freeze and final evaluation

Freeze all architecture, data mixtures, thresholds, sampling steps, selector weights, and safety rules before opening Lane-L Phase 3 predictions.

Train three fixed seeds: `42`, `123`, and `456`. The final benchmark uses the locked A1 initializer and the immutable 1,493-frame author manifest.

If seed 42 decisively violates effect-size or safety thresholds, stop the other seeds and issue NO-GO. Otherwise complete all three.

---

## 14. Starting optimization configuration

### 14.1 Main network

| Item | Starting value |
|---|---:|
| optimizer | AdamW |
| new-module learning rate | `2e-4` |
| DPoser fused-module LR if unfrozen | `2e-5` |
| selector learning rate | `1e-4` |
| weight decay | `0.05`, excluding norm/bias/embedding |
| warm-up | 5% linear |
| schedule | cosine to 10% of initial LR |
| precision | BF16 network; FP32 rotations/SMPL-X/loss aggregation |
| physical batch | 4–8 windows of 64 frames, selected by memory preflight |
| gradient accumulation | enough for effective batch 32 |
| gradient clipping | global norm `1.0` |
| dropout | `0.1` |
| EMA | `0.9999` |
| workers/CPU threads | at most 4 |
| deterministic validation | enabled with fixed masks/noise/candidate seeds |
| accepted experiment seeds | 42, 123, 456 |

### 14.2 Training budgets

| Stage | Maximum updates | Validation interval | Early stop |
|---|---:|---:|---:|
| R1 adapter/distillation | 10,000 per comparison | 1,000 | 5 validations |
| R2 relation graph | 50,000 | 2,000 | 8 validations |
| R3 masked spatial diffusion | 75,000 | 2,500 | 8 validations |
| R4 temporal-relational pretraining | 150,000 | 5,000 | 8 validations |
| R5 sign adaptation | 100,000 | 2,500 | 10 validations |
| R6 observation posterior | 50,000 | 2,500 | 8 validations |
| R7 selector | 20,000 | 1,000 | 8 validations |

These are ceilings. Profile 1,000 steps before every long run and record throughput, allocated/reserved VRAM, checkpoint size, validation time, and projected wall time. Do not reserve GPU memory merely to meet a memory target.

### 14.3 Checkpoint selection

Use a predeclared equal-region external-validation score:

$$
S = \frac{1}{3}\sum_{r\in\{U,L,R\}}
\frac{E_r^{model}}{E_r^{baseline}}
+ 0.5\sum_r\max(0, E_r^{model}/E_r^{baseline}-1.01)
+ 0.25(1-G_{hard}),
$$

where `G_hard` is clipped hard-subset relative gain. Lower is better. Contact F1 is a gate, not a hidden weight that can trade away geometry.

Save `last`, `best`, EMA, and periodic recovery checkpoints with optimizer, scheduler, scaler, RNG, manifest hashes, resolved config, git SHA, and pretrained checkpoint hashes.

---

## 15. Inference and posterior selection

### 15.1 RDP-Fast

- `K = 1`;
- deterministic probability-flow or equivalent fixed-noise path;
- 20 solver steps initially;
- direct output, with optional 5-step safe refinement only if externally accepted; and
- complete sign in one pass when `T <= 64`.

### 15.2 RDP-Best

- `K = 4` independent posterior candidates;
- 30 solver steps initially;
- four held-out evidence folds;
- evidence-based selector;
- optional 10-step observation refinement after selection; and
- groupwise initializer fallback after the final safety audit.

`K = 2` and `K = 8`, 10/20/30/50 solver steps, and guidance 1.0/1.2/1.5 are predeclared external-validation ablations. The final setting is frozen before Lane-L.

### 15.3 Long continuous sequences

For `T > 64`:

- use 64-frame windows with stride 32;
- share the same clip shape and observation normalization;
- reuse noise for the overlapping frames of a candidate;
- blend rotations with quaternion hemisphere alignment/geodesic averaging;
- blend candidate confidence with Hann weights; and
- assert exact frame coverage after merging.

The isolated author benchmark signs are processed as complete clips.

### 15.4 Baseline candidate

Always include the frozen initializer as candidate zero. This does not guarantee safety by itself; it gives the selector and fail-safe a valid unchanged alternative.

Candidate selection is whole-sequence by default. Groupwise replacement is allowed only when the wrist/forearm seam and relational graph remain valid.

---

## 16. Optional final refinement

The direct posterior output is the primary Phase 3 result. A short refinement is accepted only after a frozen external ablation.

Optimize at most 10 Adam steps over bounded local pose deltas using:

- U0-weighted held and full 2D/3D observations;
- posterior score consistency;
- wrist/forearm attachment;
- soft relation/contact terms;
- target-motion prior from the selected sequence; and
- biomechanics/non-penetration safety.

Shape, translation, global orientation, camera, face, and lower body remain frozen.

**GO:** at least one region improves by 0.2 mm externally; no region regresses over 0.1 mm; fallback remains below 1%; runtime is acceptable.

**NO-GO:** disable refinement and retain direct RDP output. The failed Phase 2 T5 result is not reused.

---

## 17. Safety and fallback

Fall back to the A1 initializer for a group/clip when:

- any parameter, score, uncertainty/reliability, joint, or vertex is non-finite;
- rotation correction exceeds externally frozen body/hand bounds;
- reliable held-out reprojection worsens beyond tolerance;
- palm, fingertip, bone, or wrist attachment becomes invalid;
- predicted contact produces penetration or high slip;
- candidate is outside the training normalization range;
- topology, frame identity, or source hash differs; or
- selector confidence is below its calibrated threshold.

Every fallback records region, frame range, candidate, and cause. More than 1% group-frame fallback on clean external validation or final Lane-L is a NO-GO.

---

## 18. Evaluation package

### 18.1 Baselines

| ID | Configuration | Question |
|---|---|---|
| A0 | original `method_hamer` | historical reference |
| A1 | frozen ensemble + HaMeR fallback | strongest accepted initializer |
| P2 | best preserved Phase 2 direct output | does deterministic refinement help? |
| R0 | DPoser-X per-frame prior only | how much comes from the public spatial prior? |
| R1 | masked diffusion without temporal/relation blocks | does generative spatial completion help? |
| R2 | temporal diffusion, no relation/contact | does whole-sequence posterior help? |
| R3 | R2 + relative graph, contact disabled | do relative features help? |
| R4 | R3 + probabilistic contact | does contact add value safely? |
| R5 | R4, `K = 1` | accepted one-path posterior |
| R6 | R4, `K = 4` + selector | does multi-hypothesis inference add value? |
| R7 | R6 + optional final refinement | is optimization still useful? |

### 18.2 Metrics

Primary:

- upper-body excluding face, left-hand, and right-hand author-style TR-V2V;
- exact frame/region coverage;
- per-sign paired difference;
- sign-clustered bootstrap 95% confidence interval;
- mean, median, and worst-decile sign error; and
- equal-region relative gain.

Posterior/relational diagnostics:

- masked 4/8/16-frame regional recovery;
- fingertip trajectory and palm-normal geodesic error;
- wrist/forearm attachment error;
- hand–hand and hand–body relative distance error;
- depth-order accuracy;
- contact precision, recall, F1, and slip;
- candidate pairwise diversity and oracle gap;
- selector top-1 accuracy and fraction of oracle gap closed;
- held-out observation NLL/risk coverage;
- MPJVE, acceleration error, and jerk;
- clean, blur, occlusion, interaction, and high-velocity subsets;
- fallback rate and causes; and
- runtime, peak VRAM, checkpoint size, steps, and `K`.

### 18.3 Statistical unit

Bootstrap by sign/clip. Never bootstrap individual vertices or frames as independent observations.

### 18.4 Evaluation lanes

- **External development:** ARCTIC, InterHand, SignAvatars/How2Sign validation. All tuning happens here.
- **Cross-language/generalization:** PHOENIX qualitative/2D consistency and an unseen sign-language subset where available.
- **Locked Lane-L:** 57 author signs / 1,493 frames, opened only after freeze.

---

## 19. Master Phase 3 GO/NO-GO gates

| Gate | GO requirements | If NO-GO |
|---|---|---|
| P3-G0 contracts/data | no leakage; disjoint splits; hashes/licenses; manual failure <10% | stop before training |
| P3-G1 pretrained prior | score adapter passes equivalence; pretrained beats random by ≥5% at equal budget | distill or train independent prior |
| P3-G2 relations/contact | relation MAE gain ≥10%; contact F1 ≥0.65 overall/≥0.60 sign; no region >1% regression | disable contact, retain safe relation features |
| P3-G3 masked spatial recovery | ≥30% recovery every region; clean regression <1% | fix representation/data; no temporal model |
| P3-G4 temporal posterior | ≥35% 4/8/16 recovery; interaction gain ≥5%; clean <1% regression | simplify temporal/relation model |
| P3-G5 sign adaptation | equal-region gain ≥3%; no region >1% worse; hard subset ≥8% | keep generic result; reject pseudo-target adaptation |
| P3-G6 observation posterior | R5 thresholds plus ≥2% additional hard gain | revert to sign prior without observation fine-tune |
| P3-G7 K-selection | ≥2% hard gain beyond K1 or ≥25% oracle gap closed; clean <0.5% regression | ship K1 |
| P3-G8 locked author benchmark | criteria below all pass | Phase 3 NO-GO; do not start Phase 4 as a claimed progression |

### P3-G8 exact locked requirements

Relative to the frozen A1 initializer:

- exactly 1,493 author frames and the expected left-hand population;
- no missing or extra prediction frame;
- no region regresses by more than 0.20 mm in any accepted seed;
- at least two regions improve with sign-clustered 95% CI excluding zero in every seed;
- equal-region relative gain is at least 3% in every seed;
- predefined occlusion/interaction/disagreement subset gain is at least 8%;
- clean low-uncertainty regression is below 1% in every region;
- fallback is below 1% of group-frames;
- three seeds complete; and
- cross-seed regional standard deviation is below 0.20 mm.

Passing P3-G8 authorizes Phase 4 phonology/phase work. Failing it preserves the best A1/RDP artifacts as a negative result and redirects effort to data/observation alignment.

---

## 20. Failure-directed pivot table

| Failure | Likely root cause | Required response |
|---|---|---|
| DPoser score mismatch | normalization/joint order/checkpoint incompatibility | fail closed; use explicit mapping or distillation |
| diffusion improves plausibility but not error | weak observation conditioning or teacher bias | strengthen independent evidence; inspect target quality |
| candidates are nearly identical | posterior collapse/guidance too strong | reduce guidance, verify independent noise and masks |
| candidates are diverse but all bad | prior/domain mismatch | improve sign adaptation; do not enlarge `K` |
| oracle K helps but selector does not | weak held-out evidence/ranker | improve selector features or ship K1 |
| relation graph helps contact but hurts hands | false contact or over-coupling | disable contact energy; retain relative tokens |
| body improves, hands worsen | region loss imbalance/generic data dominance | rebalance regions and sign/hand sampling |
| hands improve, body worsens | wrist/forearm seam or cross-part attention | freeze torso, reduce cross-part update |
| clean observations drift | insufficient full-observation batches/anchor | increase clean share and reliable anchor |
| How2Sign improves, ARCTIC/InterHand regress | pseudo-teacher overfit | lower H32 target weight; retain clean data |
| external GO, Lane failure | initializer/domain transfer mismatch | report NO-GO; never tune on Lane |
| high fallback | selector or safety mismatch | recalibrate externally; prefer candidate zero/K1 |
| slow sampling with no K gain | excessive steps/hypotheses | reduce solver steps or ship RDP-Fast |

---

## 21. Required ablations

### 21.1 Pretraining

- random spatial branch vs DPoser-X direct score vs DPoser-X distillation;
- public DPoser-X body vs local sign-body torso adapter;
- with/without optional Phase 2 ARCTIC T1 initialization;
- primary DPoser-X route vs FUSION only if permission permits; and
- frozen DPoser fused module vs last-stage low-LR adaptation.

### 21.2 Architecture

- per-frame vs temporal;
- 16/32/64-frame context;
- causal vs bidirectional;
- no relation vs relation features vs relation + contact;
- no wrist/palm tokens;
- one hand at a time vs joint two-hand modeling;
- absolute score vs residual score around DPoser-X; and
- 6 vs 8 blocks, changed only if the starting model underfits externally.

### 21.3 Data and corruption

- ARCTIC/InterHand only vs adding How2Sign vs adding SignAvatars;
- 10/25/50/100% SignAvatars scale;
- no burst masks vs 4/8/16 masks;
- no hand swap/depth ambiguity;
- pseudo-target weight 0.1/0.25/0.5;
- generic-data share 15/25/40%; and
- U0 vs H32-domain U1 diagnostic only.

### 21.4 Sampling and selection

- `K = 1, 2, 4, 8`;
- 10/20/30/50 solver steps;
- guidance 1.0/1.2/1.5;
- random vs prior-energy vs evidence selector vs oracle;
- no evidence holdout vs 20% held-out evidence;
- candidate zero excluded/included; and
- final refinement off/on.

Every ablation reports all three regions and the hard subset. Do not report only contact F1 or only oracle-`K`.

---

## 22. Additive implementation layout

Create a new package. Do not add Phase 3 branches throughout Phase 2 or DexAvatar fitting code.

```text
phase3_posterior/
  README.md
  config.py
  provenance.py
  configs/
    rdp_r2_relation.yaml
    rdp_r3_spatial_diffusion.yaml
    rdp_r4_temporal_generic.yaml
    rdp_r5_sign_adaptation.yaml
    rdp_r6_observation_posterior.yaml
    rdp_r7_selector.yaml
  data/
    cache_schema.py
    build_phase3_index.py
    build_relation_targets.py
    quality_filter.py
    evidence_split.py
    dataset.py
    corruptions.py
  geometry/
    relation_anchors.py
    contact.py
    state_adapter.py
  models/
    dposer_adapter.py
    relation_graph.py
    contact_head.py
    temporal_score.py
    relational_diffusion.py
    evidence_selector.py
  losses/
    diffusion.py
    geometry.py
    relation.py
    selector.py
  train_relation.py
  train_diffusion.py
  train_selector.py
  sample.py
  infer.py
  render.py
  evaluate.py
  gates.py
  tests/
```

Stable Phase 2 utilities may be imported read-only. If a shared utility needs a behavior change, copy or generalize it only after regression tests prove all Phase 2 behavior remains unchanged.

### 22.1 Artifact layout

```text
cache/phase3/v1/
  manifest.json
  sources/*.json
  splits/{train,val,calibration,test}.json
  relations/<source>/<clip>.npz
  quality/<source>/<clip>.json

outputs/phase3_training/<experiment>/
  resolved_config.json
  provenance.json
  best.pt
  last.pt
  checkpoints/
  validation.jsonl

outputs/phase3_gates/<gate>/<experiment>/
  decision.json
  summary.json
  per_clip.jsonl
  per_frame.csv
  hashes.sha256

outputs/phase3_rdp_<mode>/
  <sign>/smplifyx/results/*.pkl
  <sign>/smplifyx/meshes/*.obj
  <sign>/phase3_diagnostics/*.json
```

Existing output/cache directories are always read-only.

---

## 23. Command and execution contract

Long jobs run in named tmux sessions with append-only logs and at most four CPU threads.

Example command shapes:

```bash
python -m phase3_posterior.data.build_phase3_index \
  --sources phase3_posterior/configs/data_sources_v1.yaml \
  --output cache/phase3/v1

python -m phase3_posterior.train_relation \
  --config phase3_posterior/configs/rdp_r2_relation.yaml

python -m phase3_posterior.train_diffusion \
  --config phase3_posterior/configs/rdp_r5_sign_adaptation.yaml \
  --init outputs/phase3_training/rdp_r4_temporal_generic/best.pt

python -m phase3_posterior.train_selector \
  --config phase3_posterior/configs/rdp_r7_selector.yaml \
  --posterior outputs/phase3_training/rdp_r6_observation_posterior/best.pt

python -m phase3_posterior.infer \
  --config phase3_posterior/configs/rdp_r7_selector.yaml \
  --cache cache/phase2/lane_l_a1_ensemble_v1 \
  --checkpoint outputs/phase3_training/rdp_r6_observation_posterior/best.pt \
  --selector outputs/phase3_training/rdp_r7_selector/best.pt \
  --output outputs/phase3_rdp_best_seed42
```

Tmux launch pattern:

```bash
tmux new-session -d -s phase3_r5_seed42 \
  "cd /home/haipd/DexAvatar && set -o pipefail && \
   OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
   PYTHONUNBUFFERED=1 python -u -m phase3_posterior.train_diffusion \
   --config phase3_posterior/configs/rdp_r5_sign_adaptation.yaml \
   2>&1 | tee -a logs/phase3/rdp_r5_sign_seed42.txt"
```

Every command writes the git SHA, dirty-worktree status, environment, CUDA/PyTorch versions, resolved config, source manifests, pretrained hashes, seed, and output hashes.

---

## 24. Required tests before full training

### 24.1 Unit tests

- Phase 2 axis-angle/matrix/6D round-trip remains unchanged;
- 6D forward-noise and `SO(3)` projection are finite;
- DPoser-X joint-order and normalization round-trip;
- frozen-score adapter equivalence;
- source-specific supervision masks;
- relation node/edge ordering and handedness;
- palm-normal direction under mirroring;
- contact hysteresis and slip;
- conditioning/evidence split disjointness;
- masked diffusion never treats a missing label as zero;
- candidate noise independence and deterministic replay;
- selector cannot access GT fields at inference;
- quaternion/geodesic overlap merge near `pi`;
- candidate-zero identity; and
- result PKL/topology/coverage contracts.

### 24.2 Integration tests

- one clean clip reconstructs through cache -> score -> sample -> PKL -> mesh;
- residual-score disabled reproduces the frozen DPoser score;
- a fixed 8-frame hand mask is recovered on ARCTIC;
- relation conditioning changes only valid edges;
- false contact cannot force a pose when contact probability is low;
- `K = 4` emits four distinct reproducible candidates;
- held-out evidence selection never reads held-out GT;
- `K = 1` and candidate zero are valid fallbacks;
- 64-frame BF16 training has finite FP32 geometry gradients;
- repeated inference with one seed/config has identical hashes; and
- strict evaluation rejects missing/stale/duplicate frames.

### 24.3 Red-team tests

- no hands for an entire clip;
- both hands fully overlap;
- left/right identity swap;
- palm orientation flipped by 180 degrees;
- wrong camera intrinsics;
- invalid DPoser checkpoint hash;
- corrupted normalization statistics;
- frame gaps and variable FPS;
- NaN expert input;
- all contact logits high;
- all candidates collapse to one pose;
- selector confidence outside calibration range;
- stale meshes or extra result files; and
- attempted SGNify path in a train/validation manifest.

Before and after every Phase 3 change:

```bash
ruff check phase2_refiner phase3_posterior
pytest -q phase2_refiner/tests phase3_posterior/tests
python -m compileall -q phase2_refiner phase3_posterior
git diff --check
```

No Phase 3 implementation is accepted if it breaks a Phase 2 test.

---

## 25. Compute and storage strategy

The available RTX 5880 Ada is sufficient for the factorized model, but memory use must be measured rather than maximized.

- start with batch 4, 64 frames, BF16, activation checkpointing;
- increase physical batch only while retaining at least 15% memory headroom;
- target effective batch 32 through accumulation;
- keep CPU threads at or below 4;
- decode only the geometry samples required by the current loss, not every diffusion state;
- cache static relation anchors and DPoser normalization;
- batch `K` candidates along the batch dimension when memory permits;
- retain only `best`, `last`, and predeclared recovery checkpoints after a run is accepted; and
- store large candidate meshes only for gate subsets and final evaluation.

Run a 1,000-step throughput/memory preflight before each stage. A successful preflight is not an accuracy result.

---

## 26. Licensing and reproducibility gates

Record source-code and weight licenses separately for DPoser-X. Record dataset terms for SignAvatars, How2Sign, ARCTIC, InterHand2.6M, Motion-X, and any optional hand source.

WiLoR remains a frozen observation expert and must follow its released restrictions. Do not modify or redistribute derived WiLoR weights.

FUSION is optional until written modification/redistribution permission is archived. The primary RDP checkpoint must remain reproducible without FUSION.

SMPL-X and MANO model files are not redistributed unless their licenses explicitly permit it; setup instructions should require users to obtain them from the official sources.

---

## 27. Definition of done

Phase 3 is complete only when:

1. the Phase 2 NO-GO status remains documented and no failed checkpoint is presented as a validated foundation;
2. the public DPoser-X whole-body prior is hash-pinned and either directly integrated or truthfully distilled;
3. every data source has a license, split, provenance, and quality report;
4. SGNify/evaluation leakage checks pass;
5. the relational graph passes geometry/contact gates or contact is explicitly disabled;
6. masked diffusion passes decoded 4/8/16-frame recovery and clean-safety gates;
7. sign-domain posterior conditioning improves external geometry, not only smoothness;
8. `K > 1` is used only if the evidence selector passes its gate;
9. every output frame has a standard PKL, 10,475-vertex mesh, and diagnostic record;
10. the accepted final setting is frozen before Lane-L;
11. all three final seeds pass P3-G8, or an explicit Phase 3 NO-GO is issued;
12. all Phase 2 and Phase 3 tests pass; and
13. configs, checkpoints, manifests, logs, metrics, and hashes are documented.

---

## 28. Recommended immediate execution order

1. freeze and tag the current Phase 2 branch/artifacts for later reuse;
2. create the isolated `phase3_posterior/` skeleton and Phase 2 regression test command;
3. download and hash the official DPoser-X body/hand/whole-body mixed weights;
4. implement and pass the DPoser 51-joint/normalization compatibility gate;
5. materialize Phase 3 relation sidecars for ARCTIC and InterHand;
6. build the 300-clip external relation/contact audit set, excluding SGNify;
7. train and gate R2 relation/contact;
8. train R3 masked spatial diffusion and stop unless decoded recovery passes;
9. train R4 temporal-relational diffusion on clean generic sequences;
10. acquire and audit SignAvatars in parallel with the pilot, or designate an equivalent sign SMPL-X source;
11. train R5 sign adaptation and R6 observation posterior;
12. train R7 selector only after the posterior produces a useful oracle gap;
13. freeze all settings and run one seed-42 locked check;
14. if seed 42 remains viable, run seeds 123 and 456; and
15. append the complete numerical Phase 3 report to this document before deciding whether Phase 4 is authorized.

---

## 29. Final recommendation

The recommended primary Phase 3 configuration is:

> **Frozen A1 observations at evaluation + fixed U0 reliability + public DPoser-X whole-body spatial score + independently trained bidirectional temporal-relational adapters + probabilistic contact + masked sub-VP diffusion + evidence-selected `K = 4`, with candidate-zero and groupwise safety fallback.**

The ready-now local data are sufficient to implement and test the full vertical slice and to run a serious pilot. The final paper-grade model should wait for an audited SignAvatars or equivalent sign SMPL-X corpus, because How2Sign H32 alone repeats the teacher/domain weakness that blocked Phase 2.

If relations help but `K > 1` does not, release a relational `K = 1` posterior. If diffusion does not beat the deterministic and initializer baselines on spatial hard subsets, stop Phase 3 and do not hide the failure behind visual smoothness or oracle selection. If P3-G8 passes, proceed to Phase 4 and add phonology/phase as a separately measurable contribution.

---

## 30. Implementation report and readiness review (2026-08-03)

### 30.1 Delivered additive implementation

Phase 3 is implemented as a new isolated package, `phase3_posterior/`. No existing
DexAvatar fitting, expert, Phase 1, or Phase 2 source file was modified. Stable
Phase 2 cache/rotation/render/evaluation functions are imported read-only.

Implemented components:

- fail-closed YAML inheritance/validation, append-only outputs, full git/environment/
  input hashing, RNG-complete checkpoints, EMA, cosine warm-up, BF16 execution,
  gradient accumulation, clipping, and the four-worker cap;
- immutable Phase 2 clip indexes, license/source/signer split checks, forbidden
  SGNify/author-evaluation path rejection, per-clip SHA-256, Q0/Q1/Q2 reports,
  relation sidecars, and a formal P3-G0 cache audit;
- the 51-joint `SO(3)` matrix/6D adapter with region definitions;
- the fixed 32-node hand/body graph, fixed candidate edges, 16-dimensional edge
  features, torso/wrist coordinate transforms, contact hysteresis, persistence,
  depth ordering, and contact heads;
- an explicit frozen-prior interface and fail-closed DPoser-X contract auditor;
  the default route is truthfully marked `from_scratch/no_pretrained_prior`;
- a 43,215,691-parameter temporal-relational score model with eight 384-wide
  blocks, separate body/left/right spatial and temporal attention, wrist/group-only
  cross-part exchange, clipped learned relative temporal bias, relation conditioning,
  and a zero-initialized residual score projection;
- DPoser-compatible continuous sub-VP perturbation, equal-region score matching,
  masked geodesic/motion losses, focal contact/persistence and depth losses, and
  source-specific target weighting;
- the exact 20/12/12/10/10/10/8/6/6/6 corruption mixture, 4/8/16-frame bursts,
  deterministic conditioning/evidence splits, and 10% classifier-free dropout;
- R2 relation, R3-R6 diffusion, and R7 listwise-selector training CLIs with additive
  stage configs and hash-bound initializations;
- deterministic K-hypothesis sampling with independent candidate noise, candidate
  zero identity, U0-weighted observation anchoring, exact long-sequence coverage,
  shared overlap noise, quaternion hemisphere alignment, and Hann blending;
- GT-free selector inference, standard source-anchored SMPL-X result PKL export,
  finite-value fallback, rendering, strict common-manifest evaluation, and
  fail-closed P3-G0 through P3-G8 decision functions; and
- 13 Phase 3 tests covering rotations, graph ordering/features, contact hysteresis,
  leakage rejection, corruption/evidence masks, score-prior identity, gradients,
  deterministic diverse candidates, selector API isolation, and gate boundaries.

The main files are:

| Area | Files |
|---|---|
| Contracts/provenance | `config.py`, `provenance.py`, `training.py`, `README.md` |
| Data | `data/cache_schema.py`, `data/build_phase3_index.py`, `data/audit_phase3_cache.py`, `data/build_relation_targets.py`, `data/quality_filter.py`, `data/dataset.py`, `data/corruptions.py`, `data/evidence_split.py` |
| Geometry | `geometry/state_adapter.py`, `geometry/relation_anchors.py`, `geometry/contact.py` |
| Models | `models/dposer_adapter.py`, `models/relation_graph.py`, `models/contact_head.py`, `models/temporal_score.py`, `models/relational_diffusion.py`, `models/evidence_selector.py` |
| Objectives/sampling | `losses/diffusion.py`, `losses/geometry.py`, `losses/relation.py`, `losses/selector.py`, `sample.py` |
| Execution | `train_relation.py`, `train_diffusion.py`, `train_selector.py`, `infer.py`, `render.py`, `evaluate.py`, `gates.py` |
| Frozen starts | `configs/data_sources_v1.yaml`, `configs/rdp_base.yaml`, `configs/rdp_r2_relation.yaml` through `configs/rdp_r7_selector.yaml` |

### 30.2 Verification evidence

The required repository-wide checks pass:

```text
ruff check phase2_refiner phase3_posterior                         PASS
pytest -q phase2_refiner/tests phase3_posterior/tests              69 passed
python -m compileall -q phase2_refiner phase3_posterior            PASS
git diff --check                                                   PASS
```

The proposal configs all resolve and validate. The frozen starting model contains
43,215,691 trainable parameters, close to the proposed approximately 45M lower
bound without counting an external frozen DPoser-X teacher.

The DPoser intake was deliberately tested with the checked-in template contract.
Artifact:
`outputs/phase3_gates/g1/implementation_dposer_contract_audit.json`, SHA-256
`5090c7404f7a039d53a5fe1f70757e5b5038f5cca198fae42e520e6fc474a537`.
It correctly returns `passed: false`: sub-VP matches, while the official 51-joint
whole-body weights, 6D representation mapping, normalization hash, and license
record are absent. It selects `teacher_distillation_or_from_scratch` and does not
misrepresent the local body-only sign checkpoint.

### 30.3 Review decision: engineering GO, scientific gates remain NO-GO/PENDING

The additive implementation is **OK to begin R0 data materialization and external
pilot preparation**. It is **not yet OK to launch the full paper-grade R2-R8
training chain**, because ordered prerequisites have not passed.

| Gate | Current state | Exact reason |
|---|---|---|
| P3-G0 contracts/data | **NO-GO** | pilot source license IDs are explicitly marked `LOCAL_REVIEW_REQUIRED`; How2Sign signer identity is absent from current cache metadata; the append-only Phase 3 cache and 300-clip manual audit have not been completed |
| P3-G1 pretrained prior | **NO-GO** | formal audit above: 1/5 compatibility checks pass; official whole-body weights/normalizer/license are missing |
| P3-G2 relations/contact | **PENDING** | code is ready, but no R2 training or source-disjoint numerical metrics exist |
| P3-G3 masked spatial | **PENDING** | blocked by ordered G0/G1/G2; no decoded recovery metrics exist |
| P3-G4 temporal posterior | **PENDING** | blocked by G3 |
| P3-G5 sign adaptation | **PENDING** | blocked by G4 and paper-grade sign targets |
| P3-G6 observation posterior | **PENDING** | blocked by G5 |
| P3-G7 K selector | **PENDING** | blocked until a frozen posterior demonstrates a nonzero useful oracle gap |
| P3-G8 author benchmark | **LOCKED / NOT OPENED** | Lane-L remains untouched for tuning and must stay closed until G0-G7 pass |

No Phase 3 model was trained and no accuracy/recovery/contact metric is claimed in
this implementation report. Test success verifies software contracts and backward
compatibility only; it is not a scientific GO result.

### 30.4 Exact next action

Do not skip directly to training. First replace the placeholder license identifiers,
add auditable How2Sign signer IDs or construct a provably signer-disjoint sign split,
install and hash the official DPoser-X whole-body checkpoint/normalizer, and complete
the 300-clip manual target/contact report. Then build `cache/phase3/v1`, run the
formal P3-G0 and P3-G1 audits, and stop if either remains NO-GO. Only after both pass
should R2 be launched in tmux with append-only `logs/phase3/` output and the four-CPU
thread cap.

---

## 31. R0 blocker resolution and R2 launch report (2026-08-03)

This section supersedes the R0 status in Section 30.3. The work remained additive:
no Phase 1/Phase 2 cache, legacy method, author-evaluation input, or Lane-L artifact
was modified or read for tuning.

### 31.1 Correctness fixes

- The deterministic sampler now integrates the exact sub-VP probability-flow ODE,
  including the sub-VP discount and the required one-half score coefficient.
- Observation anchoring is scaled by the absolute integration step, so changing the
  number of sampler steps no longer changes anchor strength by construction.
- A 51-joint conditioning mask is mapped to the fixed 32 relation nodes. Every edge
  with a hidden endpoint is invalidated and zeroed before graph conditioning, and
  hidden edges are excluded from contact/depth supervision.
- The relation graph now has an explicit mixed-precision accumulator contract. A
  CUDA BF16 autocast regression test covers the failure found during launch.
- The Phase 3 read-only adapter initializes all current Phase 2 dataset attributes;
  a real-index item smoke test passed before relaunch.

### 31.2 Data, signer, and geometry result

How2Sign signer identity is parsed from the terminal signer field in the official
clip name. Because the previous official train/validation/calibration manifests
shared signers, Phase 3 uses signer-component splits without changing Phase 2:

| Phase 3 split | signer IDs | clips |
|---|---:|---:|
| train | 3, 5, 8 | 10,643 |
| validation | 1, 2 | 754 |
| calibration | 4, 9, 11 | 420 |

The split has 2,242 source-video groups, zero source-group overlap, and zero signer
overlap. The final append-only cache contains 14,142 train, 1,312 validation, and
420 calibration clips (15,874 total).

Relation inputs and labels use separate providers. ARCTIC and How2Sign initializer
and target poses are decoded independently through the frozen neutral SMPL-X model.
InterHand uses its official world-coordinate 3D joints, retaining true two-hand
placement instead of placing MANO rotations on a neutral SMPL-X body. Contact uses
decoded centre distance minus fixed, recorded anatomical proxy radii; these radii
only label candidate contact and are not reconstruction targets.

| source | node/keypoint coverage | torso coverage | wrist/hand coverage | edge coverage | valid contact labels | positive contact rate |
|---|---:|---:|---:|---:|---:|---:|
| ARCTIC | 100.00% | 100.00% | 100.00% | 100.00% | 95.06% | 0.4892% |
| InterHand2.6M | 75.00% | 20.00% | 100.00% | 23.46% | 20.99% | 7.1087% |
| How2Sign | 100.00% | 100.00% | 100.00% | 100.00% | 95.06% | 1.5358% |

InterHand torso coverage is intentionally limited to the two observed wrists; its
official annotations do not contain a torso, and no torso target was fabricated.

### 31.3 Visual audit and formal P3-G0

A deterministic 300-clip audit sampled 100 clips per source and inspected four
frames per clip in both XY and XZ projections. The reviewer was Codex, delegated by
the project owner. Every evidence image and relation sidecar is hash-bound in
`cache/phase3/v1/manual_quality_300.json`.

- reviewed clips: **300**;
- catastrophic failures: **0**;
- failure rate: **0.0000**, requirement `< 0.10`;
- missing/changed evidence hashes: **0**.

The first formal run was preserved as
`outputs/phase3_gates/g0/r0_cache_audit_failed_float_boundary.json`. Its only
blocker was `0.1999999999999958 < 0.20` from floating-point accumulation. The
comparison was fixed with a `1e-9` numerical tolerance without changing the 0.20
requirement. The repeated audit reports:

| P3-G0 check | result |
|---|---|
| complete relation sidecars and hashes | GO |
| source/signer/video disjoint | GO |
| forbidden author/SGNify source scan | GO |
| recorded license evidence | GO |
| relation coverage contract | GO |
| 300-clip review and `<10%` failure | GO (0.00%) |
| blocker count | **0** |

Formal decision: **P3-G0 GO**.

### 31.4 R1 route and R7 artifact dependency

The external DPoser-X contract remains unavailable, so P3-G1 is not re-labelled as
a pretrained-prior GO. The already declared `from_scratch` fallback route is used,
and no pretrained-prior claim is made. R2 relation pretraining does not consume a
DPoser checkpoint.

`outputs/phase3_training/rdp_r6_observation_seed42/selector_train.npz` cannot be
created before R6: its evidence rows must be sampled from the frozen R6 posterior,
and creating the R6 output directory early would also violate the append-only run
contract. `phase3_posterior.data.build_selector_features` is now implemented to
create the artifact after R6. A synthetic or random-posterior placeholder was not
fabricated and is not an R0 blocker.

### 31.5 R2 training launch

The frozen R2 curriculum is 30,000 generic warm-up steps on ARCTIC + InterHand,
followed by 20,000 joint-adaptation steps. The combined phase retains approximately
25% generic clips. Physical batch is 8, gradient accumulation is 8, AdamW learning
rate is `2e-4`, EMA is `0.9999`, and an atomic `last.pt` is written every 1,000
optimizer steps.

Three fail-closed launch attempts stopped before the first optimizer step: one
exposed the stale Phase 2 adapter attributes and two successive CUDA checks exposed
and then fully resolved the BF16 accumulator type mismatch. Their output directories
and logs are preserved with explicit `failed_*` names. After the fixes and complete
regression suite, the accepted launch is:

- tmux session: `phase3_r2_relation`;
- log: `logs/phase3/rdp_r2_relation_seed42_v4.txt`;
- output: `outputs/phase3_training/rdp_r2_relation_seed42`;
- initial stage: `generic_warmup`;
- step 1 total/contact/persistence/depth loss:
  **0.590228 / 0.124783 / 0.157754 / 1.007812**;
- step 200 total/contact/persistence/depth loss:
  **0.179023 / 0.013691 / 0.017824 / 0.396484**;
- observed Phase 3 GPU allocation: approximately **680 MiB**;
- Phase 3 CPU use at launch: approximately **187%**, below the 500% ceiling.

Validation after the final fixes:

```text
ruff check phase2_refiner phase3_posterior                    PASS
pytest -q phase2_refiner/tests phase3_posterior/tests         85 passed
python -m compileall -q phase2_refiner phase3_posterior       PASS
git diff --check                                               PASS
```

### 31.6 Primary artifact hashes

| artifact | SHA-256 |
|---|---|
| `cache/phase3/v1/manifest.json` | `fa71eb2f82b49689c1c62d611e9b4edac05b84ceeda9b04469790eaec4196581` |
| `cache/phase3/how2sign_signer_splits_v1/report.json` | `60545a8188875c2e73a4d55115d1b7f60437b8833fed0b118e114208bec30782` |
| `cache/phase3/v1/manual_quality_300.json` | `aca58ce9d9e0219eb4f08df1730fa4b3f91913cd99bffb26c120134173ebfb30` |
| `outputs/phase3_gates/g0/r0_cache_audit.json` | `f8a05cd439e79914aff7569ede9152861fb010bb8c109bcf2b55730b8a403df5` |
| `outputs/phase3_gates/g0/decision.json` | `f56784b35c6ad01274598bddcd802f4ef9a91a4e28eb5a8168b172ae56ce6682` |
| Phase 3 data-license evidence | `0bf83eb4b9d7c4d5dfd661211da1da2fda5deb249128d4764a6c5bc58163239f` |
| R2 config | `10f022a1c23a624442b886ea0a1b31471a12a84a8ab15d4a7884ef18f7d9ee07` |

P3-G2 and later accuracy gates remain **PENDING**. Step-1 optimization and P3-G0
GO are readiness results, not claims that relation/contact accuracy or full Phase 3
has passed.

## 32. R2 completion and fail-closed P3-G2 result (2026-08-04)

### 32.1 Completed training artifact

The R2 relation/contact run completed its frozen 50,000-update budget and exited
normally. It used the frozen 30,000-step ARCTIC+InterHand generic warm-up followed
by 20,000 joint-adaptation steps; it did not use Lane-L. The deployable weights are
the checkpoint's EMA state, consistent with `phase3_posterior.training.load_weights`.

| Item | Value |
|---|---|
| checkpoint | `outputs/phase3_training/rdp_r2_relation_seed42/best.pt` |
| completed update | 50,000 |
| final logged training loss | 0.049974 |
| final contact / persistence / depth losses | 0.004957 / 0.003900 / 0.108398 |
| checkpoint SHA-256 | `03bb7bff28a27c44c7745117f8a22943b46f728c3f5eb869ae0f9325b50a10b4` |
| frozen config SHA-256 | `10f022a1c23a624442b886ea0a1b31471a12a84a8ab15d4a7884ef18f7d9ee07` |

### 32.2 Source-disjoint validation measurement

The EMA state was evaluated deterministically on all 1,312 clips in the immutable
source/signer-disjoint validation manifest (511 ARCTIC, 47 InterHand2.6M, and 754
How2Sign clips), with 32-frame windows and the predeclared 0.5 contact threshold.
This is an evaluation of geometry-derived relation sidecars, not a final mesh-error
benchmark.

| Validation subset | Contact precision | Contact recall | Contact F1 | Non-ambiguous depth-order accuracy |
|---|---:|---:|---:|---:|
| all valid contact edges | 0.6373 | 0.7599 | **0.6932** | **0.9873** |
| ARCTIC | 0.5532 | 0.9251 | 0.6924 | 0.9973 |
| InterHand2.6M | 0.9746 | 0.6738 | 0.7967 | 0.9924 |
| How2Sign, all valid edges | 0.6442 | 0.7448 | 0.6908 | 0.9805 |
| How2Sign hand--body edges | 0.5372 | 0.4469 | **0.4879** | n/a |

The all-edge contact confusion counts are TP=30,201, FP=17,189, and FN=9,541.
For the required How2Sign hand--body subset they are TP=101, FP=87, and FN=125.

### 32.3 Formal decision: P3-G2 NO-GO

**NO-GO.** The overall contact-F1 condition passes (0.6932 >= 0.65), and the
proposal's depth-order condition passes (98.73% >= 80%). However, the mandatory
sign hand--body contact F1 is 0.4879, below its 0.60 requirement by 0.1121.

In addition, this first R2 implementation did not train/evaluate the required
geometry-only MLP comparator, relation-distance MAE, no-persistence ablation/contact
slip, or frozen-reconstruction regional-safety measurement. It also saves the final
state as `best.pt` rather than selecting it against a validation score. Those missing
measurements must be implemented before a formal machine-readable P3-G2 decision can
be emitted; they must not be represented as passes. Per the Stage R2 policy, do not
start R3 from this checkpoint. Retain its safe relative-geometry features for a
corrected R2 experiment, and keep contact energy disabled until the sign-contact gate
and all required comparator/safety measurements pass.

## 33. Corrected R2 v2 execution (2026-08-04)

### 33.1 Root causes and additive fixes

The first R2 run was not repaired by threshold tuning. Its implementation lacked
continuous target relation features, left the surface-gap input channel at zero,
diluted extremely sparse sign hand--body positives, and did not implement the
proposal's geometry-only comparator, persistence ablation, periodic validation, or
best-checkpoint selection. The original v1 cache, checkpoint, and configuration are
preserved.

The additive corrected path introduces:

- relation schema v2 with decoded initializer and independent target edge features;
- a nonzero fixed anatomical surface-gap feature and a learned distance residual;
- the frozen `55% How2Sign / 30% ARCTIC / 15% InterHand` joint mixture;
- a 35% positive-contact-clip stratum inside the How2Sign allocation, justified by
  a measured natural positive-clip rate of approximately 9.3%;
- focal positive balancing and a fixed sign hand--body edge weight;
- a geometry-only MLP trained at the same update budget;
- an identically initialized no-persistence graph ablation;
- deterministic validation every 2,000 updates and EMA best-checkpoint selection;
- explicit depth, slip-availability, relation-only safety, and distance-gain gate
  checks; and
- an end-to-end two-step smoke run that exercised optimization, validation, and
  checkpoint serialization before the long run.

No Lane-L data was opened or used for these changes.

### 33.2 Corrected cache audit

`cache/phase3/r2_relation_targets_v2` contains 15,874 clips: 14,142 train,
1,312 validation, and 420 calibration. The fail-closed audit at
`outputs/phase3_gates/g0/r2_relation_targets_v2_audit.json` reports:

| Check | Result |
|---|---:|
| blocker count | **0** |
| clips with relation schema v2 | 15,874 / 15,874 |
| How2Sign independent target provider | 11,817 / 11,817 |
| finite continuous target values | 745,950,384 |
| How2Sign hand--body positive / valid edge-frames | 2,757 / 22,688,640 |
| How2Sign hand--body positive rate | 0.01215% |

Primary hashes:

| Artifact | SHA-256 |
|---|---|
| corrected cache manifest | `b166eae34f4d68528d4b30a578afc28cc1f5a211374096112bfc87f27843cfca` |
| corrected cache audit | `adbcbef71167807e83e01f3767f5561eab17e4356304680065fe5a1d35fef165` |
| accepted v2b configuration | `4609e27c9330b6ec0c4492211f212c21cb30a25e7f3e627b970d95deb6e7a9f4` |

### 33.3 Accepted training run and early validation

An initial 100-step launch was preserved as
`rdp_r2_relation_corrected_v2_seed42_superseded_effective_batch64` after profiling
revealed that inherited accumulation produced effective batch 64 rather than the
proposal's required 32. This was corrected before the first checkpoint, without
observing Lane-L or changing a numerical gate.

The accepted run is:

- tmux: `phase3_r2_relation_v2b`;
- log: `logs/phase3/rdp_r2_relation_corrected_v2b_seed42.txt`;
- output: `outputs/phase3_training/rdp_r2_relation_corrected_v2b_seed42`;
- physical batch / accumulation / effective batch: `8 / 4 / 32`;
- curriculum: 30,000 generic warm-up updates, then 20,000 joint-adaptation updates;
- CPU use: approximately 200%, below the 500% ceiling;
- test baseline: **87 passed**, with Phase 3 lint and compilation passing.

At update 2,000, while still in generic-only warm-up, the first external validation
reported depth-order accuracy 85.52%, contact F1 0.0, sign hand--body F1 0.0,
relation-MAE gain -64.81%, and unavailable slip comparison because the EMA graph had
no true-positive contacts. This is an expected early **NO-GO snapshot**, not a final
decision. The sign-positive curriculum does not begin until update 30,001. The run
continues in tmux; P3-G2 remains **PENDING/NO-GO until a complete checkpoint passes
every formal condition**.

## 34. Corrected R2 v2b completion and formal P3-G2 decision (2026-08-04)

The corrected 50,000-update run completed normally. The predeclared selection rule
chose the EMA checkpoint at update 36,000 rather than the final update. Evaluation
used the immutable v2 source/signer-disjoint validation manifest and the fixed 0.5
contact threshold; Lane-L was not opened.

| P3-G2 condition | Best-checkpoint result | Requirement | Decision |
|---|---:|---:|---|
| relation distance MAE gain over geometry-only MLP | **15.61%** | >=10% | GO |
| overall contact F1 | **0.7049** | >=0.65 | GO |
| How2Sign hand--body contact F1 | **0.4667** | >=0.60 | **NO-GO** |
| depth-order accuracy | **98.43%** | >=80% | GO |
| slip decrease versus no-persistence ablation | **0.72%** | >=15% | **NO-GO** |
| relation-only reconstruction regression | **0.00%** | <=1% | GO |
| slip comparator available | true | true | GO |

The selected graph's contact confusion counts are TP=29,392, FP=14,254, and
FN=10,350 overall. On the required How2Sign hand--body subset they are TP=91,
FP=73, and FN=135. The failure is principally insufficient sign-contact recall,
and persistence changes predicted-contact slip by only 0.72%, far short of the
15% requirement.

**Formal P3-G2 decision: NO-GO.** The corrected relation graph is retained as a
useful relative-geometry component because it passes the distance and overall-contact
conditions. Contact energy remains disabled for later posterior stages: the sign
contact and persistence evidence does not justify enabling it. Do not start R3--R8
as a claimed Phase 3 progression from this checkpoint.

| Artifact | SHA-256 |
|---|---|
| selected `best.pt` (step 36,000) | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| final `last.pt` (step 50,000) | `5ff34cf359795e0918a85c490f0d6993ae0f1cfd2fb116e78af33130c56affaa` |
| source-disjoint evaluation | `0854bf70b8c345f840f6219e7e096b14af9da1437cecfefb99fbdcb4783600cb` |
| formal G2 decision | `5cceb65aca11d026d057d4d172dd435a81de71edc9aac7840d35ab6b986c18e8` |

## 35. Option-A geometry-only R3 progression (2026-08-04)

### 35.1 Frozen fallback decision

P3-G2 remains **NO-GO**. The failed conditions are unchanged: How2Sign hand--body
contact F1 is 0.4667 versus the 0.60 requirement, and predicted-contact slip improves
0.72% versus the required 15%. The selected update-36,000 checkpoint is used only as
a frozen relative-geometry/depth feature extractor. Its contact and persistence
predictions are not accepted downstream.

The diagnostic pipeline ID is `R2_geometry_only_R3_progression`. Its shared fallback
configuration enforces all of the following for R3--R8 descendants:

```yaml
fallback:
  mode: geometry_only
  contact_energy_enabled: false
  force_coupling_enabled: false
  persistence_constraints_enabled: false
model:
  contact_energy_enabled: false
  freeze_relation_backbone: true
loss:
  contact: 0.0
  persistence: 0.0
```

The model removes contact and persistence logits from its fallback output contract,
while retaining `relation_token`, edge tokens, distance, and depth-order predictions.
This makes accidental contact-energy or persistence use fail closed rather than
depending only on a zero scalar weight. The sampler contains no force-coupling or
contact-attraction step.

### 35.2 R3 data and initialization

R3 uses clean Tier A/B sources only. The additive manifest
`cache/phase3/r3_geometry_only_tier_ab_v1` contains 3,499 training clips and 558
validation clips from ARCTIC and InterHand2.6M. Source, signer, and source-group
identities remain disjoint. How2Sign Tier C pseudo targets are excluded from masked
spatial training and Lane-L remains unopened.

Strict initialization loads all 34 relation-backbone tensors from
`outputs/phase3_training/rdp_r2_relation_corrected_v2b_seed42/best.pt`. A two-step
GPU smoke run verified that all 34 tensors remain bit-identical after R3 optimization.

The first launch exposed an unweighted high-noise auxiliary-motion loss before any
checkpoint was written. It is preserved as
`rdp_r3_spatial_geometry_only_seed42_superseded_unweighted_aux`. The implementation
was corrected to apply the proposal's SNR-clipped auxiliary geometry weighting with
`gamma=5`; the revised smoke loss fell from 100.39 to 20.02. A corrected batch-4
pilot began with finite total loss 2.3401, but profiling showed unnecessarily low
throughput. It was stopped before its first checkpoint and preserved as
`rdp_r3_spatial_geometry_only_v2_seed42_superseded_batch4`. The accepted run changes
only the physical/accumulated batch split from 4/8 to 8/4, preserving effective
batch 32 and every optimizer, loss, model, data, and gate setting. Its batch-8 smoke
run completed with finite loss 1.4247, and the accepted long run began with finite
loss 4.5623.

### 35.3 Preflight, launch, and hashes

The revised preflight reports **GO with zero blockers** for the fallback execution
contract. The regression baseline is **90 passed**, with lint, compilation, and the
Phase 3 scoped whitespace check passing.

| Item | Value |
|---|---|
| tmux | `phase3_r3_geometry_only_v3` |
| log | `logs/phase3/rdp_r3_spatial_geometry_only_v3_seed42.txt` |
| output | `outputs/phase3_training/rdp_r3_spatial_geometry_only_v3_seed42` |
| maximum updates | 75,000 |
| physical / accumulated / effective batch | 8 / 4 / 32 |
| relation checkpoint SHA-256 | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| fallback base config SHA-256 | `afb3dd0ab17f4f11dbf5daa64d56c3febb9c884bf427fd408e6f4d0a946a43f2` |
| R3 config SHA-256 | `e1d3a61efe955e004b70f1d7dfcad1ca2ff672f6e0938bd8fee3b44e96d43feb` |
| Tier A/B manifest SHA-256 | `926935184b46c5a922e52bce05d5b6d7170013035a4f693a0681aa171d886a0d` |
| accepted preflight SHA-256 | `0bc6fde6f94d54da4efecf890cab65f2a90f3e9522d89684c32354bc526bd7df` |

R3 training is an authorized diagnostic fallback progression, not a reversal of the
P3-G2 decision. P3-G3 remains pending until decoded masked-recovery and clean-safety
evaluation completes.

The accepted tmux process was confirmed alive through update 200. At that snapshot,
the score loss had decreased from 1.0064 at update 1 to 0.5053; contact energy was
false and the relation backbone was frozen in every logged record. The process used
approximately 1.5 GiB VRAM and approximately 150% aggregate CPU, below the 500% CPU
ceiling. Periodic atomic recovery checkpoints begin at update 1,000.

### 35.4 Separate future R2 retraining backlog

A future contact-focused R2 run is kept separate from this frozen fallback. Before
that run, acquire or double-review sign hand--body contact labels, hard-mine false
negatives and confusing near-contact negatives, use a predeclared contact-positive
curriculum, and train a persistence-specific temporal head against the same-position
no-persistence ablation. It must repeat the complete source-disjoint P3-G2 decision;
no R3 result may retroactively relabel the current R2 contact NO-GO.

## 36. R3 formal evaluation repair and conditional v4b retraining (2026-08-05)

### 36.1 Completed v3 result and formal P3-G3 decision

The geometry-only v3 run completed all 75,000 updates without a numerical error.
Its final checkpoint was evaluated on all 558 immutable Tier A/B validation clips
using 30-step conditional sub-VP sampling, deterministic 35-degree corruptions,
fixed upper-body/hand/finger/wrist masks, and decoded SMPL-X regional vertices.
The corrected evaluator clamps every non-corrupted initializer joint, including
unlabelled body ancestors in hand-only InterHand samples.

| P3-G3 condition | v3 result | Requirement | Decision |
|---|---:|---:|---|
| upper-body recovery | **-105.99%** | >=30% | **NO-GO** |
| left-hand worst-mask recovery | **-177.32%** | >=30% | **NO-GO** |
| right-hand worst-mask recovery | **-239.48%** | >=30% | **NO-GO** |
| maximum clean regression | **0.00%** | <1% | GO |
| validation coverage | **558 / 558 clips** | 558 / 558 | GO |

**Formal P3-G3 decision for v3: NO-GO.** R4 remains blocked. The negative values
mean that the sampled result increased, rather than removed, the injected regional
error. Clean preservation passes because fully observed joints are restored exactly
after reverse integration.

### 36.2 Root cause

`joint_valid` in these caches describes optional decoded 3D joint-position targets;
it is not pose-observation validity. It is 0% for upper body, left hand, and right
hand across the 558-clip validation split. The original R3 trainer used this field
to construct the conditioning mask, so every observation token was zero and every
relation edge was removed. The 75,000-update v3 model therefore learned an
unconditional score model despite its masked-spatial configuration.

Two additional correctness repairs were required:

1. reverse diffusion now passes the joint mask into the score network, removes
   relation edges touching hidden endpoints, follows a fixed forward-noise path for
   observed joints, and restores those joints exactly at the end; and
2. hand-only samples retain all non-corrupted initializer rotations as conditioning,
   even where target supervision is unavailable, preventing random body ancestors
   from translating otherwise clean hand vertices.

The evaluator fail-closes on incomplete coverage and injects deterministic
corruptions before computing recovery, avoiding the undefined zero-error baseline
that results from the clean Tier A/B identity targets.

### 36.3 Validation-based checkpoint selection

Future R3 training now runs deterministic EMA validation every 2,500 updates,
writes an immutable `validation_<step>.json`, and updates `best.pt` only when the
predeclared equal-region score improves. `last.pt` is no longer copied blindly to
`best.pt`. Eight consecutive non-improving validations trigger early stopping.
The in-training selector uses a fixed masked SO(3) proxy for affordable checkpoint
ranking; the final P3-G3 decision still requires the complete decoded-vertex
evaluation above.

### 36.4 Accepted v4b corrective run

The first conditional warm-start pilot exposed dormant random observation/relation
projection weights: its step-1 loss was 72.63 and it was stopped before a checkpoint.
It is preserved as
`rdp_r3_spatial_geometry_only_v4_seed42_superseded_uncalibrated_conditioning`.
For v4b, only those two projection weight matrices are initialized to zero; their
biases and the complete learned v3 prior remain unchanged. This makes update zero
exactly reproduce the stable unconditional score while allowing conditioning to
learn. The calibrated smoke step had total loss 0.1085 instead of 72.63.

The accepted run is active with:

| Item | Value |
|---|---|
| tmux | `phase3_r3_geometry_only_v4b` |
| log | `logs/phase3/rdp_r3_spatial_geometry_only_v4b_seed42.txt` |
| output | `outputs/phase3_training/rdp_r3_spatial_geometry_only_v4b_seed42` |
| initialization | v3 EMA geometry prior plus exact frozen R2 step-36,000 graph |
| first update total / score loss | `0.12046 / 0.01263` |
| contact energy | disabled |
| relation backbone | frozen |
| regression suite | **31 passed**, lint and compilation passed |
| final preflight | **GO, 0 blockers, 16/16 checks** |

P3-G3 remains **NO-GO/PENDING** until the v4b validation-selected checkpoint passes
the complete 558-clip decoded evaluation. Do not start R4 from v3 or from an
unevaluated v4b checkpoint.

| Artifact | SHA-256 |
|---|---|
| v3 `best.pt` | `f3825c40e7cc00bd318cccaef2b6eaae9efeca40a43402be2068b88b6aec6e14` |
| corrected full v3 G3 evaluation | `f28328d35b62e749cb0bd8d28e327c2839ad7c370f4086a55087d743b4dafd05` |
| corrected formal v3 G3 decision | `fdb65ddd6be2c2475b59ac5f024a30aa92a9f528e0626e9523dc18f13db5128d` |
| accepted v4b configuration | `c06eff40e7767902164817cb92a002a5f31de6057932121af562aaf50372b003` |
| accepted v4b preflight | `cd3b5660159658f232be85a5f483cb64ff6f48aeebaaffdfb4fb36347148c13b` |

### 36.5 v4b completion and formal P3-G3 decision

The corrected v4b run stopped at update 30,000 under the predeclared patience rule.
The SO(3) checkpoint-selection score improved through update 10,000, then failed to
improve for eight consecutive 2,500-update validations. Therefore `best.pt` is the
EMA checkpoint at update 10,000; `last.pt` is the stopped update-30,000 state.

| Validation item | Result |
|---|---:|
| best update | **10,000** |
| best selection score | **2.46289** |
| update-10k upper-body proxy recovery | -64.58% |
| update-10k left-hand proxy recovery | -158.10% |
| update-10k right-hand proxy recovery | -141.18% |
| validations without improvement at stop | 8 |

The selected checkpoint was evaluated with the complete immutable 558-clip,
30-step, decoded-SMPL-X protocol:

| P3-G3 condition | v4b result | Requirement | Decision |
|---|---:|---:|---|
| upper-body recovery | **-52.81%** | >=30% | **NO-GO** |
| left-hand worst-mask recovery | **-171.05%** | >=30% | **NO-GO** |
| right-hand worst-mask recovery | **-158.42%** | >=30% | **NO-GO** |
| maximum clean regression | **0.00%** | <1% | GO |
| validation coverage | **558 / 558 clips** | 558 / 558 | GO |

Mask-level decoded errors were:

| Mask | Initial corruption | v4b prediction | Recovery |
|---|---:|---:|---:|
| upper body | 186.31 mm | 284.69 mm | -52.81% |
| left full hand | 9.98 mm | 27.05 mm | -171.05% |
| left finger chain | 2.05 mm | 4.64 mm | -126.44% |
| left wrist attachment | 25.51 mm | 44.13 mm | -73.03% |
| right full hand | 10.10 mm | 26.11 mm | -158.42% |
| right finger chain | 2.01 mm | 4.47 mm | -122.15% |
| right wrist attachment | 25.33 mm | 44.26 mm | -74.78% |

**Formal P3-G3 decision for v4b: NO-GO.** Conditional training materially reduced
several decoded prediction errors relative to v3, most notably upper body, right
full hand, both wrist-attachment cases, and both finger chains. It nevertheless
did not recover the injected pose in any required region. The remaining failure is
not a coverage or clean-safety issue. It is a posterior-quality failure: the current
zero spatial prior and Gaussian-to-pose probability-flow path produce samples worse
than the modest 35-degree corrupted initializer. Do not proceed to R4. The next R3
attempt must de-risk the spatial prior/score mapping and an initializer-centred
conditional bridge on external Tier A/B validation without changing the gate.

| Artifact | SHA-256 |
|---|---|
| v4b selected `best.pt` (update 10,000) | `30c312674214ac4f32d25b5d1012600e52689d9c31862c0715463cc6649d75b4` |
| v4b stopped `last.pt` (update 30,000) | `948c76d1f8a570fca158c9dec511f053c98326e0416b7be58d83705397a6f8ba` |
| v4b append-only training log | `f1bdf3f5942b5b153df4c70d8d4ac05bfd4880917b8fbdb4167dc4ef1c797ee2` |
| v4b full decoded G3 evaluation | `105023d0d79af2e2f3d3c0ad29acafc1cbd1f086ffdc65188fdb2109fd066203` |
| v4b formal G3 decision | `2c32bb5cc63d930e4f0ff1de88261c317379421c631a18f8a014e9d869876b2a` |

## 37. R3 initializer-centred residual completion (v5, 2026-08-05)

### 37.1 Root cause and corrective boundary

The v4b gate baseline and the learned posterior did not solve the same difficulty.
The P3-G3 evaluator perturbs selected initializer rotations by at most 35 degrees,
so its input is already a strong near-target estimate. In contrast, v4b erased the
corrupted rotations, velocities, and accelerations and initialized those joints from
Gaussian noise. It therefore attempted unconditional completion of the hardest
regions and was structurally unlikely to beat the mildly corrupted initializer.
Sampler tuning cannot repair this information mismatch.

V5 implements conditional residual completion without exposing clean targets. The
evaluator and trainer first inject the same bounded SO(3) corruption into the
initializer, recompute its rotation/motion features, and expose only those corrupted
features through a distinct `corruption_observation` projection. The normal
observation path remains restricted to trusted joints, relation edges touching a
hidden endpoint remain masked, and the two masks must be disjoint. Target-invalid
but non-corrupted initializer joints remain trusted so that hand-only samples retain
their body ancestors. The new projection is zero-initialized; consequently the v5
model exactly reproduces the selected v4b posterior before learning the residual
hint. A 5,000-update head-only warm-up prevents the pretrained geometry prior from
drifting while that input path calibrates.

This repair does not weaken any gate or use Lane-L. Training and selection remain on
the immutable Tier A/B ARCTIC/InterHand split. Contact energy, contact loss, force
coupling, persistence constraints, and persistence loss remain disabled; the R2
step-36,000 relation model remains a frozen geometry/depth feature extractor.

### 37.2 Verification and launch

The implementation passes Ruff, bytecode compilation, and **34/34 Phase 3 tests**.
New regression coverage verifies corruption/conditioning-mask disjointness, CFG
dropout behavior, zero initialization, finite sampling, and nonzero gradient flow to
the hint projection. A two-update GPU smoke run completed with finite losses and the
hint-only optimization path active.

The fail-closed v5 preflight reports **GO with 0 blockers and 16/16 checks passing**:
3,499 Tier A/B training clips, 558 source/signer/source-group-disjoint validation
clips, strict frozen relation initialization, validation-based checkpoint selection,
and the complete geometry-only fallback contract.

| Item | Value |
|---|---|
| tmux | `phase3_r3_geometry_only_v5` |
| append-only log | `logs/phase3/rdp_r3_spatial_geometry_only_v5_seed42.txt` |
| output | `outputs/phase3_training/rdp_r3_spatial_geometry_only_v5_seed42` |
| initialization | selected v4b update-10,000 checkpoint |
| frozen geometry extractor | R2 step-36,000 checkpoint |
| hint-only warm-up | 5,000 updates |
| first update total / score loss | `0.09333 / 0.01236` |
| first update hint fraction | `0.09804` |
| contact energy / relation backbone | disabled / frozen |
| CPU cap | 4 threads |

| Artifact | SHA-256 |
|---|---|
| v5 config | `587d998dafed8bcb7f6061a534bff82b5fcd30a7e3396d4b6ee8847dfde6b256` |
| v5 preflight | `17c40b6c76dacfb25da1c3057bb9dcc6133b7fd5499b244c171ca65bd00b371f` |
| v4b initialization checkpoint | `30c312674214ac4f32d25b5d1012600e52689d9c31862c0715463cc6649d75b4` |
| frozen R2 geometry checkpoint | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| frozen P3-G2 decision | `5cceb65aca11d026d057d4d172dd435a81de71edc9aac7840d35ab6b986c18e8` |

P3-G3 remains formally **NO-GO** until a validation-selected v5 checkpoint passes
the complete 558-clip decoded evaluation: at least 30% recovery in upper body and
both worst-case hands, less than 1% clean regression, and full coverage. R4 remains
blocked. This launch is a root-cause correction, not a claim that the numerical gate
has already passed.

### 37.3 Completed v5 training and formal P3-G3 result

V5 early-stopped at update 70,000 after eight consecutive validation checks without
improvement. The immutable validation selector chose the EMA checkpoint at update
50,000, before the later regressions. Its 10-step SO(3) proxy recovered 56.03% upper
body, 44.97% left hand, and 44.78% right hand, with selection score 0.642568 and no
clean regression.

The selected checkpoint was then evaluated once with the complete 558-clip Tier A/B
validation set, seven formal masks, 30 reverse steps, seed 3042, and decoded SMPL-X
regional vertices. Lane-L was not opened.

| P3-G3 condition | v5 result | Requirement | Decision |
|---|---:|---:|---|
| upper-body recovery | **65.88%** | >=30% | **GO** |
| left-hand worst-mask recovery | **48.29%** | >=30% | **GO** |
| right-hand worst-mask recovery | **48.55%** | >=30% | **GO** |
| maximum clean regression | **0.00%** | <1% | **GO** |
| validation coverage | **558 / 558 clips** | 558 / 558 | **GO** |

The complete decoded mask results are:

| Mask | Initial corruption | v5 prediction | Recovery |
|---|---:|---:|---:|
| upper body | 187.12 mm | 63.84 mm | **65.88%** |
| left full hand | 9.95 mm | 5.08 mm | **48.91%** |
| left finger chain | 2.04 mm | 1.05 mm | **48.29%** |
| left wrist attachment | 25.72 mm | 11.65 mm | **54.69%** |
| right full hand | 10.04 mm | 5.16 mm | **48.55%** |
| right finger chain | 2.06 mm | 1.04 mm | **49.33%** |
| right wrist attachment | 25.64 mm | 11.88 mm | **53.68%** |

**Formal P3-G3 decision: GO.** This resolves the masked-spatial recovery blocker.
R4 may now begin under the same geometry-only fallback boundary: contact energy,
contact loss, force coupling, and persistence constraints remain disabled. The
P3-G2 contact decision remains NO-GO and is not overridden by this result.

| Artifact | SHA-256 |
|---|---|
| v5 selected `best.pt` (update 50,000) | `9c871f259be4be3b8c4f1d3dfe368a175a8b50c760626c230dc15c3a3a1b3fc3` |
| v5 stopped `last.pt` (update 70,000) | `6885963402d20b19d5817fd726f6a50408c7006d4e28b62b5ff73b7e5461e5b1` |
| v5 append-only training log | `1ed3d00e5910d8c67773c7f9e86b768c71943a1883e3bca7ef2b941861e5ab95` |
| v5 formal evaluation log | `4d2d2dac76b0b9d58881574124cec6bf8a368d0862d3b2fe402ce86350c84b11` |
| v5 full decoded P3-G3 evaluation | `e493ec07b1706a053cd9058bae4702f8931b61aa8679b10805d6a197268cb475` |
| v5 standalone P3-G3 decision | `c920917ed4cfe37c97cb2e6b0271739b0c6bb5f8f0da618bd00d670ba6049fa4` |

## 38. P3-G2 temporal-contact recovery strategy (v3, 2026-08-06)

### 38.1 Frozen starting point and failure diagnosis

This is a separate contact-retraining branch. It does not modify or invalidate the
geometry-only R3 GO checkpoint. The source P3-G2 decision remains NO-GO with overall
contact F1 0.7049, sign hand--body F1 0.4667, and only 0.72% slip improvement. The
v2b update-36,000 checkpoint is the immutable initialization.

The observed failures have three direct causes:

1. **Extreme within-domain imbalance.** How2Sign training contains 2,506 positive
   hand--body edge-frames among 20,434,560 valid ones (0.0123%). The old 4x domain
   weight multiplies positive and negative sign edges equally and therefore does not
   balance this classification problem. Validation has only 226 positives.
2. **Frame-independent persistence.** The v2b graph processes every frame
   independently. Its persistence head has no temporal state, despite persistence
   being defined across adjacent frames.
3. **Objective/inference disconnect.** Persistence was supervised over all valid
   edges, where the trivial non-contact class dominates, and its logit was never used
   to select contacts. Slip could improve only indirectly through shared features.
   Conditional on a true sign hand--body contact, 49.1% of training labels are
   persistent, which supplies a much healthier temporal target.

Threshold sweeps on the old model are retained only as diagnosis. V3 does not tune a
threshold on the formal validation set: the threshold is frozen at 0.5 before the
new run.

### 38.2 Additive v3 architecture and losses

The geometry/depth relation backbone is loaded from v2b and permanently frozen. Its
distance, depth, edge-token, and relation-token outputs bypass the new contact
adapter, so the already-passing 15.61% distance-MAE gain and 98.43% depth accuracy
cannot drift during contact recovery.

Each fixed edge receives a bidirectional GRU over its 32-frame sequence. A
zero-initialized residual projection makes the temporal adapter an exact identity at
update zero. A copied contact/persistence head is then optimized with:

- the original overall focal contact loss;
- a separately normalized sign hand--body loss containing all positives and at most
  eight hardest negatives per positive;
- contact-conditional persistence focal loss, evaluated only on true contacts; and
- explicit persistence-aware scoring
  `guided_contact_logit = contact_logit + 2 * persistence_logit`.

The no-persistence comparator has the same temporal architecture, the same frozen
v2b initialization, sampling, optimizer, and contact losses. Its only differences
are zero persistence loss and zero fusion weight. This makes the slip ablation
causal rather than comparing unrelated checkpoints.

### 38.3 Data, curriculum, and frozen hyperparameters

The source/signer/source-group-disjoint v2 relation cache is reused without mutation.
Lane-L and author evaluation data remain forbidden. Training samples 70% How2Sign,
20% ARCTIC, and 10% InterHand; 65% of sampled How2Sign clips are contact-positive.
This changes sampling frequency but never relabels an edge.

| Parameter | Frozen value |
|---|---:|
| maximum updates | 20,000 |
| physical / accumulated / effective batch | 8 / 4 / 32 |
| learning rate / weight decay | 1e-4 / 0.01 |
| temporal hidden width | 128 bidirectional |
| sign-contact loss weight | 4.0 |
| hard negatives per sign positive | 8 |
| conditional persistence weight | 1.0 |
| persistence fusion weight | 2.0 |
| contact threshold | 0.5, frozen |
| validation interval / patience | 1,000 / 8 validations |
| CPU workers | 4 |

Checkpoint selection remains fail closed on the complete P3-G2 vector. Formal GO
still requires relation MAE gain >=10%, overall contact F1 >=0.65, sign hand--body F1
>=0.60, depth accuracy >=0.80, slip gain >=15%, a valid no-persistence comparator,
and <=1% regional regression. No individual metric may substitute for this decision.

### 38.4 Verification and launch readiness

Ruff, compilation, and **36/36 Phase 3 tests** pass. Tests cover exact identity
initialization, frozen geometry outputs, finite stratified contact gradients, and
contact-conditional persistence gradients. The two-update GPU smoke completed a
full 1,312-clip validation twice. Its update-2 diagnostic retained 15.61% relation
MAE gain, 98.43% depth accuracy, zero reconstruction regression, and measured 5.32%
slip separation before meaningful temporal training.

The fail-closed recovery preflight reports **GO, 13/13 checks, 0 blockers**, with
14,142 training clips, 1,312 validation clips, 2,506 training and 226 validation sign
hand--body positive edge-frames, locked update-36,000 initialization, disjoint
identities, frozen threshold, and no Lane-L path.

| Artifact | Value |
|---|---|
| config | `phase3_posterior/configs/rdp_r2_contact_recovery_v3.yaml` |
| output | `outputs/phase3_training/rdp_r2_contact_recovery_v3_seed42` |
| tmux | `phase3_r2_contact_recovery_v3` |
| log | `logs/phase3/rdp_r2_contact_recovery_v3_seed42.txt` |
| config SHA-256 | `14ed63e8346cb31961f2aaaab2442ba6d41fa0128908939db077b2e2d2a912bc` |
| frozen v2b SHA-256 | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| source P3-G2 decision SHA-256 | `5cceb65aca11d026d057d4d172dd435a81de71edc9aac7840d35ab6b986c18e8` |

This section freezes the recovery strategy before any v3 formal validation result.
P3-G2 remains NO-GO until the complete numerical decision passes every condition.

### 38.5 V3 and v4 execution results

V3 validated the temporal-persistence hypothesis but not the sign-contact
classification hypothesis. The selected update-7,000 checkpoint passed slip with
20.73% gain and retained overall F1 0.6614, but sign F1 was only 0.4644. Later
updates increased slip while reducing both overall and sign contact quality. The run
was stopped after update 8,000 rather than spending the remaining budget on the
wrong trade-off.

V4 gave contact a trainable copy of the relation encoder while preserving the
separate frozen geometry/depth provider. Its fail-closed preflight passed 15/15
checks after catching and repairing a copied `requires_grad=False` flag. The invalid
smoke is preserved as `rdp_r2_contact_recovery_v4_smoke_superseded_frozen_encoder`.
The accepted smoke and **37/37 tests** passed.

V4 selected update 1,500. The complete source-disjoint formal evaluation is:

| P3-G2 condition | V4 result | Requirement | Decision |
|---|---:|---:|---|
| relation-distance MAE gain | **15.61%** | >=10% | GO |
| overall contact F1 | **0.6537** | >=0.65 | GO |
| sign hand--body contact F1 | **0.4673** | >=0.60 | **NO-GO** |
| depth-order accuracy | **98.43%** | >=80% | GO |
| slip gain vs no-persistence | **23.06%** | >=15% | GO |
| slip comparator available | true | true | GO |
| maximum reconstruction regression | **0.00%** | <=1% | GO |

The sign confusion counts are TP=132, FP=207, FN=94: precision 0.3894 and recall
0.5841. **Formal V4 P3-G2 decision: NO-GO**, with sign contact as the sole failed
condition. Contact energy remains disabled. R3's geometry-only GO result remains
valid and unchanged.

### 38.6 Evidence-domain blocker

Further capacity or threshold tuning is not justified on the current cache. A
diagnostic threshold sweep of the selected temporal model cannot exceed sign F1
0.49. The two dominant sign edges have initializer-gap ROC-AUC approximately 0.994,
but their extreme class imbalance leaves average precision only 0.58 and 0.47. A
nonlinear 49-feature temporal probe trained on all 2,506 sign positives reached
training F1 0.958, then collapsed on unseen validation signers to F1 0.268 (217 TP,
1,178 FP, 9 FN). This is strong evidence of cross-signer target/evidence shift, not
an optimizer or threshold problem.

The current geometry-derived sign targets are also too sparse for a paper-grade
contact claim: only 2,506 positive train edge-frames and 226 validation edge-frames,
concentrated mainly on two fingertip--chest proxy edges. The 420-clip calibration
split has only 25 positive sign hand--body edge-frames. It cannot support stable
threshold or edge-specific calibration. Reusing formal validation to choose these
values would be leakage.

### 38.7 Data-centric strategy required for a defensible P3-G2 GO

The next attempt is **R2 sign-contact target v3**, and it must stop at each ordered
gate below rather than launching another model on the current labels.

1. **D0: contact audit and annotation.** Build at least 300 source/signer-disjoint
   How2Sign/PHOENIX clips, deliberately stratified over hand--face, hand--torso,
   two-hand, near-contact negative, fast transition, occlusion, and no-contact cases.
   Double-review at least 10%; require catastrophic target error below 10% and
   inter-review agreement (Cohen's kappa >=0.75). Record onset, persistence, release,
   contacted body region, and visibility. **NO-GO** if agreement/support fails.
2. **D1: mesh-surface targets.** Replace fixed joint-sphere proxies with nearest
   hand-vertex to body-surface distance, body-part identity, surface normal alignment,
   tangential velocity, and the frozen 12/20-mm onset/release hysteresis. Use video
   evidence to reject depth-only pseudo contacts. Preserve original v2 sidecars;
   materialize a new cache and audit every hash.
3. **D2: supported signer-disjoint splits.** Require at least 2,000 positive training,
   500 calibration, and 1,000 validation sign contact edge-frames, with at least 50
   positive clips per represented signer group. Freeze calibration and validation
   before feature/model experiments. **NO-GO** if any positive-support minimum fails.
4. **D3: frozen observation features.** Use only deployable Phase-1 evidence:
   WiLoR/HaMeR hand geometry, SMPLer-X body geometry, Sapiens 2D/body confidence,
   hand/body crop embeddings already available locally, endpoint reliability,
   surface gap, approach velocity, and visibility. Do not use R3 output or GT at
   inference, avoiding circular R2-to-R3 ordering.
5. **D4: feature sufficiency probe.** Train a small contact probe on train only and
   freeze its threshold on calibration. It must obtain sign F1 >=0.65 and precision
   and recall each >=0.60 on the untouched signer-disjoint validation split.
   **NO-GO** here means improve targets/features; do not launch the relation graph.
6. **D5: relation/contact retraining.** Pretrain geometry on ARCTIC/InterHand, then
   run sign-balanced adaptation with contact-positive clip sampling, per-region hard
   negatives, an AUPRC/listwise ranking term, contact-conditional persistence, and
   the identical no-persistence comparator. Keep geometry/depth bypassed and frozen.
   Select checkpoints on calibration, never on formal validation.
7. **D6: formal P3-G2.** Freeze threshold, fusion, and checkpoint; run the existing
   complete gate once on the untouched signer-disjoint validation set. Require every
   numerical condition, then repeat seeds 123 and 456 as a robustness audit. Contact
   energy may be enabled downstream only if all formal conditions pass and the sign
   F1 seed standard deviation is below 0.03.

This strategy does not block geometry-only R4 progression, but it does block any
claim that contact energy is safe. The correct immediate action is target/evidence
construction through D0--D4, not additional hyperparameter tuning on the 226-positive
formal validation set.

| Artifact | SHA-256 |
|---|---|
| v3 config | `14ed63e8346cb31961f2aaaab2442ba6d41fa0128908939db077b2e2d2a912bc` |
| v3 preflight | `cead29ed0f6baab79a017aeef423b6a899bf55c73f404bb02ad1be60acb01856` |
| v3 selected checkpoint | `ba50500c7b0d4f7403040dc913d47a042867e0e47926de48a6ca393593e821ee` |
| v4 config | `f5d2e7582158b2abc9a93c9a7bf9ce2487447a68840c8ea7bbe35eaa7b009a31` |
| v4 preflight | `d0e3ecfb7fa659e03b69ae58a8750c216e5ee7a3e6df776c66baabf3e640ad91` |
| v4 selected checkpoint | `5021cc6295780e72e1348467707fa00c00c3c1b84f9a9c60e4fd7af9463fd754` |
| v4 formal evaluation | `1ed880906175ebdceb2e6d8b82e8f9250a4b72088161a0ea64450717104dc90f` |
| v4 formal P3-G2 decision | `47eb07d253801d76bddb96845bf8cedcb4e51278f100fa6584aa9496383c7909` |

## 39. Observation-branch recovery and sealed signer-10 audit (2026-08-06)

### 39.1 Root cause and additive data repair

The V4 failure was not caused by insufficient network capacity or by the frozen
0.5 threshold. Threshold sweeps of the old model remained below 0.49 sign-contact
F1, while a high-capacity diagnostic probe reached 0.958 training F1 and only 0.268
on unseen signers. The original How2Sign training partition contained only the
connected signer component `{3, 5, 8}`. Signers 1 and 2 share source-video groups,
so treating them as independent source groups would also violate the required
source-disjoint contract.

The deployable signal missing from the temporal-contact model was the Phase-2
observed-minus-projected 2D residual. A train-only sufficiency audit quantified the
effect before model construction:

| Diagnostic evidence | Unseen-signer sign F1 |
|---|---:|
| relation features only | 0.4459 |
| + observed 2D motion/reliability | 0.5475 |
| + regional reprojection residual | 0.5726 |

The residual is useful but not sufficient by itself. It was therefore introduced
through a contained contextual graph branch: the passing V4 base is frozen, the
new branch is zero-initialized, and an exactly zero residual leaves the old logits
unchanged. V6 established that the former EMA value of 0.9999 lagged the contained
branch severely (at update 1,000, live delta norm 0.0774 versus EMA delta norm
0.00268). V7 changes only this recovery mechanism to EMA 0.99 and trains the
observation branch while leaving the geometry/depth provider and base contact path
frozen. V8 hand-body-only containment was implemented and smoke-tested as a reserve,
but was deliberately **not trained** after V7 crossed the development gate.

All changes are additive. Legacy Phase 2/3 caches, configs, checkpoints, and methods
were not overwritten. Lane-L and the author's 1,493-frame evaluation set were not
read, trained on, or used for selection.

### 39.2 Source/signer-disjoint expansion and sealed test construction

Official How2Sign test signer 10 was extracted independently. Of 247 eligible clips
(7,904 frames), 220 passed temporal target refinement and reprojection enrichment;
27 were rejected by the existing fail-closed quality rules. The final partition is:

| Partition | Signers | Relation clips | Role |
|---|---|---:|---|
| train | 3, 4, 5, 8, 9, 11 plus generic ARCTIC/InterHand | 14,562 | fitting |
| development validation | 1, 2 plus generic ARCTIC/InterHand | 1,312 | selection |
| sealed How2Sign test | 10 only | 220 | one-time transfer audit |

The sealed set contains 38 contact-positive clips and 123 positive hand--body
edge-frames among 422,400 valid edge-frames. Train, development, and test signer and
source groups are disjoint. The test manifest was hashed before evaluation. It was
opened exactly once after the V7 update-5,000 checkpoint, EMA policy, score fusion,
threshold, and formal P3-G2 decision code had all been frozen.

### 39.3 V5--V7 development results

V5 allowed the new evidence to update the full contact encoder. It improved neither
the transfer feature nor containment and remained NO-GO. V6 froze the base but was
stopped after confirming EMA lag; it is diagnostic and has no formal claim. V7 is
the accepted contained-branch experiment.

| P3-G2 condition | V5 | V7 development | Requirement |
|---|---:|---:|---:|
| relation-distance MAE gain | 15.61% | **15.61%** | >=10% |
| overall contact F1 | 0.6501 | **0.6619** | >=0.65 |
| sign hand--body F1 | 0.4783 | **0.6032** | >=0.60 |
| depth-order accuracy | 98.43% | **98.43%** | >=80% |
| slip gain | 24.08% | **26.79%** | >=15% |
| maximum reconstruction regression | 0.00% | **0.00%** | <=1% |

The fail-closed V7 preflight had zero blockers, and the complete development
decision passed every condition. This is recorded as **P3-G2 development GO**, not
as full transfer GO, because the old validation split had already been used during
the recovery investigation.

### 39.4 One-time sealed signer-10 result

The immutable signer-10 evaluation completed all 220 clips. Its formal gate uses
`contact_logits + 2.0 * persistence_logits`, exactly as frozen before the test.

| P3-G2 condition | Sealed result | Requirement | Decision |
|---|---:|---:|---|
| relation-distance MAE gain | **-3.81%** | >=10% | **NO-GO** |
| overall contact F1 | **0.6188** | >=0.65 | **NO-GO** |
| sign hand--body F1 | **0.5669** | >=0.60 | **NO-GO** |
| depth-order accuracy | **97.45%** | >=80% | GO |
| slip gain vs no-persistence | **13.92%** | >=15% | **NO-GO** |
| slip comparator available | true | true | GO |
| maximum reconstruction regression | **0.00%** | <=1% | GO |
| relation-only reconstruction unchanged | true | true | GO |

The contact-only diagnostic (persistence fusion disabled) obtains sign F1 0.6108,
but it is not the predeclared gate score and cannot replace it after observing the
test. The graph hand--hand MAE is 0.03331 m versus 0.03209 m for the frozen geometry
MLP, which explains the negative relation gain. Together with the development-to-
test drops in contact and slip, this identifies a signer/domain geometry shift and
a persistence-fusion transfer failure rather than a remaining optimizer bug.

**Final P3-G2 status: sealed-transfer NO-GO.** Contact energy and all contact-driven
attraction, persistence, and force-coupling constraints remain disabled. The earlier
geometry-only P3-G3 GO result remains valid and may continue downstream under the
documented fallback.

### 39.5 Valid next recovery experiment

Signer 10 is now consumed and must never become a tuning or checkpoint-selection
set. A defensible next attempt must create a new development/test protocol before
training:

1. materialize PHOENIX hand/body observations and mesh-surface contact targets, then
   split by signer and connected source-video component;
2. reserve one component as a newly sealed final audit and use the remaining
   PHOENIX components only for feature sufficiency and checkpoint selection;
3. normalize 2D residuals by torso scale and camera crop, augment camera/crop noise,
   and require train-only probes to pass F1 >=0.65 with both precision and recall
   >=0.60 on the new development component;
4. pretrain the contained observation/persistence branches across How2Sign and
   PHOENIX, keeping relation geometry bypassed and frozen; select fusion and threshold
   on development only;
5. require three development seeds to pass the complete P3-G2 vector before opening
   the new sealed component once.

No V8 training or post-test V7 tuning was launched in this cycle. This stop is
intentional: another run selected from signer-10 feedback could produce a number
above threshold but would no longer be a valid transfer result.

### 39.6 Verification and immutable artifacts

Ruff passes for `phase2_refiner` and `phase3_posterior`; both packages compile.
Tests pass **67/67** for Phase 2 and **42/42** for Phase 3.

| Artifact | SHA-256 |
|---|---|
| V7 config | `84a93772a47b74ecd84ea83e30ee21ccd548678aa93c881adff336f477f91d56` |
| V7 selected checkpoint (update 5,000) | `da398ab1aa6399c38705d14d6559150d152a55a2495729292c149aee7d3a840a` |
| V7 preflight | `eac48e2ba041b1478bbc9fd5ba6ade3b4a93a6cadf200814a25f281eed4a761b` |
| V7 development evaluation | `569ab1575787fb0400f1376372c1466334431b9abcff62472951023966be9b2c` |
| V7 development decision | `d3eb3927a037d8794b17f965f76b31ab2f4655a6b2dc99673985c77b74bbb5ba` |
| expanded train manifest | `eeda81b36c79eafa6dcb6c7fecac89380c2bb7b44ba548166b7e18801ca5c4ca` |
| expanded development manifest | `6827f7148a4e28ca35b735e075e08ddff49692cc4267da6e0e81b55e8cd46ef6` |
| sealed signer-10 manifest | `17921e781ccd586a198f47399a120d43c608222172d916848b7d2370a93f3fb7` |
| signer-10 relation-cache manifest | `056ec93ef42b2ed8f1e46fc1cdf4fa3b80442748aa26a6ddbc078af5e7eada17` |
| signer-10 reprojection report | `908bee45f972e9f5d42d857b510bf461464e05ebb1592341d786fb8b6fcc5723` |
| sealed evaluation | `cfc02fd8d829dde6a1628cdff3f20b92595ae5fb66de970a19511826d3df45d7` |
| sealed formal decision | `ebb46861f75c7b2a729b7e0cd64bd1e394aed9bd26f349dac8934bb676f2a9e6` |
| sealed append-only log | `9438b389e3cd17cc9aa4962de4cc9ceaa5e1b55786927ec52e1e0a3df4b708cc` |
