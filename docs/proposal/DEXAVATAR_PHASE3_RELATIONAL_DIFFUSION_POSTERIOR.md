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
