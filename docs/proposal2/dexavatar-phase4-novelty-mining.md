# Phase 4 — Novelty Mining and Research-Hypothesis Ranking

**Target:** monocular RGB video → accurate SMPL-X reconstruction of sign language  
**Primary metrics:** SGNify TR-V2V — upper body excluding face, left hand, right hand  
**Evidence cut-off:** 10 August 2026  
**Status:** hypothesis mining only. No idea in this report is declared novel, and no final architecture is selected.

---

## Executive conclusion

Ten falsifiable hypotheses were generated from the Phase-3 gaps. After checking the closest direct and adjacent primary-source precedents, the five candidates worth taking into a dedicated novelty-verification phase are:

1. **H1 — Visibility-conditioned bidirectional sign-motion inference**
2. **H3 — Temporally ordered hand–hand interaction reconstruction**
3. **H4 — Body-anchored hand–body contact reconstruction**
4. **H2 — Calibrated component-wise multi-hypothesis SMPL-X reconstruction**
5. **H10 — Reliability-certified pseudo-GT correction coupled to hybrid inference**

This shortlist does **not** mean these ideas are novel. It means their causal motivation, expected geometric effect, and ablation structure are strong enough to justify an exhaustive closest-work search.

The following generic contributions are already too weak:

- “use a Transformer for temporal consistency”;
- “use diffusion to generate plausible signing motion”;
- “replace HaMeR with a newer detector”;
- “add cross-hand attention”;
- “add a contact loss”;
- “add a recognition loss”;
- “fine-tune at test time”;
- “jointly optimize the camera.”

Each has clear prior art. A defensible contribution must identify **which missing sign-specific variable makes current reconstruction fail**, and demonstrate that the proposed mechanism changes the corresponding 3D geometry rather than only plausibility, smoothness, or recognition.

---

## 1. What the 2026 literature changes

### 1.1 Temporal and generative reconstruction are no longer open by themselves

- [DanceHMR](https://arxiv.org/abs/2605.18102) performs temporal, hand-aware SMPL-X recovery from monocular video.
- [MoRo](https://arxiv.org/abs/2601.16079) uses masked modeling and visual/motion priors for human motion recovery under occlusion.
- [RoHM](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RoHM_Robust_Human_Motion_Reconstruction_via_Diffusion_CVPR_2024_paper.html) uses diffusion for robust motion reconstruction.
- [HandFlow](https://arxiv.org/abs/2607.11221) performs fully generative 4D hand recovery with confidence-aware masking.
- [StableHand](https://arxiv.org/abs/2605.18553) conditions bimanual flow matching on learned, component-wise observation quality and targets long missing spans.

**CONCLUSION:** sequence models, masked completion, bidirectional context, diffusion/flow matching, and confidence-conditioned hand-motion reconstruction are established adjacent directions. Sign-specific conditioning and evaluation may still be open; the generic mechanism is not.

### 1.2 Uncertainty and multiple plausible poses are established adjacent ideas

- [ProHMR](https://arxiv.org/abs/2108.11944) models a conditional distribution of human poses.
- [POCO](https://arxiv.org/abs/2308.12965) predicts uncertainty and uses it for pseudo-label selection/video inpainting.
- [MaskHand](https://arxiv.org/abs/2412.13393) uses confidence-guided generative hand reconstruction under ambiguity.
- StableHand predicts separate observation-quality channels for wrists and fingers of both hands.

**CONCLUSION:** “predict uncertainty” or “sample several hands” is not novel. The open question is whether uncertainty is calibrated for sign-specific SMPL-X errors and whether automatic hypothesis selection improves geometry rather than only best-of-K oracle performance.

### 1.3 Contact-aware reconstruction is established outside sign reconstruction

- [TUCH / On Self-Contact and Human Pose](https://arxiv.org/abs/2104.03176) uses contact supervision and contact-aware fitting for human pose.
- [CHOIR](https://arxiv.org/abs/2605.20992) reconstructs monocular hand–object interaction with contact correspondences, generative spatial rectification, and contact-aware joint optimization.
- [ARCTIC](https://arctic.is.tue.mpg.de/) provides dense dynamic contact for hand–object interaction.
- SGNify's supplement already contains generic self-contact/interpenetration terms.

**CONCLUSION:** a generic contact penalty is not novel. Potential sign-specific headroom lies in **which articulators touch, which surface is in front, and when approach/touch/separate events occur**.

### 1.4 Phonology and semantic motion constraints also have prior art

- [SGNify](https://arxiv.org/abs/2304.10482) uses linguistic symmetry and hand-pose invariance constraints during reconstruction.
- [Toward Phonology-Guided Sign Language Motion Generation](https://arxiv.org/abs/2603.17388) conditions SMPL-X sign generation on handshape, finger configuration, location, movement, and flexion attributes.
- [Meaningful Pose-Based Sign Language Evaluation](https://arxiv.org/abs/2510.07453) finds that carefully chosen pose distances and back-translation likelihoods correlate with native-signer judgments.
- Sign recognition and sign-production work already uses semantic and recognition embeddings.

**CONCLUSION:** “use phonology” is not novel. A reconstruction contribution would have to show that uncertain geometry is resolved by specific phonological variables without hallucinating a canonical sign or leaking ground-truth gloss.

### 1.5 Test-time adaptation, hybrid fitting, and camera optimization are mature ideas

- [CycleAdapt](https://arxiv.org/abs/2308.06554) cyclically adapts a mesh regressor and motion denoiser on a test video.
- [STRIDE](https://arxiv.org/abs/2312.16221) adapts a masked motion prior on an individual occluded test video.
- [Meta-HMR](https://arxiv.org/abs/2401.14121) trains a regressor to be friendly to test-time optimization.
- [SPIN](https://arxiv.org/abs/1909.12828), [EFT](https://arxiv.org/abs/2004.03686), [SMPLify-X](https://arxiv.org/abs/1904.05866), and [ScoreHMR](https://arxiv.org/abs/2403.09623) establish regression–optimization, pseudo-label refinement, and learned-prior fitting paradigms.
- GLAMR, WHAM, GVHMR, and other world-grounded methods jointly reason about humans and cameras.

**CONCLUSION:** test-time adaptation, hybrid regression plus optimization, and camera–pose optimization cannot be primary novelty. They can be enabling components only if tied to a newly validated sign-specific failure.

---

## 2. Coverage of the 16 required directions

| Required direction | Hypotheses that explore it |
|---|---|
| 1. Sequence-level sign motion priors | H1, H6 |
| 2. Bidirectional temporal inference | H1 |
| 3. Uncertainty-aware reconstruction | H2, H10 |
| 4. Multi-hypothesis hand reconstruction | H2, H6 |
| 5. Explicit hand–hand contact reasoning | H3 |
| 6. Hand–body contact reasoning | H4 |
| 7. Occlusion-order modeling | H3, H4 |
| 8. Sign phonology-aware reconstruction | H5 |
| 9. Sign-recognition features as supervision | H5 |
| 10. Conditional generative/diffusion motion prior | H6 |
| 11. Signer adaptation | H7 |
| 12. Test-time adaptation | H8 |
| 13. Joint camera–pose optimization | H9 |
| 14. Learned observation confidence | H2, H8 |
| 15. Pseudo-GT correction | H10 |
| 16. Hybrid regression + optimization | H9, H10 |

---

## 3. Ten research hypotheses

## H1 — Visibility-conditioned bidirectional sign-motion inference

**Current failure.** During hand–hand/body occlusion or blur, DexAvatar has missing or biased hand evidence. Its released temporal term links only the current and previous body pose; it has no hand sequence model and cannot use future reappearance to repair earlier frames.

**Why existing methods fail.** Framewise fitting chooses a prior-favoured hand/depth mode. First-order smoothing propagates the preceding error and can lag true fast motion. Generic temporal models may smooth the hand but are not trained or evaluated for sign phases, bimanual coordination, holds, repetitions, or contact transitions.

**Missing information.** Per-frame visibility/reliability, future observations after the occlusion, sign-motion phase, and the joint hand–wrist–arm trajectory over the whole window.

**Proposed mechanism.** Infer the full body-and-hand SMPL-X sequence from both temporal directions while conditioning the prior and observation fusion on explicit per-joint/component visibility. The causal claim is that future reappearance provides geometric evidence about the hidden interval. “Transformer” is not the contribution; bidirectional evidence use under sign-specific missingness is.

**Expected geometric improvement.** Fewer hand identity/depth switches, improved hidden finger pose, coherent wrist/forearm orientation, lower post-occlusion discontinuity, and less motion lag than fixed smoothing.

**Expected TR-V2V effect.** Potentially **moderate-to-large** on left/right hands in occluded spans; **moderate** upper-body improvement when wrist/forearm errors propagate through the arm; small visible-frame change. Aggregate benefit depends on how many benchmark frames are genuinely occluded.

**Falsifiable test.** Compare otherwise matched framewise, forward-only, backward-only, and bidirectional inference under real and synthetic contiguous masks. Measure TR-V2V, depth order, recovery latency, velocity/acceleration error, and lag. **Refute H1** if bidirectional evidence gives no paired improvement over forward-only inference at equal capacity/compute, or improves smoothness by attenuating true motion.

**Closest prior-work pressure.** DanceHMR, MoRo, GLAMR, STRIDE, HandFlow, StableHand, and pose-guided temporal hand enhancement. The novelty claim would have to be narrower than “temporal completion” or “bidirectional model.”

---

## H2 — Calibrated component-wise multi-hypothesis SMPL-X reconstruction

**Current failure.** Monocular overlap can support several plausible finger, wrist, hand-depth, and arm configurations. DexAvatar emits a single fit; detector confidence is not calibrated 3D uncertainty.

**Why existing methods fail.** A single deterministic optimization collapses ambiguity to the initialization/prior mode. A single confidence per detection conflates wrist placement, finger articulation, left/right hand quality, and body evidence. Sampling alone does not help unless the correct hypothesis can be selected without ground truth.

**Missing information.** Component-wise error probability; multiple coherent body/hand sequence modes; calibration targets; evidence that can rank alternatives, such as future observations, contact/order, silhouettes, and reprojection.

**Proposed mechanism.** Represent a small set or distribution of coherent SMPL-X sequence hypotheses with separate uncertainty for left/right wrist, left/right fingers, arms, and camera. Preserve reliable components while allowing only uncertain components to branch. Rank or marginalize hypotheses using image evidence, temporal reappearance, and interaction consistency.

**Expected geometric improvement.** Reduced mode-collapse errors in depth, wrist twist, hand identity, and hidden finger articulation; fewer cases in which one wrong hand corrupts an otherwise reliable body estimate.

**Expected TR-V2V effect.** **Moderate-to-large** hand reduction if automatic selection closes much of the oracle best-of-K gap; likely **small-to-moderate** upper-body improvement. If only oracle best-of-K improves, practical TR-V2V will not.

**Falsifiable test.** Report calibration, negative log-likelihood/coverage, risk–coverage curves, best-of-K and automatically selected-K TR-V2V, diversity, and component-wise error correlation. Compare with HaMeR confidence and deterministic fitting. **Refute H2** if uncertainty is not better calibrated than detector confidence, hypotheses lack meaningful diversity, or the oracle–automatic selection gap remains large.

**Closest prior-work pressure.** ProHMR, POCO, MaskHand, ScoreHMR, KNOWN-Hand, CondiMen, HandFlow, and StableHand. “Multi-hypothesis” and “uncertainty-aware” are established; sign-specific component calibration and useful selection are the only plausible open contribution.

---

## H3 — Temporally ordered hand–hand interaction reconstruction

**Current failure.** Under bimanual overlap, current methods do not explicitly determine whether the hands are separate, which hand is in front, which surface pair touches, how long contact persists, or when separation occurs.

**Why existing methods fail.** Independent hand estimates lack relative bimanual geometry. Collision repulsion only says that meshes should not penetrate; it does not say which contact is intended. A per-frame contact loss can flicker or choose the wrong surface pair without a temporal state transition model.

**Missing information.** Bimanual visibility order, contacting surface pairs, approach/touch/separate state, contact duration, relative velocity, and left/right identity through overlap.

**Proposed mechanism.** Infer a temporally coherent interaction state sequence—separate, left-in-front, right-in-front, touching with a specified surface pair, crossing, and separating—and use that state as a geometric constraint on relative hand depth, signed distance, normals, and identity. The state representation, not “cross-attention” or an added collision loss, is the candidate contribution.

**Expected geometric improvement.** Correct relative hand depth and identity, more accurate finger arrangement near contact, fewer penetrations and false separations, and temporally correct contact onset/offset.

**Expected TR-V2V effect.** Likely **moderate** hand improvement on bimanual/contact frames but possibly only **small-to-moderate** aggregate improvement because contact frames may be a minority. Contact F1, depth-order accuracy, and penetration should improve more strongly than mean TR-V2V.

**Falsifiable test.** Annotate bimanual order/contact states on multi-view or mocap sequences. Compare independent hands, collision-only, unordered contact, and ordered state inference. **Refute H3** if ordered states do not improve relative depth/contact metrics or paired hand TR-V2V beyond unordered contact, or if state errors introduce worse geometry.

**Closest prior-work pressure.** SGNify/TUCH self-contact, IntagHand, Dyn-HaMR, interacting-hand generative models, StableHand, and CHOIR's explicit contact correspondences. Novelty would require a sign-specific bimanual state and evidence that it improves reconstruction.

---

## H4 — Body-anchored hand–body contact reconstruction

**Current failure.** When a hand touches or passes near the face, chest, shoulder, neck, or opposite arm, 2D overlap cannot determine whether the hand is in front, behind, or touching the body. Repulsion can keep meshes apart while placing the hand at the wrong depth.

**Why existing methods fail.** Current fitting treats the body mainly as another articulated mesh and obstacle. It lacks a sign-specific intended contact pair and temporal contact event. Generic pose priors do not encode that a particular palm/fingertip should touch a particular anatomical region.

**Missing information.** Contacting hand and finger/palm region, anatomical body target, front/back order, signed distance, surface normal alignment, contact onset/duration, and whether the body contact is linguistically meaningful or incidental.

**Proposed mechanism.** Treat anatomically indexed body surfaces as 3D anchors for uncertain hands. Infer a temporally persistent hand-part ↔ body-region relation and constrain hand location/orientation only when that relation is supported. This is stronger than “add a contact loss”: it uses contact identity to resolve monocular depth.

**Expected geometric improvement.** More accurate hand depth and global orientation relative to the torso/face, reduced hand–body penetration, correct body location, and improved wrist/arm configuration through the kinematic chain.

**Expected TR-V2V effect.** **Moderate** hand and possibly upper-body improvement on body-contact signs; **small-to-moderate** aggregate effect unless such signs are common. Contact precision/recall and body-region location accuracy should show the clearest gain.

**Falsifiable test.** Label hand-part/body-region contacts and compare collision-only, generic contact, and indexed persistent contact. Match contact and non-contact frames by speed/visibility. **Refute H4** if contact identity does not improve hand depth/orientation or TR-V2V, or if it increases false contact on near-body motion.

**Closest prior-work pressure.** TUCH, PROX/RICH-style human contact, dense human-scene contact work, CHOIR/HOI contact reconstruction, and SGNify's self-contact terms. The sign-specific articulator-to-body relation is the potential distinction.

---

## H5 — Ambiguity-gated phonological and recognition supervision

**Current failure.** A geometrically plausible but incorrect handshape, orientation, location, dominance pattern, or movement may fit noisy 2D evidence and still change the sign. TR-V2V weights vertices uniformly rather than by linguistic importance.

**Why existing methods fail.** DexAvatar's sign priors learn motion distributions without explicit phonological labels. SGNify uses only coarse symmetry/invariance rules. A generic recognition feature may be invariant to geometric details needed for reconstruction or may impose a canonical sign that ignores signer-specific execution.

**Missing information.** Soft estimates of handshape, orientation, location, movement, dominance, and contact; which attributes are reliable; whether the video supplies a known gloss; and how semantic feature distance relates to true 3D error.

**Proposed mechanism.** Use soft phonological/recognition evidence only on ambiguous components and frames, never as an unconditional global reconstruction target. The hypothesis is that recognition features provide disambiguating constraints when image geometry is weak. Ground-truth gloss should not be required at test time unless the task explicitly permits it.

**Expected geometric improvement.** Fewer handshape and orientation mode errors, better dominant/non-dominant coordination, correct sign location and movement phase, and reduced catastrophic semantic failures.

**Expected TR-V2V effect.** Probably **small-to-moderate** aggregate hand improvement but potentially large worst-sign/tail improvement. It could worsen TR-V2V if the semantic constraint canonicalizes signer-specific geometry.

**Falsifiable test.** Compare no semantics, predicted phonology, ground-truth phonology, shuffled labels, and recognition embedding supervision. Gate by calibrated ambiguity. Report TR-V2V, per-phoneme errors, recognition, and signer-specific deviation. **Refute H5** if only ground-truth gloss helps, shuffled labels perform similarly, or recognition improves while geometric error worsens.

**Closest prior-work pressure.** SGNify, phonology-guided sign generation, text-driven sign hand-motion generation, semantics-aware test-time adaptation, MASA, and meaningful pose evaluation. Reviewer risk is high because semantic supervision can leak labels or hallucinate a canonical motion.

---

## H6 — Observation-guided conditional generative sign-motion posterior

**Current failure.** Deterministic priors and smoothers may regress to the mean during long occlusion, while several sign-motion continuations remain plausible.

**Why existing methods fail.** Local latent priors lack dynamics; deterministic sequence predictors collapse alternatives; unconditional sign generation ignores the actual video; generic diffusion can create plausible but image-inconsistent motion.

**Missing information.** A distribution of sign-specific SMPL-X trajectories conditioned on visible observations, missingness, hand/body coordination, and possibly interaction/phonological state.

**Proposed mechanism.** Learn a conditional generative posterior over full SMPL-X motion and guide it with reliable frame/component observations. Diffusion or flow matching is an implementation choice, not the novelty. The scientific claim is that a distributional sequence prior recovers ambiguous hidden spans more accurately than a capacity-matched deterministic prior.

**Expected geometric improvement.** Plausible alternative trajectories through occlusion, coherent fast motion, fewer frozen hands, and improved long-gap recovery.

**Expected TR-V2V effect.** Potentially **moderate** hand improvement in long gaps; uncertain aggregate effect. Best-of-K may improve substantially while selected single-output TR-V2V remains unchanged.

**Falsifiable test.** Compare deterministic and generative priors at matched data, capacity, conditioning, and compute. Report selected and best-of-K geometry, calibration, diversity, temporal fidelity, and image consistency. **Refute H6** if samples collapse, selection fails, or benefits disappear after matching capacity/compute.

**Closest prior-work pressure.** Neural Sign Actors, SignAvatar/phonology-guided sign generation, ScoreHMR, RoHM, DiffMesh, MoRo, HandFlow, StableHand, and Dyn-HaMR. This is highly crowded and cannot be sold as “diffusion for sign reconstruction.”

---

## H7 — Signer-specific geometric and motion calibration

**Current failure.** Universal shape, range-of-motion, neutral wrist, articulation-speed, and style assumptions may over-regularize a new signer. The primary SGNify benchmark cannot reveal this because it has one signer.

**Why existing methods fail.** A clip-mean shape does not personalize pose dynamics. Global joint limits and priors mix anatomy with style. Adapting to all frames can learn observation errors rather than signer characteristics.

**Missing information.** Stable signer properties estimated from reliable frames: shape/limb proportions, neutral wrist convention, feasible articulation range, dominant hand, movement speed/style, and confidence that a property is persistent.

**Proposed mechanism.** Infer a low-dimensional signer calibration from high-visibility frames and hold it consistent across clips, separating persistent anatomy/style from per-frame pose. Only parameters proven stable should adapt.

**Expected geometric improvement.** Better bone/shape consistency, wrist alignment, signer-specific valid extremes, reduced prior pull on unusual articulation, and fewer temporal shape changes.

**Expected TR-V2V effect.** Likely **small** on the single-signer SGNify setting once personalized shape is already used; potentially **moderate** cross-signer improvement and reduced worst-signer error.

**Falsifiable test.** Use multi-signer mocap with leave-one-signer-out evaluation. Compare no adaptation, shape-only, motion-only, and joint calibration. **Refute H7** if persistent parameters do not generalize across clips or do not reduce held-out signer error beyond shape normalization.

**Closest prior-work pressure.** Personalized pose/shape refinement, anthropometric HMR, subject-specific avatar fitting, and general domain personalization. The single-signer benchmark makes this difficult to publish without new data.

---

## H8 — Reliability-restricted test-time adaptation

**Current failure.** Off-the-shelf observations and priors can be mismatched to a new camera, signer, resolution, or sign language. Naive adaptation to noisy 2D keypoints can amplify depth and occlusion errors.

**Why existing methods fail.** Test-time objectives lack 3D truth and can overfit detector noise. Existing Cyclic TTA/STRIDE-style systems target generic body pose and do not separate reliable wrist/finger/body channels or sign phases.

**Missing information.** Which frames/components are safe adaptation anchors; stable target-domain statistics; a mechanism to prevent catastrophic drift; and sign-specific validation under domain shift.

**Proposed mechanism.** Adapt only narrowly defined observation-reliability or signer-prior parameters using high-confidence, multi-frame-consistent evidence; freeze ambiguous components and validate through held-out frames in the same sequence.

**Expected geometric improvement.** Better camera/crop calibration, observation weighting, and target-signer fit without corrupting hidden hands.

**Expected TR-V2V effect.** Probably **small** on in-domain SGNify; potentially **moderate** under controlled cross-domain shift. It may worsen error if confidence is miscalibrated.

**Falsifiable test.** Compare no adaptation, all-frame adaptation, reliability-restricted adaptation, and oracle reliability across increasing shift. Include corrupted-2D stress tests. **Refute H8** if adaptation gains vanish with realistic estimated confidence or if drift harms unseen frames.

**Closest prior-work pressure.** CycleAdapt, STRIDE, Meta-HMR, semantics-aware TTA, and online pose adaptation. Test-time adaptation is not a novelty claim.

---

## H9 — Observability-gated joint camera–shape–pose refinement

**Current failure.** DexAvatar fixes the SMPLer-X camera; camera errors may be absorbed by shape, depth, and pose, especially in cropped upper-body videos.

**Why existing methods fail.** Always fixing the camera prevents correction; unconstrained joint optimization can instead create camera–pose degeneracy. Translation-aligned TR-V2V still retains rotation, scale, and articulation compensation errors.

**Missing information.** Camera observability from the sequence, reliable torso/face anchors, calibrated intrinsics where available, and a decomposition of camera versus pose error.

**Proposed mechanism.** Jointly refine clip-level camera, consistent shape, and per-frame pose only when the sequence supplies sufficient anchors; freeze or regularize unobservable degrees of freedom. Hybrid regression supplies initialization, optimization supplies sequence consistency.

**Expected geometric improvement.** Better upper-body scale/orientation, shoulder–elbow–wrist geometry, and less pose compensation for camera bias.

**Expected TR-V2V effect.** Potentially **moderate** upper-body improvement if the fixed camera is wrong; likely **small** hand-articulation improvement. The oracle-camera ablation may show negligible headroom.

**Falsifiable test.** Compare fixed, calibrated, optimized, and deliberately perturbed camera settings while controlling shape/observations. **Refute H9** if calibrated/optimized cameras do not improve upper-body TR-V2V or if joint optimization only lowers reprojection while worsening 3D geometry.

**Closest prior-work pressure.** SMPLify-X, GLAMR, WHAM, GVHMR, SynCHMR, ScoreHMR, and many metric/world-grounded HMR systems. This is an important baseline correction, not a plausible standalone novelty.

---

## H10 — Reliability-certified pseudo-GT correction coupled to hybrid inference

**Current failure.** SignBPoser learns from pseudo-SMPL-X labels; filtering improves upper-body TR-V2V, proving label quality matters. Hard filtering removes implausible poses but neither repairs systematic bias nor distinguishes rare valid signing from annotation error. Strong initialization can then preserve those biases at test time.

**Why existing methods fail.** A regressor or prior trained on one fitting pipeline reproduces its errors. Hard thresholds discard coverage. Pure optimization remains sensitive to initialization; pure regression cannot enforce sequence/contact evidence at test time.

**Missing information.** Per-label reliability, cross-view/temporal consistency, contact/biomechanical violations, disagreement between annotators/estimators, and a clean subset that distinguishes error from rare valid motion.

**Proposed mechanism.** Assign pseudo-GT reliability using independent evidence; correct or downweight only demonstrably inconsistent body/hand components; train a fast regressor/prior on the resulting labels; then use its output and uncertainty as initialization for structured sequence optimization. The contribution cannot be “hybrid regression plus fitting”; it must be a validated sign-specific label-correction and reliability process.

**Expected geometric improvement.** Less systematic arm/wrist/finger bias, better coverage of rare valid poses than hard filtering, improved initialization, and reduced prior over-regularization.

**Expected TR-V2V effect.** Potentially **moderate-to-large** upper-body improvement because DexAvatar already shows a 3.78 mm UBody(-F) gain from body-data filtering; possible **moderate** hand improvement if hand pseudo labels are corrected. No numeric future gain is assumed.

**Falsifiable test.** Train identical models on raw, hard-filtered, reliability-weighted, corrected, and smaller clean subsets; use a fixed downstream fitter and held-out signers. Compare pure regression, pure optimization, and hybrid inference. **Refute H10** if correction does not improve independent mocap error or downstream TR-V2V beyond filtering, or if it removes rare valid motions and worsens generalization.

**Closest prior-work pressure.** EFT, SPIN, POCO pseudo-label selection, SignAvatars curation, DexAvatar filtering/correction, Meta-HMR, and general label-denoising work. Novelty must reside in independently validated sign-specific correction and its causal effect.

---

## 4. Generic variants explicitly rejected

| Rejected variant | Reason for rejection | Minimum change needed to become testable research |
|---|---|---|
| Temporal Transformer over SMPL-X | DanceHMR and many video HMR methods already do temporal sequence modeling. | Tie temporal evidence to measured sign-specific visibility gaps and prove motion fidelity rather than only smoothness. |
| Bidirectional encoder | Bidirectional/masked modeling is established in motion recovery. | Demonstrate causal recovery from future hand reappearance under sign-specific occlusion. |
| Diffusion prior for signing | Sign generation and reconstruction already use diffusion/flow matching. | Compare against a capacity-matched deterministic posterior and solve automatic hypothesis selection. |
| Replace HaMeR with WiLoR/another detector | SOKE and Tamaththul3D already show this substitution pattern. | Use the detector only as a controlled baseline or observation-ceiling experiment. |
| Cross-hand attention | IntagHand and many interacting-hand systems already exchange hand features. | Represent and supervise relative depth/contact state with measurable geometric consequences. |
| Collision/contact loss | SMPLify-X, TUCH, SGNify, and HOI reconstruction already use contact/collision terms. | Infer intended surface pairs and temporal contact order; report contact precision/recall and penetration. |
| Recognition loss | Semantics/recognition supervision is widespread and can canonicalize motion. | Gate specific phonological constraints by calibrated geometric ambiguity and use predicted, not leaked, semantics. |
| Confidence head | POCO, MaskHand, StableHand, and others model confidence/uncertainty. | Calibrate component-wise 3D error and demonstrate automatic multi-hypothesis selection. |
| Test-time fine-tuning | CycleAdapt, STRIDE, Meta-HMR, and related methods establish TTA/TTO. | Identify which persistent sign/signer variables are safely observable and prevent noisy-2D drift. |
| Joint camera optimization | Foundational fitting and world-grounded HMR already do this. | Treat as an ablation unless a new sign-specific observability result is established. |
| Hybrid regression plus optimization | SPIN, EFT, Meta-HMR, ScoreHMR, and many fitters are hybrid. | Demonstrate a new causal role for sign-specific reliability or interaction state. |
| Pseudo-GT filtering | DexAvatar already filters/corrects prior data. | Independently estimate label uncertainty, repair rather than only discard, and validate against accurate 3D truth. |

---

## 5. Ranking

### Scoring convention

All scores are ordinal from 1 to 5.

- **N, S, T, F, A:** higher is better — novelty potential, expected SOTA improvement, technical depth, feasibility, and ablatability.
- **D, C, R:** lower is better — dataset requirement, compute requirement, and reviewer risk.
- **Adjusted total:**

\[
N+S+T+F+A+(6-D)+(6-C)+(6-R),
\]

with a maximum of 40. The total is a triage aid, not evidence of novelty or expected numerical gain.

| Rank | Hypothesis | N | S | T | F | A | D burden | C burden | R risk | Adjusted /40 | Provisional status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **H1 Visibility-conditioned bidirectional sign inference** | 4 | 5 | 5 | 4 | 5 | 2 | 3 | 2 | **34** | Strong causal candidate; generic temporal novelty already occupied. |
| 2 | **H3 Ordered hand–hand interaction** | 5 | 4 | 5 | 3 | 5 | 4 | 3 | 3 | **30** | High sign-specific novelty potential; contact labels are the bottleneck. |
| 3 | **H4 Body-anchored hand–body contact** | 5 | 4 | 5 | 3 | 5 | 4 | 3 | 3 | **30** | High sign-specific potential; aggregate TR effect uncertain. |
| 4 | **H2 Calibrated component-wise multi-hypothesis reconstruction** | 4 | 5 | 5 | 3 | 5 | 3 | 4 | 3 | **30** | Strong impact potential; adjacent 2026 work creates heavy novelty pressure. |
| 5 | **H10 Reliability-certified pseudo-GT + hybrid inference** | 3 | 4 | 5 | 3 | 5 | 4 | 4 | 3 | **27** | Direct DexAvatar evidence supports the failure; method novelty is not yet clear. |
| 6 | H8 Reliability-restricted test-time adaptation | 2 | 3 | 4 | 3 | 5 | 2 | 3 | 4 | **26** | TTA is crowded; likely an enabling component. |
| 7 | H9 Observability-gated camera–pose refinement | 1 | 3 | 3 | 5 | 5 | 1 | 3 | 5 | **26** | Important baseline, weak standalone novelty. |
| 8 | H5 Ambiguity-gated phonology/recognition | 4 | 3 | 4 | 3 | 4 | 3 | 3 | 5 | **25** | Sign-specific but high leakage/canonicalization risk. |
| 9 | H6 Conditional generative sign-motion posterior | 3 | 4 | 4 | 3 | 4 | 3 | 5 | 4 | **24** | Very crowded after HandFlow/StableHand/MoRo; mechanism alone is weak. |
| 10 | H7 Signer-specific calibration | 3 | 3 | 4 | 3 | 4 | 5 | 2 | 4 | **24** | Scientifically valid, but requires multi-signer accurate 3D data. |

### Ranking interpretation

- H1 leads because it targets the best-supported combination of occlusion and temporal gaps, can be falsified with controlled masking, and is likely to affect hand TR-V2V directly.
- H3 and H4 have the strongest sign-specific distinction from adjacent work, but new contact/order annotations are required and aggregate TR-V2V may under-reward them.
- H2 may have high accuracy impact, but HandFlow and StableHand make a generic uncertainty/generative claim untenable. Calibration and automatic selection must be central.
- H10 has weaker pure novelty but unusually strong direct evidence: DexAvatar's own data-filtering ablation establishes that label quality changes downstream upper-body error.
- H5 is deliberately kept outside the top five despite its sign-specific nature. SGNify and recent phonology/semantic work are close, and reviewers can reasonably object that semantics hallucinate a canonical sign rather than recover observed geometry.
- H9 should be implemented as a diagnostic baseline if Phase-3's oracle-camera experiment shows headroom, not positioned as the main contribution.

---

## 6. Top five selected for novelty verification

The next phase should conduct backward, forward, recency, and code-level closest-work searches for these five candidates.

### Candidate V1 — H1: visibility-conditioned bidirectional sign inference

**Provisional novelty boundary:** not temporal modeling, not masking, and not bidirectionality. The possible contribution is joint SMPL-X hand/body inference that uses explicit sign-specific visibility and future reappearance while preserving fast-motion fidelity.

**Must distinguish from:** DanceHMR, MoRo, RoHM, GLAMR, STRIDE, HandFlow, StableHand, Dyn-HaMR, and temporal low-resolution hand recovery.

**Kill criterion:** a prior method already performs visibility-conditioned bidirectional full SMPL-X sign reconstruction, or a controlled oracle study shows future context provides negligible geometry gain.

### Candidate V2 — H2: calibrated component-wise multi-hypothesis reconstruction

**Provisional novelty boundary:** not pose distributions, uncertainty prediction, or K samples. The possible contribution is calibrated wrist/finger/arm/camera uncertainty for signing plus non-oracle selection that improves TR-V2V.

**Must distinguish from:** ProHMR, POCO, MaskHand, ScoreHMR, KNOWN-Hand, CondiMen, HandFlow, StableHand, and uncertainty-aware video HMR.

**Kill criterion:** an existing method already provides component-wise calibrated sign-SMPL-X hypotheses with effective selection, or selected-K does not outperform a deterministic baseline.

### Candidate V3 — H3: ordered hand–hand interaction

**Provisional novelty boundary:** not cross-hand attention, collision avoidance, or generic contact. The possible contribution is an explicit temporal bimanual state with depth order, surface-pair identity, and contact transition timing for sign reconstruction.

**Must distinguish from:** SGNify/TUCH, IntagHand, InterHand-family reconstruction, Dyn-HaMR, interacting-hand generative models, StableHand, and contact-aware HOI systems such as CHOIR.

**Kill criterion:** prior work already reconstructs the same ordered bimanual state from monocular video, or contact/order supervision improves plausibility metrics but not geometry/perceptual correctness.

### Candidate V4 — H4: body-anchored hand–body contact

**Provisional novelty boundary:** not self-contact or collision loss. The possible contribution is a sign-specific hand-part ↔ anatomical-body-region relation used as a depth/orientation anchor over time.

**Must distinguish from:** TUCH, PROX/RICH/self-contact work, dense human-scene contact, human-object contact reconstruction, SGNify, and any sign-phonology location/contact models.

**Kill criterion:** the same articulator-to-body region inference exists for monocular sign SMPL-X, or body-contact labels do not reduce relative hand geometry error.

### Candidate V5 — H10: reliability-certified pseudo-GT and hybrid inference

**Provisional novelty boundary:** not pseudo-label filtering, correction, or regression plus fitting. The possible contribution is independent, component-wise reliability/correction of sign SMPL-X labels whose causal downstream effect is validated under fixed inference.

**Must distinguish from:** SPIN, EFT, POCO, SignAvatars, Neural Sign Actors/SOKE curation, DexAvatar's filtering/correction, Meta-HMR, ScoreHMR, and general noisy-label learning.

**Kill criterion:** closely matching reliability/correction already exists for sign SMPL-X, or corrected labels do not improve independent mocap and fixed downstream TR-V2V over hard filtering.

---

## 7. Minimum evidence required before choosing a final method

| Decision gate | Required experiment | Hypotheses affected |
|---|---|---|
| Observation ceiling | Detector versus oracle 2D and initialization | H1, H2, H6, H8, H10 |
| Future-context value | Forward versus bidirectional under equal capacity and real/synthetic gaps | H1, H6 |
| Uncertainty usefulness | Calibration and automatic selected-K versus deterministic | H2, H6 |
| Interaction prevalence | Fraction and error of hand–hand/body contact and occlusion-order frames | H3, H4 |
| Contact causal value | Collision-only versus unordered versus ordered/indexed contact | H3, H4 |
| Semantic causal value | Predicted versus GT versus shuffled phonology under ambiguity gating | H5 |
| Signer variance | Leave-one-signer-out accurate 3D evaluation | H7, H8 |
| Camera headroom | Fixed versus calibrated/oracle camera | H9 |
| Label-quality headroom | Raw versus filtered versus independently corrected pseudo-GT | H10 |
| Protocol validity | One evaluator/manifest for DexAvatar, SOKE, and newer candidates | All |

**Decision rule:** if the corresponding oracle or controlled intervention shows negligible headroom, the associated hypothesis should be dropped before architecture design.

---

## 8. Primary-source shortlist used in this phase

### Direct sign reconstruction and semantics

- [DexAvatar](https://arxiv.org/abs/2512.21054) · [official code](https://github.com/kaustesseract/DexAvatar)
- [SGNify](https://arxiv.org/abs/2304.10482) · [official code](https://github.com/MPForte/SGNify)
- [Neural Sign Actors](https://arxiv.org/abs/2312.02702)
- [SignAvatars](https://arxiv.org/abs/2310.20436)
- [SOKE](https://arxiv.org/abs/2411.17799)
- [Toward Phonology-Guided Sign Language Motion Generation](https://arxiv.org/abs/2603.17388)
- [Meaningful Pose-Based Sign Language Evaluation](https://arxiv.org/abs/2510.07453)

### Temporal, occlusion, uncertainty, and hand reconstruction

- [DanceHMR](https://arxiv.org/abs/2605.18102)
- [MoRo](https://arxiv.org/abs/2601.16079)
- [RoHM](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RoHM_Robust_Human_Motion_Reconstruction_via_Diffusion_CVPR_2024_paper.html)
- [HandFlow](https://arxiv.org/abs/2607.11221)
- [StableHand](https://arxiv.org/abs/2605.18553)
- [Dyn-HaMR](https://arxiv.org/abs/2412.12861)
- [MaskHand](https://arxiv.org/abs/2412.13393)
- [ProHMR](https://arxiv.org/abs/2108.11944)
- [POCO](https://arxiv.org/abs/2308.12965)
- [ScoreHMR](https://arxiv.org/abs/2403.09623)

### Contact, optimization, adaptation, and pseudo labels

- [TUCH / On Self-Contact and Human Pose](https://arxiv.org/abs/2104.03176)
- [CHOIR](https://arxiv.org/abs/2605.20992)
- [ARCTIC](https://arctic.is.tue.mpg.de/)
- [CycleAdapt](https://arxiv.org/abs/2308.06554)
- [STRIDE](https://arxiv.org/abs/2312.16221)
- [Meta-HMR](https://arxiv.org/abs/2401.14121)
- [SMPLify-X](https://arxiv.org/abs/1904.05866)
- [SPIN](https://arxiv.org/abs/1909.12828)
- [EFT](https://arxiv.org/abs/2004.03686)

---

## Final Phase-4 conclusion

**FACT:** every generic mechanism requested in this phase already has substantial direct or adjacent prior art.

**INFERENCE:** the strongest remaining research hypotheses are defined by variables that direct sign reconstruction does not currently represent or evaluate: visibility through time, component-wise uncertainty, ordered bimanual interaction, articulator-to-body contact identity, and independently validated pseudo-label reliability.

**HYPOTHESIS:** one or more of H1–H4 could reduce hand error on the currently hidden hard strata, while H10 could improve broader upper-body accuracy through cleaner priors and initialization. This remains unproven until the Phase-3 oracle studies and Phase-5 novelty verification are complete.

**DECISION:** advance H1, H2, H3, H4, and H10 to exhaustive novelty verification. Do not combine them into a final architecture yet.
