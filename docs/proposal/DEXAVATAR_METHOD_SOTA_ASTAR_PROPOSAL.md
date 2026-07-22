# Method-Centered Research Proposal for SOTA 3D Sign Language Reconstruction

**Working paper concept:** **SignPoste      rior4D: Phonology- and Interaction-Conditioned Whole-Sequence Posterior Reconstruction**

**Primary benchmark:** SGNify, using the standard TR-V2V protocol used by SGNify and DexAvatar

**Target output:** SMPL-X upper-body and hand motion reconstructed from monocular sign-language video

**Research target:** a technically defensible, public, reproducible method suitable for an A*-level vision conference

**Audit date:** 13 July 2026

---

## 1. Scope and research decision

This document treats the SGNify TR-V2V evaluation as the fixed benchmark. It does **not** propose changing, criticizing, or optimizing around the evaluator. The task is to build a genuinely better reconstruction method that lowers:

- upper-body excluding face TR-V2V;
- left-hand TR-V2V; and
- right-hand TR-V2V.

The analysis is based on:

1. the [DexAvatar WACV 2026 paper and supplement](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html);
2. the original [official DexAvatar repository](https://github.com/kaustesseract/DexAvatar);
3. the pristine original code contained in this repository's history at commit `7e97916`, rather than the later NLF, WiLoR, DPoser-X, or other experimental additions in the current working tree; and
4. the [SGNify paper](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html), which introduced the benchmark protocol.

### Central recommendation

Do **not** make the paper's main claim “we replace HaMeR with a newer hand estimator” or “we replace a VAE with diffusion.” Those are useful baselines, but no longer sufficient novelty.

The recommended paper should reformulate reconstruction as **whole-sequence posterior inference under a structured sign-language motion model**:

> Given uncertain body, hand, image, and keypoint observations, infer the most probable coordinated SMPL-X sequence while preserving the sign's handshape, palm orientation, location, movement phase, symmetry/dominance, and contact relations.

The proposed method, provisionally called **SignPosterior4D (SP4D)**, has one coherent main idea:

> A phonology-conditioned relational diffusion prior jointly reconstructs upper body, wrists, and both hands over the complete sign, while an uncertainty model decides when to trust visual observations and when to infer through temporal and cross-part context.

This direction directly targets DexAvatar's largest methodological limitations and remains distinguishable from recent generic temporal whole-body recovery, hand-body stitching, and generic pose diffusion work.

---

## 2. Competitive target

### 2.1 Published and newly reported results

| Method | Upper body | Left hand | Right hand | Main idea |
|---|---:|---:|---:|---|
| SGNify | 55.63 | 19.22 | 17.50 | Hand symmetry and pose-invariance constraints |
| Neural Sign Actors | 46.42 | 16.17 | 15.23 | Curated SMPL-X annotations for sign production |
| EVA* | 40.38 | 13.73 | 13.68 | Multiple off-the-shelf pseudo-supervision sources |
| DexAvatar | **30.13** | **13.53** | **13.08** | Sign-specific static body and hand VAEs plus fitting |
| Tamaththul3D preprint | 29.28 | 10.65 | 8.90 | WiLoR-to-SMPL-X conversion, geometric wrist/forearm alignment, shoulder refinement |

DexAvatar's official target is `30.13 / 13.53 / 13.08`. However, the June 2026 [Tamaththul3D v2 preprint](https://arxiv.org/html/2605.05367v2) reports `29.28 / 10.65 / 8.90` on SGNify. Treat it as the strongest modern result to reproduce with the project's fixed TR-V2V implementation, then aim to beat **both** it and DexAvatar.

### 2.2 Proposed success gates

These are engineering targets, not promised results.

| Gate | Upper body | Left hand | Right hand | Meaning |
|---|---:|---:|---:|---|
| DexAvatar reproduction | approximately 30.13 | approximately 13.53 | approximately 13.08 | Confirms the original pipeline |
| Strong modern baseline | < 29.3 | < 10.7 | < 8.9 | Competitive with the strongest currently reported result |
| Paper-worthy target | **< 27.5** | **< 9.5** | **< 8.0** | Clear improvement with room beyond estimator replacement |
| Stretch target | < 26.5 | < 9.0 | < 7.5 | Strong SOTA result if supported by generalization experiments |

A small numerical lead alone is not enough for an A*-level submission. The target is a method whose gains are largest precisely on blur, occlusion, hand interaction, and rapid transitions—and whose ablations demonstrate why.

---

## 3. What original DexAvatar actually does

DexAvatar is an optimization pipeline, not an end-to-end video reconstruction network.

### 3.1 Observation and initialization pipeline

For each video frame, the original release obtains:

- SMPL-X body, camera, shape, and initial hand parameters from SMPLer-X;
- 2D whole-body joints from Sapiens;
- hand crops, 2D/3D hand joints, and MANO estimates from HaMeR; and
- the one-handed/two-handed sign class from SGNify's classifier.

It then optimizes low-dimensional latent variables decoded by SignBPoser and SignHPoser and renders an SMPL-X mesh.

### 3.2 Learned priors

DexAvatar trains two three-layer VAE priors:

- **SignBPoser**, a 33-dimensional body latent trained using filtered pseudo-SMPL-X sequences derived from sign videos; and
- **SignHPoser**, a 23-dimensional hand latent trained from glove/Vicon fingerspelling captured from eight signers.

The priors are frame-pose distributions. They are not sequence models and do not jointly represent body, wrist, left hand, and right hand.

### 3.3 Original fitting objective

The paper expresses the fitting loss as

$$
\mathcal{L}_{\text{Dex}} =
\mathcal{L}_{\text{joint}}
+ \lambda_b \mathcal{L}_{\text{B-prior}}
+ \lambda_h \mathcal{L}_{\text{H-prior}}
+ \lambda_{\text{pen}} \mathcal{L}_{\text{pen}}
+ \lambda_t \mathcal{L}_{\text{temp}}
+ \lambda_{bb} \mathcal{L}_{\text{body-bio}}
+ \lambda_{hb} \mathcal{L}_{\text{hand-bio}}.
$$

In the original code snapshot:

- the optimizer primarily updates the SignBPoser and SignHPoser latent variables;
- body and hand latents begin at zero for each frame;
- the decoded poses are strongly anchored to the SMPLer-X and HaMeR initial parameters;
- the available hand-joint 3D term compares wrist-relative, independently normalized **depth coordinates** rather than full metric XYZ geometry, and the released YAML sets all of its `data_3d_weights` to zero; and
- the temporal term is a first-order penalty on the body pose against the preceding processed frame.

Relevant original paths are [fitting.py](../dexavatar_fitting/smplifyx/fitting.py), [fit_single_frame.py](../dexavatar_fitting/smplifyx/fit_single_frame.py), [main.py](../dexavatar_fitting/smplifyx/main.py), and [fit_smplx_vposer_x.yaml](../dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml). Because the current files include later experiments, use `git show 7e97916:<path>` when checking the released implementation.

### 3.4 Why DexAvatar made a large advance

DexAvatar's core insight remains correct: a signing-specific pose distribution is much more useful than a generic daily-motion prior. Its filtering improved body error from `34.06` to `30.28`, and corrected hand-prior training improved the three reported errors from `31.34 / 14.19 / 13.92` to `30.17 / 13.55 / 13.06` before the final biomechanical term.

The next paper should preserve this sign-specific insight while replacing the independent, static, deterministic prior with a coordinated, temporal, uncertainty-aware model.

---

## 4. Method weaknesses that directly limit TR-V2V

This section analyzes reconstruction weaknesses only. They are ordered by likely impact on the three benchmark scores.

### 4.1 The method does not reconstruct a sign as a sequence

DexAvatar processes frames sequentially and uses the previous body pose as a local smoothness reference. It never sees future frames when resolving the current frame.

Consequences:

- a blurred frame cannot borrow evidence from a clear frame five frames later;
- a hand temporarily hidden by the other hand cannot be inferred from the entry and exit configuration jointly;
- errors can propagate causally from one frame to the next;
- first-order smoothing penalizes legitimate fast transitions; and
- finger articulation receives no equivalent learned whole-sequence constraint.

This is the most important weakness for a video problem.

### 4.2 Body and hands are statistically independent

SignBPoser and SignHPoser are separately trained and separately decoded. The two hands also use independent latent vectors.

But signing is coordinated:

- wrist orientation constrains plausible forearm rotation;
- arm trajectory predicts where and how a hand will appear;
- symmetric signs correlate left and right handshape and motion;
- passive-hand poses depend on the active hand and contact geometry; and
- hand-to-face, hand-to-torso, and hand-to-hand relations persist across time.

An independent prior can generate anatomically plausible parts that are jointly inconsistent. This affects upper-body TR-V2V through arm/wrist errors and hand TR-V2V through wrong palm orientation and articulation.

### 4.3 SignHPoser cannot learn wrist–forearm coordination

The supplement states that wrist rotations from the MANUS setup could not be transferred reliably because of rig T-pose and bone-roll incompatibility. Consequently, SignHPoser learns finger articulation but not the full global hand orientation relationship to the wrist and forearm.

This is a structural limitation, not merely insufficient data. Precise palm orientation is linguistically important and contributes strongly to vertex error even when finger angles look plausible.

### 4.4 Hand supervision discards most metric 3D information

The original fitting source implements a hand-joint term that selects the depth channel, makes it wrist-relative, and normalizes prediction and observation independently. The released configuration sets this term's scheduled weights to zero, so the effective hand supervision comes mainly from 2D joints and strong HaMeR pose-parameter anchors. Even if enabled, the implemented term would provide only a weak cue about the actual metric 3D hand configuration.

The optimization does not fully exploit:

- metric XYZ bone geometry;
- palm plane and palm normal;
- wrist-to-MCP orientation;
- scale-consistent fingertip positions;
- left/right relative hand pose; or
- hand-to-body position.

A modern method should consume full 3D hypotheses and express their uncertainty instead of reducing them to normalized depth.

### 4.5 The optimizer is strongly tied to a possibly wrong initializer

The decoded body and hand poses are explicitly supervised by SMPLer-X and HaMeR parameters, with large initialization weights in all fitting stages. When an initializer fails under blur or occlusion, the learned prior is asked to remain close to that error.

The system therefore behaves more like constrained denoising than true posterior inference. It has no mechanism to maintain several plausible hypotheses and choose one using evidence elsewhere in the video.

### 4.6 The pose priors are unimodal static VAEs

A small Gaussian VAE is efficient, but it tends to average ambiguous configurations. This is especially damaging for:

- finger flexion with similar 2D projection;
- palm-facing versus palm-away ambiguity;
- crossed or occluded hands;
- thumb opposition; and
- rare but valid handshapes.

The solution is not “diffusion” by name alone. The useful advance is a **conditional sequence distribution** that can represent several plausible motions and use video observations to collapse the posterior.

### 4.7 Temporal consistency is not phase aware

Signing contains holds, transitions, repetitions, and rapid directional movements. A uniform zero-velocity or derivative penalty treats these differently meaningful phases as the same process.

It can:

- oversmooth a rapid change between handshapes;
- erase the apex of a motion;
- retain jitter during an intended hold if the data term is strong; and
- introduce lag because only past motion is used.

A phase-aware model should be stiff during holds and permissive during meaningful transitions.

### 4.8 Hard sign decisions discard weak but useful motion

For a predicted one-handed sign, the original fitting disables the non-dominant arm and hand. In real signing, the nominally inactive side can still show posture adjustment, preparatory movement, or stabilization.

A probabilistic dominance variable is preferable to a hard switch. It can strongly regularize a passive side without forcing it to remain unchanged.

### 4.9 Contact is treated as collision avoidance, not a sign relation

The interpenetration loss prevents invalid geometry, but valid signing frequently contains intentional contact. Collision avoidance alone cannot tell the system:

- which hand is in front;
- which fingertip touches which surface;
- when contact begins and ends;
- whether contact should persist through a hold; or
- how two hands move together after contact.

Contact should be a predicted relational state, not only a repulsion term.

### 4.10 Observation confidence is incomplete

DexAvatar weights 2D keypoints by detector confidence, but it does not estimate calibrated uncertainty for SMPL-X pose, MANO articulation, palm orientation, or temporal consistency.

Detector confidence alone is insufficient: a hand estimator can be confidently wrong under mirror ambiguity, severe occlusion, or a poor crop. Cross-estimator disagreement and temporal inconsistency provide valuable additional uncertainty cues.

### 4.11 Training data covers pose but not the complete reconstruction failure process

The prior sees pose samples, but is not explicitly trained to correct the burst errors generated by image estimators under blur, occlusion, truncation, and hand interaction. Random Gaussian corruption is not an adequate substitute for realistic estimator failure.

The new model should be trained with residuals produced by the same public initializers used at inference, plus synthetic burst masks and blur. This aligns training with the actual inverse problem.

---

## 5. Why obvious upgrades are not enough for the paper

### 5.1 WiLoR substitution is a required baseline, not the novelty

[WiLoR](https://github.com/rolpotamias/WiLoR) is a strong public CVPR 2025 hand localizer and reconstructor. Tamaththul3D already shows that replacing the hand parameters and aligning the wrist can reduce the reported hand errors substantially. Therefore:

- use WiLoR as a strong initializer;
- reproduce direct MANO-to-SMPL-X conversion;
- reproduce a differentiable wrist/forearm alignment baseline; but
- do not present this alone as the new paper.

### 5.2 Generic temporal body–hand fusion is already occupied

The May 2026 [DanceHMR preprint](https://arxiv.org/html/2605.18102) jointly fuses body and hand observations before temporal modeling, adds close-up augmentation, visibility-aware supervision, and fingertip-focused training. A generic “temporal Transformer for body and hands” would overlap heavily.

Our differentiation must be sign-specific and explicit: compositional phonology, sign-phase dynamics, relational contact, calibrated posterior sampling, and benchmarked semantic preservation.

### 5.3 Generic whole-body diffusion is already occupied

[DPoser-X](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html) already uses masked diffusion to model full-body part dependencies, while [FUSION](https://arxiv.org/abs/2601.03959) models joint body-and-hand motion. Replacing SignBPoser with either prior is useful, but not sufficient novelty.

The proposed contribution is a **sign posterior**, not a generic motion prior: it conditions on inferred phonological structure and explicit hand–hand/hand–body relations and is optimized against uncertain video observations.

### 5.4 Post-hoc smoothing is already occupied and can hurt signs

Tamaththul3D uses velocity, acceleration, and jerk smoothing. Temporal hand methods and DanceHMR also address jitter. Uniform smoothing does not resolve ambiguity and can erase meaningful fast articulation.

The paper should instead learn when a frame belongs to a hold, transition, repetition, or contact interval and adjust the posterior dynamics accordingly.

### 5.5 Stronger biomechanical limits have low remaining upside

DexAvatar's final biomechanical hand term improves the left hand marginally and slightly worsens the right hand. Harder joint limits may improve visual plausibility without reducing TR-V2V. Use biomechanics as a safety regularizer, not the central research direction.

---

## 6. Proposed method: SignPosterior4D

### 6.1 Problem formulation

Given a monocular RGB sequence

$$
\mathbf{I}_{1:T} = \{I_1, \ldots, I_T\},
$$

recover a temporally coherent SMPL-X sequence

$$
\mathbf{X}_{1:T} = \{X_1, \ldots, X_T\}
$$

that represents upper-body pose, both wrist orientations, both hand articulations, shared body shape, and camera parameters.

Instead of solving every frame independently, infer

$$
p(\mathbf{X}_{1:T} \mid \mathbf{O}_{1:T}, \mathbf{Z}_{\text{ph}}, \mathbf{R}_{1:T}),
$$

where:

- $\mathbf{O}_{1:T}$ are uncertain observations from RGB, 2D keypoints, body estimators, and hand estimators;
- $\mathbf{Z}_{\text{ph}}$ is a probabilistic phonological representation inferred from the video; and
- $\mathbf{R}_{1:T}$ contains hand–hand and hand–body relational states.

The important shift is from **fitting one deterministic pose to noisy estimates** to **inferring a distribution over a complete coordinated sign and selecting the sample best supported by the video**.

### 6.2 Sequence state representation

Use continuous 6D rotations internally and convert to axis-angle only when calling SMPL-X. Define each frame as

$$
X_t = [
\theta_t^{\text{torso}},
\theta_t^{\text{arms}},
R_{t,w}^{L}, R_{t,w}^{R},
\theta_{t,h}^{L}, \theta_{t,h}^{R},
\beta,
\pi_t
].
$$

Recommended canonicalization:

- upper-body rotations and joint positions in a torso-centered frame;
- hand articulation in a wrist-centered frame;
- palm orientation retained explicitly relative to the forearm and torso;
- shared shape $\beta$ across the sequence; and
- camera only in the observation model, not inside the sign-motion prior.

This separation prevents camera variation from consuming motion-prior capacity while retaining the wrist orientation needed for accurate mesh vertices.

### 6.3 Observation bank

The initial implementation should use frozen public experts:

- **body:** SMPLer-X as the required baseline; optionally SMPLest-X or Hand4Whole++ as an additional hypothesis;
- **hands:** WiLoR as the primary MANO hypothesis; optionally HaMeR, NLF joints, MaskHand samples, or OmniHands for diversity;
- **2D evidence:** Sapiens or RTMPose whole-body keypoints and high-resolution hand keypoints;
- **appearance:** full-body features plus independent left/right hand-crop features; and
- **motion:** short-term feature tracks or optical flow around wrists, fingertips, and visible hand edges.

The required configuration must depend only on released code and checkpoints—at minimum, public SMPLer-X and WiLoR. Additional experts such as MaskHand or OmniHands should remain optional unless compatible weights and licenses are available. The paper and release must make the single-expert public configuration fully reproducible; optional expert ensembles belong in a separate ablation.

For every joint and frame, construct an observation token

$$
O_{t,j} = [\hat{x}_{t,j}^{2D}, \hat{x}_{t,j}^{3D},
\hat{R}_{t,j}, f_{t,j}, c_{t,j}, v_{t,j}, d_{t,j}],
$$

containing the estimate, image feature, detector confidence, visibility, and disagreement between experts.

The method should not concatenate estimates blindly. A learned reliability head predicts heteroscedastic uncertainty:

$$
(\mu_{t,j}, \log \sigma^2_{t,j}) = g_{\omega}(O_{t,j}, O_{t-1:t+1,j}).
$$

Train this head on datasets with 3D supervision and realistic corruptions. At inference, high uncertainty reduces an observation's influence and lets the sign prior and neighboring frames fill the gap.

### 6.4 Compositional phonology latent

The phonology representation is the main sign-specific distinction from generic video HMR. It must not require a known gloss at test time.

Infer a soft latent

$$
Z_{\text{ph}} = [
H^L, H^R,
O^L, O^R,
L^L, L^R,
M^L, M^R,
S, D, C
],
$$

where:

- $H$: handshape tokens;
- $O$: palm and finger orientation;
- $L$: location in torso-centered signing space;
- $M$: movement type and direction;
- $S$: symmetry relationship;
- $D$: dominance/activity probability; and
- $C$: hand–hand or hand–body contact type.

#### How to obtain supervision

Use three complementary sources:

1. **Geometry-derived labels.** Extract palm normals, handshape clusters, trajectory direction, symmetry, activity, and contact directly from curated 3D sequences.
2. **HamNoSys supervision.** The [SignAvatars dataset](https://signavatars.github.io/) includes HamNoSys, word, and sentence prompts for part of its 70,000 sequences and can supply explicit phonological structure.
3. **Video self-supervision.** Predict the attributes from the RGB sequence, reconstruct the 3D sequence, and require the deterministic geometry extractor applied to the reconstruction to recover the same attributes.

This gives a semantic–kinematic cycle:

$$
\hat{Z}_{\text{ph}} = q_{\phi}(I_{1:T}),
\qquad
\tilde{Z}_{\text{ph}} = F_{\text{geom}}(X_{1:T}),
$$

$$
\mathcal{L}_{\text{ph-cycle}} =
D(\hat{Z}_{\text{ph}}, \tilde{Z}_{\text{ph}}).
$$

Unlike SGNify's hard six-group selection, this representation is compositional, probabilistic, and time aware. It can express uncertainty—for example, `0.65 symmetric` rather than forcing a binary decision.

### 6.5 Relational hand–body graph

Build a small relational graph at every frame. Nodes represent:

- both wrists;
- palm centers and normals;
- fingertips and MCP joints;
- head, chin, chest, shoulders, and upper arms; and
- optional dense contact anchors on SMPL-X.

Edges encode:

- wrist-to-wrist vector;
- fingertip-to-fingertip and palm-to-palm distances;
- hand-to-face and hand-to-torso signed distances;
- relative palm orientation;
- front/back ordering; and
- contact probability and persistence.

These relations are more stable under global translation and more informative under occlusion than two independent hand poses. A relational encoder produces $R_t$, which conditions both the prior and the final refinement.

For a predicted contact edge $(a,b)$, use a soft target distance rather than pure repulsion:

$$
\mathcal{L}_{\text{contact}} =
\sum_{t,a,b} c_{t,a,b}
\rho\left(d(V_{t,a}, V_{t,b}) - \delta_{a,b}\right)
+ \lambda_{\text{slip}} c_{t,a,b}c_{t-1,a,b}
\left\|\Delta r_{t,a,b}\right\|_1.
$$

Retain a separate non-penetration term for vertex pairs that are not in contact.

### 6.6 Phonology-conditioned masked diffusion prior

Train a hierarchical spatio-temporal denoiser over complete sign sequences. A practical architecture is:

1. **intra-part graph blocks** for torso/arms, left hand, and right hand;
2. **cross-part relational attention** through wrist and contact tokens;
3. **bidirectional temporal attention** over the complete central sign; and
4. **cross-attention to soft phonology and uncertain observation tokens**.

At diffusion step $\tau$:

$$
\hat{\epsilon} =
\epsilon_{\theta}(X^{\tau}_{1:T}, \tau,
Z_{\text{ph}}, R_{1:T}, O_{1:T}).
$$

The training corruption process must resemble reconstruction failures:

- mask individual fingers;
- mask a complete hand for a burst of frames;
- perturb wrist orientation;
- inject realistic SMPLer-X, HaMeR, and WiLoR residuals;
- swap or mirror ambiguous hand hypotheses;
- simulate image blur and keypoint dropout;
- truncate hands at image boundaries; and
- perturb arm and hand jointly to reproduce broken-wrist cases.

Use mixed masked training so that high-quality hand-only datasets improve finger modeling without requiring full-body labels, following the useful principle demonstrated by DPoser-X. SignAvatars then adapts the generic model to signing dynamics and hand–body coordination.

### 6.7 Phase-aware dynamics

Predict a per-frame distribution over `hold`, `transition`, `repetition`, and `contact-transition`:

$$
p_t = \operatorname{softmax}(h_{\text{phase}}(O_{1:T}, Z_{\text{ph}})).
$$

Use the phase to control dynamics:

$$
\mathcal{L}_{\text{phase-dyn}} =
\sum_t p_t^{\text{hold}}\|\Delta X_t\|_1
+ p_t^{\text{transition}}
\rho(\Delta^2 X_t)
+ p_t^{\text{contact}}
\|\Delta R_t^{\text{contact}}\|_1.
$$

The model is therefore strongly stable during a hold, smooth but responsive during a transition, and relationally stable during sustained contact. It avoids the central failure of uniform smoothing: treating meaningful velocity as noise.

### 6.8 Posterior sampling and final refinement

At inference:

1. run the frozen observation experts;
2. convert all candidates into the shared SMPL-X/torso canonicalization;
3. predict observation uncertainty, phonology, phase, and contact;
4. sample $K$ complete sequence hypotheses from the conditioned diffusion posterior;
5. refine each hypothesis with differentiable SMPL-X observation and relation losses; and
6. select the hypothesis with the best video-evidence likelihood.

The final energy can be written as

$$
E(X) =
\lambda_{\text{obs}}E_{\text{uncertain-obs}}
+ \lambda_{2D}E_{\text{reproj}}
+ \lambda_{\text{img}}E_{\text{image}}
+ \lambda_{\text{rel}}E_{\text{relation}}
+ \lambda_{\text{ph}}E_{\text{ph-cycle}}
+ \lambda_{\text{dyn}}E_{\text{phase-dyn}}
+ \lambda_{\text{bio}}E_{\text{biomech}}
+ \lambda_{\text{pen}}E_{\text{penetration}}.
$$

For heteroscedastic 3D observations:

$$
E_{\text{uncertain-obs}} =
\sum_{t,j,e}
\frac{\rho(X_{t,j} - \mu^{e}_{t,j})}{2(\sigma^{e}_{t,j})^2}
+ \frac{1}{2}\log (\sigma^{e}_{t,j})^2,
$$

where $e$ indexes observation experts. This lets the method use strong WiLoR evidence on a clear right hand while rejecting a confident but temporally inconsistent prediction for an occluded left hand.

Use only video evidence for sample selection—never benchmark ground truth. A useful selection test is masked-observation prediction: temporarily withhold a subset of reliable keypoints or features and choose the hypothesis that best predicts them.

For the isolated-sign SGNify benchmark, the model can process each supplied sign clip as one sequence. For continuous signing, run overlapping temporal windows and merge them with shared-frame posterior confidence; a learned phase/boundary token can prevent a required manual gloss segmentation from becoming an inference assumption.

### 6.9 Two inference modes

The research should expose two modes from the same model.

#### SP4D-Fast

- one deterministic denoising path;
- one short SMPL-X refinement;
- suitable for dataset annotation and broad qualitative studies.

#### SP4D-Best

- 4–8 posterior hypotheses only where uncertainty is high;
- bidirectional complete-sign context;
- longer differentiable refinement;
- used for the main TR-V2V table.

This makes the paper useful beyond the benchmark while allowing the highest-accuracy configuration to demonstrate the method's ceiling.

---

## 7. Why the proposed method should reduce each score

| DexAvatar failure | SP4D mechanism | Expected score affected |
|---|---|---|
| Incorrect hand under blur | Bidirectional temporal infilling plus handshape token | Left/right hand |
| Wrong palm orientation | Explicit palm/forearm state and full XYZ observation | Hand and upper body |
| Broken wrist after hand replacement | Joint wrist–forearm state and kinematic refinement | Upper body and hand |
| Occluded interacting hands | Relational graph, contact state, multi-hypothesis posterior | Both hands |
| Jitter during held handshape | Hold-aware dynamics | Both hands |
| Oversmoothed rapid transition | Transition-aware dynamics | All three |
| Confident bad initializer | Calibrated uncertainty and expert disagreement | All three |
| Independent left/right guesses | Symmetry/dominance-conditioned cross-hand attention | Both hands |
| Arm error around hand contact | Hand-to-body relations back-propagated through the arm chain | Upper body |
| Rare sign pose pulled to VAE mean | Multimodal conditional prior | All three |

The expected gain sequence is:

1. a strong modern hand expert makes the large first reduction in hand TR-V2V;
2. full XYZ and wrist/forearm integration prevent losing that gain when inserting hands into SMPL-X;
3. temporal posterior inference corrects frames where the expert is wrong;
4. phonology prevents temporally plausible but semantically wrong handshape or orientation; and
5. relational reasoning improves two-hand/contact cases and feeds better wrist constraints into the upper body.

---

## 8. Training strategy

### 8.1 Data sources

#### Sign-specific whole-body motion

- **SignAvatars:** 70,000 sequences, 153 signers, and 8.34 million SMPL-X frames with multiple prompt types. Use its highest-quality subset after automatic quality filtering.
- **How2Sign-derived public annotations:** useful for continuous ASL and co-articulation if their license and release permit the selected representation.
- **DexAvatar SignBPoser data/checkpoint:** useful as a teacher or initialization, but the new method should not depend on unavailable private hand mocap.

#### High-quality hands and coordination

- InterHand2.6M for interacting hands;
- ARCTIC for accurate hand articulation and body context;
- DexYCB or HanCo for visible hand geometry;
- WHIM from WiLoR for in-the-wild hands;
- AssemblyHands-X if the data license and release allow use; and
- MANO-compatible synthetic motion rendered with realistic sign-video crops.

#### General whole-body motion

- Motion-X and other SMPL-X datasets can pretrain body–hand topology;
- use them at lower sampling probability during sign adaptation so everyday gestures do not dominate signing dynamics.

The SGNify quantitative benchmark must remain evaluation-only.

### 8.2 Quality filtering

SignAvatars is large but automatically annotated. Avoid training a denoiser to reproduce annotation errors.

Retain or upweight frames/sequences with:

- agreement between two body/hand estimators;
- low 2D reprojection error;
- stable identity and shape;
- valid joint rotations and no mesh collapse;
- plausible hand bone lengths;
- temporally consistent palm orientation; and
- image/mesh silhouette agreement where available.

Use the remaining lower-quality sequences only with confidence-weighted supervision or as noisy inputs paired with a teacher-refined target.

### 8.3 Four-stage curriculum

#### Stage A: generic structured motion pretraining

Train intra-part and cross-part blocks with mixed whole-body and hand-only data. Use masked-part training so that missing labels do not become zero poses.

#### Stage B: sign-language adaptation

Train on SignAvatars with higher sampling of two-hand interaction, fast articulation, upper-body crops, and rare handshapes. Add geometry-derived phonology and phase supervision.

#### Stage C: reconstruction-corruption training

Run the exact planned observation experts over training videos. Learn to map their real residual distributions to refined sequences. Add burst occlusion, blur, crop loss, hand swapping, and wrist perturbation.

#### Stage D: posterior and evidence calibration

Train uncertainty and multi-hypothesis selection on held-out signers and datasets. Calibrate predicted variance using negative log likelihood and coverage curves, not only regression error.

### 8.4 Losses during training

The complete training loss can be:

$$
\mathcal{L}_{\text{train}} =
\lambda_{\epsilon}\mathcal{L}_{\text{diff}}
+ \lambda_v\mathcal{L}_{\text{vertex}}
+ \lambda_j\mathcal{L}_{\text{joint}}
+ \lambda_r\mathcal{L}_{\text{rotation}}
+ \lambda_{tip}\mathcal{L}_{\text{fingertip}}
+ \lambda_{rel}\mathcal{L}_{\text{relation}}
+ \lambda_c\mathcal{L}_{\text{contact}}
+ \lambda_p\mathcal{L}_{\text{phonology}}
+ \lambda_{cyc}\mathcal{L}_{\text{ph-cycle}}
+ \lambda_{phase}\mathcal{L}_{\text{phase}}
+ \lambda_u\mathcal{L}_{\text{uncertainty}}.
$$

Important implementation choices:

- supervise vertices and fingertip joints, not rotations alone;
- give palm orientation an explicit geodesic rotation loss;
- balance left and right hands rather than allowing the body vertex count to dominate;
- sample signs rather than frames to preserve rare sequences;
- mirror sequences only with correct handedness and dominance transformations; and
- keep shape shared across the sequence.

---

## 9. Practical development ladder

The full method should be reached through publishable intermediate baselines. Each stage must use the same benchmark inputs and TR-V2V script.

### Phase 0: original release reproduction

Goal: reproduce DexAvatar from commit `7e97916` and archive its meshes and scores.

Deliverables:

- exact original configurations;
- per-sign error table;
- per-frame predictions for paired statistical comparisons; and
- qualitative failure gallery grouped by blur, occlusion, contact, and fast motion.

### Phase 1: strongest non-novel baseline

Build:

- SMPLer-X body;
- WiLoR hands;
- correct MANO-to-SMPL-X conversion;
- differentiable wrist/forearm alignment;
- full XYZ and 2D observation refinement; and
- simple bidirectional smoothing.

This should establish whether the codebase can reach the modern `approximately 29 / 10.6 / 8.9` operating region. It is essential, but it is not the final method.

### Phase 2: uncertainty-aware whole-sequence refiner

Add:

- complete-sequence windows;
- observation tokens and reliability head;
- realistic burst corruption;
- joint body/wrist/hand state; and
- deterministic temporal denoising.

This is the fastest way to test whether temporal inference reduces TR-V2V beyond the strong initializer.

### Phase 3: relational diffusion posterior

Replace deterministic denoising with masked diffusion and add:

- cross-hand and hand–body graph;
- contact prediction;
- $K$-hypothesis inference; and
- evidence-based sample selection.

This should produce the largest gain on occluded and interacting hands.

### Phase 4: phonology and sign phase

Add:

- soft handshape/orientation/location/movement representation;
- semantic–kinematic cycle loss;
- probabilistic symmetry and dominance;
- hold/transition/repetition/contact phase; and
- phase-controlled dynamics.

This is the paper-defining sign-specific contribution.

### Phase 5: efficiency and release

Distill the best posterior into SP4D-Fast, document all preprocessing, release pretrained models and predictions, and provide a single reproducible benchmark command.

---

## 10. Required ablation study

The main table should report all three standard TR-V2V regions for every row.

| Row | Configuration | Scientific question |
|---|---|---|
| A0 | Original DexAvatar | Published reference |
| A1 | Modern body + WiLoR | How much comes from newer observations? |
| A2 | A1 + wrist/forearm integration + full XYZ | Does geometrically correct fusion help? |
| A3 | A2 + deterministic bidirectional refiner | Does whole-sequence context help? |
| A4 | A3 + uncertainty weighting | Can the model reject bad observations? |
| A5 | A4 + relational graph/contact | Do interactions improve hands and arms? |
| A6 | A5 + diffusion posterior, K=1 | Does a generative prior help without best-of-K? |
| A7 | A6 + K hypotheses and evidence selection | Does ambiguity modeling help? |
| A8 | A7 + phonology conditioning | Is the method specifically sign aware? |
| A9 | A8 + phase-aware dynamics | Does it preserve motion while stabilizing holds? |

Additional controlled ablations:

- no future context versus bidirectional context;
- generic motion prior versus sign-adapted prior;
- hard SGNify-style class versus soft compositional phonology;
- no handshape token, no orientation token, no movement token, no relation token;
- no burst corruption training;
- single estimator versus multi-expert observations;
- predicted uncertainty versus detector confidence only;
- `K = 1, 2, 4, 8` hypotheses;
- uniform smoothing versus phase-aware dynamics;
- no contact, collision-only, and predicted contact;
- 16-, 32-, 64-, and full-sign temporal context; and
- SignAvatars data-scale curves at 10%, 25%, 50%, and 100%.

Never report only the full stack. Reviewers must be able to see that phonology, interaction, and posterior inference each provide value beyond a stronger hand estimator.

---

## 11. Evaluation package for an A*-level paper

### 11.1 Primary quantitative result

- standard SGNify TR-V2V on all 2,872 central frames;
- upper body excluding face, left hand, and right hand;
- comparison against original DexAvatar and every reproducible public baseline;
- per-sign paired differences and bootstrap 95% confidence intervals; and
- mean, median, and worst-decile sign error in the supplement.

### 11.2 Diagnostic subsets

Create annotations without altering the benchmark metric:

- one-handed versus two-handed;
- symmetric versus asymmetric;
- contact versus non-contact;
- hand–hand versus hand–body contact;
- visible versus partially occluded versus severely occluded;
- low versus high blur;
- hold versus transition;
- slow versus fast wrist/fingertip velocity; and
- left/right estimator disagreement.

Report the same TR-V2V within each subset. The proposed method should show its largest relative improvement on severe occlusion, contact, and transition cases.

### 11.3 Cross-dataset generalization

The SGNify benchmark is small. A strong paper should also test:

- UBody or ARCTIC for whole-body/hand reconstruction;
- InterHand2.6M for interacting-hand reconstruction;
- How2Sign, PHOENIX-2014T, WLASL, and MM-WLAuslan for qualitative and 2D-consistency evaluation where 3D ground truth is unavailable; and
- at least one signer/language held out from sign-prior training.

Do not tune the method separately for every language. The compositional attributes should transfer even when gloss vocabularies do not.

### 11.4 Robustness tests

Apply controlled perturbations to frames with valid ground truth:

- Gaussian and motion blur;
- 10%, 25%, and 40% hand occlusion;
- complete hand dropout for 4, 8, and 16 consecutive frames;
- crop truncation;
- 2D keypoint noise;
- wrong hand hypothesis injected into one expert; and
- frame-rate reduction.

Plot TR-V2V against corruption severity. This directly tests the posterior and uncertainty claims.

### 11.5 Temporal and semantic metrics

TR-V2V remains the main benchmark, but the paper's temporal and phonological claims require additional evidence:

- MPJVE, acceleration error, and jerk;
- fingertip trajectory error;
- palm-normal geodesic error;
- contact precision/recall and contact slip;
- handshape/orientation/location attribute accuracy;
- sign recognition accuracy from reconstructed 3D motion; and
- a Deaf-signer perceptual study measuring sign identity and naturalness.

The semantic evaluation is important: a smoother mesh is not automatically a more correct sign.

### 11.6 Efficiency

Report:

- expert preprocessing time;
- SP4D-Fast and SP4D-Best runtime per frame/sign;
- GPU memory;
- number of diffusion steps and hypotheses; and
- accuracy/runtime trade-off.

Tamaththul3D explicitly emphasizes runtime, so accuracy must not be the only comparison.

---

## 12. Publication novelty and positioning

### 12.1 Recommended title direction

**SignPosterior4D: Phonology-Conditioned Relational Diffusion for 3D Sign Language Reconstruction**

Alternative titles:

- **Beyond Framewise Priors: Whole-Sequence Posterior Inference for 3D Signing Avatars**
- **PhonoSign4D: Uncertainty-Aware 4D Hand–Body Reconstruction from Monocular Signing Video**
- **RelSign4D: Interaction- and Phase-Aware 3D Sign Reconstruction**

### 12.2 Paper thesis

> Existing sign reconstruction methods use signing-specific constraints or pose priors but treat uncertain observations, body–hand coordination, and time incompletely. We introduce a structured posterior that combines compositional sign phonology with relational 4D motion, allowing visible frames and coordinated body parts to disambiguate occluded hand articulation without oversmoothing meaningful transitions.

### 12.3 Defensible contributions

1. **A compositional sign posterior.** The first reconstruction prior conditioned jointly on inferred handshape, orientation, location, movement, symmetry/dominance, and contact without requiring gloss labels at inference.
2. **Relational whole-sequence inference.** A masked diffusion model that jointly represents upper body, wrist orientation, both hands, and hand–body contact over complete signs.
3. **Uncertainty- and phase-aware reconstruction.** Calibrated expert fusion and hold/transition-aware dynamics that infill occlusions without suppressing linguistically meaningful motion.
4. **A rigorous reconstruction study.** Standard TR-V2V plus cross-dataset, corruption, temporal, contact, and semantic evaluations.

If the full method is too large for one paper, keep contributions 1–3 and treat efficient distillation as future work.

### 12.4 Clear distinction from nearby work

| Nearby work | Already contributes | SP4D must contribute beyond it |
|---|---|---|
| SGNify | Hard symmetry and pose-invariance classes | Soft compositional phonology, learned phase, relational posterior |
| DexAvatar | Static sign-specific body/hand VAEs | Unified body–wrist–hand sequence distribution |
| Tamaththul3D | WiLoR conversion and geometric arm alignment | Learned temporal disambiguation and sign semantics |
| Hand4Whole++ | Single-frame body/hand feature conditioning and rigid alignment | Complete-sign probabilistic reconstruction |
| DanceHMR | Generic temporal hand-aware whole-body regression | Sign phonology, contact graph, phase-aware posterior, hypotheses |
| DPoser-X | Generic masked whole-body diffusion prior | Sign-conditioned temporal and relational inverse problem |
| FUSION | Generic body-and-hand motion diffusion | Video-conditioned reconstruction and sign structure |
| MaskHand | Probabilistic single-image hand recovery | Coordinated two-hand/body sequence inference |
| OmniHands | Relation-aware interactive 4D hands | Explicit signing semantics and full SMPL-X upper body |
| PAD-Hand | Physics-aware hand motion diffusion | Signing phonology, two-hand/body relations, RGB posterior |

### 12.5 Claims to avoid

Do not claim:

- “first temporal whole-body hand reconstruction”;
- “first diffusion prior for body and hands”;
- “first multi-hypothesis hand reconstruction”;
- “first hand–body integration”;
- “first linguistic prior for sign reconstruction”; or
- “SOTA” before every public method has been run under the fixed protocol.

The novel intersection is narrower and stronger: **compositional phonology-conditioned relational posterior inference for complete-sign 3D reconstruction**.

---

## 13. Reviewer risks and how to neutralize them

### Risk 1: “The gain comes only from WiLoR.”

Required response: show A1/A2, then consistent additional reductions from temporal posterior, relations, phonology, and phase. Include HaMeR and WiLoR initializations to demonstrate initializer-agnostic refinement.

### Risk 2: “This is DanceHMR adapted to signs.”

Required response: demonstrate compositional phonology, sign phase, contact graph, and multi-hypothesis posterior ablations. Evaluate attribute preservation and sign recognition, which generic HMR does not target.

### Risk 3: “The sign prior memorizes SGNify signs.”

Required response: keep SGNify evaluation-only, deduplicate training sequences by source/gloss/signer, evaluate unseen signers and languages, and publish the split hashes.

### Risk 4: “Large pseudo-3D data only reproduces teacher bias.”

Required response: quality filter with multiple experts, use real 3D hand datasets, show results by training-target quality, and demonstrate improvement over every teacher.

### Risk 5: “Diffusion makes the method slow and complicated.”

Required response: report SP4D-Fast, use uncertainty-triggered sampling only for difficult intervals, and distill the multi-sample posterior into a deterministic refiner.

### Risk 6: “Temporal smoothness improves appearance but not accuracy.”

Required response: report TR-V2V, fingertip trajectory error, hold/transition subsets, and the uniform-smoothing ablation. The method must lower spatial error as well as jitter.

### Risk 7: “Phonology labels are noisy or language specific.”

Required response: use geometry-derived continuous attributes, report calibration/attribute accuracy, and test across languages. Treat HamNoSys as privileged training information, not a required inference input.

### Risk 8: “Contact constraints force wrong interactions.”

Required response: make contact probabilistic and uncertainty weighted, ablate oracle/predicted/no contact, and disable contact energy when its posterior is diffuse.

---

## 14. Recommended first experiments

The fastest evidence-producing order is:

1. reproduce original DexAvatar meshes and scores;
2. build a clean SMPLer-X + WiLoR + wrist-alignment baseline;
3. replace normalized depth-only hand supervision with confidence-weighted full XYZ, palm-normal, and fingertip observations;
4. optimize 32–64 frame bidirectional windows while sharing shape;
5. train a deterministic corruption-to-clean temporal refiner on SignAvatars;
6. add reliability prediction and measure improvements specifically on bad-initializer frames;
7. add the relational graph and contact head;
8. convert the refiner into a masked conditional diffusion posterior;
9. add soft phonology and phase conditioning; and
10. run the complete ablation table before tuning for a final headline number.

### Go/no-go checkpoints

- If step 2 cannot approach the strongest modern hand result, fix MANO/SMPL-X coordinate conversion before training any new model.
- If step 4 reduces jitter but not TR-V2V, inspect temporal alignment and oversmoothing before adding diffusion.
- If the deterministic refiner cannot improve its frozen teachers on supervised validation data, more posterior complexity will not solve the data problem.
- If phonology gives no gain on occluded/blurred subsets, improve attribute supervision rather than hiding it inside the full model.
- If the relational graph helps only contact metrics but hurts TR-V2V, use contact uncertainty to gate its energy.

---

## 15. Minimal viable paper versus full paper

### Minimum viable strong paper

The smallest coherent publishable version is:

- strong public body and hand initialization;
- joint upper-body/wrist/hand canonicalization;
- uncertainty-aware masked sequence diffusion;
- soft handshape/orientation/location/movement conditioning;
- phase-aware posterior refinement; and
- standard TR-V2V plus robustness and semantic evaluation.

Contact can be an auxiliary module if implementation time is limited.

### Full paper

The full version adds:

- explicit hand–hand/hand–body graph;
- contact and depth-order prediction;
- evidence-selected multi-hypothesis sampling;
- cross-language generalization;
- Deaf-signer perceptual study; and
- distilled fast inference.

The full version is better suited to CVPR/ICCV/ECCV because it provides both a stronger technical contribution and a broader evaluation story.

---

## 16. Final recommendation

The highest-value route is not to continue incrementally modifying DexAvatar's per-frame LBFGS loss. Use its observations and SMPL-X infrastructure to establish the baseline, then move the research core to **whole-sign posterior reconstruction**.

The recommended final system is:

> **Modern public observations + calibrated uncertainty + unified body/wrist/two-hand state + relational masked diffusion + inferred phonology + phase-aware posterior refinement.**

This direction has a realistic mechanism for beating both the original `30.13 / 13.53 / 13.08` DexAvatar result and the stronger modern hand-integration baseline. More importantly, it supports an A*-level scientific claim that remains meaningful even after off-the-shelf estimators improve: the reconstruction is guided by the compositional linguistic and relational structure of the sign, not merely by whichever framewise regressor is newest.

---

## 17. Primary research sources

### Sign reconstruction and data

- [DexAvatar, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html)
- [Official DexAvatar repository](https://github.com/kaustesseract/DexAvatar)
- [SGNify, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html)
- [SignAvatars, ECCV 2024 project and data](https://signavatars.github.io/)
- [Neural Sign Actors, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html)
- [Tamaththul3D v2, June 2026 preprint](https://arxiv.org/html/2605.05367v2)

### Whole-body and hand initialization

- [SMPLer-X official repository](https://github.com/caizhongang/SMPLer-X)
- [SMPLest-X official repository](https://github.com/MotrixLab/SMPLest-X)
- [WiLoR official repository](https://github.com/rolpotamias/WiLoR)
- [HaMeR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html)
- [Hand4Whole++, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html)
- [OmniHands official project](https://omnihand.github.io/)

### Temporal and generative priors

- [DanceHMR, May 2026 preprint](https://arxiv.org/html/2605.18102)
- [DPoser-X, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html)
- [FUSION full-body motion prior](https://arxiv.org/abs/2601.03959)
- [MaskHand, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html)
- [Pose-Guided Temporal Enhancement, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Pose-Guided_Temporal_Enhancement_for_Robust_Low-Resolution_Hand_Reconstruction_CVPR_2025_paper.html)
- [Dyn-HaMR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html)
- [PAD-Hand, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html)

---

## 18. Validated pretrained-versus-scratch training strategy

This section audits the trainable modules proposed above and separates:

1. components that should use public pretrained models;
2. components that should remain frozen;
3. components that require new DexAvatar-specific training; and
4. models that are technically useful but have availability or licensing risks.

The main conclusion is:

> **Do not train the complete SP4D system from random initialization. Use pretrained perception, spatial-pose, and sign-video representations, while training the uncertainty, relational, phase, conditioning, and hypothesis-selection components for this task.**

This is both the highest-probability route to strong reconstruction accuracy and the cleanest scientific design. Training all backbones from zero would require substantially more clean data and compute while making it harder to determine whether gains came from the proposed posterior formulation or from basic representation learning.

### 18.1 Module-level decision table

| Proposed component | Initialization decision | Recommended source | What should be trainable |
|---|---|---|---|
| Image-to-body/hand observation expert | Use pretrained and freeze | Hand4Whole++ | No estimator fine-tuning in the main experiment; train only downstream adapters |
| Spatial SMPL-X pose prior | Transfer and selectively adapt | DPoser-X whole-body checkpoint | Temporal adapters, condition projections, and optionally upper diffusion blocks at low learning rate |
| Whole-body temporal diffusion prior | Transfer if permission permits | FUSION; otherwise DPoser-X plus a new temporal backbone | Sign-specific temporal layers and conditioning modules |
| Sign-video representation | Transfer and selectively adapt | SHuBERT | New phonology and phase heads; optionally LoRA or late-layer fine-tuning |
| Appearance/image features | Use pretrained and freeze initially | SHuBERT visual streams or another public self-supervised image encoder | Small feature projections only |
| Observation reliability and variance | Train for this system | No directly compatible pretrained head | Complete reliability head and post-training calibration |
| Phonological attribute heads | Train for this task | SHuBERT features plus sign annotations | Handshape, orientation, location, movement, symmetry, dominance, and contact heads |
| Sign phase head | Train for this task | Shared sign/temporal features | Hold, transition, repetition, and contact-transition classifier |
| Hand–hand/hand–body relational graph | Hybrid initialization | DPoser-X/FUSION pose features | Relation edges, contact/depth-order heads, and graph updates |
| Multi-hypothesis evidence selector | Train for this system | No directly transferable model | Complete candidate scorer/ranker |
| Final SMPL-X refinement | Start with optimization | SMPL-X differentiable geometry | No extra learned network initially |

### 18.2 Primary observation model: freeze Hand4Whole++

The strongest practical main initializer is Hand4Whole++ rather than independently combining SMPLer-X and WiLoR. Hand4Whole++ already addresses coherent wrist, hand, and upper-body integration and produces SMPL-X-compatible output.

Recommended use:

- run the released checkpoint as the primary framewise observation expert;
- preserve its native confidence signals and intermediate hand/body features where available;
- keep the estimator frozen in the main SP4D experiment;
- train SP4D to correct its sequence-level failures rather than silently changing the image estimator; and
- retain the original SMPLer-X + WiLoR alignment pipeline as a required baseline.

Freezing the observation model is scientifically important. It allows the paper to claim that gains come from the proposed posterior reconstruction, uncertainty model, and linguistic/relational reasoning rather than from a newly retrained front end.

WiLoR's released assets use a restrictive non-commercial, no-derivatives license. Therefore:

- do not fine-tune or redistribute modified WiLoR weights;
- use WiLoR only through an allowed frozen inference path;
- report the exact checkpoint and license in the paper and repository; and
- obtain written clarification before any use that is not unambiguously covered by the release terms.

Relevant official resources:

- [Hand4Whole++ official repository and checkpoint](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE)
- [WiLoR official repository and license information](https://github.com/rolpotamias/WiLoR)

### 18.3 Spatial pose prior: initialize from DPoser-X

DPoser-X is the most compatible public starting point for the proposed SMPL-X pose prior. It models whole-body pose while exploiting both whole-body and part-only training data, making it better suited than a random spatial denoiser for body, wrist, and hand completion.

Recommended architecture strategy:

1. load the public DPoser-X whole-body checkpoint;
2. preserve its per-frame spatial denoising representation;
3. add temporal attention or temporal adapter blocks;
4. add projections for observation uncertainty, phonology, phase, and relational features;
5. initially freeze or heavily down-weight updates to the pretrained spatial blocks;
6. train the new modules on generic motion and hand data; and
7. jointly adapt the later spatial blocks during sign-specific fine-tuning.

If the final temporal architecture is too different for direct weight transfer, use DPoser-X as a teacher:

- denoise corrupted SMPL-X poses with DPoser-X;
- distill its clean-pose score or reconstruction into the new model;
- preserve its spatial prior while learning the new temporal posterior; and
- compare weight initialization against teacher distillation in an ablation.

The repository already contains a locally sign-fine-tuned DPoser-X checkpoint at:

`DPoser-X/checkpoints/dposer/sign/sign_body_ft/last.ckpt`

That checkpoint is useful for integration and preliminary comparisons, but the local staged training set is too small to establish the final paper result. The paper model should be adapted using a substantially larger, quality-filtered sequence corpus and evaluated against the original public checkpoint.

Relevant resources:

- [DPoser-X ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.pdf)
- [DPoser-X official repository](https://github.com/careless-lu/DPoser)
- [DPoser-X public pretrained weights](https://huggingface.co/Moon-bow/DPoser-X)

The DPoser-X code and released weights do not use identical licenses. Record both the source-code license and the model-weight license in the release documentation.

### 18.4 Temporal prior: FUSION is technically strong but requires a license gate

FUSION is a particularly relevant temporal initialization because it is a unified body-and-hand motion diffusion prior. Technically, it is a better temporal warm start than learning a full whole-body motion transformer from zero.

However, its published license restricts modification and redistribution of the released software/model. A fine-tuned SP4D checkpoint derived from FUSION may therefore be impossible to publish without explicit permission.

Use a two-track strategy:

#### Track A: permission-approved FUSION initialization

- contact the authors before making FUSION a critical dependency;
- request explicit permission to fine-tune the checkpoint;
- request permission to release derived weights or adapters;
- preserve proof of the granted terms; and
- report FUSION-initialized results separately from the fully redistributable route.

#### Track B: independent temporal model

- initialize spatial components from DPoser-X;
- add and train temporal adapters or a temporal transformer;
- pretrain the new temporal modules on generic body/hand sequences;
- perform sign-specific sequence adaptation; and
- release this model as the primary reproducible configuration.

FUSION can also be evaluated as a frozen teacher or baseline when permitted. The project must not depend on obtaining permission late in the publication schedule.

Relevant resources:

- [FUSION official repository](https://github.com/enesduran/FUSION)
- [FUSION project license](https://fusion.is.tue.mpg.de/license.html)

### 18.5 Phonology and phase: transfer SHuBERT representations, train new heads

Training a sign-video encoder from zero is not recommended. SHuBERT provides sign-specific self-supervised features learned from a large sign-language video corpus and uses hand, face, and upper-body streams that align well with the proposed conditioning variables.

Recommended use:

- initialize the visual/sign encoder from the public SHuBERT weights;
- freeze early layers during the first sign-specific stage;
- train new attribute and phase heads;
- then apply low-learning-rate late-layer adaptation or LoRA if validation improves;
- use the resulting soft probabilities, not hard linguistic labels, as SP4D conditions; and
- mask unreliable or out-of-domain attributes using predicted confidence.

SHuBERT is not a 3D geometric model and should not replace the pose prior. It should supply contextual sign features that help disambiguate pose observations.

New heads should predict:

- handshape;
- palm/finger orientation;
- body-relative location;
- movement type and direction;
- one-handed versus two-handed structure;
- symmetry and dominance;
- contact likelihood; and
- hold, transition, repetition, and contact-transition phases.

HamNoSys annotations should not be treated as exact frame-level phase labels. They describe sign structure, but temporal phase boundaries should be obtained from:

- hand and wrist velocity;
- acceleration and direction changes;
- contact onset/offset;
- repetition structure;
- temporal alignment to sign boundaries; and
- a manually reviewed subset.

Relevant resources:

- [SHuBERT official repository and public weights](https://github.com/ShesterG/SHuBERT)
- [SignAvatars project](https://signavatars.github.io/)
- [SignAvatars official repository](https://github.com/ZhengdiYu/SignAvatars)
- [ASL-LEX 2.0](https://asl-lex.org/about.html)
- [WLASL-LEX, ACL 2022](https://aclanthology.org/2022.acl-short.49/)

ASL-specific lexical resources are useful supplementary supervision but must not be the sole linguistic source when evaluating DGS or cross-language generalization.

### 18.6 Modules that must be trained for the proposed system

#### 18.6.1 Observation reliability and heteroscedastic uncertainty

The uncertainty head must be trained against the exact estimators and corruption processes used by SP4D. Generic confidence predictors will not be calibrated for Hand4Whole++, WiLoR, SMPLer-X, or their characteristic failures.

Training examples should include:

- motion blur and low spatial resolution;
- partial and complete hand occlusion;
- hand-over-face and hand-over-torso occlusion;
- left/right hand swaps;
- interacting-hand identity swaps;
- incorrect wrist attachment;
- incorrect hand depth;
- implausible finger articulation;
- missed detections;
- temporal jumps; and
- disagreement among observation experts.

Use both synthetic corruptions and real estimator residuals on clean 3D validation data. Calibrate the predicted variance after training using a held-out calibration split. Report expected calibration error, negative log likelihood, error-versus-confidence curves, and selective accuracy when low-confidence observations are rejected.

#### 18.6.2 Relational and contact graph

The lower-level pose features can be initialized from DPoser-X or the temporal prior, but relation edges and contact predictions should be learned specifically for signing.

Recommended pretraining data:

- InterHand2.6M for interacting-hand articulation and identity;
- ARCTIC for synchronized full-body, two-hand, and contact-rich sequences; and
- Contact4D for contact visibility and contact-sensitive feature learning.

Relevant resources:

- [InterHand2.6M official project](https://mks0601.github.io/InterHand2.6M/)
- [ARCTIC official project](https://arctic.is.tue.mpg.de/)
- [Contact4D, 3DV 2026](https://openreview.net/forum?id=5DPvfQtAjm)

These datasets emphasize hand–hand or hand–object motion and do not directly solve sign-language hand–face, hand–torso, or language-dependent contact. Fine-tune the contact graph using geometry-derived sign contacts and a manually verified sign subset.

Contact must remain probabilistic. The contact energy should be weakened or disabled when:

- contact probability is low;
- the posterior over contact location is diffuse;
- the image evidence contradicts contact; or
- the candidate would require an implausible body or finger deformation.

#### 18.6.3 Multi-hypothesis evidence selector

The evidence selector must be trained after the generative model produces meaningful candidates. It needs to learn the behavior of SP4D's own posterior samples and the exact observation bank used at inference.

Train it using:

- supervised pose error on external training and validation data;
- held-out observation prediction;
- temporal consistency;
- calibrated observation likelihood;
- physical plausibility;
- relational consistency; and
- uncertainty-aware contact agreement.

Do not tune the selector on SGNify test ground truth. The selector should score independently produced candidates and must not become an indirect benchmark-specific oracle.

#### 18.6.4 Final refinement

Begin with differentiable SMPL-X optimization rather than another large learned network. Optimize:

- reliable 2D/3D observations;
- posterior consistency;
- hand/body attachment;
- temporal dynamics;
- soft relational/contact terms;
- joint limits; and
- shared shape.

Only replace this optimizer with a learned refiner if profiling demonstrates that runtime is a publication-critical limitation.

### 18.7 Data strategy and supervision quality

SignAvatars is suitable for large-scale sign adaptation, but its reconstructed 3D annotations are pseudo-ground truth rather than uniformly clean marker-based motion capture. Training exclusively on these labels can reproduce the annotation teacher's biases.

Use three supervision tiers:

#### Tier 1: clean geometric and motion pretraining

- public DPoser-X training domains where licensing permits;
- InterHand2.6M;
- ARCTIC;
- available high-quality SMPL-X motion data; and
- carefully validated in-house or licensed motion sequences.

#### Tier 2: large sign-specific adaptation

- SignAvatars with quality filtering;
- How2Sign where the required data and annotations are licensed;
- DexAvatar training data;
- signer-disjoint pseudo-labels generated by strong frozen experts; and
- phonology/lexical resources compatible with each sign language.

#### Tier 3: manually verified sign validation

Create a small but high-quality set containing:

- accurate hand and wrist poses;
- body-relative hand location;
- left/right identity;
- hand–hand and hand–body contact;
- depth ordering;
- phase boundaries; and
- visible failure labels for the observation experts.

This set is necessary to:

- calibrate uncertainty;
- measure teacher bias;
- validate contact and phase predictions;
- choose model checkpoints;
- diagnose whether phonology helps genuinely difficult frames; and
- prevent pseudo-label quality from becoming the hidden evaluation target.

All splits should be signer- and source-disjoint where possible. SGNify must remain evaluation-only.

### 18.8 Revised training curriculum

#### Stage 0: evaluator and coordinate audit

1. Reproduce the official DexAvatar evaluation protocol exactly.
2. Verify units, global orientation, camera coordinates, SMPL-X joint conventions, MANO-to-SMPL-X mapping, and temporal alignment.
3. Report per-frame and aggregate metrics using the same frame set as prior work.
4. Freeze the evaluator before method development.

Go/no-go condition: do not train SP4D until the released baseline and local evaluation agree within an explainable tolerance.

#### Stage 1: strongest frozen observation baseline

1. Run the released Hand4Whole++ checkpoint.
2. Preserve the SMPLer-X + WiLoR alignment baseline.
3. Evaluate both under the exact protocol.
4. Categorize failures by blur, occlusion, interaction, wrist attachment, depth, and temporal instability.

Go/no-go condition: if the modern initializer does not produce a competitive baseline, correct preprocessing and coordinate conversion before training downstream models.

#### Stage 2: deterministic pretrained temporal refiner

1. Initialize the spatial prior from DPoser-X.
2. Add temporal adapters and full-XYZ observation conditioning.
3. Include wrist-chain, palm-normal, fingertip, and shared-shape constraints.
4. Train corruption-to-clean reconstruction before introducing diffusion sampling.

Go/no-go condition: the deterministic model must improve the frozen teacher on external supervised validation data, not only on its pseudo-label training set.

#### Stage 3: calibrated uncertainty

1. Generate realistic corruption and estimator residual examples.
2. Train the reliability/variance head.
3. Calibrate on held-out real sequences.
4. Verify improvements specifically on low-confidence and failure subsets.

Go/no-go condition: predicted uncertainty must correlate with actual error and improve reconstruction when used for weighting.

#### Stage 4: relational and contact conditioning

1. Pretrain interaction features on InterHand2.6M and ARCTIC.
2. Add probabilistic relation/contact heads.
3. Adapt to sign-language hand–hand and hand–body relations.
4. Measure contact accuracy separately from TR-V2V.

Go/no-go condition: relational conditioning must not reduce geometric accuracy through incorrect forced contacts.

#### Stage 5: masked diffusion posterior

1. Convert the successful deterministic refiner into masked conditional diffusion.
2. Use FUSION initialization only if permission is secured.
3. Otherwise retain DPoser-X spatial initialization and independently trained temporal blocks.
4. Start with one posterior sample during model development.

Go/no-go condition: diffusion must improve difficult/ambiguous subsets or calibrated likelihood, not merely generate smoother-looking motion.

#### Stage 6: phonology and phase

1. Initialize the sign encoder from SHuBERT.
2. Train soft phonology and phase heads.
3. Condition SP4D using probabilities and confidence masks.
4. Evaluate gains on occluded, blurred, interaction-heavy, and cross-language subsets.

Go/no-go condition: phonology must improve geometry or semantic correctness on targeted difficult cases. If it does not, retain it as auxiliary supervision rather than a headline contribution.

#### Stage 7: multi-hypothesis selection

1. Generate a small candidate set, such as `K = 2` or `K = 4`.
2. Train the evidence selector on held-out training/validation sequences.
3. Compare selection against mean pose, highest likelihood, and oracle selection.
4. Increase `K` only if the selector closes a meaningful fraction of the oracle gap.

### 18.9 Required initialization and transfer ablations

The final ablation table should include:

1. random spatial initialization versus DPoser-X initialization;
2. DPoser-X initialization versus DPoser-X teacher distillation;
3. DPoser-X temporal route versus FUSION initialization, if legally available;
4. random sign encoder versus frozen SHuBERT versus adapted SHuBERT;
5. frozen Hand4Whole++ versus the older SMPLer-X + WiLoR baseline;
6. fixed observation confidence versus learned uncertainty;
7. uncertainty before and after calibration;
8. no relational graph versus relation-only versus relation-plus-contact;
9. no phonology versus predicted soft phonology;
10. no phase versus kinematic phase versus learned phase;
11. deterministic refiner versus diffusion posterior;
12. `K = 1`, selected `K > 1`, and oracle `K > 1`; and
13. optimization refinement on/off.

These ablations are needed to demonstrate that the paper's result is not explained solely by a stronger public observation checkpoint.

### 18.10 Main reproducible configuration and optional research configuration

#### Main reproducible configuration

The preferred release target is:

> **Frozen Hand4Whole++ observations + DPoser-X spatial initialization + independently trained temporal adapters + calibrated uncertainty + sign-specific relation/phase/phonology modules + evidence-based refinement.**

This route minimizes dependency on models whose licenses prohibit distributing derived weights.

#### Optional highest-performance configuration

If written permission is obtained:

> **Frozen Hand4Whole++ observations + FUSION temporal initialization + SHuBERT sign features + the proposed SP4D uncertainty, relation, phase, and selection modules.**

Report this separately if its checkpoint cannot be released under the same terms as the primary model.

Optional OmniHands, SAM 3D Body, SMPLest-X, or other experts should appear as supplementary hypotheses or ablations rather than required parts of the central method. Models without verified public checkpoints or clear licenses should not become critical dependencies.

### 18.11 A*-level claim and scope control

Pretrained models should be treated as controlled foundations, not as the paper's main contribution. The central claim should be:

> **Sign-conditioned, uncertainty-calibrated posterior inference produces temporally and relationally consistent whole-body reconstruction under ambiguous hand observations.**

The strongest defensible contributions are:

1. estimator-specific heteroscedastic observation modeling;
2. phonology-conditioned masked whole-sequence inference;
3. explicit probabilistic hand–hand and hand–body relational reasoning;
4. phase-aware temporal dynamics; and
5. evidence-based selection from a calibrated posterior.

The paper should avoid presenting a large ensemble of public models as its novelty. Use one primary frozen observation model, one geometric/motion prior, and one sign encoder in the main method. Place additional experts in supplementary experiments.

An A*-conference outcome cannot be guaranteed by model choice alone. The work will additionally require:

- exact and transparent benchmark reproduction;
- strong modern baselines;
- clean signer-disjoint validation;
- extensive targeted ablations;
- robustness and calibration evaluation;
- semantic and perceptual evaluation in addition to geometric error;
- runtime and reproducibility reporting; and
- release terms that allow other researchers to evaluate the principal configuration.

### 18.12 Final implementation decision

The recommended decision for this repository is:

- **do not train the observation backbone from zero;**
- **do not train the spatial SMPL-X prior from zero;**
- **do not train the sign-video representation from zero;**
- **train the temporal adapters and SP4D conditioning pathway;**
- **train the uncertainty, phonology, phase, relation/contact, and evidence-selection heads;**
- **use FUSION only behind an explicit permission and releaseability gate;**
- **treat the existing small sign-fine-tuned DPoser-X checkpoint as an integration checkpoint rather than the final research model;** and
- **build a manually verified sign validation set before scaling the full posterior.**

This strategy provides the best balance among achievable accuracy, scientific novelty, reproducibility, licensing safety, and the evidence expected in a competitive CVPR/ICCV/ECCV submission.
