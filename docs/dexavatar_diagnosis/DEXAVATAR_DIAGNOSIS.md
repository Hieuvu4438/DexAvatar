# DexAvatar: Technical Diagnosis of the Pipeline, Failure Modes, Error Budget, and Unvalidated Assumptions

**Paper:** Kundu et al., *DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors*, arXiv:2512.21054 / WACV 2026  
**Scope:** Main paper and Supplementary Material, plus a consistency check against the released fitting and evaluation code in this repository.  
**Task boundary:** Diagnosis only. This document intentionally does **not** propose replacement methods or a future research design.

---

## 0. Reading conventions and headline verdict

I use the following evidence labels throughout:

- **[READ]**: directly stated, shown, or mathematically defined in the paper/supplement.
- **[CODE]**: directly observed in the released implementation or evaluator.
- **[INFERENCE]**: a technical conclusion implied by the objective, protocol, or missing experiment, but not explicitly claimed by the authors.
- **[NOT IDENTIFIABLE]**: the requested numerical decomposition cannot be recovered from the reported evidence.

### Headline diagnosis

DexAvatar is a **framewise, initializer-anchored SMPL-X optimization pipeline** whose main novelty is to replace a generic body prior with a sign-domain body VAE (**SignBPoser**) and add independent sign-domain hand VAEs (**SignHPoser**) for the two hands. The priors reduce implausible poses, but they do not condition on pixels, do not model a sequence, do not model body–hand coupling, and do not explicitly model desired contact. The written temporal term covers body pose only.

The large Table 1 gain is principally an **upper-body benchmark gain**: EVA* falls from 40.38 to 30.13 mm, a 10.25 mm or 25.38% reduction. The hand deltas against EVA* are much smaller: 0.20 mm on the left and 0.60 mm on the right. The paper provides no confidence intervals, per-sign distribution, repeated runs, reference-noise calibration, or metric ablation, so the residual hand score cannot be decomposed numerically into “true reconstruction error,” “SGNify reference error,” and “metric-design error.” Those quantities are not additive observables in the reported experiment.

A necessary correction to the question is that **TR-V2V is translation-removed, not translation-sensitive**. In the released evaluator, prediction and reference are independently centered for each evaluated vertex region before vertex distances are calculated (`data/evaluation_from_author/evaluate_new_fitting.py:159-169`, `data/evaluation_from_author/evaluate_new_fitting.py:380-395`). Consequently, constant translation of an evaluated hand region contributes exactly **0 mm** to its hand TR-V2V score. Rotation, scale, articulation, palm/wrist orientation, and relative geometry within the region remain penalized.

---

# 1. Exact pipeline reconstruction as a dataflow

## 1.1 Paper-level dataflow

```text
Monocular sign-language video frames
        │
        ├── SMPLer-X
        │     ├── initial SMPL-X body pose
        │     ├── initial shape / expressive parameters
        │     └── initial camera parameters
        │
        ├── Sapiens
        │     └── 2D body keypoints + confidences
        │
        └── HaMeR
              ├── initial left/right hand pose estimates
              ├── 2D hand keypoints
              └── 3D hand-joint information used by the implementation

One-/two-handed sign decision from SGNify sign classifier / metadata
        │
        ├── lower-body keypoint weights set to zero
        └── for one-handed signs, non-dominant shoulder/elbow/wrist/hand disabled

Latent-variable fitting
        │
        ├── SignBPoser latent ζ̄ (33-D selected configuration)
        │       └── decoder → 21-joint body pose θ_b
        │
        ├── SignHPoser left latent ε_l (23-D selected configuration)
        │       └── decoder → 15-joint left finger pose
        │
        └── SignHPoser right latent ε_r (23-D selected configuration)
                └── decoder → 15-joint right finger pose

SMPL-X forward kinematics / skinning
        │
        └── body + hands + shape + camera → projected joints and 10,475-vertex mesh

LBFGS minimization of Eq. (12)
        │
        ├── 2D reprojection evidence
        ├── initializer-anchored body and hand priors
        ├── self-penetration penalty
        ├── previous-frame body-pose consistency
        └── body and hand joint-range penalties

Optimized per-frame SMPL-X parameters and mesh
```

**Primary paper citations:** Fig. 2 and its caption; Secs. 3.1–3.4; Eqs. (2), (12)–(15); Supplement S1.

## 1.2 What each off-the-shelf component contributes

| Component | Direct role | Important dependency or limitation |
|---|---|---|
| **SMPLer-X** | Supplies the initial SMPL-X estimate and camera; its body estimate \(\hat\theta_b\) is also used as the target inside \(L_{bprior}\). | It is not merely a starting point. Eq. (13) explicitly anchors the decoded SignBPoser result to SMPLer-X, so SMPLer-X errors can persist as supervised optimization targets. [READ: Sec. 3.4, Eq. (13)] |
| **HaMeR** | Supplies hand initialization, hand 2D keypoints, and the target hand pose \(\hat\theta_h\) in Eq. (14). | HaMeR influences both the observation and the prior target. The code also uses normalized wrist-relative hand depth from HaMeR (`dexavatar_fitting/smplifyx/fitting.py:599-643`). This creates correlated evidence rather than independent verification. [CODE] |
| **Sapiens** | Supplies body 2D keypoints and confidences used by \(L_{joint}\). | The fitting has no direct dense pixel, silhouette, texture, optical-flow, or multiview term. Sapiens failures therefore weaken the principal image-conditioned body constraint. [READ: Fig. 2; Eq. (2)] |
| **SMPL-X** | Provides the joint hierarchy, MANO-based articulated hands, FLAME-based face, shape, pose blend shapes, and final mesh. | Supplement S1 gives \(N=10{,}475\) vertices and \(K=54\) articulated joints plus global rotation; Eqs. (16)–(17) define the forward model. |
| **SGNify sign classifier / sign class** | Determines one- versus two-handed processing; a non-dominant side can be disabled. | No classification accuracy, confusion matrix, or classifier-error sensitivity is reported. The code reads precomputed class labels and separately infers active side from available hand tracks rather than showing an end-to-end classifier call. [READ + CODE] |

## 1.3 What is actually optimized

The paper describes DexAvatar at the level of Eq. (12), but does not provide a complete optimized-variable set for that equation. Fig. 2 and Eqs. (13)–(14) imply that the primary learned variables are:

- body latent \(\bar\zeta\), decoded by SignBPoser;
- independent left/right hand latents \(\epsilon_l,\epsilon_r\), decoded by SignHPoser;
- other SMPL-X/camera variables inherited from the SMPLify-X fitting structure, depending on stage/configuration.

**[CODE]** In the released fitting snapshot, the body and hand latents are central optimization variables. The implementation also retains generic shape, angle, face/expression, jaw, and collision terms beyond the seven-term abstraction in Eq. (12). The YAML uses three optimization stages and 30 maximum iterations (`dexavatar_fitting/smplifyx/cfg_files/fit_smplx_vposer_x.yaml:45-89`).

**[INFERENCE]** The method is therefore not “video → learned prior → mesh” in an amortized sense. It is “video → several external estimates → constrained local optimization near those estimates.”

---

# 2. Complete loss ledger for Eq. (12)

The paper defines

\[
\mathcal L = \mathcal L_{joint}
+ \lambda_1\mathcal L_{bprior}
+ \lambda_2\mathcal L_{hprior}
+ \lambda_3\mathcal L_{pen}
+ \lambda_4\mathcal L_{temp}
+ \lambda_5\mathcal L_{bbiomech}
+ \lambda_6\mathcal L_{hbiomech}.
\tag{12}
\]

## 2.1 \(L_{joint}\): robust 2D reprojection

Paper definition:

\[
\mathcal L_{joint}=\frac{1}{|J|}\sum_{i\in J}
\gamma_i\omega_i\,\psi\left(P(D_i)-K_i\right).
\tag{2}
\]

- \(D_i\): a 3D SMPL-X joint.
- \(P(\cdot)\): camera projection.
- \(K_i\): detected 2D keypoint.
- \(\omega_i\): detector confidence or an explicit zero mask.
- \(\gamma_i\): predefined joint importance.
- \(\psi\): robust Geman–McClure function.

**Role:** connects the optimized 3D avatar to image evidence while limiting the influence of noisy keypoints.

**Body/hand observations:** Sapiens supplies body keypoints; HaMeR supplies hand keypoints (Fig. 2; text following Eq. 12).

**Special masks:** lower-body joint weights are set to zero; for one-handed signs, the non-dominant shoulder, elbow, wrist, and hand are set to zero (Sec. 3.4).

**Implementation caveat:** the code computes a weighted robust reprojection term but does not literally reproduce Eq. (2): it squares effective weights, omits the explicit \(1/|J|\), and adds a normalized, wrist-relative **depth-only** hand term (`dexavatar_fitting/smplifyx/fitting.py:599-643`, `dexavatar_fitting/smplifyx/fitting.py:682-704`).

## 2.2 \(L_{bprior}\): SignBPoser manifold plus SMPLer-X anchor

\[
\mathcal L_{bprior}=\psi\left(\theta_b-\hat\theta_b\right)
+\lambda_{\bar\zeta}\mathcal L_{\bar\zeta}.
\tag{13}
\]

- \(\bar\zeta\): low-dimensional body latent.
- \(\theta_b\): body pose decoded by SignBPoser.
- \(\hat\theta_b\): SMPLer-X body estimate.
- \(L_{\bar\zeta}\): zero-mean Gaussian latent penalty analogous to Eq. (3).

**Role:** restricts body articulation to the learned sign-pose manifold while preventing the fit from departing too far from SMPLer-X.

**What it actually constrains:** frame-level rotations for 21 body joints. It is not a temporal prior and does not encode hand shape, hand–hand relation, contact, sign identity, or image appearance.

**Training distribution:** SignAvatars 3D body estimates derived from How2Sign, after frames violating shoulder, elbow/forearm, wrist, clinical range-of-motion, or signer-space constraints are rejected (Sec. 3.2.1; Fig. 3; Supplement S2).

**Implementation caveat:** the code uses latent L2 plus an L1 decoder-to-initializer anchor, split into core and non-core joints (`dexavatar_fitting/smplifyx/fitting.py:706-712`), rather than an explicitly documented Geman–McClure vector penalty.

## 2.3 \(L_{hprior}\): independent SignHPoser manifolds plus HaMeR anchors

\[
\mathcal L_{hprior}=\psi\left(\theta_h-\hat\theta_h\right)
+\lambda_{\epsilon_l}\mathcal L_{\epsilon_l}
+\lambda_{\epsilon_r}\mathcal L_{\epsilon_r}.
\tag{14}
\]

- \(\epsilon_l,\epsilon_r\): independent left/right hand latents.
- \(\theta_h\): decoded hand pose.
- \(\hat\theta_h\): HaMeR hand estimate.
- latent penalties are Gaussian priors analogous to Eq. (3).

**Role:** keeps finger articulation close to a sign-specific hand-pose manifold while anchoring it to HaMeR.

**What it actually constrains:** 15 articulated finger joints per hand, especially bending/splaying/twisting patterns represented in the mocap-derived training data.

**What it does not constrain:**

- global hand translation;
- reliable global wrist rotation or palm orientation relative to the arm;
- shoulder–elbow–wrist kinematic coordination;
- inter-hand contact identity or desired contact location;
- left/right coordination;
- temporal hand dynamics;
- lexical or phonological correctness.

This exclusion is not merely conceptual: Supplement S3 states that wrist rotations **could not be transferred** during MANUS-to-SMPL-X retargeting because of T-pose/bone-roll incompatibility.

**Implementation caveat:** for each active hand, the code uses latent L2, an L1 decoded-pose anchor, and an additional robust anchor to the same target (`dexavatar_fitting/smplifyx/fitting.py:828-875`).

## 2.4 \(L_{pen}\): anti-interpenetration, not positive contact modeling

Eq. (4) detects colliding face pairs and penalizes bidirectional intrusion depth.

**Role:** prevents mesh parts from occupying the same volume.

**Critical distinction:** a non-penetration term can say “these surfaces must not pass through each other.” It cannot say:

- which fingertips should touch;
- that two hands should remain in contact;
- which hand is in front;
- where contact occurs;
- whether a near-contact is linguistically required.

Therefore, the paper’s broad “contact-aware” language should be read narrowly. In Eq. (12), the explicit physical interaction term is anti-collision, not an attraction or contact-preservation objective.

The code uses BVH collision detection and a penetration-distance loss (`dexavatar_fitting/smplifyx/fitting.py:951-969`).

## 2.5 \(L_{temp}\): first-order body-pose consistency

\[
\mathcal L_{temp}=\psi\left(\theta_b-\theta_b^{pre}\right).
\tag{15}
\]

**Role:** discourages abrupt changes between the current body pose and the previous fitted body pose.

**Scope:** Eq. (15) explicitly names body pose \(\theta_b\), not hand pose, contact state, camera, velocity, acceleration, or a sequence latent.

**Implementation:** the code compares 21 current body axis-angle joints to a carried previous pose and multiplies the robustified difference by a hard-coded factor of 2000 (`dexavatar_fitting/smplifyx/fitting.py:583-583`, `dexavatar_fitting/smplifyx/fitting.py:641-643`). Sequential state is updated after each successful frame (`dexavatar_fitting/smplifyx/main.py:225-227`, `dexavatar_fitting/smplifyx/main.py:242-344`).

**Implication:** the model has first-order causal smoothing, not sequence reconstruction. It cannot infer a long occluded interval from future observations, and no written temporal term regularizes finger motion directly.

## 2.6 \(L_{bbiomech}\) and \(L_{hbiomech}\): box-like angle constraints

The common paper definition is

\[
\mathcal L_{biomech}=\sum_{j=1}^{J}
\left\|\max(\theta_j-\theta_{j,max},\theta_{j,min}-\theta_j,0)\right\|_2^2.
\tag{11}
\]

- \(J=6\) selected body joints for SignBPoser.
- \(J=15\) hand joints for SignHPoser.
- penalty is zero inside the range and grows quadratically outside it.

**Role:** excludes joint angles outside predefined anatomical ranges.

**Body source:** clinical ROM plus signer-space constraints on shoulders, elbow/forearm, and wrists (Supplement S2; Figs. S1–S3).

**Hand source:** per-joint bending, splaying, and twisting constraints, after axis alignment to account for MANO coordinate differences (Sec. 3.2.2; Fig. 4).

**Implementation discrepancy:** the released body term converts selected body rotations to Euler XYZ and applies a mean **linear hinge** violation (`dexavatar_fitting/smplifyx/fitting.py:691-696`), which is not Eq. (11)’s summed squared penalty. In the author-origin fitting path inspected here, there is no clear formula-faithful 15-joint hand-range counterpart to Eq. (11). This is a major reproducibility concern.

## 2.7 Additional implementation terms omitted from Eq. (12)

The released fitting loss also contains generic terms inherited from SMPLify-X, including:

- shape prior;
- generic elbow/knee angle prior;
- expression prior;
- jaw prior;
- configuration-dependent hand priors and later experimental terms.

See `dexavatar_fitting/smplifyx/fitting.py:928-969`.

**Diagnosis:** Eq. (12) is a conceptual summary, not a complete executable specification of the released optimizer.

---

# 3. What the learned priors really learn

## 3.1 SignBPoser

| Property | Diagnosis |
|---|---|
| Architecture | VAE, three linear layers, 512-unit embedding layers (Sec. 4 implementation details). |
| Selected latent | 33 dimensions in the preferred/reported configuration (Supplement S4, Table S1). |
| Input/output representation | Per-joint rotation matrices during prior training; decoded body pose is used as axis-angle in fitting. |
| Training source | SignAvatars 3D body estimates derived from How2Sign. These are pseudo-ground truth, not marker-based body mocap. |
| Preprocessing | Rejects frames outside selected upper-limb ROM and signer-space envelopes (Fig. 3; Supplement S2). |
| Actual statistical object | A frame-level distribution over filtered sign-like upper-body poses. |
| Not learned | Motion dynamics, signer identity, sign semantics, hand pose, hand–body contact, two-hand coordination, camera/image likelihood. |

## 3.2 SignHPoser

| Property | Diagnosis |
|---|---|
| Architecture | VAE matched to SignBPoser, three linear layers and 512 units. |
| Selected latent | 23 dimensions for the filtered preferred configuration (Supplement S4, Table S2 and text). |
| Training source | Vicon + MANUS glove recordings from 8 signers: 6 Auslan-proficient and 2 ASL-fluent participants, each fingerspelling a curated list of 93 words (Sec. 3.2.2). |
| Preprocessing | Retarget to SMPL-X in Blender; correct implausible finger configurations using bending/splaying/twisting limits (Fig. 4; Supplement S3). |
| Actual statistical object | A frame-level distribution over corrected fingerspelling-derived finger rotations. |
| Critical missing signal | Wrist rotations were not reliably transferred during retargeting (Supplement S3). |
| Not learned | Lexical DGS handshape distribution as such, wrist/palm orientation relative to arm, two-hand coupling, hand–body contact, temporal coarticulation, image evidence. |

## 3.3 Prior-training loss, Eq. (5)

The prior VAEs use:

\[
\mathcal L = c_1L_{KL}+c_2L_{recon}+c_3L_{mesh}+c_4L_{orth}+c_5L_{reg}+c_6L_{biomech}.
\tag{5}
\]

- \(L_{KL}\), Eq. (6): posterior-to-standard-normal KL.
- \(L_{recon}\), Eq. (7): squared axis-angle reconstruction error.
- \(L_{mesh}\), Eq. (8): squared vertex reconstruction error.
- \(L_{orth}\), Eq. (9): rotation orthogonality constraint.
- \(L_{reg}\), Eq. (10): parameter L2 regularization.
- \(L_{biomech}\), Eq. (11): out-of-range angle penalty.

The stated coefficients are:

- **SignBPoser:** 0.001, 0.999, 0.999, 0.01, 0.0001, 1.5.
- **SignHPoser:** 0.0001, 0.999, 0.999, 0.01, 0.0001, 1.5.

These are **prior-training** coefficients. They are not the missing fitting weights \(\lambda_{1:6}\) in Eq. (12).

---

# 4. Failure-mode taxonomy

Table 1 has only aggregate regional scores. It does not contain occlusion-, blur-, contact-, speed-, or handedness-stratified columns. The mappings below identify where each failure would enter the reported columns, not where the paper directly measured a failure-specific effect.

## 4.1 Summary matrix

| Error source | Intended Eq. (12) defense | Why it plausibly fails | Table 1 manifestation |
|---|---|---|---|
| Self-occlusion | \(L_{joint}\), \(L_{bprior}\), \(L_{hprior}\), \(L_{pen}\), \(L_{temp}\), biomechanical terms | Hidden 3D articulation is not observed; priors choose a likely pose but do not recover scene-specific overlap order; anti-penetration does not encode intended contact; hand temporal state is absent | Mainly LHand/RHand; UBody(-F) through wrists/forearms/arms |
| Motion blur | Robust \(L_{joint}\), learned priors, \(L_{temp}\), biomechanical terms | No blur likelihood or multi-frame image aggregation; fast hand motion corrupts both HaMeR keypoints and HaMeR target pose; Eq. (15) does not regularize fingers | Mainly LHand/RHand; UBody(-F) when wrist/arm detections shift |
| Hand–hand contact | \(L_{pen}\), hand prior, biomechanics | \(L_{pen}\) only discourages intersection; it does not reward correct touching, contact pair, overlap order, or relative depth | Both hand columns jointly; UBody(-F) secondarily |
| Depth ambiguity | Priors, \(L_{pen}\), \(L_{temp}\), biomechanical limits supplement 2D reprojection | Multiple 3D poses share the same 2D projection; priors impose population likelihood, not instance-specific depth; code’s HaMeR depth is normalized and scale-free | All three columns, especially crossing/contact hands |
| One-handed sign detection | Sign-class gating inside \(L_{joint}\) | Hard masking magnifies classifier/side errors; no classifier accuracy is reported; disabling a side prevents image evidence from correcting it | Disabled L/R hand; UBody(-F) via shoulder/elbow/wrist |
| Initializer noise | Robust \(L_{joint}\), priors, biomechanics, collision | SMPLer-X/HaMeR are also explicit prior targets; HaMeR supplies multiple correlated signals; strong code weights can preserve bad initial estimates | All columns |
| Temporal drift | \(L_{temp}\) | First-order, causal, body-only parameter smoothing; no hand sequence model, acceleration term, future context, or long-horizon constraint; frame failures can interrupt state | UBody(-F) directly; hands indirectly via arm chain |

## 4.2 Self-occlusion

**Observed challenge:** the introduction explicitly identifies frequent hand–hand and hand–body self-occlusion as a central failure source. Supplement S7.2 and Fig. S9 present three qualitative examples.

**Supposed defenses:** robust reprojection, sign-specific pose manifolds, anti-penetration, previous-frame body consistency, and joint limits.

**Failure mechanism:**

1. Occlusion reduces or corrupts the 2D evidence driving Eq. (2).
2. A robust loss can reduce an outlier’s influence, but cannot infer which hidden finger configuration generated the visible pixels.
3. Independent per-hand priors do not model relative hand pose.
4. Anti-penetration cannot infer front/back ordering. Supplement S7.2 itself discusses overlap-order errors in SGNify, demonstrating that plausibility and overlap order are separate issues.
5. Eq. (15) offers no explicit finger-motion continuity.

**Metric visibility:** finger shape and palm orientation affect LHand/RHand. Wrist and arm changes affect UBody(-F). Because hand regions are separately centered, absolute hand placement relative to the body is largely not measured by hand TR-V2V itself.

## 4.3 Motion blur

**Observed challenge:** highlighted in the abstract/introduction; Supplement S7.1 and Fig. S8 show qualitative examples.

**Supposed defenses:** robust \(L_{joint}\), priors, biomechanical limits, and temporal body consistency.

**Failure mechanism:**

- HaMeR hand keypoints and HaMeR hand parameters can fail together under blur.
- Eq. (14) can then anchor the optimizer toward the same corrupted source that caused the bad observation.
- The written temporal term excludes hand pose.
- There is no reported blur severity, subset score, or sampling protocol. Three examples do not establish a population robustness effect.

**Metric visibility:** mainly LHand/RHand; fast arm motion also changes UBody(-F).

## 4.4 Hand–hand contact

**Observed challenge:** the paper stresses hand–hand/hand–body interaction and qualitatively describes “clean stable contact” in Supplement S7.

**Supposed defense:** \(L_{pen}\), with priors and joint limits supplying plausibility.

**Failure mechanism:** nonpenetration and contact are mathematically different. A collision penalty has a minimum when surfaces are separated, even if the sign requires them to touch. It contains no target contact map, fingertip pairing, attraction, or overlap-order condition.

**Metric visibility:** both hand columns. Yet TR-V2V has no contact-specific term, so two meshes can have similar mean vertex error but different contact correctness.

## 4.5 Monocular depth ambiguity

**Observed challenge:** the introduction explicitly notes that different 3D configurations can project to identical 2D keypoints.

**Supposed defense:** sign-domain priors, anti-collision, temporal smoothing, and biomechanical limits.

**Failure mechanism:** these terms remove some impossible solutions but do not uniquely identify the correct instance. The code’s additional hand 3D signal selects only depth, makes it wrist-relative, and independently standardizes prediction and target (`dexavatar_fitting/smplifyx/fitting.py:599-643`), discarding absolute depth scale.

**Metric visibility:** rotation and within-region depth configuration remain penalized in all columns. Constant regional translation is removed.

## 4.6 One-handed sign detection

**Paper behavior:** for one-handed signs, \(\omega_i=0\) for the non-dominant arm and hand (Sec. 3.4).

**Failure mechanism:** a hard classification error changes the optimization problem itself. False one-handed classification suppresses valid evidence; false two-handed classification allows a visually unreliable inactive hand to be optimized. No classifier accuracy or per-handedness result is reported.

**Code nuance:** class metadata determines one-hand mode, while active side is inferred from hand-motion/detection availability. Frames without required active detections are filtered (`dexavatar_fitting/smplifyx/data_parser.py:185-237`).

**Metric visibility:** the author evaluator skips LHand on class-0 signs and removes left-hand vertices from other regions (`data/evaluation_from_author/evaluate_new_fitting.py:380-395`). SGNify has 15 class-0 signs (0a + 0b), so LHand covers 42 two-handed signs while RHand covers all 57 under the right-handed protocol.

## 4.7 Initializer noise and anchoring

**Supposed defenses:** robust keypoint fitting and learned priors.

**Failure mechanism:**

- Eq. (13) treats SMPLer-X as body supervision.
- Eq. (14) treats HaMeR as hand supervision.
- HaMeR also supplies hand keypoints; code adds HaMeR-derived depth.
- The released YAML uses body and hand initialization weights of 1200 in all three stages (`dexavatar_fitting/smplifyx/cfg_files/fit_smplx_vposer_x.yaml:58-73`).

**[INFERENCE]** This can make DexAvatar a plausibility-corrected version of the initializers rather than an independent reconstruction. When image evidence is weak, the optimizer may preserve a confident but incorrect proposal.

## 4.8 Temporal drift, error propagation, and frame selection

**Supposed defense:** Eq. (15).

**Failure mechanism:**

- only the previous body pose is used;
- no future frame can repair the current fit;
- no explicit hand temporal term exists;
- no velocity/acceleration/contact trajectory is modeled;
- skipped or failed frames alter the temporal chain.

The parser drops frames lacking HaMeR output or initialization and requires both hands for two-handed signs (`dexavatar_fitting/smplifyx/data_parser.py:185-237`). The main loop carries state only after successful fitting (`dexavatar_fitting/smplifyx/main.py:301-344`).

**[INFERENCE]** The difficult blur/occlusion frames most likely to need temporal inference are also the frames most likely to disappear from the usable sequence, producing coverage bias.

---

# 5. Quantitative error-budget analysis

## 5.1 Table 1 arithmetic

| Region | EVA* | DexAvatar | Absolute reduction | Relative reduction | Residual as fraction of EVA* |
|---|---:|---:|---:|---:|---:|
| UBody(-F) | 40.38 | 30.13 | **10.25 mm** | **25.38%** | **74.62%** |
| LHand | 13.73 | 13.53 | **0.20 mm** | **1.46%** | **98.54%** |
| RHand | 13.68 | 13.08 | **0.60 mm** | **4.39%** | **95.61%** |

The question’s “40.38 → 30.13 (-25%)” and “13.73 → 13.53 (-1.5%)” are correct up to rounding.

The paper’s headline **35.11%** upper-body improvement is a different comparator: Neural Sign Actors 46.42 → DexAvatar 30.13, not EVA* → DexAvatar (Table 1; Sec. 5.1).

## 5.2 What the residual hand millimetres geometrically mean

For a hand region \(R\), the released evaluator computes

\[
E_{TR}^{R}=\frac{1}{|R|}\sum_{i\in R}
\left\|(\hat v_i-\bar{\hat v}_{R})-(v_i-\bar v_R)\right\|_2.
\]

This follows directly from `transl_point_error` (`data/evaluation_from_author/evaluate_new_fitting.py:159-169`). Therefore, the remaining 13.53/13.08 mm can live in:

- incorrect finger articulation;
- palm and wrist orientation;
- within-hand relative depth;
- hand scale or shape mismatch;
- relative vertex geometry caused by pose/shape blend effects;
- rotation of the hand region;
- disagreement with artifacts in the SGNify fitted reference.

It does **not** include a constant translation of the entire evaluated hand region. The hand may be misplaced relative to the torso yet receive the same hand TR-V2V after hand-region centering. That placement can still influence UBody(-F), depending on the upper-body region mask.

## 5.3 Why the three requested sources cannot be assigned percentages

Let:

- \(T\): unknown true physical surface;
- \(G\): SGNify fitted “ground-truth” mesh;
- \(P\): DexAvatar prediction;
- \(d_R\): the region-centered TR-V2V metric.

The table reports \(d_R(P,G)\). It does not report \(d_R(P,T)\) or \(d_R(G,T)\).

### (a) Genuine reconstruction error

**Reported numerical amount:** **unknown**.

If SGNify were exact and TR-V2V captured every relevant aspect, then the entire 13.53/13.08 mm would be reconstruction mismatch. Neither premise is established.

If \(q_R=d_R(G,T)\) denotes unknown reference error, triangle inequality yields a conditional interval:

\[
\max(0,d_R(P,G)-q_R)\le d_R(P,T)\le d_R(P,G)+q_R.
\]

No \(q_R\) is reported, so this does not produce a numerical partition.

### (b) Noise or implausibility in the SGNify reference

**Reported numerical amount:** **unknown**.

Supplement S6 explicitly states that the reference occasionally contains collapsed fingers, irregular knuckle spacing, and anatomically inconsistent postures; Fig. S7 shows examples. However:

- there is no percentage of corrupted frames;
- no hand-specific reference uncertainty in millimetres;
- no repeated fit or independent annotation;
- no marker residual converted into vertex uncertainty;
- no expert adjudication of every evaluated frame.

Reference error is not necessarily a positive additive “penalty.” It can **inflate or reduce** a method’s reported error depending on whether a prediction moves toward or away from the fitted artifact.

### (c) Metric design

**Reported numerical amount:** not expressible as an additive millimetre component.

One exact statement is possible:

- **constant per-region translation contribution = 0 mm**, because TR-V2V removes it.

Everything else requires recomputation under alternative metrics. TR-V2V:

- retains rotation and scale mismatch;
- ignores the hand region’s global translation relative to the torso;
- averages vertices, so large local errors can be diluted;
- does not represent contact correctness;
- does not represent sign intelligibility or phonological correctness;
- does not explicitly score overlap order;
- uses different effective populations for LHand and RHand under the one-handed protocol.

These are properties of what is and is not observed, not separable terms that sum to 13.53 mm.

## 5.4 A strict answer to “how much is each?”

| Requested component | Defensible quantitative answer from the paper |
|---|---|
| Genuine reconstruction error | **Not identifiable.** It lies in a conditional interval depending on unknown SGNify reference error. |
| SGNify reference noise/implausibility | **Not identifiable.** The supplement shows existence, not prevalence or magnitude. |
| Metric-design contribution | **Not an additive component.** Constant regional translation contributes exactly 0 mm; the effects of alternate alignment/contact/semantic metrics were not measured. |

Any exact three-way percentage would be fabricated.

## 5.5 What the ablations do identify

### Body prior, Table 2

| Change | UBody(-F) | Interpretation |
|---|---:|---|
| BPu → BPf | 34.06 → 30.28, **−3.78 mm / −11.10%** | Association between filtered body-prior training configuration and lower final benchmark score. It does not isolate which removed frames or biomechanical rule caused the gain. |
| BPf → BPf+bio during prior training | 30.28 → 30.44, **+0.16 mm / worse** | The extra training-time biomechanical regularizer mildly over-regularizes according to the authors. |
| BPf with fitting-time biomechanics → final | 30.28 → 30.13, **−0.15 mm / −0.50% from printed values** | Very small final difference. The paper text says 0.33%, which is not reconciled by the two-decimal table values. |

### Hand prior, Table 3

| Change | UBody(-F) | LHand | RHand |
|---|---:|---:|---:|
| HPu → HPf | 31.34 → 30.17 (**−1.17**) | 14.19 → 13.55 (**−0.64**) | 13.92 → 13.06 (**−0.86**) |
| HPf → HPf+bio | 30.17 → 30.13 (**−0.04**) | 13.55 → 13.53 (**−0.02**) | 13.06 → 13.08 (**+0.02, worse**) |

The meaningful hand-prior ablation gain is primarily from the **full uncorrected-data configuration to the corrected-data configuration**, not from adding the final biomechanics term. The final term’s changes are hundredths of a millimetre and directionally inconsistent between hands.

## 5.6 Why the UBody gain cannot be added to the hand gains

The reported columns are not a partition of one total error:

1. UBody(-F) is a separate vertex region and can overlap with hand vertices.
2. Each region is independently centered.
3. Mean Euclidean distances are nonlinear.
4. The LHand and RHand scores use different sign populations because class-0 left hands are skipped (`data/evaluation_from_author/evaluate_new_fitting.py:380-395`).
5. UBody(-F)’s vertex set changes on class-0 signs because left-hand vertices are removed.

Therefore, statements such as “X mm of upper-body improvement came from the hands” cannot be derived from Table 1.

## 5.7 Statistical scale of the hand result

The paper reports point estimates only. There are no:

- confidence intervals;
- standard errors;
- per-sign paired deltas;
- bootstrap results;
- repeated prior-training seeds;
- repeated fitting runs;
- test–retest GT fits.

A 0.20 mm LHand delta and a 0.60 mm RHand delta may be real, but the paper does not establish their statistical stability. The final 0.02 mm biomechanical ablation changes are particularly below any demonstrated uncertainty floor.

---

# 6. Assumptions that are not validated

“Not validated” means the paper does not directly test, quantify, or stress-test the assumption. It does not mean the assumption is necessarily false.

## 6.1 Data and domain assumptions

1. **Fingerspelling transfers to lexical signs.** SignHPoser is trained on 93 fingerspelled words but evaluated on isolated lexical German signs. No lexical-versus-fingerspelling or handshape-coverage analysis is reported. [Sec. 3.2.2; Sec. 4]
2. **Auslan/ASL handshape statistics transfer to DGS.** Six Auslan-proficient and two ASL-fluent signers train the prior; the quantitative benchmark is DGS. No cross-language ablation is reported.
3. **Eight signers are sufficient to represent hand-pose variation.** No signer-held-out prior evaluation, demographic description, or scaling curve is reported.
4. **Ninety-three words sufficiently cover the relevant hand manifold.** No vocabulary-coverage, unseen-handshape, or nearest-neighbor analysis is shown.
5. **Fingerspelling motion is representative of natural lexical/coarticulated signing.** Fingerspelling emphasizes rapid letter sequences and may have different handshape transitions and two-hand statistics; this is not tested.
6. **Pseudo-ground-truth SignAvatars body pose is accurate enough after filtering.** The authors acknowledge residual noise and bias, but there is no retained-data accuracy audit against body mocap.
7. **How2Sign/SignAvatars body statistics transfer to the single DGS benchmark signer.** No cross-dataset or cross-language body-prior analysis is shown.
8. **Isolated central sign frames represent continuous signing.** Evaluation excludes onset/offset and contains only isolated signs; no quantitative continuous-sign test is reported.
9. **One benchmark signer is representative.** SGNify captures one native right-handed DGS signer. No inter-signer variance is measurable from Table 1. [SGNify dataset description]
10. **A 57-sign corpus is broad enough for general claims.** The signs cover SGNify’s selected classes but cannot validate broad signer/language/style robustness by itself.
11. **2,872 central frames are independent evidence.** Frames within a sign are highly correlated; the paper does not report sign-level uncertainty or use sign-level statistical units.
12. **Qualitative MM-WLAuslan examples establish external robustness.** S7 selects three examples for blur, three for occlusion, and three for Gaussian noise, with no sampling protocol or numerical external evaluation.

## 6.2 Prior representation and hyperparameter assumptions

13. **A 33-D body latent is the appropriate capacity.** Supplement S4 compares only 31/32/33 for specific settings; no capacity/generalization curve beyond this narrow range is shown.
14. **A 23-D hand latent is the appropriate capacity.** Supplement S4 compares 22/23/24; the preferred dimension depends on the corrected-data setting and is not externally validated.
15. **Three-layer 512-unit MLP VAEs are sufficient.** No architectural comparison establishes that the chosen VAE family captures multimodal signing pose distributions.
16. **An isotropic Gaussian latent is appropriate for sign pose.** This is imposed by the VAE but not tested against latent mismatch, multimodality, or rare handshape collapse.
17. **Axis-angle, rotation-matrix, mesh, orthogonality, and biomechanical losses have the chosen relative scales.** Eq. (5) weights are empirical; no broad sensitivity or confidence interval is reported.
18. **The same lightweight prior architecture is appropriate for body and hands.** The domains have different dimensionality, symmetry, multimodality, and contact structure, yet architecture parity is not justified experimentally.
19. **Independent left/right hand latents are sufficient.** No test compares independent hand priors with a coupled representation; two-hand coordination is assumed unnecessary for the prior.
20. **A frame-level pose prior can handle a temporal language.** Neither SignBPoser nor SignHPoser learns motion trajectories; this sufficiency is not validated.
21. **Selecting hyperparameters using DEV and TEST is acceptable.** Supplement S4 explicitly says the best hyperparameter is selected on “Evaluation (DEV) and TEST data.” Even if these are prior-training splits rather than SGNify, test-set model selection compromises an unbiased test estimate unless a separate untouched test is identified.
22. **The selected fitting weights generalize.** The paper does not report \(\lambda_{1:6}\), their search space, sensitivity, or an external validation protocol.

## 6.3 Mocap and retargeting assumptions

23. **MANUS glove rotations are sufficiently accurate for prior learning.** Sensor and calibration uncertainty are acknowledged generically but not quantified.
24. **The Blender/Rokoko retargeting preserves anatomical finger rotations.** No retargeting-error metric or independent joint-angle validation is reported.
25. **Deleting parent-container keyframes preserves the intended finger signal without distortion.** Supplement S3 describes the operation but offers no quantitative verification.
26. **Differences in finger length do not materially affect learned pose.** The supplement states they did not substantially affect retargeting, but provides no measurement.
27. **Manual alignment and joint spacing are reproducible.** The process required careful Blender alignment; no deterministic script, inter-operator agreement, or retargeting tolerance is reported.
28. **Missing wrist-rotation transfer does not undermine SignHPoser’s downstream usefulness.** This is particularly important because TR-V2V retains rotation, and hand orientation is phonologically meaningful.
29. **Inverse-kinematics arm following yields valid arm–hand relationships.** The prior is still trained primarily on hand articulation and no arm–wrist consistency metric is reported.

## 6.4 Biomechanical assumptions

30. **Clinical ROM maps correctly into the chosen SMPL-X Euler convention.** Supplement S2 uses deterministic conversion and a brief visual sanity check, not quantitative rig validation.
31. **Mirroring signed ranges across the sagittal plane is correct for every left/right joint convention.** No synthetic round-trip or joint-axis error is reported.
32. **The selected six body joints are sufficient for sign biomechanics.** Other torso, scapular, clavicular, neck, and hand–arm couplings are not included in the stated body constraint.
33. **Fifteen independent hand-joint bounds capture plausible hand biomechanics.** Coupled tendon constraints, inter-joint dependencies, and contact-conditioned configurations are not validated.
34. **Box-like per-angle limits correspond to natural sign articulation.** A pose can satisfy every independent interval yet remain globally implausible; conversely, signer-specific valid poses can approach or exceed generic limits.
35. **Signer-space bounds transfer across languages, bodies, camera views, and discourse styles.** The shoulder envelope is not validated across target signers.
36. **Anatomical plausibility correlates with linguistic correctness.** No Deaf-expert intelligibility or phonological evaluation confirms this assumption.
37. **Plausibility constraints do not oversmooth or suppress rare valid signs.** No rare-pose recall or false-rejection rate is measured.

## 6.5 Observation and initializer assumptions

38. **Sapiens body keypoints are calibrated for low-resolution signing.** No detector accuracy or confidence calibration is reported on SGNify.
39. **HaMeR hand keypoints and hand pose are reliable under blur and self-occlusion.** These are exactly the failure conditions motivating the paper, yet no initializer-only failure analysis is presented.
40. **SMPLer-X camera and body estimates are close enough for local optimization.** No initialization perturbation or basin-of-convergence study is reported.
41. **Using HaMeR as both observation and prior target does not create circular confirmation.** No ablation separates HaMeR keypoints from HaMeR pose anchoring.
42. **Using SMPLer-X as initializer and body target does not lock in systematic errors.** No anchor-weight sensitivity or target-removal result is shown.
43. **Geman–McClure robustification is sufficient for all detector noise.** The robust scale and alternative losses are not studied in the paper.
44. **Detector confidence is a valid uncertainty estimate.** No reliability diagram or confidence-versus-error analysis is shown.
45. **Normalized wrist-relative depth retains the 3D evidence needed for hands.** This code behavior discards absolute scale and image-plane 3D geometry and is not documented or ablated in the paper.
46. **Dropping frames with missing proposals does not bias evaluation or qualitative claims.** The parser’s filtering behavior is not accompanied by coverage statistics.

## 6.6 One-handedness and body-part assumptions

47. **The one-/two-handed classifier is accurate.** No accuracy is reported.
48. **A sign has a fixed one-/two-handed class across all relevant frames.** Transitional or temporarily inactive hands are not analyzed.
49. **The non-dominant side can safely be disabled.** Some one-handed signs use the other hand as a location or weak articulator; the hard-mask assumption is not tested here.
50. **Right-handed benchmark conventions generalize.** The evaluation’s asymmetric handling of left/right hands is not validated for left-handed signers.
51. **Lower-body motion is negligible.** Lower-body weights are set to zero, but no effect on global posture, pelvis, torso, or upper-body kinematics is measured.
52. **Face can be excluded without undermining the task claim.** The metric excludes the face, although non-manual facial signals are linguistically important. The paper’s “sign language reconstruction” claim is therefore evaluated on a manual/upper-body subset.

## 6.7 Temporal assumptions

53. **First-order body-pose smoothing is enough for temporal consistency.** No comparison with longer temporal context is reported.
54. **Hand dynamics need no explicit temporal term.** Eq. (15) regularizes body only.
55. **Axis-angle parameter differences are a suitable motion distance.** No geodesic or representation-boundary analysis is supplied.
56. **A fixed temporal weight is appropriate across fast and slow signing.** No speed-conditioned evaluation is reported.
57. **Causal fitting does not accumulate drift.** No long-horizon drift metric or forward/backward consistency test is reported.
58. **Central-frame evaluation adequately tests temporal behavior.** It deliberately omits portions where transitions and initialization effects are strongest.
59. **Parallel/chunked or resumed fitting preserves the same temporal state.** The paper does not specify execution-history invariance; the code’s temporal initialization can depend on processing order.

## 6.8 Contact and physical-interaction assumptions

60. **Anti-penetration is an adequate proxy for correct contact.** No contact attraction or contact-pair objective appears in Eq. (12).
61. **Independent hand priors can recover coordinated two-hand configurations.** No explicit bilateral relation is learned.
62. **Correct overlap order follows from 2D keypoints plus priors.** The objective contains no explicit depth-order term.
63. **Qualitative “clean contact” demonstrates contact accuracy.** There are no 3D contact labels, precision/recall, penetration volume, or contact-distance distributions.
64. **Hand–body contact is handled by the same mechanism as hand–hand collision.** No region-specific contact validation is reported.

## 6.9 Evaluation and metric assumptions

65. **SGNify fitted meshes are sufficiently accurate to serve as ground truth.** Supplement S6 directly documents implausible hand configurations.
66. **Translation-only alignment is the appropriate geometric criterion.** No alternate alignment analysis is reported.
67. **Per-region centering is equivalent to the intended task metric.** It removes hand location relative to the body from hand scores; this semantic consequence is not discussed in the paper.
68. **Mean per-vertex L2 tracks sign correctness.** It ignores phonology, intelligibility, contact identity, and local functional importance.
69. **Every vertex should contribute equally.** Fingertips and linguistically decisive joints are not weighted differently.
70. **The mean is robust to catastrophic frames.** No median, tail percentile, or per-sign maximum is reported.
71. **LHand and RHand columns are directly comparable.** They use different effective sign populations under the class-0 protocol.
72. **UBody(-F), LHand, and RHand can be read as independent subsystem errors.** They overlap and are independently centered; they are not additive.
73. **Point estimates at two decimals imply meaningful sub-millimetre differences.** No uncertainty floor is reported.
74. **The same 2,872 frames are evaluated for every method without coverage differences.** The paper does not publish a per-method frame manifest or failure count.
75. **Modification of EVA to EVA* preserves a fair baseline.** No EVA-versus-EVA* ablation or implementation detail quantifies the effect of this modification.
76. **Central-frame micro-averaging is representative across signs.** Longer signs contribute more frames, and no macro per-sign score is shown.
77. **Qualitative examples are representative rather than selected successes.** Selection criteria are absent.

## 6.10 Efficiency and reproducibility assumptions

78. **Runtime is acceptable.** The paper reports RTX 4090/24 GB, 64 GB CPU memory, PyTorch, LBFGS, and architecture details, but no seconds/frame, seconds/sign, preprocessing cost, or end-to-end runtime.
79. **Memory and compute scale beyond the 57-sign benchmark.** No scaling experiment is reported.
80. **Eq. (12) is sufficient for reproduction.** The fitting \(\lambda\) values are omitted, while code contains additional terms and nonliteral implementations.
81. **The released code implements the paper equations.** The inspected body-biomechanics form differs from Eq. (11), and a formula-faithful hand-biomechanics term is not evident in the author-origin path.
82. **Training and fitting are deterministic/stable.** No random seeds, multi-seed variance, or convergence-failure rate is reported.

---

# 7. Five weakest points a hostile reviewer would attack

## 7.1 The hand contribution is numerically small and statistically unsupported

The core paper framing emphasizes fine-grained hand reconstruction, but against the strongest adapted baseline EVA*, Table 1 improves:

- LHand by only **0.20 mm (1.46%)**;
- RHand by **0.60 mm (4.39%)**.

The final HPf → HPf+bio ablation changes are −0.02 mm left and +0.02 mm right. There are no confidence intervals, sign-level paired tests, seed variance, or reference-noise calibration. A hostile reviewer can argue that the principal hand claim is carried more by qualitative plausibility than by a demonstrated robust quantitative effect.

## 7.2 The written method and released optimizer do not match closely enough

Eq. (12) omits executable details and fitting weights. The implementation:

- adds normalized HaMeR depth not shown in Eq. (2);
- uses L1 initializer anchors where Eqs. (13)–(14) suggest a generic robust penalty;
- duplicates L1 and robust hand anchors;
- implements body biomechanics as a linear mean hinge rather than Eq. (11)’s squared sum;
- does not expose a clear formula-faithful 15-joint hand biomechanical loss in the inspected author-origin fitting path;
- retains additional generic SMPLify-X terms outside Eq. (12).

This allows a reviewer to challenge both reproducibility and whether the reported gains can be attributed to the advertised objective.

## 7.3 The evaluation is too narrow for the breadth of the claims

Quantitative evaluation uses:

- one native right-handed DGS signer;
- 57 isolated signs;
- 2,872 central frames;
- no quantitative continuous-sign evaluation;
- no signer-held-out or language-held-out test;
- asymmetric one-hand treatment between LHand and RHand;
- no failure-mode stratification.

The external blur/occlusion/noise evidence consists of selected qualitative examples. A hostile reviewer can characterize “robust in-the-wild sign reconstruction” as under-validated.

## 7.4 SignHPoser has a severe domain and retargeting mismatch

The prior is learned from 8 Auslan/ASL signers fingerspelling 93 words, then used on lexical DGS signs. Supplement S3 further admits that wrist rotations could not be transferred because of rig incompatibility. Yet wrist/palm orientation is central to sign phonology and remains penalized by translation-removed V2V. No cross-language test, unseen-handshape analysis, or retargeting-error measurement is supplied.

A hostile reviewer can therefore argue that SignHPoser is a narrow finger-pose regularizer, not a validated general sign-language hand prior.

## 7.5 The physical/contact and robustness claims exceed what the objective measures

Eq. (12) contains an anti-penetration term but no explicit desired-contact term. Eq. (15) regularizes body pose only, not hands. Occlusion, blur, and contact are evaluated qualitatively, and TR-V2V does not score contact or phonological correctness. The SGNify reference itself contains implausible hands.

A hostile reviewer can argue that:

- “contact-aware” is overstated;
- “bio-mechanically accurate” is not independently measured;
- visually plausible divergence from flawed GT cannot be quantified by the reported metric;
- robustness claims are anecdotal rather than statistically demonstrated.

---

# 8. Paper–code and reporting inconsistencies worth flagging explicitly

1. **TR-V2V wording versus evaluator behavior:** the paper says meshes are translationally aligned; the evaluator centers each evaluated region independently. This distinction changes what hand scores measure.
2. **Body final-gain percentage:** Table 2’s printed BPf 30.28 and final 30.13 imply about 0.50%, while the text reports 0.33%.
3. **Table reference typo:** Sec. 5.1 says “Table S2” when discussing the main-paper SignHPoser ablation shown as Table 3.
4. **Test-set selection wording:** Supplement S4 says best hyperparameters are selected on DEV and TEST data.
5. **Missing fitting weights:** \(\lambda_{1:6}\), \(\lambda_{\bar\zeta}\), and \(\lambda_{\epsilon_l,\epsilon_r}\) are not reported as a complete fitting specification.
6. **Geman–McClure details:** the paper does not define whether \(\psi\) acts componentwise or on a vector norm, nor report its scale.
7. **Hand notation:** Eq. (14) uses singular \(\theta_h\), although Fig. 2 uses independent left/right latents; masking and bilateral aggregation are not formally defined.
8. **Contact terminology:** “contact-aware terms” in the contribution prose are not matched by a positive contact term in Eq. (12).
9. **Hand biomechanical implementation:** the released author-origin fitting path does not visibly match Eq. (11)’s 15-joint hand formulation.
10. **Runtime:** hardware is reported, but no runtime or throughput is reported.

---

# 9. Bottom-line answers to the four technical questions

## 9.1 What is the pipeline?

A monocular frame is initialized by SMPLer-X and HaMeR, fitted to Sapiens body and HaMeR hand keypoints, and regularized by independent body/left-hand/right-hand pose VAEs, anti-penetration, previous-frame body smoothing, and anatomical angle limits. The result is a per-frame optimized SMPL-X mesh. The priors constrain pose plausibility; they do not themselves infer pose from pixels or model a whole sign sequence.

## 9.2 Why do the listed failure modes remain?

Every difficult condition weakens the same external observations that initialize and supervise the optimizer. The priors narrow the solution space but cannot uniquely recover hidden scene-specific geometry. Contact is represented only through nonpenetration, hand temporal dynamics are absent, and hard one-hand masking can suppress valid evidence.

## 9.3 Where do the remaining hand millimetres live?

They are mean per-vertex mismatches after independent hand-centroid removal. They can reflect articulation, orientation, scale/shape, within-hand depth, and disagreement with SGNify mesh artifacts. They do not include constant translation of the whole hand region. The paper cannot quantify how much is true reconstruction error versus reference error versus metric choice.

## 9.4 What is the central review risk?

The strongest result is a large upper-body score reduction. The hand score improvement is modest, the benchmark is narrow, the reference is acknowledged to be imperfect, and the written objective is not a complete or fully faithful description of the released fitting code. These limitations weaken causal claims about the learned hand prior and broad claims of physically/contact-correct sign reconstruction.

---

# 10. Source map

## Paper and supplementary

- Main Fig. 1: qualitative task framing.
- Main Fig. 2: full DexAvatar pipeline.
- Main Fig. 3: biomechanical body-data filtering.
- Main Fig. 4: biomechanical hand-data correction.
- Main Fig. 5: SGNify qualitative comparisons.
- Main Eqs. (1)–(4): SMPLify-X baseline terms.
- Main Eqs. (5)–(11): prior-training losses and biomechanics.
- Main Eq. (12): DexAvatar fitting objective.
- Main Eqs. (13)–(15): body prior, hand prior, temporal loss.
- Main Table 1: benchmark comparison.
- Main Table 2: SignBPoser ablation.
- Main Table 3: SignHPoser ablation.
- Supplement S2 / Figs. S1–S3: body ROM and signer space.
- Supplement S3 / Figs. S4–S6: MANUS-to-SMPL-X retargeting and wrist-rotation limitation.
- Supplement S4 / Tables S1–S2: latent and biomechanics hyperparameter sweeps.
- Supplement S5 / Table S3: SignHPoser with VPoser ablation.
- Supplement S6 / Fig. S7: SGNify reference limitations.
- Supplement S7 / Figs. S8–S10: qualitative blur, self-occlusion, and Gaussian-noise cases.

## Repository evidence

- TR-V2V centering: `data/evaluation_from_author/evaluate_new_fitting.py:159-169`.
- One-hand region/evaluation behavior: `data/evaluation_from_author/evaluate_new_fitting.py:380-395`.
- SGNify capture and metric description: `data/paper/SGNify_Paper.md:121-151`.
- HaMeR normalized depth and temporal term: `dexavatar_fitting/smplifyx/fitting.py:568-704`.
- SignBPoser initializer anchor: `dexavatar_fitting/smplifyx/fitting.py:706-712`.
- SignHPoser latent and initializer terms: `dexavatar_fitting/smplifyx/fitting.py:828-875`.
- Generic priors/collision: `dexavatar_fitting/smplifyx/fitting.py:928-969`.
- Frame filtering and active-side behavior: `dexavatar_fitting/smplifyx/data_parser.py:185-237`.
- Fitting-stage weights: `dexavatar_fitting/smplifyx/cfg_files/fit_smplx_vposer_x.yaml:45-89`.
- Sequential temporal state and frame failures: `dexavatar_fitting/smplifyx/main.py:225-352`.
