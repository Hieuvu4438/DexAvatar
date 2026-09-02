# SignEFT-X: Exact-Incumbent Asymmetric Multi-Expert Rescue for Monocular 3D Sign Language Reconstruction

> **Paper-method dossier, not a camera-ready claim.** This document is written in the style of a computer-vision paper so that the method, novelty, and empirical support can be audited before manuscript submission. All reported numbers come from the attached SignEFT-X artifacts. The frozen H1 experiment is confirmatory; EI-AMER/H15-v2 was formulated after an earlier holdout had been opened and is therefore reported as exploratory until it is evaluated on a new prospectively sanctioned set.

## Abstract

Monocular 3D sign language reconstruction is especially sensitive to hand failures: a generic whole-body estimate can be geometrically plausible while losing the finger configurations that distinguish signs. We present **SignEFT-X with Exact-Incumbent Asymmetric Multi-Expert Rescue (EI-AMER)**, an inference-time refinement method for an existing SMPL-X reconstruction. The method first maps WiLoR and HaMeR hand observations into a palm-centered, scale-normalized coordinate system and retargets only the 15 SMPL-X finger rotations while holding body pose, wrist orientation, shape, expression, and camera fixed. A frozen WiLoR-based estimator serves as an immutable incumbent. HaMeR may propose a rescue only for hand sides rejected by that incumbent; the rescue must be supported independently by a Sapiens heatmap likelihood and HaMeR canonical 3D evidence, while contradictory WiLoR evidence vetoes it. An expert-specific centroid regularizer is applied to HaMeR rescues but not to the validated WiLoR path. Finally, an artifact-level construction guarantees that non-rescued states are exact incumbent copies and rejected states are exact baseline fallbacks.

On the attached 57-sign/1,493-frame protocol, the frozen WiLoR stage improves all six official translation-aligned errors over the A3f control, with fully negative paired-sign bootstrap intervals. The exploratory EI-AMER stage further changes the aggregate errors from 42.0696/25.8053/29.1131/39.6254/12.5219/11.9180 mm to 42.0640/25.7991/29.1057/39.6121/12.5060/11.8431 mm for All, UBody, UBody-F, UBody-H, LHand, and RHand, respectively. Its incremental effect over the frozen incumbent is small: only All and UBody-H have fully negative 95% percentile intervals. The evidence therefore supports a paper contribution about **conservative, factorized multi-expert hand refinement**, but does not yet support an unbiased generalization or state-of-the-art claim for EI-AMER.

## 1. Introduction

Sign languages convey lexical and grammatical information through coordinated handshape, orientation, location, movement, and non-manual signals. Recovering a plausible human mesh is therefore insufficient: the reconstruction must preserve fine finger articulation while remaining consistent with the signer’s body and the input image. Monocular video makes this difficult because hands are small, frequently self-occluded, blurred, and observed under depth ambiguity.

Existing sign-specific systems attack this problem with linguistic or learned pose priors. SGNify introduces universal linguistic priors for expressive signing avatars [1], while DexAvatar learns SignHPoser and SignBPoser to constrain hand and body pose [2]. More recently, Tamaththul3D combines a whole-body reconstruction with WiLoR hand estimates and 2D-supervised optimization [3]. These systems establish that sign-domain priors and specialized hand estimators are useful, but they leave a practical question unresolved: **how can a new hand expert improve an already validated SMPL-X reconstruction without silently damaging frames or body regions that were already correct?**

SignEFT-X treats this as a selective refinement problem rather than a full-body replacement problem. Frozen vision models generate observations, a constrained SMPL-X optimizer generates candidate finger articulations, and a factorized gate decides whether each hand side should retain the baseline, use the validated incumbent, or accept a rescue expert. The method is deliberately framewise and ground-truth-free at inference. It also separates two notions that are often conflated: metric non-regression on a finite benchmark and structural non-regression of the output artifact. The latter is enforced by copying the incumbent state exactly whenever no rescue is accepted.

The paper-level contributions that can be defended from the current implementation are:

1. **Canonical cross-model finger retargeting.** MANO-like hand observations are converted into a proper palm coordinate system, normalized to the shared-beta SMPL-X bone lengths, and fitted through bounded residuals on only 15 finger rotations.
2. **Exact-incumbent rescue.** A validated WiLoR path is frozen at the state and decision level; HaMeR is allowed to act only on incumbent rejects. This is a structurally monotonic cascade, not a regeneration that merely preserves aggregate acceptance counts.
3. **Asymmetric proposal geometry with factorized evidence.** The rescue expert receives a centroid-neutral articulation constraint, whereas the incumbent expert remains unmodified. Sapiens and the proposing expert must both show uncertainty-normalized improvement, and WiLoR is used only as a conflict veto for a HaMeR rescue.
4. **Artifact-level safety contract.** Non-rescued outputs are exact H1 copies, and complete rejects are exact A3f fallbacks. The contract is verified over pose arrays, meshes, decisions, and implementation hashes.
5. **A complete negative-result study.** Whole-body heatmap, NLF, coupling, segmentation, TTA, per-joint variance, hard cross-expert veto, SignHPoser veto, wrist unlock, compositional fingers, trust-radius, and symmetric-centroid alternatives are retained as ablations rather than omitted.

The contribution is **not** WiLoR, HaMeR, Sapiens, SMPL-X, SignHPoser, or SignBPoser. EI-AMER does not call DexAvatar’s SignHPoser or SignBPoser after the frozen H1 state has been produced.

## 2. Related Work and Search-Bounded Novelty

### 2.1 Monocular 3D sign language reconstruction

SGNify formalized sign language capture from monocular video and showed the value of linguistic priors, including perceptual evaluation of comprehensibility and naturalness [1]. Neural Sign Actors uses reconstructed 3D signing motion as part of a diffusion-based sign production framework [4]. SignAvatar studies word-level 3D sign reconstruction and generation [5]. DexAvatar improves sign reconstruction with learned hand and body pose priors [2]. Tamaththul3D introduces a Saudi Sign Language resource and integrates SMPL-X-compatible body and MANO-compatible hand estimates through forearm inverse kinematics followed by 2D-supervised shoulder refinement; its reported instantiation includes WiLoR hand estimates [3].

Tamaththul3D is the closest prior work for the narrow idea of adding WiLoR hand evidence to a sign reconstruction pipeline. Consequently, neither “using WiLoR” nor “optimizing hands under 2D keypoints” is a valid novelty claim for SignEFT-X. The defensible delta is the combination of canonical shared-shape retargeting, an immutable incumbent, rescue-only expert routing, expert-specific proposal regularization, factorized accept/veto roles, and artifact-level exactness.

### 2.2 General-purpose hand and human observations

WiLoR performs end-to-end in-the-wild hand localization and reconstruction [6], while HaMeR scales transformer-based hand reconstruction using large training corpora [7]. Sapiens provides whole-human foundation models, including 2D pose heatmaps [8]. These models are used here as frozen sources of heterogeneous evidence; no claim is made that SignEFT-X improves their representation learning.

### 2.3 Selective prediction and multi-expert deferral

Selective prediction and learning-to-defer methods formalize abstention and routing between predictors or experts [9,10]. Hence, “using a cascade” or “deferring to an expert” is not novel in machine learning generally. The subfield-specific question addressed here is how to implement a conservative cascade over coupled SMPL-X artifacts when an accepted change to one hand can unintentionally perturb other state variables or evaluation regions.

### 2.4 Novelty-search protocol and boundary

The novelty assessment is based on targeted searches performed on **2026-09-01/02** across CVF Open Access, arXiv, PMLR, and official project repositories. Queries combined terms for *3D sign language reconstruction*, *WiLoR/HaMeR sign hand refinement*, *multi-expert hand reconstruction*, *selective prediction*, *deferral*, and *incumbent-preserving refinement*. It is not a systematic review.

The search found prior examples of sign-specific priors, WiLoR-based sign reconstruction, 2D-supervised hand optimization, general multi-expert deferral, and uncertainty-aware selection. It did **not** identify a prior sign-reconstruction method that explicitly combines all of the following: (i) a frozen SMPL-X incumbent state, (ii) rescue-only cross-model finger proposals, (iii) expert-specific proposal geometry, (iv) proposal-source evidence plus an independent image likelihood and a conflicting-expert veto, and (v) an audited guarantee that all non-rescue artifacts are byte-/array-equivalent to the incumbent or fallback. This is a **search-bounded differentiation**, not an absolute “first” claim.

| Candidate claim | Novelty assessment | Paper-safe wording |
|---|---|---|
| Use WiLoR to improve signing hands | Low; directly anticipated by Tamaththul3D | Do not claim as novel |
| Fit external hand joints to SMPL-X fingers | Low-to-moderate; optimization/retargeting is established | Describe as a necessary technical component |
| Palm-canonical, shared-beta, wrist-locked retargeting | Moderate as a specific integration | “A canonical retargeting formulation designed to avoid cross-model wrist/shape leakage” |
| Multi-expert cascade | Low in general ML | Do not claim generic deferral as novel |
| Exact frozen incumbent plus reject-only rescue | Moderate-to-high in this task, subject to broader review | “An exact-incumbent rescue construction for coupled SMPL-X outputs” |
| Expert-specific centroid geometry and factorized accept/veto roles | Moderate-to-high as a combined method | “An asymmetric proposal-and-evidence design motivated and validated by ablation” |
| Artifact-level non-regression audit | Moderate and practically distinctive | “A verifiable output contract,” not “semantic safety” |

The relation to the three nearest sign-reconstruction families is summarized below. The last column is the manuscript’s intended technical delta, not a claim that earlier systems are deficient outside this specific design objective.

| Prior family | Primary contribution of that work | Overlap with SignEFT-X | Remaining delta claimed here |
|---|---|---|---|
| SGNify [1] | Linguistic priors and expressive monocular signing-avatar capture | Same task family and SMPL-X-style output | Frozen general hand experts, selective finger-only proposals, and exact rollback rather than new linguistic priors |
| DexAvatar [2] | Learned SignHPoser/SignBPoser priors for sign reconstruction | Supplies the upstream reconstruction being refined | Post-H1 EI-AMER uses neither prior; contribution is proposal retargeting, routing, and invariants |
| Tamaththul3D [3] | Modular body/hand integration, forearm alignment, and 2D-supervised refinement | External hand estimator, including WiLoR, improves sign reconstruction | Wrist-locked shared-beta finger retargeting plus immutable-incumbent, reject-only HaMeR rescue and artifact identity |

## 3. Problem Formulation

For frame \(t\), let the baseline SMPL-X state be

\[
\Theta_t^0 = (\theta_t^{B},\theta_t^{W,L},\theta_t^{W,R},
\theta_t^{H,L},\theta_t^{H,R},\beta_t,\psi_t,c_t),
\]

where \(\theta^B\) is body pose, \(\theta^{W,s}\) is wrist orientation, \(\theta^{H,s}\in SO(3)^{15}\) is the finger pose for side \(s\in\{L,R\}\), \(\beta\) is shape, \(\psi\) is facial state, and \(c\) denotes camera parameters. The decoder \(M(\Theta)\) produces the SMPL-X mesh and joints.

The objective is not to re-estimate the full state. For each side, the method selects one of three actions:

\[
a_{t,s}\in\{\text{baseline},\text{WiLoR-incumbent},\text{HaMeR-rescue}\},
\]

subject to

\[
(\theta^B,\theta^{W,L},\theta^{W,R},\beta,\psi,c)
=
(\theta^{B,0},\theta^{W,L,0},\theta^{W,R,0},\beta^0,\psi^0,c^0).
\]

Only the selected side’s 15 finger rotations may change. The inference objective uses neither ground-truth meshes nor temporal pose terms. Neighboring frames are not used as pose targets.

## 4. Method

### 4.1 Overview

The method has two nested stages. **H1** constructs a WiLoR finger proposal for each available hand and accepts it only when independent Sapiens 2D evidence and WiLoR 3D evidence both support the change. This stage was frozen before evaluation on the untouched 45-sign partition. **EI-AMER** then loads the resulting H1 state and decision as an immutable incumbent. If H1 accepted a side, EI-AMER copies it exactly. If H1 rejected the side, HaMeR may generate a rescue proposal under a separate gate. If the rescue fails, the output remains the exact H1 artifact, which itself contains exact A3f fallback on H1-rejected sides.

### 4.2 Palm-canonical cross-model representation

Let \(J\in\mathbb{R}^{21\times3}\) be hand joints ordered as wrist plus four joints for each of five fingers. Root-centering removes global translation:

\[
\bar J_i = J_i-J_0.
\]

Using the index and little-finger metacarpophalangeal joints (indices 5 and 17), the method constructs

\[
x = \operatorname{norm}(\bar J_5-\bar J_{17}),
\]

\[
\tilde y=\tfrac12(\bar J_5+\bar J_{17}),\qquad
y=\operatorname{norm}(\tilde y-(\tilde y^\top x)x),
\]

\[
z=\operatorname{norm}(x\times y),\qquad y=z\times x.
\]

The proper palm frame is \(R_P=[x\;y\;z]\), and every accepted frame must satisfy \(\det(R_P)>0.999\). Scale is defined by the wrist-to-middle-MCP distance,

\[
s=\lVert\bar J_9\rVert_2,
\]

giving the canonical hand

\[
\mathcal C(J)=\bar J R_P/s.
\]

This operation removes global hand translation, palm rotation, and scale before comparing a MANO-like expert output with SMPL-X. It is central to preventing an external hand model’s camera, root, or shape convention from leaking into the whole-body state.

### 4.3 Shared-beta bone normalization

Even after palm canonicalization, expert and SMPL-X hands can have different proportions. Let \(E=\mathcal C(J^{expert})\) and \(R=\mathcal C(J^{SMPLX,0})\). Along every kinematic finger edge \((p(i),i)\), the target retains the expert direction but imposes the reference SMPL-X length:

\[
T_i=T_{p(i)}+
\frac{E_i-E_{p(i)}}{\lVert E_i-E_{p(i)}\rVert_2}
\lVert R_i-R_{p(i)}\rVert_2.
\]

Thus, candidate articulation follows the expert while morphology remains consistent with the baseline’s shared \(\beta\). The method does not transplant the expert MANO mesh or its wrist pose.

### 4.4 Bounded SO(3) finger fitting

For one side, let \(Q_j^0\in SO(3)\) be the 15 baseline finger rotations and \(\delta_j\in\mathbb{R}^3\) their Lie-algebra residuals. A candidate is decoded as

\[
Q_j(\delta)=\exp([\delta_j]_\times)Q_j^0.
\]

The fitting objective is

\[
\mathcal L_{fit}=
\frac1{20}\sum_{i=1}^{20}\rho\!\left(\mathcal C(J(\delta))_i-T_i\right)
+\lambda_e\left\lVert
\mu(\mathcal C(J(\delta))_{1:20})-\mu(R_{1:20})
\right\rVert_2^2
+0.2\frac1{15}\sum_j\lVert\delta_j\rVert_2^2,
\]

where \(\rho\) is smooth L1 and \(\mu\) is the joint centroid. Optimization uses Adam for 40 steps, learning rate 0.03, and cosine annealing. Search residuals are bounded at 12°, after which the best proposal is projected to an 8° production trust region. The final asymmetric configuration uses

\[
\lambda_{WiLoR}=0,\qquad \lambda_{HaMeR}=0.5.
\]

This asymmetry is intentional. A symmetric centroid penalty improved development metrics but perturbed the validated incumbent proposal and failed one held-out regional metric. EI-AMER therefore leaves the incumbent geometry untouched and regularizes only the new rescue family.

### 4.5 Sapiens heatmap evidence

For each hand landmark \(i\), the Sapiens heatmap is normalized to a probability map. Its reliability weight is

\[
w_i=q_i\left(1-\frac{H_i}{\log(64\cdot48)}\right)v_i,
\]

where \(q_i\) is detector score, \(H_i\) is heatmap entropy, and \(v_i\) indicates validity. Projected candidate joints are evaluated by heatmap negative log-likelihood (NLL). A frozen per-hand 2D nuisance translation handles detector/camera offset without modifying the SMPL-X camera.

Let \(E^{2D}_0\) and \(E^{2D}_c\) be the weighted baseline and candidate NLL. The evidence delta is

\[
\Delta^{2D}=E^{2D}_c-E^{2D}_0.
\]

The uncertainty \(\sigma^{2D}\) is estimated from weighted joint-level variation and effective sample size. Negative delta favors the candidate.

### 4.6 Expert-specific canonical 3D evidence

The canonical residual to expert \(e\) is

\[
E^e(J)=\frac1{20}\sum_{i=1}^{20}
\left\lVert\mathcal C(J)_i-T_i^e\right\rVert_2.
\]

For candidate \(c\),

\[
\Delta^e=E^e(J_c)-E^e(J_0).
\]

Its uncertainty is a jointwise standard-error estimate, divided by detector confidence clipped below at 0.25 and lower-bounded numerically. This makes a weak expert require a larger observed improvement.

### 4.7 Frozen H1 incumbent gate

A WiLoR proposal is accepted only if it is available and wins against the baseline by two estimated standard errors under both evidence families:

\[
A^{H1}_{t,s}=
\mathbf 1[available]
\mathbf 1[\Delta^{2D}<-2\sigma^{2D}]
\mathbf 1[\Delta^{W}<-2\sigma^{W}].
\]

Body, wrist, shape, camera, face, and the opposite hand are fixed. Numerical drift audits require opposite-hand, face, and lower-body changes to remain below 0.01 mm. If the gate fails, exact rollback returns the A3f artifact.

### 4.8 Exact-incumbent asymmetric rescue

EI-AMER does not recompute H1 acceptance. It reads the frozen H1 decision \(A^{H1}_{t,s}\). A HaMeR proposal is eligible only when the incumbent rejected that side. It is accepted when:

\[
A^{R}_{t,s}=
(1-A^{H1}_{t,s})
\mathbf 1[HaMeR\ available]
\mathbf 1[\Delta^{2D}_H<-2\sigma^{2D}_H]
\mathbf 1[\Delta^{H}<-2\sigma^{H}]
\mathbf 1[\Delta^{W}_H\leq\sigma^{W}_H].
\]

The first two evidence terms require agreement between the image heatmap and the proposing HaMeR expert. The final WiLoR term is not counted as a third positive vote: it is a **conflict veto** that rejects HaMeR candidates strongly contradicted by WiLoR. The action is

\[
a_{t,s}=
\begin{cases}
\text{WiLoR-incumbent}, & A^{H1}_{t,s}=1,\\
\text{HaMeR-rescue}, & A^{H1}_{t,s}=0\land A^R_{t,s}=1,\\
\text{baseline}, & \text{otherwise}.
\end{cases}
\]

This factorization avoids treating correlated expert outputs as independent evidence. It does not eliminate correlation—WiLoR, HaMeR, and Sapiens may share training data—but it assigns each signal a distinct operational role.

### 4.9 Artifact-level exactness contract

Metric averages cannot prove that an incumbent was preserved. EI-AMER therefore enforces the following construction:

1. If neither side is rescued, copy the incumbent OBJ and NPZ exactly.
2. If one side is rescued, load the incumbent NPZ, overwrite only that side’s hand-pose array, regenerate vertices, and preserve every other state array.
3. If H1 rejected a side and no rescue is accepted, that side remains the exact A3f/H1 fallback.
4. Every decision stores input/output hashes, expert-observation hashes, implementation hash, accepted and rescued sides, evidence deltas, and fixed-variable flags.

This contract concerns **computational identity and scope of modification**. It should not be described as a guarantee of linguistic correctness, fairness, or absence of all reconstruction errors.

### 4.10 Inference algorithm

```text
Input: RGB frame, A3f SMPL-X state, frozen H1 state/decision,
       Sapiens heatmaps, WiLoR joints, HaMeR joints

for each hand side s:
    if frozen H1 decision accepts s:
        preserve H1 side exactly
    else:
        canonicalize HaMeR and baseline SMPL-X joints
        normalize HaMeR bones to baseline shared-beta lengths
        fit a centroid-regularized, wrist-locked 15-finger proposal
        compute Sapiens, HaMeR, and WiLoR delta/sigma evidence
        accept rescue iff Sapiens and HaMeR win at 2 sigma
                         and WiLoR does not strongly conflict

if no side is rescued:
    exact-copy the H1 artifact
else:
    overwrite only rescued hand-pose arrays in the H1 state
    regenerate the mesh and audit all protected arrays

Output: canonical SMPL-X OBJ, NPZ state, and decision record
```

## 5. Experimental Protocol

### 5.1 Data partitions and claim status

The attached benchmark contains 57 isolated signs and 1,493 paired frames. The engineering partition contains 12 signs/298 frames; the remaining partition contains 45 signs/1,195 frames. The full57 result combines both partitions and is not an independent test set.

| Partition | Signs | Frames | Role for H1 | Role for EI-AMER |
|---|---:|---:|---|---|
| Engineering12 | 12 | 298 | Development/ablation | Development/ablation |
| Untouched/Exploratory45 | 45 | 1,195 | Prospective confirmation after H1 freeze | Exploratory; already opened before H15 hypothesis |
| Full57 | 57 | 1,493 | Aggregate confirmed protocol result | Exploratory aggregate |

There is an unresolved protocol-size discrepancy: the attached manifest contains 1,493 frames, fewer than counts reported in some prior-paper settings. Until the pairing and exclusion rules are reconciled, every claim must be scoped to this attached 57-sign/1,493-frame protocol.

### 5.2 Metrics

The official evaluator reports translation-aligned mean vertex error in millimetres; lower is better. Each region is centered independently before vertex distances are measured.

| Metric | Region |
|---|---|
| All | All 10,475 SMPL-X vertices |
| UBody | Above-pelvis upper body |
| UBody-F | Upper body excluding face |
| UBody-H | Upper body excluding head |
| LHand | Left MANO-to-SMPL-X hand vertices; omitted for one-handed class-0 frames |
| RHand | Right MANO-to-SMPL-X hand vertices |

The main aggregates use the author’s vertex-micro protocol. An independent audited evaluator reconstructs the same regions and agrees with the official evaluator after rounding. Statistical comparisons use 10,000 paired bootstrap replicates over signs, not over individual vertices or frames, with seed 20260901.

### 5.3 Controls and leakage restrictions

All variants use the same manifest, A3f states, topology, camera parameters, beta, evaluator, and frozen observation caches. Ground truth and evaluator segment assets are used only after fitting. The optimization objective contains no ground-truth or temporal-pose term. The official evaluator hash is `2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300`.

### 5.4 Promotion rule

On Engineering12, a candidate must be non-worse than its reference on all six metrics, improve at least one target hand metric, and pass all provenance/invariant tests. A frozen candidate is then evaluated without retuning on the external partition. This rule selected H1. It cannot retrospectively turn EI-AMER into a confirmatory result because the H15 hypothesis was motivated after examining H14 on the 45-sign partition.

## 6. Main Results

### 6.1 Frozen H1 result

| Method | Split | All | UBody | UBody-F | UBody-H | LHand | RHand |
|---|---|---:|---:|---:|---:|---:|---:|
| A3f/C0 | Engineering12 | 41.1539 | 24.4695 | 27.7074 | 38.9223 | 12.3310 | 11.9162 |
| H1 | Engineering12 | **41.1480** | **24.4635** | **27.7001** | **38.9090** | **12.0412** | **11.6415** |
| A3f/C0 | Untouched45 | 42.3287 | 26.1718 | 29.5062 | 39.8910 | 12.9821 | 12.1802 |
| H1 | Untouched45 | **42.3001** | **26.1411** | **29.4671** | **39.8056** | **12.6482** | **11.9869** |
| A3f/C0 | Full57 | 42.0936 | 25.8311 | 29.1458 | 39.6963 | 12.8466 | 12.1275 |
| H1 | Full57 | **42.0696** | **25.8053** | **29.1131** | **39.6254** | **12.5219** | **11.9180** |

H1 improves Full57 over C0 by -0.0241, -0.0258, -0.0327, -0.0709, -0.3247, and -0.2096 mm for the six metrics. All six paired-sign 95% intervals are negative: All [-0.0355,-0.0129], UBody [-0.0391,-0.0133], UBody-F [-0.0481,-0.0176], UBody-H [-0.0967,-0.0400], LHand [-0.4674,-0.2025], and RHand [-0.2906,-0.1209] mm. H1 accepted 991 hand sides in 756/1,493 frames and returned exact fallback in 737 frames.

### 6.2 Exploratory EI-AMER result

| Method | Split | All | UBody | UBody-F | UBody-H | LHand | RHand |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen H1 | Engineering12 | 41.1480 | 24.4635 | 27.7001 | 38.9090 | 12.0412 | 11.6415 |
| EI-AMER | Engineering12 | **41.1458** | **24.4597** | **27.6955** | **38.9032** | **12.0184** | **11.5082** |
| Frozen H1 | Exploratory45 | 42.3001 | 26.1411 | 29.4671 | 39.8056 | 12.6482 | 11.9869 |
| EI-AMER | Exploratory45 | **42.2937** | **26.1344** | **29.4590** | **39.7905** | **12.6341** | **11.9266** |
| Frozen H1 | Full57 | 42.0696 | 25.8053 | 29.1131 | 39.6254 | 12.5219 | 11.9180 |
| EI-AMER | Full57 | **42.0640** | **25.7991** | **29.1057** | **39.6121** | **12.5060** | **11.8431** |

Full57 aggregate deltas for EI-AMER minus H1 are -0.0056, -0.0062, -0.0074, -0.0133, -0.0159, and -0.0749 mm. These improvements are consistent in direction across Engineering12 and Exploratory45, but the external evidence is post-hoc and the effect is small.

### 6.3 Paired-sign uncertainty

| Metric | Mean sign delta, EI-AMER-H1 (mm) | 95% percentile CI | Improved/Worse signs | \(P(\Delta\geq0)\) | Interpretation |
|---|---:|---:|---:|---:|---|
| All | -0.0060 | [-0.0134, -0.0005] | 24/22 of 57 | 0.0138 | Interval excludes zero |
| UBody | -0.0065 | [-0.0162, 0.0013] | 24/22 of 57 | 0.0559 | Inconclusive |
| UBody-F | -0.0078 | [-0.0189, 0.0007] | 24/22 of 57 | 0.0425 | Percentile interval crosses zero |
| UBody-H | -0.0144 | [-0.0306, -0.0016] | 26/20 of 57 | 0.0121 | Interval excludes zero |
| LHand | -0.0112 | [-0.0400, 0.0182] | 17/17 of 42 | 0.2186 | Inconclusive |
| RHand | -0.0500 | [-0.1138, 0.0076] | 27/19 of 57 | 0.0470 | Percentile interval crosses zero |

Only All and UBody-H have fully negative percentile intervals versus H1. In contrast, EI-AMER versus C0 has fully negative intervals for all six metrics: All [-0.0430,-0.0162], UBody [-0.0490,-0.0156], UBody-F [-0.0600,-0.0212], UBody-H [-0.1159,-0.0508], LHand [-0.4866,-0.2088], and RHand [-0.3591,-0.1468] mm. The correct reading is that the complete SignEFT-X pipeline is reliably better than C0 on this protocol, whereas the added benefit of the HaMeR rescue over H1 is promising but not uniformly resolved.

### 6.4 Exactness and coverage

| Audit item | Engineering12 | Exploratory45 | Full57 |
|---|---:|---:|---:|
| Frames | 298 | 1,195 | 1,493 |
| H1/WiLoR sides preserved | 193 | 798 | 991 |
| HaMeR rescue sides | 99 | 330 | 429 |
| Baseline sides | 304 | 1,262 | 1,566 |
| Exact incumbent frames | 209 | 910 | 1,119 |
| Exact fallback frames | 109 | 414 | 523 |
| Invariant violations | **0** | **0** | **0** |

Some counts overlap by frame: an exact incumbent frame can contain an H1-accepted side and a baseline side, while “fallback exact” denotes frames with no accepted side. On Full57, all 991 frozen WiLoR sides are preserved and 429 HaMeR sides are added. The run resumed all 1,493 artifacts with one implementation hash and no rewrites. The EI-AMER implementation hash is `e627e54c460c400c87ca4c9d73fde59e087d5c6631c72790d61eb82c64f79ac0`.

## 7. Ablation Studies

The ablations below are grouped by scientific question rather than presented as an engineering timeline. Every metric is in millimetres and lower is better.

### 7.1 Can generic upper-body evidence improve the A3f state?

| Variant | Added mechanism | All | UBody | UBody-F | UBody-H | LHand | RHand | Accepted frames | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | Exact A3f control | 41.1539 | 24.4695 | 27.7074 | 38.9223 | 12.3310 | 11.9162 | 0 | Control |
| C1 | Sapiens heatmap likelihood | 41.1539 | 24.4695 | 27.7074 | 38.9223 | 12.3310 | 11.9162 | 0 | Reject: no two-family consensus |
| C2 | C1 + NLF 3D bone vectors | 41.1539 | 24.4695 | 27.7074 | 38.9223 | 12.3310 | 11.9162 | 0 | Reject: no consensus |
| C3-initial | Wrist-protected hand-body coupling | 41.1539 | 24.4695 | 27.7074 | 38.9223 | 12.3310 | 11.9162 | 0 | Reject: no activation |
| C3-lite-v3 | Relaxed protected coupling | 41.1645 | 24.4776 | 27.7424 | 39.2344 | 12.3322 | **11.9151** | 59 | Reject: 5/6 regress |
| C4-v3 | Sapiens2 part probabilities + soft point splat | 41.2270 | 24.5572 | 27.8197 | 39.2162 | 12.3311 | **11.9156** | 48 | Reject: upper-body regressions |
| C5 | Pointmap depth/surface evidence | — | — | — | — | — | — | — | Not run: gated on C4 promotion |
| C6 | Pointmap-derived normals | — | — | — | — | — | — | — | Not run: gated on C5 |

The result falsifies the original assumption that broad upper-body evidence would yield easy gains. C4 is a probability-map/soft-point-splat implementation rather than the exact triangle rasterizer originally proposed; it is therefore evidence against that tested realization, not every possible segmentation formulation. Because C4 failed its predefined promotion gate, C5 and C6 were not executed and must not be described as ablated results.

### 7.2 Which initial hand formulation should be promoted?

| Variant | Hand mechanism | All | UBody | UBody-F | UBody-H | LHand | RHand | Accepted frames | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| H0 | A3f hand control | 41.1539 | 24.4695 | 27.7074 | 38.9223 | 12.3310 | 11.9162 | 0 | Control |
| H1 | Canonical WiLoR fingers; wrist locked | **41.1480** | **24.4635** | **27.7001** | **38.9090** | **12.0412** | 11.6415 | 142 | Promote |
| H2 | WiLoR TTA medoid | 41.1471 | 24.4630 | 27.6995 | 38.9080 | 12.0695 | 11.6801 | 137 | Reject: both hands worse than H1 |
| H3 | Per-joint low-variance transfer | 41.1534 | 24.4688 | 27.7067 | 38.9225 | 12.3147 | 11.9094 | 56 | Reject: too selective/weak |
| H4 | HaMeR as hard cross-expert consensus | 41.1484 | 24.4643 | 27.7014 | 38.9130 | 12.0737 | 11.6496 | 129 | Reject: does not beat H1 |
| H5 | DPoser-X veto after H4 | — | — | — | — | — | — | — | Not run: optional dependency not promoted |
| H6 | Tiny wrist unlock | 41.1486 | 24.4641 | 27.7010 | 38.9113 | 12.0442 | **11.6325** | 145 | Reject: 5/6 regress |

H6 is an informative negative control: unlocking the wrist can improve one hand metric yet spread error into the coupled body/hand regions. The frozen method therefore keeps both wrists fixed. H4 also shows that HaMeR’s failure as a universal veto does not imply that HaMeR lacks complementary proposals; this motivated the later rescue formulation.

### 7.3 Do local finger composition or SignHPoser improve H1?

| Variant | Mechanism | All | UBody | UBody-F | UBody-H | LHand | RHand | Accepted frames | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| H1 | Frozen whole-hand candidate | 41.1480 | 24.4635 | 27.7001 | 38.9090 | 12.0412 | 11.6415 | 142 | Reference |
| H7 | Compositional fingers | 41.1484 | 24.4637 | 27.7006 | 38.9107 | 12.0444 | 11.6574 | 155 | Reject: local candidate can displace good whole hand |
| H7b | Monotonic compositional rescue | 41.1479 | 24.4634 | 27.7000 | 38.9089 | **12.0396** | **11.6378** | 155 | Dev pass; external reject |
| H8 | H7b + SignHPoser KL veto | **41.1471** | **24.4613** | **27.6973** | **38.9035** | **12.0309** | **11.6357** | 148 | Dev pass; external reject |

H7 demonstrates that a strong local per-finger score is not globally rank-equivalent to a whole-hand score. H7b fixes the logical error by allowing local candidates to rescue only H1 rejects. H8 uses the original DexAvatar SignHPoser only as a veto; it improves Engineering12 but regresses all six metrics on the 45-sign partition. This is evidence that the tested prior threshold captured development-specific bias rather than reliable uncertainty. SignHPoser is therefore excluded from EI-AMER.

### 7.4 How sensitive is H1 to the SO(3) trust radius?

| Production radius | All | UBody | UBody-F | UBody-H | LHand | RHand | Accepted hands | Accepted solutions at bound |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4° | 41.1501 | 24.4651 | 27.7024 | 38.9139 | 12.0738 | 11.6775 | 196 | 185 |
| 6° | 41.1491 | 24.4647 | 27.7017 | 38.9115 | 12.0558 | 11.6467 | 194 | 98 |
| 8° | 41.1480 | 24.4635 | 27.7001 | 38.9090 | 12.0412 | 11.6415 | 193 | 34 |
| 10° | 41.1471 | 24.4627 | 27.6991 | 38.9073 | 12.0325 | 11.6407 | 193 | 11 |
| 12° | **41.1470** | **24.4626** | **27.6990** | **38.9071** | **12.0311** | **11.6407** | 193 | 0 |

The engineering curve indicates underfitting at 8°, but the frozen 12° candidate changed untouched45 RHand from 11.9869 to 11.9871 mm, a +0.0002 mm regression, and was rejected under the strict gate. The radius study is mechanistic evidence, not a promoted contribution. EI-AMER retains the 8° production radius.

### 7.5 Why use asymmetric rescue geometry?

| Variant | Proposal/gate change relative to H1 | All | UBody | UBody-F | UBody-H | LHand | RHand | Accepted hands | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| H1 | WiLoR incumbent only | 41.1480 | 24.4635 | 27.7001 | 38.9090 | 12.0412 | 11.6415 | 193 | Reference |
| H13 | Add HaMeR rescue; no centroid term | 41.1485 | 24.4624 | 27.6990 | 38.9112 | 12.0235 | **11.5035** | 291 | Reject/repair: All and UBody-H regress |
| H14 | Centroid weight 0.5 for WiLoR and HaMeR | **41.1431** | **24.4561** | **27.6915** | **38.8977** | 12.0377 | 11.5398 | 292 | Dev pass; external UBody-H reject |
| H15-v1 | Weight 0 for WiLoR, 0.5 for HaMeR; regenerate both | 41.1458 | 24.4597 | 27.6955 | 38.9032 | **12.0184** | 11.5082 | 292 | Reject: 389 non-rescue sides differ from frozen H1 |
| H15-v2 / EI-AMER | Load exact H1; fit/overwrite rescue sides only | 41.1458 | 24.4597 | 27.6955 | 38.9032 | **12.0184** | 11.5082 | 292 | Exploratory pass; zero invariant violations |

H13 proves that HaMeR supplies complementary right-hand proposals: it adds 98 rescue sides on Engineering12 and reduces RHand by 0.1380 mm relative to H1. However, a hand can improve after hand-specific translation alignment while its centroid shift damages All or UBody-H. H14 adds centroid protection symmetrically and improves all development metrics, but it also changes the incumbent WiLoR proposal geometry and fails external UBody-H. H15 makes the regularizer expert-specific.

H15-v1 and H15-v2 have identical aggregate metrics but are not equivalent methods. H15-v1 regenerated the WiLoR path; the artifact audit found 389 non-rescue hand sides different from frozen H1 even though the total WiLoR acceptance count remained 991. H15-v2 consumes the frozen H1 state directly and achieves zero violations. This ablation is the principal evidence that exact-incumbent construction is more than bookkeeping.

### 7.6 External rejection ledger

| Variant | All | UBody | UBody-F | UBody-H | LHand | RHand | External interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen H1 | 42.3001 | 26.1411 | 29.4671 | 39.8056 | 12.6482 | 11.9869 | Confirmatory reference |
| H7b | 42.3000 | 26.1411 | 29.4670 | 39.8055 | 12.6486 | 11.9861 | Reject: LHand +0.0004 |
| H8 | 42.3041 | 26.1444 | 29.4725 | 39.8254 | 12.6574 | 12.0361 | Reject: all six regress |
| H12, 12° | 42.2999 | 26.1410 | 29.4668 | 39.8044 | 12.6450 | 11.9871 | Reject: RHand +0.0002 |
| H14 | 42.2990 | 26.1397 | 29.4660 | 39.8073 | 12.6457 | 11.9543 | Reject: UBody-H +0.0017 |
| H15-v1 | 42.2937 | 26.1344 | 29.4590 | 39.7905 | 12.6341 | 11.9266 | Metric pass, invariant fail |
| H15-v2 / EI-AMER | **42.2937** | **26.1344** | **29.4590** | **39.7905** | **12.6341** | **11.9266** | Exploratory metric/invariant pass |

This ledger is important for publication integrity. H7b, H8, H12, and H14 were not silently discarded; each falsifies a plausible mechanism. H15-v2 cannot use the same 45 signs as an unbiased confirmation set because its design was informed by H14’s result on those signs.

## 8. Interpretation

### 8.1 What actually produces the gain?

The largest reliable gain comes from H1, not from EI-AMER. H1 corrects finger articulation only when Sapiens 2D likelihood and WiLoR canonical 3D geometry agree. Its hand improvements are substantially larger than the changes in whole-body averages and survive the clean 45-sign confirmation.

EI-AMER addresses H1’s recall limitation. HaMeR is not used as a universal judge; it acts as a proposal generator only where H1 abstained. The Sapiens/HaMeR two-sigma rule filters unsupported proposals, while WiLoR vetoes strong conflicts. The centroid term protects regional alignment without changing the incumbent. The resulting improvement is concentrated on RHand, consistent with the added-rescue distribution, but per-sign uncertainty shows that the incremental gain is not yet conclusive for both hand regions.

### 8.2 Why not use SignHPoser and SignBPoser?

DexAvatar’s A3f/H1 input may already reflect the upstream influence of its learned priors, but the post-H1 EI-AMER stage does not invoke either SignHPoser or SignBPoser. SignBPoser is irrelevant because body pose is frozen. SignHPoser was tested as an H8 veto and rejected after all six external metrics regressed. Therefore, the proposed contribution is not a new use of DexAvatar’s priors; it is a frozen-expert refinement and selection method operating on the produced SMPL-X state.

### 8.3 What do the negative results teach?

1. Broad upper-body evidence can activate without improving the target region; availability is not usefulness.
2. More expert agreement is not always better. HaMeR as a hard veto suppresses useful WiLoR changes, whereas HaMeR as a reject-only proposal generator adds complementary coverage.
3. A sign-domain prior score is not automatically calibrated uncertainty. H8 improves development data but fails externally.
4. A larger trust region can improve development averages yet fail a strict external no-regression rule.
5. Identical aggregate counts and metrics do not imply identical decisions or artifacts. H15-v1 is the critical counterexample.

## 9. Is This a Publishable Contribution?

### 9.1 Verdict

**Yes, the method is conceptually appropriate for a 3D sign language reconstruction paper, but the current evidence supports a strong method/ablation paper draft—not yet a clean final submission claim for EI-AMER.**

The most defensible framing is:

> *Conservative inference-time hand refinement for monocular SMPL-X sign reconstruction, using canonical cross-model proposals, factorized evidence, and an exact-incumbent rescue contract.*

The weakest framing would be:

> *A new hand estimator that achieves state of the art by combining WiLoR and HaMeR.*

The latter is inaccurate because the component estimators are prior work, Tamaththul3D already integrates WiLoR into sign reconstruction, no broad state-of-the-art comparison has been completed, and the incremental EI-AMER result is exploratory.

### 9.2 Contribution-strength audit

| Dimension | Current assessment | Reason |
|---|---|---|
| Problem relevance | Strong | Hand articulation is linguistically central and remains a known reconstruction bottleneck |
| Technical novelty | Moderate | Individual components are known; the exact-incumbent/asymmetric/factorized combination is distinctive in the targeted literature |
| Methodological rigor | Strong for internal artifacts | Frozen H1, negative-result ledger, paired-sign bootstrap, evaluator parity, hashes, and invariant audit |
| Empirical effect of H1 | Strong on attached protocol | All six Full57 intervals versus C0 are negative; clean Untouched45 confirmation |
| Incremental EI-AMER effect | Modest and partly unresolved | Small deltas; only All and UBody-H CIs fully exclude zero versus H1 |
| External validity | Insufficient | One attached benchmark, protocol-count discrepancy, H15 is post-holdout |
| Sign-language validity | Insufficient for a strong application claim | No new comprehensibility, naturalness, recognition, or Deaf-user evaluation |
| Main-conference readiness | Not yet | Needs prospective confirmation, stronger comparisons, qualitative/failure analysis, and runtime |
| Workshop/short-paper readiness | Plausible | If framed transparently around conservative refinement and negative results |

### 9.3 Required experiments before a strong submission

| Priority | Experiment | Purpose | Minimum acceptable outcome |
|---:|---|---|---|
| 1 | Prospectively freeze H15-v2 and evaluate on a new external dataset/split | Remove post-holdout bias | Directionally consistent hand gains with paired uncertainty and zero invariant violations |
| 2 | Reconcile 1,493-frame protocol with prior reported counts | Ensure benchmark comparability | Document every inclusion/exclusion and reproduce an official common split |
| 3 | Direct comparison with current sign-reconstruction baselines, especially Tamaththul3D where feasible | Establish field-level value | Same data/evaluator or a carefully qualified non-identical-protocol comparison |
| 4 | Qualitative success/failure panels under blur, occlusion, crossing hands, and body contact | Show when rescue helps or fails | Predefined cases, not cherry-picked only successes |
| 5 | Runtime, memory, and coverage | Establish practical cost | Per-frame timings for observations, fitting, gating, and artifact materialization |
| 6 | Perceptual/intelligibility or sign-recognition evaluation | Connect millimetres to linguistic value | Blinded user study or frozen recognizer with statistical analysis |
| 7 | Sensitivity of two-sigma threshold and HaMeR centroid weight on development only | Show method stability | A broad plateau rather than a single tuned optimum |
| 8 | Cross-language/signer analysis | Test domain robustness | Per-signer and per-sign breakdown; ideally another sign language |

If only one new experiment can be funded, Priority 1 is mandatory. Without it, the manuscript must keep H1 as the confirmatory final method and present EI-AMER as an exploratory extension.

## 10. Limitations and Threats to Validity

1. **Post-holdout formulation.** EI-AMER was motivated after examining H14 on the 45-sign partition. Full57 includes those data and cannot repair the bias.
2. **Small incremental magnitude.** Improvements over H1 are below 0.1 mm in every aggregate and unresolved for four of six percentile intervals.
3. **Single attached protocol.** Generalization across languages, signer demographics, camera conditions, and continuous signing is unknown.
4. **Protocol discrepancy.** The 1,493 attached frames do not yet match larger counts reported elsewhere.
5. **Correlated experts.** Sapiens, WiLoR, and HaMeR may have overlapping training sources; their evidence is operationally factorized but not statistically independent.
6. **Framewise inference.** The method ignores temporal consistency, coarticulation, and motion dynamics. This avoids temporal leakage but can produce jitter.
7. **Frozen wrist and body.** EI-AMER cannot repair incorrect hand location, wrist orientation, arm pose, torso pose, or face. It should be presented as hand-articulation refinement for a full-body reconstruction, not complete reconstruction from scratch.
8. **Metric-to-language gap.** Translation-aligned vertex errors do not directly measure sign comprehensibility, naturalness, or semantic preservation.
9. **Safety terminology.** Exact rollback guarantees artifact identity only; it does not guarantee that the baseline is correct or culturally/linguistically appropriate.
10. **Recent preprint landscape.** Tamaththul3D appeared in 2026, and the rapidly changing literature may contain additional contemporaneous work not captured by the targeted search.

## 11. Reproducibility and Artifact Provenance

The principal artifacts are:

- Frozen H1 result card: `SignEFT-X/reports/final_result_card.md`
- EI-AMER result card: `SignEFT-X/reports/h15v2_exact_incumbent_result_card.md`
- Core ablation ledger: `SignEFT-X/reports/engineering12_core_ablation.json`
- Hand ablation ledger: `SignEFT-X/reports/engineering12_hand_ablation.json`
- Full EI-AMER run: `SignEFT-X/runs/signeft_h15v2_exact_incumbent_full57/`
- Full invariant audit: `SignEFT-X/runs/signeft_h15v2_exact_incumbent_full57/incumbent_rescue_audit.json`
- Paired bootstrap versus H1: `SignEFT-X/runs/signeft_h15v2_exact_incumbent_full57/audited_metrics/paired_bootstrap_vs_h1.json`
- Paired bootstrap versus C0: `SignEFT-X/runs/signeft_h15v2_exact_incumbent_full57/audited_metrics/paired_bootstrap_vs_c0.json`

The fitting code records implementation and observation hashes per decision. Official and audited evaluators agree after rounding. Before public release, the paper should additionally report hardware, wall-clock runtime, dependency versions, model licenses, and an executable command manifest.

## 12. Ethical and Accessibility Considerations

3D sign reconstruction can support accessible content creation, avatar communication, education, and data annotation, but geometric fidelity alone does not ensure linguistic or cultural fidelity. Evaluation and deployment should include Deaf signers and native sign-language users, with explicit consent and attention to identity, biometric data, signer representation, and the risks of incorrect signs. A reconstructed avatar should not be presented as an authoritative translation without linguistic validation.

## 13. Conclusion

SignEFT-X demonstrates that conservative hand refinement can improve an existing sign-specific SMPL-X reconstruction without changing body, wrist, shape, face, or camera. The frozen H1 contribution is empirically confirmed on the attached partitioning: canonical WiLoR finger proposals selected by independent Sapiens and WiLoR evidence improve all six metrics over A3f. EI-AMER extends this principle with an exact incumbent, asymmetric HaMeR rescue geometry, factorized accept/veto evidence, and artifact-level invariants. It obtains the best verified aggregates on the attached protocol, but its incremental benefit is small and its external evaluation is post-hoc. The method is therefore a credible paper contribution when framed around conservative multi-expert refinement and rigorous non-regression, provided that a new prospective evaluation is completed before making a final generalization or state-of-the-art claim.

## References

[1] Forte et al. [Reconstructing Signing Avatars From Video Using Linguistic Priors (SGNify)](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html). CVPR, 2023.

[2] Kundu et al. [DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html). WACV, 2026.

[3] Alghamdi et al. [Tamaththul3D: High-Fidelity 3D Saudi Sign Language Avatars from Monocular Video](https://arxiv.org/abs/2605.05367). arXiv, 2026.

[4] Baltatzis et al. [Neural Sign Actors: A Diffusion Model for 3D Sign Language Production from Text](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html). CVPR, 2024.

[5] Dong et al. [SignAvatar: Sign Language 3D Motion Reconstruction and Generation](https://arxiv.org/abs/2405.07974). FG, 2024.

[6] Potamias et al. [WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html). CVPR, 2025.

[7] Pavlakos et al. [Reconstructing Hands in 3D with Transformers (HaMeR)](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html). CVPR, 2024.

[8] Khirodkar et al. [Sapiens: Foundation for Human Vision Models](https://arxiv.org/abs/2408.12569). ECCV, 2024.

[9] Geifman and El-Yaniv. [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html). ICML, 2019.

[10] Mao, Mohri, and Zhong. [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html). ICML, 2024.

## Manuscript Disclosure Note

This dossier was organized and drafted with AI assistance from repository code, result cards, audit artifacts, and primary-source literature links. All numerical claims should be rechecked against the cited machine-readable artifacts before submission. The authors remain responsible for the scientific claims, source interpretation, benchmark permissions, and final manuscript text.
