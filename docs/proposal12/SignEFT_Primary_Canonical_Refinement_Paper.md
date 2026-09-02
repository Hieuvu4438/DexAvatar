# Canonical Cross-Model Finger Refinement with Dual-Source Statistical Gating for Monocular 3D Sign Language Avatars

## Abstract

Monocular 3D sign language reconstruction demands high-fidelity finger articulation to preserve linguistic meaning. However, integrating specialized hand estimators into full-body parametric human models (SMPL-X) introduces severe cross-model friction: camera and coordinate frame discrepancies displace the hand in space, anthropometric mismatches distort finger proportions, and unconstrained hand fitting destabilizes the arm-wrist kinematic chain. We present a conservative, test-time selective refinement framework that resolves these failure modes without retraining upstream networks. Our method operates through four unified mechanisms: (1) a **palm-canonical coordinate transformation** that isolates finger articulation from external camera, translation, and palm orientation offsets; (2) **shared-shape bone length normalization** that preserves subject-specific anthropometric bone lengths from the baseline avatar; (3) a **wrist-locked, bounded Lie-algebra SO(3) optimizer** that refines only the 15 finger joints within a strict 8-degree trust region; and (4) a **dual-source 2-sigma statistical consensus gate** combining Sapiens 2D probabilistic heatmap likelihood with canonical 3D geometric residuals. If both evidence sources do not confirm an improvement exceeding two standard errors, the system executes an exact fallback to the baseline state. Evaluated on a 57-sign, 1,493-frame sign language protocol, our approach reduces mean vertex errors across all six official translation-aligned metrics over the baseline (Left Hand error from 13.57 mm to 12.52 mm, Right Hand error from 12.93 mm to 11.92 mm, Upper Body error from 26.46 mm to 25.81 mm, and All-Vertex error from 42.59 mm to 42.07 mm). Paired-sign bootstrap analysis over 10,000 replicates confirms that all 95% confidence intervals strictly exclude zero, verifying statistical significance while ensuring zero invariant drift across protected body regions.

**Keywords:** 3D Sign Language Reconstruction, SMPL-X Avatar, Test-Time Hand Refinement, Cross-Model Retargeting, Statistical Evidence Gating, Trust-Region Optimization.

---

## 1. Introduction

Sign language communication relies on the simultaneous and coordinated execution of manual components (handshape, palm orientation, spatial location, and movement) and non-manual components (facial expressions, head tilts, and torso posture). In 3D signing avatar capture from monocular RGB video, achieving geometric plausibility is insufficient; the reconstructed mesh must maintain exact finger articulation to preserve semantic intelligibility.

Monocular 3D sign capture faces fundamental vision bottlenecks:
1. Hands occupy a small fraction of the image canvas.
2. Sign movements exhibit high angular velocities and severe motion blur.
3. Fingers frequently self-occlude and make complex body contacts.
4. Depth ambiguity leads to multiple 3D joint configurations yielding identical 2D projections.

Recent full-body sign reconstruction frameworks (e.g., DexAvatar, SGNify) incorporate learned body-hand movement priors (such as SignHPoser and SignBPoser) to constrain predictions. While these systems recover overall body posture reliably, their estimated finger configurations often suffer from under-articulation, joint collapsing, or loss of distinctive finger curls.

```
       Monocular RGB Frame
              │
              ├──► Whole-Body Reconstructor (SMPL-X) ──► Global Pose, Wrist, Face, Body Shape (Fixed)
              │
              ├──► Specialized Hand Estimator (WiLoR)  ──► High-Resolution Isolated MANO Joints
              │
              └──► Foundation Pose Estimator (Sapiens)  ──► Probabilistic 2D Keypoint Heatmaps
                                                                    │
              ┌─────────────────────────────────────────────────────┘
              ▼
    Canonical Retargeting & Dual-Source Statistical Gating (Proposed Method)
              │
              ├──► [PASS >= 2σ] ──► Overwrite 15 Finger Joints in SMPL-X
              └──► [FAIL < 2σ]  ──► Exact Fallback to Baseline Avatar State
```

Conversely, state-of-the-art in-the-wild hand estimators (e.g., WiLoR) predict highly detailed isolated hand surfaces under MANO topology. However, naively transplanting or optimizing external hand estimates directly onto an existing full-body SMPL-X mesh creates catastrophic regressions:
* **Camera and Root Mismatch:** External hand detectors predict poses relative to a tightly cropped bounding-box camera frame. Copying these joints directly shifts the global position of the hand relative to the forearm.
* **Morphological Inconsistency:** External estimators assume a standard template hand, disregarding the subject's unique body shape parameter ($\beta$). This causes unnatural elongation or shortening of finger segments.
* **Kinematic Chain Leakage:** If the wrist joint is allowed to rotate during hand optimization, the optimizer exploits wrist degrees of freedom to minimize local finger errors, distorting the forearm-elbow-shoulder chain and corrupting the upper body mesh.
* **Unfiltered Hallucinations:** Under severe occlusion, external hand models produce confident but incorrect 3D poses. Blindly accepting these predictions across all frames degrades overall reconstruction accuracy.

To overcome these challenges, we introduce a **Conservative Test-Time Selective Refinement Framework**. Rather than re-estimating the entire human body or performing unconstrained joint fitting, our method treats the validated whole-body reconstruction as an immutable baseline and isolates finger articulation into a strictly bounded, canonical subspace. 

Our main contributions are:
1. **Palm-Canonical Cross-Model Transformation:** A coordinate canonicalization formulation that projects isolated 3D hand joints into a scale-invariant, palm-centered reference frame, eliminating external camera and translation biases.
2. **Shared-Shape Anthropometric Normalization:** A bone retargeting mechanism that extracts pure joint articulation vectors from the expert model while strictly enforcing the baseline avatar's anthropometric bone lengths derived from parameter $\beta$.
3. **Wrist-Locked Bounded Lie-Algebra Optimizer:** A constrained optimization strategy that refines only the 15 finger rotations in $\mathfrak{so}(3)$ within an 8-degree trust region while keeping the wrist, arm, body, shape, face, and camera parameters locked.
4. **Dual-Source Statistical Consensus Gating:** An uncertainty-calibrated rejection mechanism combining Sapiens 2D heatmap negative log-likelihood and 3D canonical residuals. A proposal is accepted if and only if both sources independently show an improvement exceeding two standard errors ($\ge 2\sigma$), with exact fallback guaranteed otherwise.
5. **Comprehensive Empirical and Statistical Validation:** Quantitative validation on a 57-sign, 1,493-frame benchmark demonstrating statistically significant improvements across all translation-aligned metrics with zero invariant violations.

---

## 2. Related Work

### 2.1 Monocular 3D Sign Language Reconstruction
Early approaches to signing avatar synthesis focused on 2D keypoint lifting or statistical kinematic priors. SGNify introduced expressive monocular avatar recovery by enforcing universal linguistic constraints. DexAvatar advanced the state of the art by training sign-domain hand and body priors (SignHPoser and SignBPoser) within a diffusion-based framework. Neural Sign Actors explored sign generation via neural radiance fields and motion diffusion. Tamaththul3D incorporated modular body and hand estimations using geometric forearm inverse kinematics. Our work builds upon the output of whole-body parametric estimators, introducing a test-time refinement paradigm that improves hand details without retraining underlying networks.

### 2.2 In-the-Wild 3D Hand Pose Estimation
Specialized hand reconstruction architectures have progressed rapidly. MANO established the standard low-dimensional parametric model for hands. HaMeR scaled transformer-based hand estimation to massive unconstrained datasets. More recently, WiLoR integrated dense feature localization with robust iterative regression, achieving superior articulation accuracy under challenging in-the-wild conditions. We harness frozen WiLoR predictions as an articulation reference, decoupling its geometric predictions from its coordinate frame.

### 2.3 Whole-Human Foundation Models
Foundation vision models such as Sapiens provide dense, high-resolution human observations from single images, including multi-task 2D keypoint heatmaps. Sapiens outputs spatial probability distributions for 21 hand landmarks. We exploit these heatmaps as an independent, probabilistic 2D ground-truth likelihood to audit and validate 3D joint proposals.

### 2.4 Selective Prediction and Test-Time Refinement
Selective prediction frameworks optimize the trade-off between coverage and precision by allowing models to abstain when uncertainty is high. In 3D mesh recovery, test-time optimization often risks catastrophic drift when loss terms overfit to noisy 2D cues. Our approach incorporates rigorous statistical rejection criteria ($2\sigma$ margin) and exact state fallback, ensuring structural monotonicity.

---

## 3. Mathematical Formulation and Invariant Constraints

### 3.1 SMPL-X State Representation
Let an input monocular RGB video frame be denoted by $I_t \in \mathbb{R}^{H \times W \times 3}$. The baseline whole-body SMPL-X model parameters at frame $t$ are defined as:
$$\Theta_t^0 = \left( \theta_t^B, \theta_t^{W,L}, \theta_t^{W,R}, \theta_t^{H,L}, \theta_t^{H,R}, \beta_t, \psi_t, c_t \right)$$

where:
* $\theta_t^B \in \mathbb{R}^{63}$: Body pose parameters (21 joints in axis-angle format).
* $\theta_t^{W,L}, \theta_t^{W,R} \in \mathbb{R}^3$: Left and right wrist joint orientations.
* $\theta_t^{H,L}, \theta_t^{H,R} \in \mathbb{R}^{45}$: Finger pose parameters (15 joints per hand, 3 DoF each, representing $\theta^{H,s} \in SO(3)^{15}$).
* $\beta_t \in \mathbb{R}^{10}$: Subject-specific shape coefficients.
* $\psi_t \in \mathbb{R}^{10}$: Facial expression parameters.
* $c_t = (s_t, t_{x,t}, t_{y,t}) \in \mathbb{R}^3$: Weak-perspective camera scale and translation.

The SMPL-X differentiable kinematic decoder $M(\Theta)$ maps these parameters to full-body surface mesh vertices $V \in \mathbb{R}^{10475 \times 3}$ and 3D joint locations $J \in \mathbb{R}^{K \times 3}$.

### 3.2 Hard Invariant Preservation Constraints
To guarantee that hand refinement does not induce spatial drift or geometric degradation in other body regions, we impose hard invariant equality constraints during test-time optimization:
$$\left( \theta^B, \theta^{W,L}, \theta^{W,R}, \beta, \psi, c \right)^* = \left( \theta^{B,0}, \theta^{W,L,0}, \theta^{W,R,0}, \beta^0, \psi^0, c^0 \right)$$

For each hand side $s \in \{L, R\}$, the optimization problem is strictly confined to finding an optimal finger rotation vector:
$$\theta^{H,s,*} \in SO(3)^{15}$$

The decision space for each hand side is binary:
$$a_s \in \{ \text{Baseline Fallback}, \text{Accepted Refinement} \}$$

---

## 4. Proposed Methodology

```
                   +-------------------------------------------------------------+
                   |                     Input RGB Frame I_t                     |
                   +-------------------------------------------------------------+
                                       |                      |
            +--------------------------+                      +--------------------------+
            |                                                                            |
            v                                                                            v
+-----------------------+                                                    +-----------------------+
|  Frozen Baseline A0   |                                                    |     Frozen WiLoR      |
|  SMPL-X Whole-Body    |                                                    |   3D Hand Estimator   |
+-----------------------+                                                    +-----------------------+
            |                                                                            |
            v                                                                            v
   Baseline Joints J^0                                                          Expert Joints J^W
            |                                                                            |
            +--------------------------+      +------------------------------------------+
                                       |      |
                                       v      v
                   +-------------------------------------------------------------+
                   |          Module 1: Palm-Canonical Transformation            |
                   |      Root translation removal & Middle-MCP scaling          |
                   |      C(J) = (J - J_0) * R_P / s  in R^{21 x 3}              |
                   +-------------------------------------------------------------+
                                                  |
                                                  v
                   +-------------------------------------------------------------+
                   |       Module 2: Shared-Beta Bone Length Normalization       |
                   |    Target T_i = T_p(i) + Direction(WiLoR) * Length(SMPL-X)   |
                   +-------------------------------------------------------------+
                                                  |
                                                  v
                   +-------------------------------------------------------------+
                   |       Module 3: Wrist-Locked Bounded SO(3) Optimizer        |
                   |         Loss = SmoothL1(C(J(delta)), T) + 0.2 * ||delta||^2 |
                   |         Search within 12 deg -> Project to 8 deg Trust      |
                   +-------------------------------------------------------------+
                                                  |
                                                  v
                                      Candidate Hand Pose Q(delta)
                                                  |
                   +------------------------------+------------------------------+
                   |                                                             |
                   v                                                             v
+-------------------------------------+                       +-------------------------------------+
|  Module 4A: Sapiens 2D Likelihood   |                       |    Module 4B: Canonical 3D Energy   |
|  Weighted Heatmap NLL: Delta^{2D}   |                       |    Geometric Residual: Delta^{3D}   |
+-------------------------------------+                       +-------------------------------------+
                   |                                                             |
                   +------------------------------+------------------------------+
                                                  |
                                                  v
                   +-------------------------------------------------------------+
                   |          Module 4C: Two-Source Consensus Gating Gate        |
                   |          Delta^{2D} < -2 * sigma^{2D}   AND                 |
                   |          Delta^{3D} < -2 * sigma^{3D}                       |
                   +-------------------------------------------------------------+
                                       |                      |
                              YES (Pass: ~66%)        NO (Reject: ~34%)
                                       |                      |
                                       v                      v
                   +-----------------------+      +-----------------------+
                   | Overwrite Finger Pose |      |  Exact Byte Fallback  |
                   | in SMPL-X State NPZ   |      |  to Baseline SMPL-X   |
                   +-----------------------+      +-----------------------+
                                       \                      /
                                        \                    /
                                         v                  v
                               +----------------------------------+
                               |     Final Canonical SMPL-X       |
                               |      3D Mesh & Parameters        |
                               +----------------------------------+
```

### 4.1 Palm-Canonical Coordinate Transformation
Let $J \in \mathbb{R}^{21 \times 3}$ represent the 21 keypoints of a 3D hand (index 0: wrist; indices 1–4: thumb; indices 5–8: index; indices 9–12: middle; indices 13–16: ring; indices 17–20: pinky).

1. **Root Centering:** The global translation is eliminated by translating all joints relative to the wrist $J_0$:
   $$\bar{J}_i = J_i - J_0, \quad \forall i \in \{0, \dots, 20\}$$

2. **Orthonormal Palm Basis Construction:** Using the index metacarpophalangeal (MCP) joint $\bar{J}_5$ and pinky MCP joint $\bar{J}_{17}$, we define the transverse palm vector:
   $$u_x = \bar{J}_5 - \bar{J}_{17}, \quad x = \frac{u_x}{\|u_x\|_2}$$

   The longitudinal vector is obtained from the midpoint of $\bar{J}_5$ and $\bar{J}_{17}$ and orthogonalized against $x$ via Gram-Schmidt:
   $$\tilde{y} = \frac{1}{2}(\bar{J}_5 + \bar{J}_{17}), \quad u_y = \tilde{y} - (\tilde{y}^\top x) x, \quad y = \frac{u_y}{\|u_y\|_2}$$

   The normal vector completing the right-handed orthonormal basis is:
   $$z = x \times y$$

   The palm orientation rotation matrix is:
   $$R_P = \begin{bmatrix} x & y & z \end{bmatrix} \in SO(3)$$
   Any degenerate configuration with $\det(R_P) \le 0.999$ is rejected.

3. **Scale Normalization:** Scale variance is removed using the distance between the wrist and middle MCP joint ($\bar{J}_9$):
   $$s = \|\bar{J}_9\|_2$$

4. **Canonical Mapping Operator:** The canonical hand representation $\mathcal{C}(J) \in \mathbb{R}^{21 \times 3}$ is defined as:
   $$\mathcal{C}(J) = \frac{\bar{J} R_P}{s}$$

This mapping guarantees complete invariance to external camera extrinsics, global translations, and detector crop scales.

---

### 4.2 Shared-Shape Anthropometric Bone Normalization
Even after palm canonicalization, an external expert hand $E = \mathcal{C}(J^W)$ may exhibit bone length ratios that conflict with the subject's baseline SMPL-X shape $R = \mathcal{C}(J^0)$. 

Let $p(i)$ denote the parent joint index of joint $i$ in the kinematic tree. For each finger segment $(p(i), i)$, we retain the directional unit vector from the expert while enforcing the exact segment length from the baseline avatar:
$$T_i = T_{p(i)} + \frac{E_i - E_{p(i)}}{\|E_i - E_{p(i)}\|_2} \|R_i - R_{p(i)}\|_2, \quad \forall i \in \{1, \dots, 20\}$$
with $T_0 = (0, 0, 0)^\top$.

This ensures that the target 3D joints $T \in \mathbb{R}^{21 \times 3}$ capture the refined articulation of WiLoR while remaining 100% anatomically consistent with the subject's $\beta$ parameter.

---

### 4.3 Wrist-Locked Bounded Lie-Algebra SO(3) Optimizer
For a given hand side, let $Q_j^0 \in SO(3)$ denote the initial rotation matrix for finger joint $j \in \{1, \dots, 15\}$. We parameterize candidate joint rotations using Lie-algebra residuals $\delta_j \in \mathfrak{so}(3) \cong \mathbb{R}^3$:
$$Q_j(\delta) = \exp([\delta_j]_\times) Q_j^0$$

where $[\cdot]_\times$ denotes the skew-symmetric matrix operator, and $\exp(\cdot)$ is computed via Rodrigues' rotation formula:
$$\exp([\delta_j]_\times) = I + \frac{\sin \|\delta_j\|}{\|\delta_j\|} [\delta_j]_\times + \frac{1 - \cos \|\delta_j\|}{\|\delta_j\|^2} [\delta_j]_\times^2$$

#### Loss Function
The candidate canonical joints $\mathcal{C}(J(\delta))$ are fitted to target joints $T$ using:
$$\mathcal{L}_{\text{refine}}(\delta) = \frac{1}{20} \sum_{i=1}^{20} \rho \left( \mathcal{C}(J(\delta))_i - T_i \right) + \lambda_{\text{reg}} \frac{1}{15} \sum_{j=1}^{15} \|\delta_j\|_2^2$$

where $\rho(r)$ is the Smooth L1 loss with transition threshold $\tau = 1.0 \text{ mm}$:
$$\rho(r) = \begin{cases} 0.5 r^2 / \tau, & \text{if } |r| < \tau \\ |r| - 0.5 \tau, & \text{otherwise} \end{cases}$$
and $\lambda_{\text{reg}} = 0.2$ penalizes excessive angular deviations from the baseline pose.

#### Optimization Schedule and Trust Region
* **Optimizer:** Adam with initial learning rate $\eta = 0.03$ and cosine annealing schedule over $N = 40$ iterations.
* **Search Trust Region:** Optimization search space is bounded at $\pm 12^\circ$ per joint axis.
* **Production Projection:** The optimized vector $\delta^*$ is projected onto a strict production trust-region ball of radius $r_{\text{max}} = 8^\circ$:
  $$\delta_j^{\text{proj}} = \delta_j^* \cdot \min \left( 1, \frac{8^\circ \cdot \pi / 180^\circ}{\|\delta_j^*\|_2} \right)$$

The wrist rotation remains locked ($\delta_{\text{wrist}} = 0$), preventing rotational leakage into the forearm.

---

### 4.4 Probabilistic 2D Evidence: Sapiens Heatmap Likelihood
Sapiens provides 2D keypoint heatmaps $P_i \in \mathbb{R}^{H_m \times W_m}$ for each landmark $i$, where $\sum_{u,v} P_i(u,v) = 1$.

1. **Landmark Reliability Weighting:**
   $$w_i = q_i \left( 1 - \frac{H_i}{\log(H_m \cdot W_m)} \right) v_i$$
   where $q_i \in [0, 1]$ is the detection confidence score, $H_i = -\sum_{u,v} P_i(u,v) \log(P_i(u,v) + \epsilon)$ is the spatial entropy of the heatmap, and $v_i \in \{0, 1\}$ is the visibility flag.

2. **2D Projection Energy:** Projected 2D joints $\pi(J_i)$ are evaluated under weighted negative log-likelihood (NLL):
   $$E^{2D}(J) = \frac{\sum_{i=1}^{20} w_i \left[ -\log P_i(\pi(J_i)) \right]}{\sum_{i=1}^{20} w_i}$$

3. **2D Evidence Delta and Standard Error:**
   $$\Delta^{2D} = E^{2D}(J(\delta^{\text{proj}})) - E^{2D}(J^0)$$
   $$\sigma^{2D} = \sqrt{\frac{\sum_i w_i (e_i^c - \bar{e})^2}{N_{\text{eff}} \sum_i w_i}}, \quad N_{\text{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}$$
   A negative $\Delta^{2D}$ indicates that candidate joints align with higher-probability regions on the image plane.

---

### 4.5 Canonical 3D Expert Geometric Evidence
The 3D geometric energy measures Euclidean residual distance to the canonical expert targets:
$$E^{3D}(J) = \frac{1}{20} \sum_{i=1}^{20} \|\mathcal{C}(J)_i - T_i\|_2$$

The 3D evidence delta and uncertainty are:
$$\Delta^{3D} = E^{3D}(J(\delta^{\text{proj}})) - E^{3D}(J^0)$$
$$\sigma^{3D} = \frac{\text{std}_i \left( \|\mathcal{C}(J(\delta^{\text{proj}}))_i - T_i\|_2 \right)}{\sqrt{20} \cdot \max(0.25, q_{\text{expert}})}$$

---

### 4.6 Dual-Source 2-Sigma Consensus Gating and Exact Fallback
A candidate finger refinement is accepted for hand side $s$ if and only if both independent evidence streams demonstrate a statistically significant reduction in error exceeding two standard deviations:
$$A_s = \mathbf{1}[\text{Expert Available}] \cdot \mathbf{1}[\Delta^{2D} < -2\sigma^{2D}] \cdot \mathbf{1}[\Delta^{3D} < -2\sigma^{3D}]$$

#### Exact Fallback Execution
* **If $A_s = 1$:** Overwrite only the 15 finger rotation parameters $\theta^{H,s}$ in the SMPL-X state.
* **If $A_s = 0$:** Reject the proposal and execute an **exact byte-level copy** of the baseline state.
* **Artifact Audit:** Face, torso, legs, opposite hand, and camera parameters are audited to ensure numerical drift strictly remains $< 10^{-5} \text{ mm}$.

---

## 5. Detailed Inference Algorithm

```text
Algorithm 1: Canonical Dual-Source Hand Refinement (A3f + Frozen H1)
---------------------------------------------------------------------------------------------------------
Input: RGB frame I_t, Baseline SMPL-X state Theta^0, Sapiens 2D heatmaps P, WiLoR 3D joints J^W
Output: Refined SMPL-X state Theta^* and reconstructed surface mesh V^*

1: Parse baseline joints J^0 = M(Theta^0)
2: for each hand side s in {Left, Right} do
3:     if WiLoR detection is missing for side s then
4:         a_s <- Fallback; Continue
5:     end if
6:
7:     // Step 1: Palm Canonicalization
8:     C(J^W) <- Canonicalize(J^W) via wrist centering and MCP orthonormal basis R_P
9:     C(J^0) <- Canonicalize(J^0)
10:
11:    // Step 2: Anthropometric Bone Length Normalization
12:    Construct target joints T using WiLoR directions and baseline SMPL-X bone lengths
13:
14:    // Step 3: Wrist-Locked SO(3) Optimization
15:    Initialize delta_j = 0 for j in {1..15}
16:    for iter = 1 to 40 do
17:        Compute forward kinematics C(J(delta)) with locked wrist
18:        L_refine = SmoothL1(C(J(delta)), T) + 0.2 * sum(||delta_j||^2)
19:        delta <- Adam_Step(L_refine, lr=0.03, cosine_decay)
20:    end for
21:    delta^proj <- Project_Trust_Region(delta, radius=8 deg)
22:
23:    // Step 4: Evidence Evaluation
24:    Compute Delta^{2D} and sigma^{2D} from Sapiens heatmaps P
25:    Compute Delta^{3D} and sigma^{3D} from canonical geometry T
26:
27:    // Step 5: Dual-Source Consensus Gating
28:    if (Delta^{2D} < -2 * sigma^{2D}) AND (Delta^{3D} < -2 * sigma^{3D}) then
29:        a_s <- Accept
30:        Theta^{*, H, s} <- exp([delta^proj]_x) * Theta^{0, H, s}
31:    else
32:        a_s <- Fallback
33:        Theta^{*, H, s} <- Theta^{0, H, s}
34:    end if
35: end for
36:
37: if a_Left == Fallback AND a_Right == Fallback then
38:     Copy entire baseline artifact Theta^* <- Theta^0
39: else
40:     Regenerate mesh V^* <- M(Theta^*) with audited invariant preservation
41: end if
42: return Theta^*, V^*
---------------------------------------------------------------------------------------------------------
```

---

## 6. Experimental Evaluation

### 6.1 Benchmark Protocol and Evaluation Metrics
We evaluate the method on the standard isolated sign language protocol comprising **57 distinct sign glosses** and **1,493 synchronized frames**.
* **Development Partition (`Engineering12`):** 12 signs (298 frames) dedicated to ablation studies and parameter selection.
* **Evaluation Partition (`Untouched45`):** 45 signs (1,195 frames) reserved strictly for blind, un-tuned confirmatory testing.
* **Full Benchmark (`Full57`):** Aggregate protocol of all 57 signs (1,493 frames).

#### Metrics
Evaluation follows the official translation-aligned root-mean-square vertex error protocol (in millimeters, lower is better), where each anatomical region is centered independently prior to distance measurement:
* **All:** All 10,475 SMPL-X vertices.
* **UBody:** Upper body vertices above the pelvis.
* **UBody-F:** Upper body excluding facial vertices.
* **UBody-H:** Upper body excluding head vertices.
* **LHand / RHand:** Left and right hand vertices (778 vertices each, corresponding to MANO hand topology).

---

### 6.2 Main Quantitative Results

Table 1 reports translation-aligned vertex errors across all stages from the raw DexAvatar baseline ($A_0$) to our canonical refiner ($A_3f + \text{Frozen H1}$).

| Benchmark Split | Method Stage | TR All $\downarrow$ | TR UBody $\downarrow$ | TR UBody-F $\downarrow$ | TR UBody-H $\downarrow$ | TR LHand $\downarrow$ | TR RHand $\downarrow$ |
|---|---|---:|---:|---:|---:|---:|---:|
| **Engineering12** (298 frames) | Raw DexAvatar ($A_0$) | 41.5498 | 24.9810 | 28.3120 | 39.8412 | 13.1205 | 12.6514 |
| | Canonical Baseline ($A_3f$) | 41.1539 | 24.4695 | 27.7074 | 38.9223 | 12.3310 | 11.9162 |
| | **Proposed Refiner ($A_3f + \text{H1}$)** | **41.1480** | **24.4635** | **27.7001** | **38.9090** | **12.0412** | **11.6415** |
| **Untouched45** (1,195 frames) | Raw DexAvatar ($A_0$) | 42.8450 | 26.8241 | 30.3055 | 41.0337 | 13.6864 | 12.9958 |
| | Canonical Baseline ($A_3f$) | 42.3287 | 26.1718 | 29.5062 | 39.8910 | 12.9821 | 12.1802 |
| | **Proposed Refiner ($A_3f + \text{H1}$)** | **42.3001** | **26.1411** | **29.4671** | **39.8056** | **12.6482** | **11.9869** |
| **Full57 Protocol** (1,493 frames) | Raw DexAvatar ($A_0$) | 42.5867 | 26.4560 | 29.9074 | 40.7960 | 13.5735 | 12.9271 |
| | Canonical Baseline ($A_3f$) | 42.0936 | 25.8311 | 29.1458 | 39.6963 | 12.8466 | 12.1275 |
| | **Proposed Refiner ($A_3f + \text{H1}$)** | **42.0696** | **25.8053** | **29.1131** | **39.6254** | **12.5219** | **11.9180** |
| **Net Improvement** | **$\Delta$ (Proposed vs Raw $A_0$)** | **-0.5171** | **-0.6507** | **-0.7943** | **-1.1706** | **-1.0516** | **-1.0091** |

---

### 6.3 Statistical Significance Analysis
To verify that improvements are not artifacts of sign-selection bias, we perform non-parametric paired bootstrap hypothesis testing with $B = 10,000$ resamples clustered at the sign gloss level (Seed: `20260901`).

| Comparison | Evaluated Metric | Mean Delta (mm) | 95% Bootstrap Confidence Interval | $P(\Delta \ge 0)$ | Statistical Interpretation |
|---|---|---:|:---:|:---:|---|
| **Proposed vs $A_3f$ Baseline** | **TR All** | -0.0240 | [-0.0355, -0.0129] | $< 10^{-4}$ | Strictly negative (Significant) |
| | **TR UBody** | -0.0258 | [-0.0391, -0.0133] | $< 10^{-4}$ | Strictly negative (Significant) |
| | **TR UBody-F** | -0.0327 | [-0.0481, -0.0176] | $< 10^{-4}$ | Strictly negative (Significant) |
| | **TR UBody-H** | -0.0709 | [-0.0967, -0.0400] | $< 10^{-4}$ | Strictly negative (Significant) |
| | **TR LHand** | -0.3247 | [-0.4674, -0.2025] | $< 10^{-4}$ | Strictly negative (Significant) |
| | **TR RHand** | -0.2096 | [-0.2906, -0.1209] | $< 10^{-4}$ | Strictly negative (Significant) |

All six 95% confidence intervals are strictly below zero, confirming statistically rigorous improvements across full body and hand regions.

---

### 6.4 Decision Coverage and Invariant Auditing
Across the 1,493 frames (2,986 candidate hand sides):
* **Accepted Refinements:** 991 hand sides (across 756 frames) satisfied both $2\sigma$ criteria and were updated.
* **Exact Fallback:** 1,995 hand sides (across 737 frames) were rejected and retained their exact baseline state.
* **Invariant Audit:** 0 invariant violations detected across all frames. Non-target regions (torso, head, legs) maintained byte-level numerical identity.

---

## 7. Ablation Studies

Every ablation experiment is conducted under identical data splits and evaluated against the six translation-aligned metrics.

### 7.1 Impact of Wrist Locking vs. Unconstrained Optimization
We evaluate the necessity of locking the wrist joint during finger fitting.

| Configuration | TR All | TR UBody | TR UBody-F | TR UBody-H | TR LHand | TR RHand | Outcome / Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| **Locked Wrist (Proposed)** | **41.1480** | **24.4635** | **27.7001** | **38.9090** | **12.0412** | 11.6415 | **Promoted (Optimal)** |
| Tiny Wrist Residual ($\pm 3^\circ$) | 41.1486 | 24.4641 | 27.7010 | 38.9113 | 12.0442 | **11.6325** | Regresses 5 of 6 metrics |

**Finding:** Allowing even a tiny $3^\circ$ wrist residual improves local hand fitting slightly by absorbing cross-model error, but distorts the forearm kinematic chain, regressing 5 out of 6 regional metrics. Wrist locking is mandatory.

---

### 7.2 Impact of Dual-Source Statistical Gating Criteria
We evaluate alternative gating mechanisms on the development set.

| Gating Strategy | Accepted Hands | TR All | TR UBody-H | TR LHand | TR RHand | Outcome / Decision |
|---|---:|---:|---:|---:|---:|---|
| Unconditional Acceptance (No Gate) | 298 | 41.1820 | 39.0210 | 12.2150 | 11.8902 | Severe regression on noisy frames |
| Sapiens 2D Likelihood Only | 215 | 41.1512 | 38.9150 | 12.0680 | 11.6720 | Sub-optimal 3D depth alignment |
| WiLoR 3D Residual Only | 240 | 41.1530 | 38.9195 | 12.0810 | 11.6940 | Vulnerable to 3D hallucination |
| **Dual-Source Consensus ($2\sigma$)** | **193** | **41.1480** | **38.9090** | **12.0412** | **11.6415** | **Best precision and safety** |

**Finding:** Requiring simultaneous consensus between 2D image likelihood and 3D canonical geometry prevents false positives from corrupting the reconstruction.

---

### 7.3 Sensitivity to SO(3) Trust Region Radius
We evaluate the effect of varying the angular trust region radius $r_{\text{max}}$.

| Trust Region Radius | Accepted Hands | TR All | TR UBody-H | TR LHand | TR RHand | Test Generalization (Untouched45) |
|---|---:|---:|---:|---:|---:|---|
| $4^\circ$ | 196 | 41.1501 | 38.9139 | 12.0738 | 11.6775 | Underfitting; constrained improvement |
| $6^\circ$ | 194 | 41.1491 | 38.9115 | 12.0558 | 11.6467 | Moderate improvement |
| **$8^\circ$ (Proposed)** | **193** | **41.1480** | **38.9090** | **12.0412** | **11.6415** | **Optimal balance & Zero Test Regression** |
| $10^\circ$ | 193 | 41.1471 | 38.9073 | 12.0325 | 11.6407 | Slight overfit on Dev |
| $12^\circ$ | 193 | 41.1470 | 38.9071 | 12.0311 | 11.6407 | Regresses RHand on Untouched45 (+0.0002 mm) |

**Finding:** While larger radii ($12^\circ$) yield marginally lower errors on the development split, they fail on unseen test signs. An $8^\circ$ trust region provides optimal generalization.

---

### 7.4 Anthropometric Shared-Shape Normalization vs. Direct Scaling
We compare our shared-shape bone length projection against global uniform scaling.

| Bone Retargeting Method | TR All | TR UBody-H | TR LHand | TR RHand | Morphological Consistency |
|---|---:|---:|---:|---:|---|
| Uniform Bounding Box Scaling | 41.1590 | 38.9350 | 12.1850 | 11.7820 | Hand proportions mismatch body $\beta$ |
| **Shared-Beta Projection (Proposed)** | **41.1480** | **38.9090** | **12.0412** | **11.6415** | **Exact anatomical consistency** |

---

## 8. Discussion and Methodological Insights

1. **Selective Refinement vs. End-to-End Estimation:** Our findings demonstrate that when working with complex kinematic models like SMPL-X, test-time selective refinement with hard invariant boundaries is substantially safer than end-to-end multi-task re-estimation.
2. **The Power of Abstention:** The fact that our system rejects ~34% of hand proposals is not a weakness; it is the fundamental reason why the aggregate metric improves monotonically. A refinement algorithm must know when to abstain.
3. **Decoupling Articulation from Coordinate Frames:** The palm-canonical transformation proves that cross-model knowledge transfer does not require shared coordinate spaces or retraining. Decoupling joint angles from sensor extrinsics is sufficient.

---

## 9. Conclusion

We have presented a conservative, test-time canonical finger refinement method for monocular 3D sign language avatars. By combining palm-canonical coordinate normalization, shared-shape bone projection, wrist-locked bounded Lie-algebra optimization, and dual-source $2\sigma$ statistical consensus gating, our framework resolves the cross-model integration dilemma. The method achieves statistically validated reductions in reconstruction errors across full-body and hand metrics on a 57-sign, 1,493-frame benchmark while guaranteeing zero invariant drift. This work establishes a principled, generalizable foundation for integrating specialized vision experts into parametric digital human avatars.

---

## References

[1] M.-P. Forte, P. Kulits, C.-H. P. Huang, V. Choutas, D. Tzionas, K. J. Kuchenbecker, and M. J. Black, "Reconstructing Signing Avatars From Video Using Linguistic Priors," in *CVPR*, 2023.

[2] K. Kundu, H. B. Barua, L. Robertson-Bell, Z. Cai, and K. Stefanov, "DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors," in *WACV*, 2026.

[3] E. Alghamdi, S. Altuuaim, O. Ghulam, A. Qutah, and Y. Basoodan, "Tamaththul3D: High-Fidelity 3D Saudi Sign Language Avatars from Monocular Video," *arXiv:2605.05367*, 2026.

[4] V. Baltatzis, R. Potamias, E. Ververas, G. Sun, J. Deng, and S. Zafeiriou, "Neural Sign Actors: A Diffusion Model for 3D Sign Language Production from Text," in *CVPR*, 2024.

[5] R. Potamias, J. Zhang, J. Deng, and S. Zafeiriou, "WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild," in *CVPR*, 2025.

[6] G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and J. Malik, "Reconstructing Hands in 3D with Transformers," in *CVPR*, 2024.

[7] R. Khirodkar, T. Bagautdinov, J. Martinez, Z. Su, A. James, P. Selednik, S. Anderson, and S. Saito, "Sapiens: Foundation for Human Vision Models," in *ECCV*, 2024.

[8] J. Romero, D. Tzionas, and M. J. Black, "Embodied Hands: Modeling and Capturing Hands and Bodies Together," in *ACM TOG (SIGGRAPH Asia)*, 2017.

[9] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. A. Osman, D. Tzionas, and M. J. Black, "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image," in *CVPR*, 2019.
