# SIGNAL-4D V7: literature review and material-gain integration plan

**Review date:** 2026-08-21

**Priority:** UBody-F → UBody → LHand/RHand

**Constraint:** no novelty claim based only on swapping SMPLer-X for a larger expert

**Frozen reference:** SIGNAL-4D V6 remains unchanged

## 1. Research question and answer

The question is not “which new expert replaces SMPLer-X?” It is:

> How can SIGNAL-4D combine heterogeneous body, dense-localization and hand
> evidence over time so that the final SMPL-X sequence improves materially,
> remains kinematically coherent, and can abstain from unreliable experts?

The literature and the completed NLF audit support this answer:

1. Do not replace SMPLer-X/V6 wholesale with NLF.
2. Use NLF's dense non-parametric points and calibrated uncertainty as a new
   upper-body observation factor.
3. Use a clean-room SOKE-style arm reprojection stage, but preserve **global**
   wrist orientation, not merely local wrist pose.
4. Activate an interacting-hand motion/contact prior only in bimanual
   occlusion/contact windows; do not impose it on every sign.
5. Make every factor uncertainty- and residual-gated, with multi-scale temporal
   support and a coherent SMPL-X forward pass.

This is a method contribution about uncertainty-conditioned factor fusion and
kinematic boundary preservation, not an expert replacement.

## 2. Review method

The search covered official NeurIPS, CVF/CVPR/ICCV, ECVA/ECCV and SIGGRAPH Asia
proceedings, project pages and official repositories. Searches combined:

- 3D sign language reconstruction, signing avatar fitting, SGNify;
- monocular human mesh recovery, whole-body SMPL-X fitting and pose priors;
- 4D/temporal human mesh recovery and uncertainty-aware fitting;
- hand reconstruction, interacting hands, occlusion, biomechanical constraints;
- public source code, checkpoints and license terms.

Inclusion required a reputable archival venue or a directly relevant official
implementation. Preference was given to code plus checkpoint availability.
Claims below are based on primary papers/repositories, not survey summaries.
Repository commits and licenses checked on the review date are locked in
`signal4d_v7_nlf_fusion/artifacts/sources.lock.json`.

## 3. Evidence matrix

| Work | Relevant mechanism | Public implementation | Fit for SIGNAL-4D | Decision |
|---|---|---|---|---|
| [SGNify, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html) | Linguistic hand priors for sign-specific ambiguity | Research code/data page | Defines target domain and official evaluation | Retain protocol and sign-specific priors |
| [NLF, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd23a1f3bc89e042d70960b466dc20e8-Abstract-Conference.html) | Arbitrary canonical point localization; 2D/3D dense queries; uncertainty-weighted SMPL-X fitting | [MIT code; noncommercial models](https://github.com/isarandi/nlf) | Strong complementary UBody signal, weak direct hands | Integrate as uncertainty-bearing factor, never wholesale hand expert |
| [Signs as Tokens/SOKE, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html) | WiLoR hand pose/global orientation substitution; 2D shoulder/arm reprojection; mesh/joint temporal loss | [CC BY-NC-ND repo](https://github.com/2000ZRL/SOKE); fitting code is not present | Direct same-dataset evidence: 10.55/8.94 mm hand MPVPE in supplement | Clean-room equations only; no code copying/modification |
| [WiLoR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html) | Detection plus iterative high-fidelity MANO refinement | [Official code/checkpoints](https://github.com/rolpotamias/WiLoR) | Already the strongest hand expert in current pipeline | Retain expert; improve temporal/kinematic integration |
| [Dyn-HaMR, CVPR 2025 Highlight](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html) | Three-stage temporal optimization, interacting motion prior, biomechanical and penetration constraints | [MIT code/checkpoints](https://github.com/ZhengdiYu/Dyn-HaMR) | Useful on bimanual contact/occlusion, excessive for all static-camera frames | Adapt only prior/residuals on detected interaction windows |
| [InterWild, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Bringing_Inputs_to_Shared_Domains_for_3D_Interacting_Hands_Recovery_CVPR_2023_paper.html) | Separate per-hand mesh and relative translation estimation | [CC BY-NC; archived](https://github.com/facebookresearch/InterWild) | Useful fallback for relative two-hand geometry | Optional black-box observation, lower priority than Dyn-HaMR prior |
| [MS-MANO, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_MS-MANO_Enabling_Hand_Pose_Tracking_with_Biomechanical_Constraints_CVPR_2024_paper.pdf) | Musculoskeletal MANO and simulation-in-the-loop pose refinement | Paper/research assets require audit | Strong plausibility idea but high integration cost | Later ablation; not first implementation |
| [HaWoR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_HaWoR_World-Space_Hand_Motion_Reconstruction_from_Egocentric_Videos_CVPR_2025_paper.html) | Egocentric SLAM and hand-motion infilling | [CC BY-NC-ND code](https://github.com/ThunderVVV/HaWoR) | Camera assumptions mismatch static SGNify; infiller may help missing hands | Do not integrate now |
| [DPoser-X, ICCV 2025 Oral](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html) | Whole-body/part diffusion prior and truncated timestep schedule | [MIT code/checkpoints](https://github.com/moonbow721/DPoser-X) | Already evaluated in V6 with only a small gain | Keep V6 component; not a new V7 contribution |
| [ScoreHMR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Stathopoulos_Score-Guided_Diffusion_for_3D_Human_Recovery_CVPR_2024_paper.html) | Task observation guidance during diffusion denoising | Public research code/models | Supports factor-guided refinement rather than blind prior replacement | Borrow observation-guidance formulation, not another full expert |
| [TokenHMR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Dwivedi_TokenHMR_Advancing_Human_Mesh_Recovery_with_a_Tokenized_Pose_Representation_CVPR_2024_paper.html) | Threshold-Adaptive Loss Scaling avoids forcing small/noisy 2D and pseudo-GT residuals | Research code/models | Directly relevant to noisy sign 2D keypoints | Adapt thresholding to uncertainty-calibrated residual factors |
| [GVHMR, SIGGRAPH Asia 2024](https://github.com/zju3dv/GVHMR) | Gravity-view coordinates and temporal global body recovery | Public noncommercial research code/checkpoint | Good torso/motion auxiliary; lacks expressive hand detail | Optional torso observation after NLF factor ablation |
| [SignAvatars, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00653.pdf) | 70k videos, 153 signers and 8.34M holistic 3D frames | Dataset/project access under its terms | Best source for external sign-domain router/prior training | Primary external training/calibration corpus |

### Direct same-domain evidence from SOKE

The [SOKE supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zuo_Signs_as_Tokens_ICCV_2025_supplemental.pdf)
describes this curation pipeline:

1. OSX initializes body pose.
2. WiLoR hand pose **and global orientation** replace OSX hand parameters.
3. MediaPipe shoulder/arm 2D joints drive L1 reprojection refinement.
4. Mesh and joint differences between adjacent frames enforce temporal
   coherence.
5. L2 pose regularization rejects irregular upper-body pose.

On the 57-sign SGNify mocap set, its Table S1 reports body/LH/RH mean per-vertex
errors of 46.73/10.55/8.94 mm; NSA reports 46.42/16.17/15.23. This is the
strongest direct evidence found for a large hand gain in this domain.

Those values are **not directly comparable** to our author TR-V2V values:
SOKE labels them MPVPE and does not document the same region-centering,
one-hand exclusion and vertex-micro wrapper. They justify the mechanism, not a
numerical SOTA comparison.

The official SOKE repository contains generation/tokenization code and cites
OSX/WiLoR, but the supplement's fitting pipeline is not present. Its license is
CC BY-NC-ND 4.0. Therefore SIGNAL-4D must implement the published equations
independently and must not copy or modify SOKE source.

## 4. Proposed method: SIGNAL-4D Field-Guided Kinematic Fusion

Working name: **SIGNAL-4D FGKF**. The name is provisional and must not be used
as a final novelty claim until the confirmatory gates pass.

### 4.1 State and observations

For frame `t`, optimize one coherent SMPL-X state:

```math
z_t = (\theta_t^{body},\theta_t^{lh},\theta_t^{rh},\theta_t^{face},
       \beta,\psi_t,c_t).
```

Observations are not interchangeable predictions:

- frozen V6 pose, risk and uncertainty;
- NLF non-parametric 3D/2D upper-body points `q_{t,i}` and uncertainty
  `σ_{t,i}`;
- WiLoR MANO pose, global wrist orientation, 2D landmarks and confidence;
- existing body/arm 2D keypoints and confidence;
- optional Dyn-HaMR interaction-prior score only in selected windows.

The identity `β` is signer-level and remains frozen from DexAvatar/V6 unless a
separate identity calibration experiment proves a gain. NLF's per-frame betas
must not leak into the final sequence.

### 4.2 Uncertainty-tempered NLF field factor — primary UBody contribution

Select canonical NLF queries on torso, clavicle, shoulder, upper/lower arm and
elbow surfaces; exclude face, hands and lower body. Fit them after a robust
camera-space alignment:

```math
E_{field}=\sum_{t,i}g_{t,i}\,
\rho\!\left(\frac{\|A_tV_i(z_t)-q_{t,i}\|_2}{\max(\sigma_{t,i},\sigma_{min})}\right).
```

`A_t` is a robust torso alignment whose scale is fixed or tightly regularized;
otherwise it can absorb pose error. `ρ` is Geman–McClure or Charbonnier. Start
from NLF's official `σ^-1.5` weighting as an ablation, then calibrate exponent
only on training/calibration.

The gate combines uncertainty, NLF–V6 disagreement, temporal innovation and
visibility. It is continuous at optimization time; hard routing is retained as
a reproducible ablation.

### 4.3 Threshold-adaptive reprojection — SOKE × TokenHMR

Use shoulder, elbow and wrist 2D evidence, but do not minimize noisy residuals
indefinitely. Let `r` be normalized reprojection error and `τ(c,σ)` an
uncertainty/confidence threshold:

```math
w(r)=\begin{cases}
1,&r>\tau(c,\sigma),\\
\epsilon,&r\le\tau(c,\sigma).
\end{cases}
```

This adapts TokenHMR's TALS principle to multi-expert sign fitting. Gross
misalignment receives full correction; already plausible joints receive only
a small force, avoiding the 2D-overfitting/3D-degradation failure mode.

### 4.4 Kinematic boundary condition — preserve global wrists

Body refinement and hand refinement meet at each wrist. Preserving a local
wrist angle is incorrect after shoulder/elbow changes. The boundary constraint
is defined in global rotations:

```math
E_{wristR}=\sum_{t,h}\omega_{t,h}
d_{SO(3)}(R^{G}_{w,t}(z_t),\widehat R^{G,WiLoR}_{w,t})^2.
```

For high-confidence WiLoR frames, solve the local wrist exactly from its new
parent global rotation. For uncertain/occluded frames, relax the equality and
let temporal/interaction priors contribute. This module already reduced the
alpha-1 hand regression from +1.499/+0.892 mm to +0.011/−0.068 mm.

### 4.5 Multi-scale temporal factor

Signing has different frequency bands: torso is slow, arms intermediate and
fingers fast. A single smoothness coefficient oversmooths fingers or
under-regularizes torso. Use region-specific scales:

```math
E_{temp}=\sum_{r\in\{torso,arm,hand\}}\sum_{k\in\{1,2,4\}}
\lambda_{r,k}\,g^r_{t,k}\,
d_{SO(3)}(R^r_t,R^r_{t-k})^2.
```

The temporal gate decreases at true motion boundaries and increases under
blur/occlusion. Boundary evidence comes from robust 2D velocity, expert
disagreement and uncertainty; never from evaluation GT.

### 4.6 Selective interacting-hand factor — hand contribution

Run a bimanual event detector using only observable quantities:

- overlap/IoU of expanded hand boxes;
- 2D keypoint distance and crossing;
- current 3D inter-hand distance;
- hand uncertainty, missing detection and temporal handedness flips.

Only selected windows invoke the Dyn-HaMR-style interacting motion prior,
relative-hand translation factor and penetration/contact penalties:

```math
E_{hands}=g^{interact}(E_{motion-prior}+E_{relative}+E_{contact}+E_{penetration}).
```

For non-interacting, high-confidence frames, keep WiLoR/V6 output. This
prevents an interaction prior from distorting independent two-hand signs.

The first implementation should consume Dyn-HaMR's prior as an isolated
black-box score/checkpoint with a pinned MIT commit. Do not import its SLAM or
dynamic-camera stage into static SGNify.

### 4.7 Unified objective

```math
E = \lambda_{field}E_{field}
  + \lambda_{2D}E_{2D-TALS}
  + \lambda_{wrist}E_{wristR}
  + \lambda_{temp}E_{temp}
  + \lambda_{hand}E_{hands}
  + \lambda_{prior}E_{DPoser-X}
  + \lambda_{anchor}E_{V6}.
```

Optimization is on SO(3), uses one SMPL-X forward pass for the final mesh, and
has explicit trust regions around V6. No vertex splicing, GT-time selection or
per-sign manual tuning is allowed.

## 5. Why this is more novel than changing experts

Potential paper contributions, conditional on successful ablations:

1. **Uncertainty-tempered dense field fitting for signing:** arbitrary NLF
   surface localizers become heteroscedastic residual factors inside a temporal
   expressive SMPL-X optimizer, instead of serving as another initialization.
2. **Global-wrist kinematic boundary fusion:** upper-body articulation can
   change without rotating an already reliable hand; confidence controls exact
   versus soft boundary enforcement.
3. **Frequency- and event-aware factor activation:** torso/arm/hand temporal
   scales differ, and the interacting-hand prior activates only under observed
   bimanual ambiguity.
4. **GT-free risk control:** selection is learned externally or calibrated
   without test labels, with explicit abstention back to frozen V6.

The paper should not claim “we combine NLF, SOKE and Dyn-HaMR.” The scientific
claim is the coherent uncertainty/kinematic formulation and its demonstrated
component-wise effect.

## 6. End-to-end implementation plan

All new work stays under `signal4d_v7_nlf_fusion/`. V5/V6 code and artifacts
remain read-only.

### Stage 0 — completed: NLF replacement audit

- Pin NLF v0.3.2 source/model and hashes.
- Export 1,493 dense observations and uncertainties.
- Reject direct NLF replacement.
- Demonstrate complementary UBody errors.
- Implement coherent SO(3) body routing and global-wrist preservation.
- Achieve exploratory full UBody-F gain of 2.633 mm.

Evidence: `SIGNAL-4D_V7_NLF_empirical_audit_2026-08-21.md`.

### Stage 1 — convert hard router to dense uncertainty factor

New modules:

```text
signal4d_v7_nlf_fusion/factors/nlf_field.py
signal4d_v7_nlf_fusion/factors/robust_alignment.py
signal4d_v7_nlf_fusion/factors/tals_reprojection.py
signal4d_v7_nlf_fusion/optimize_upper_body.py
```

Tasks:

1. Define and unit-test the canonical upper-body query set.
2. Estimate torso-only alignment with scale trust region.
3. Implement uncertainty calibration plots and clipping.
4. Optimize root/spine/clavicle/shoulder/elbow only; keep wrist global boundary.
5. Compare hard route, continuous blend and field optimization.
6. Save per-factor residuals and convergence traces.

Exit gate on calibration/out-of-fold data:

- UBody-F gain ≥2.0 mm;
- UBody gain ≥1.5 mm;
- L/R hand regressions each <0.1 mm;
- no increase in temporal acceleration error over V6.

### Stage 2 — clean-room arm/wrist reprojection

New modules:

```text
signal4d_v7_nlf_fusion/factors/arm_reprojection.py
signal4d_v7_nlf_fusion/factors/global_wrist.py
signal4d_v7_nlf_fusion/calibration/reprojection_thresholds.py
```

Tasks:

1. Reuse existing 2D shoulder/elbow/wrist observations; add MediaPipe only as
   an optional independent observation, not a required expert swap.
2. Implement SOKE equations S1–S3 independently from the supplement.
3. Replace raw L1 with confidence/uncertainty TALS and robust loss.
4. Verify exact global wrist invariance with forward-kinematics tests.
5. Run a controlled ablation: reprojection, +TALS, +global wrist, +temporal.

Do not vendor SOKE code because its license prohibits derivatives and its
fitting implementation is not released.

### Stage 3 — selective bimanual hand refinement

New isolated dependency lane:

```text
signal4d_v7_nlf_fusion/external/dyn_hamr.lock.json
signal4d_v7_nlf_fusion/hands/interaction_detector.py
signal4d_v7_nlf_fusion/hands/dyn_hamr_prior_adapter.py
signal4d_v7_nlf_fusion/hands/contact_losses.py
```

Tasks:

1. Pin Dyn-HaMR commit and audit every checkpoint/MANO/transitive license.
2. Identify bimanual windows without GT and log every activation.
3. Convert WiLoR/SMPL-X hand states to the prior's MANO convention with
   round-trip tests below `1e-5` rotation error.
4. Use static/identity camera inputs; bypass SLAM.
5. Optimize hand articulation and relative translation inside windows while
   respecting global wrist boundary and body seams.
6. Abstain to V6 whenever convergence, handedness or uncertainty checks fail.

Hand exit gate:

- LHand and RHand each improve by ≥0.5 mm;
- no UBody-F regression >0.2 mm;
- collision rate and acceleration error do not worsen;
- gains remain under per-sign, not just per-frame, aggregation.

### Stage 4 — GT-independent router/risk training

The current exploratory random forest must not be the paper model because it
uses SGNify development errors as targets.

Preferred training path:

1. Use SignAvatars signer-disjoint clips for sign-domain motion/uncertainty
   calibration, subject to its access terms.
2. Generate controlled corruptions of body/arm/hand pose, blur, crop,
   occlusion and handedness; the known corruption defines risk supervision.
3. Train a small temporal risk network to predict factor reliability and
   improvement, not final pose.
4. Calibrate abstention on a non-test SGNify calibration partition only.
5. Freeze feature normalization, network hash, threshold and factor weights.

Fallback if external GT is insufficient: use sign-level K-fold out-of-fold
predictions. For every evaluated sign, train the router on other signs only;
nest all threshold selection inside each training fold. This is weaker than a
new independent holdout but eliminates direct target leakage.

### Stage 5 — strict evaluation and visualization

Required runs:

```text
A0  Frozen V6
A1  NLF direct replacement (negative control)
A2  Hard NLF body router
A3  Dense NLF uncertainty factor
A4  A3 + TALS arm reprojection
A5  A4 + global-wrist boundary
A6  A5 + multi-scale temporal factor
A7  A6 + selective interaction prior (full method)
```

For every run:

- same 57 signs and 1,493 frames;
- official author evaluator and exact source/assets hashes;
- All, UBody, UBody-F, LHand, RHand;
- per-frame, per-sign and author vertex-micro tables;
- paired sign-cluster bootstrap 95% confidence intervals;
- temporal velocity/acceleration and failure/abstention rates;
- reconstruction renders for best, median and worst signs;
- complete runtime, GPU, source/checkpoint and configuration provenance.

## 7. Confirmatory success criteria

The target is deliberately larger than a tiny numerical gain:

| Metric | Minimum gain versus frozen V6 | Additional condition |
|---|---:|---|
| UBody-F | ≥2.0 mm | 95% paired sign-cluster CI entirely below zero |
| UBody | ≥1.5 mm | no dominant single-sign effect |
| LHand | ≥0.5 mm | official one-hand exclusion unchanged |
| RHand | ≥0.5 mm | official region unchanged |
| All | no regression | report even if not primary |
| Coverage | exactly 1,493/1,493 | no fallback or duplicated frame |

If Stage 1 passes UBody gates but Stage 3 fails hand gates, publish/retain it as
an upper-body method and make no hand-improvement claim. Do not hide the hand
ablation or average L/R hands into one favorable number.

## 8. Ablations needed for attribution

1. SMPLer-X/V6 versus direct NLF proves replacement is insufficient.
2. NLF parametric pose versus dense non-parametric field isolates field value.
3. Uniform weights versus NLF uncertainty versus calibrated uncertainty.
4. Local-wrist copy versus global-wrist compensation.
5. Framewise versus single-scale versus multi-scale temporal factors.
6. Plain L1 reprojection versus TALS reprojection.
7. Always-on versus event-gated interacting-hand prior.
8. Router trained on SGNify (exploratory upper bound) versus external/out-of-fold
   router (valid claim).
9. V6 beta versus blended NLF beta; current evidence favors V6 identity.
10. Equal-frame and equal-sign aggregation to expose long-clip domination.

## 9. Risks and stop rules

| Risk | Detection | Response |
|---|---|---|
| NLF uncertainty is miscalibrated in sign videos | reliability curve/PCC by region | temperature/isotonic calibration on non-test data; cap weights |
| Dense field alignment absorbs pose error | scale/rotation drift diagnostics | torso anchors, fixed scale, trust region |
| Arm gain rotates good hands | global wrist deviation | exact/soft global-wrist constraint |
| Interaction prior oversmooths fast fingers | hand acceleration and high-motion subset | event gating and shorter windows |
| 2D fit improves pixels but worsens 3D | calibration 2D-vs-3D trade-off | TALS and early stopping |
| Router exploits dataset identity | signer/sign OOD performance | external training and sign-disjoint folds |
| License prevents adaptation/redistribution | source lock audit | clean-room equations or black-box use; never vendor ND code |
| Test-set overfitting | provenance log shows result inspection | new holdout or nested OOF; label current numbers exploratory |

Stop a module if it misses its material gate in two independently seeded or
folded evaluations, even if one aggregate number improves slightly. Preserve
the best earlier artifact and document the negative result.

## 10. Recommended execution order

1. Promote the current global-wrist implementation as the V7 research base,
   not as final SOTA.
2. Implement the dense NLF field factor and TALS arm reprojection first; this
   has both direct NLF headroom and same-domain SOKE evidence.
3. Build the GT-independent/out-of-fold gate before any new test claim.
4. Integrate Dyn-HaMR's interaction prior only after the upper-body pipeline is
   frozen and only for detected bimanual windows.
5. Run all ablations and strict author evaluation once configurations are
   frozen.

This order maximizes expected UBody-F gain, protects the already strong hands,
and reserves expensive hand-prior integration for frames where it can solve a
specific ambiguity.
