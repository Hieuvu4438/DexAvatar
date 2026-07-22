# DexAvatar TR-V2V SOTA Research Report

**Audit date:** 2026-07-13  
**Scope:** DexAvatar code, WACV 2026 paper and supplement, SGNify benchmark, author-supplied evaluator, current local outputs, and public research available by the audit date.  
**Primary target:** reduce translation-removed vertex-to-vertex (TR-V2V) error below DexAvatar's published **30.13 / 13.53 / 13.08 mm** for upper body minus face / left hand / right hand, using a reproducible and publication-safe protocol.

## Executive conclusion

The best route is **not** another isolated pose-prior replacement. The largest, most defensible opportunity is an **uncertainty-aware, whole-sequence body-hand reconstruction system** that:

1. preserves the complete 3D hand observation and global wrist orientation from a strong public hand estimator;
2. integrates the hand into the SMPL-X arm chain with explicit coordinate conversion and differentiable wrist/forearm alignment;
3. optimizes an entire sign bidirectionally instead of fitting each frame causally;
4. uses a joint body-two-hand masked motion prior, with soft phonological and contact constraints; and
5. trains/evaluates against the actual region-centred TR-V2V geometry while never dropping difficult frames.

The immediate engineering priority is to build a canonical 2,872-frame evaluator and reproduce the published protocol. The local scores currently stored in `outputs/` use roughly half-rate, non-identical frame sets and a different one-hand treatment. They are useful diagnostics but **cannot support a SOTA claim**.

The closest new competitor is the June 2026 Tamaththul3D preprint, which reports **29.28 / 10.65 / 8.90 mm** on SGNify. Its paper defines PA-MPVPE (rotation and translation removed), yet its baseline table reproduces the exact published TR-V2V values. This makes protocol equivalence uncertain. The numbers should be treated as a competitive target that must be independently recomputed with the author TR-only evaluator, not as a verified apples-to-apples result.

### Recommended score gates

| Gate | UBody-F | LHand | RHand | Meaning |
|---|---:|---:|---:|---|
| Reproduction | within ±0.2 of 30.13 | within ±0.2 of 13.53 | within ±0.2 of 13.08 | Same 2,872 frames, regions, exclusions, and aggregation |
| Minimum verified improvement | < 30.13 | < 13.53 | < 13.08 | Strictly beats published DexAvatar on exact TR-V2V |
| 2026 competitive target | < 29.28 | < 10.65 | < 8.90 | Beats the preprint numbers after exact TR-only recomputation |
| Defensible-margin design target | ≤ 27.5 | ≤ 10.0 | ≤ 8.5 | Aspirational engineering target with paired 95% confidence intervals |

These are research gates, not promised outcomes.

---

## 1. What the benchmark actually measures

### 1.1 Official test set

The SGNify mocap benchmark contains 57 isolated DGS signs from one native, right-handed signer. The video was reduced to 514×300 at 30 fps and cropped above the pelvis. The central intervals contain **2,872 frames**. The local [`segment.json`](../data/evaluation_from_author/segment.json) confirms this exactly when intervals are treated as end-exclusive:

- `sum(end - start) = 2,872`;
- `sum(end - start + 1) = 2,929`;
- 42 signs have class `~0` and 15 have one-hand class `0` in [`signs.txt`](../data/evaluation_from_author/signs.txt).

The source definition is in the [SGNify CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf).

### 1.2 Exact TR-V2V computation

For region \(R\), the author evaluator computes:

\[
E_{TR}^{R}=\frac{1}{|R|}\sum_{i\in R}
\left\|
(\hat v_i-\bar{\hat v}_{R})-(v_i-\bar v_R)
\right\|_2.
\]

This behavior is implemented in [`evaluate_new_fitting.py`](../data/evaluation_from_author/evaluate_new_fitting.py), lines 159–169. The prediction and ground truth are centred **independently for each evaluated vertex region**. Consequently:

- translation is removed;
- rotation and scale are **not** removed;
- finger articulation, palm orientation, wrist orientation, hand scale, and relative arm geometry still affect the score;
- the hand scores are not pelvis-aligned and are not whole-mesh-translation scores.

The function named `point_error_common_center` at lines 172–181 is algebraically the same centroid-removed error because the same centre is added to both sets. The printed `V2V left/right wrist` labels therefore do not represent a distinct wrist-aligned metric.

### 1.3 Region and one-hand semantics

The paper reports:

| Method | UBody-F | LHand | RHand |
|---|---:|---:|---:|
| DexAvatar | **30.13** | **13.53** | **13.08** |

In the author evaluator, class-`0` signs receive special handling at lines 380–395:

- the left-hand metric is skipped;
- left-hand vertices are removed from every other evaluated region;
- the right hand is still evaluated.

Thus the reported left-hand result covers only the 42 two-handed signs, while the right hand covers all 57 signs under the evaluator's right-handed assumption. UBody-F also changes its vertex set for the 15 class-`0` signs. Any new evaluator must reproduce this behavior for table compatibility and separately report a cleaner fixed-region analysis.

### 1.4 Why PA-MPVPE must remain separate

PA-MPVPE uses a Procrustes transformation and can remove global rotation as well as translation. It is easier to improve with a hand whose articulation is correct but palm/wrist orientation is wrong. TR-V2V deliberately retains that orientation error. Never place PA-MPVPE and TR-V2V values in one comparison column.

The [Tamaththul3D v2 preprint](https://arxiv.org/html/2605.05367v2) defines PA-MPVPE, then reports the exact historical TR-V2V baseline values, including DexAvatar's 30.13/13.53/13.08. Until predictions are available and rescored, its 29.28/10.65/8.90 should be labelled **reported, protocol not verified**.

---

## 2. Evaluation and reproducibility weaknesses

These issues must be fixed before interpreting any model improvement.

| Severity | Weakness | Evidence | Consequence | Required correction |
|---|---|---|---|---|
| Critical | Folder lists are sorted independently and paired by array index | Author evaluator lines 305–307 and 510–516 | A missing or extra directory can silently pair the wrong sign | Join by exact sign name and assert a one-to-one mapping |
| Critical | Predictions are paired with GT by list position, not frame ID | Lines 342–361; prediction filenames are only numerically sorted | Missing prediction frame shifts every subsequent correspondence | Parse frame IDs and inner-join against a signed manifest; missing frames must be failures, not shifts |
| Critical | Central-frame logic is fragile | `--central` is parsed at lines 480–483 but never used; GT intervals are multiplied by two and iterated inclusively at lines 231–249 | Off-by-one and frame-rate mismatches can change the test set | Define one canonical 30-fps, end-exclusive manifest with exactly 2,872 rows |
| Critical | Difficult frames are removed before fitting | [`data_parser.py`](../dexavatar_fitting/smplifyx/data_parser.py), lines 185–234, drops missing HaMeR/init frames and requires both detected hands for `~0` | Coverage bias: a method looks better by avoiding its failures | Keep every benchmark frame; propagate a visibility mask and reconstruct missing observations temporally |
| High | NaN predictions are skipped | Author evaluator lines 364–366 | Catastrophic failures lower apparent mean error | Count NaN/missing as failures, publish coverage, and use a predefined failure penalty or separate failure rate |
| High | Longer signs dominate | Errors are concatenated over all vertices and frames at lines 432–461 | Micro-average can hide sign-specific collapse | Preserve official micro mean, add per-sign macro mean, median, p90 and paired intervals |
| High | Hard-coded author paths | Lines 521–571 point to a private home directory | Released evaluator is not directly portable | Resolve MANO/SMPL-X/segmentation assets from CLI or repository-relative paths and hash them |
| Medium | Region labelling is ambiguous | Several region variants are computed; the paper selects upper-body-minus-face | Easy to report the wrong column | Emit named JSON fields and assert the exact vertex-index SHA-256 |
| Medium | Test-time model selection risk | The supplement states that the best hyperparameter is selected on DEV and TEST | Inflated result and weak review defensibility | Freeze configuration using external validation only; run SGNify test once after preregistration |

### Canonical evaluation manifest

Create a versioned file with one row per expected frame:

```text
sign, class, rgb_frame_id, gt_frame_id, prediction_frame_id,
left_active, right_active, upper_region_hash, left_region_hash, right_region_hash
```

The evaluator should fail closed unless it sees exactly 57 signs and 2,872 rows. It should report:

- expected, reconstructed, missing, NaN, and evaluated frames;
- official micro TR-V2V;
- macro per-sign TR-V2V;
- paired bootstrap 95% confidence intervals;
- one-hand/two-hand and visibility/blur/occlusion strata.

### Local results are not paper-comparable

[`all_methods_comparison_final.csv`](../outputs/all_methods_comparison_final.csv) contains local results such as:

| Local method | TR UBody | TR LHand | TR RHand | Frames |
|---|---:|---:|---:|---:|
| DexAvatar-Hand2D | 24.86 | 20.83 | 12.19 | 1,450 |
| DexAvatar-Biomech | 24.89 | 20.76 | 12.17 | 1,450 |
| DexAvatar-Baseline | 24.90 | 20.96 | 12.23 | 1,429 |
| DexAvatar-Direct | 25.27 | 22.64 | 12.10 | 1,439 |

The frame counts differ by method, are near half of the official 2,872 frames, and the local left-hand calculation includes behavior that differs from the author class-`0` exclusion. The low upper-body numbers therefore do not prove a 5 mm improvement over the paper. They should be labelled **development diagnostics only**.

One useful diagnostic survives: inactive class-`0` left hands strongly inflate the local LHand mean, whereas the author table excludes them. This confirms that exact region/class handling is a first-order issue.

---

## 3. DexAvatar model weaknesses that limit TR-V2V

The repository [`README.md`](../README.md) and [DexAvatar WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html) describe this core flow:

| Stage | Released role | Main audited code |
|---|---|---|
| 2D observation | Sapiens whole-body keypoints | preprocessing and [`data_parser.py`](../dexavatar_fitting/smplifyx/data_parser.py) |
| Whole-body initialization | SMPLer-X SMPL-X parameters | parser and [`fit_single_frame.py`](../dexavatar_fitting/smplifyx/fit_single_frame.py) |
| Hand observation | HaMeR hand detection/3D joints | parser and fitting loss |
| Sign-aware priors | SignBPoser and SignHPoser VAEs | `signbposer/`, `signhposer/`, and fitting loss |
| Refinement | 2D/3D data, priors, temporal, collision and biomechanics | [`fitting.py`](../dexavatar_fitting/smplifyx/fitting.py) |
| Evaluation | author TR-V2V script | [`data/evaluation_from_author/`](../data/evaluation_from_author) |

The current working tree also contains NLF, DPoser-X, VQ-VAE/direct and uncertainty experiments. Findings below explicitly distinguish released/default behavior from those ongoing branches; no experimental branch is assumed to have a verified paper-protocol score.

The paper ablations show where its gains came from and where they saturated:

- filtering the body-prior training data improves UBody-F from 34.06 to 30.28 mm;
- adding biomechanical filtering during prior training changes this to 30.44 mm, while final optimization constraints reach 30.13 mm;
- the unfiltered-to-filtered hand prior changes L/R from 14.19/13.92 to 13.55/13.06 mm;
- final biomechanics reaches 13.53/13.08, slightly worsening the right hand relative to the filtered-prior row.

This is a strong baseline, but the tiny final biomechanics delta indicates that further progress is more likely to come from observations, body-hand coordinate fusion, and temporal inference than from increasingly rigid angle limits.

### 3.1 Only normalized hand depth is used by the released 3D hand loss

In [`fitting.py`](../dexavatar_fitting/smplifyx/fitting.py), lines 599–643:

- only coordinate `2:3` (depth) is selected;
- the values are made wrist-relative;
- prediction and target are independently standardized by their mean and standard deviation;
- the loss discards image-plane 3D geometry and absolute hand scale.

This throws away much of HaMeR/WiLoR's useful 3D output. Independent standardization also makes the observation insensitive to depth scale. TR-V2V, in contrast, penalizes all three coordinates and retains scale and rotation after centring.

**Correction:** use all wrist-relative \(x,y,z\) joints or vertices in a shared camera/body coordinate system, preserve metric scale, add explicit confidence/covariance, and supervise global wrist rotation geodesically.

### 3.2 Hand-body orientation is under-modelled

The official supplement documents that wrist rotations could not be transferred during the SignHPoser retargeting because of T-pose/bone-roll incompatibility. Finger articulation is therefore learned without a reliable global wrist/body relationship. This matters because TR-V2V does not remove rotation.

The recent Tamaththul3D ablation supports the same diagnosis: replacing SMPLer-X fingers with WiLoR after coordinate conversion accounts for nearly all its reported hand improvement, while forearm/wrist alignment repairs kinematic consistency. The 2026 [Hand4Whole++ paper](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html) independently targets this supervision gap by combining hand-specific features with whole-body context and differentiable rigid alignment.

**Correction:** explicitly solve the MANO-to-SMPL-X convention, left-hand reflection, MANO mean pose, global-to-local wrist rotation, and shoulder-elbow-wrist forward kinematics. Validate the conversion with synthetic rotations before optimizing on video.

### 3.3 The temporal model is body-only, causal, and first order

`fitting.py`, lines 583 and 641, compares the current 21-joint body pose to the previous pose with a fixed weight. It does not model:

- hand-pose dynamics;
- velocity, acceleration, or jerk;
- future frames;
- uncertainty-dependent temporal support;
- two-hand coordination;
- an explicit motion manifold.

[`main.py`](../dexavatar_fitting/smplifyx/main.py), lines 225–344, walks frames causally. Resuming a partially completed run can skip frames before refreshing temporal state (lines 300–303), so the result can depend on execution history.

**Correction:** optimize complete signs or overlapping bidirectional windows, with rotation-geodesic velocity/acceleration terms and observation-aware smoothing. Cache-free deterministic re-runs should yield identical results.

### 3.4 Body and hands are independent per-frame VAEs

SignBPoser and SignHPoser are small MLP VAEs. Body, left hand, and right hand are decoded separately, so the prior cannot learn:

- arm-to-palm coordination;
- dominant/non-dominant hand relationships;
- hand-hand contact and symmetry;
- temporal coarticulation;
- multi-modal completions under occlusion.

The paper's own ablations show that the filtered priors help, but biomechanical terms provide only marginal gains and slightly worsen the reported right-hand score in the final step. This suggests the remaining error is observation/fusion/sequence limited, not simply fixed by stronger joint-angle thresholds.

**Correction:** use a joint body-plus-two-hands sequence prior and treat missing observations as a masked completion problem.

### 3.5 SignHPoser training coverage is narrow

The supplement describes 8 signers (6 Auslan, 2 ASL) performing 93 fingerspelled words, retargeted through a partly manual Blender process. This is valuable high-quality capture, but it is small relative to natural signing variation and is concentrated on fingerspelling rather than continuous lexical/coarticulated motion. The paper also notes imperfect/collapsed fingers in some benchmark GT frames.

**Correction:** combine high-quality hand capture with much larger pseudo-labelled sign corpora, but retain confidence filtering and source-balanced sampling. Do not train a plausibility prior to copy benchmark annotation artifacts.

### 3.6 Strong initialization anchors can lock in proposal error

The baseline YAML applies initialization weights of 1,200 in every stage for body and both hands ([`fit_smplx_vposer_x.yaml`](../dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml), lines 58–73). `fitting.py`, lines 706–712 and 830–841, adds strong L1/robust anchors to initial body/hand poses.

These prevent divergence but can make the optimizer a denoised copy of SMPLer-X/hand initialization. A better observation or prior cannot correct a confident but wrong initializer if the anchor does not anneal with uncertainty.

**Correction:** learn or calibrate proposal uncertainty, use a mixture of proposal experts, and anneal each anchor by joint, frame, estimator agreement, visibility, and optimization stage.

### 3.7 Released hand-latent initialization is accidentally discarded

At repository commit `531de48`, `fit_single_frame.py` first regresses hand latents toward the initialization, then resets the active latent(s) to zero immediately before optimization (HEAD lines 618–630). Zero is the VAE mean, so this discards the expensive initializer and makes convergence within a small iteration budget harder.

The current working tree already removes that reset. This is a sensible fix, but it must be evaluated on the canonical protocol and isolated as its own ablation.

### 3.8 The optimized state is too restricted

The default fitting stages primarily optimize body and hand latent variables. Global orientation and translation are frozen unless current experimental flags are enabled; shape is initialized and regularized rather than jointly estimated as one clip-level identity. Yet upper-body TR-V2V is sensitive to root orientation, body proportions, shoulders, and arm geometry.

**Correction:** optimize a structured state:

- clip-shared shape \(\beta\), scale/camera calibration, and optional bone-length correction;
- slowly varying camera/root variables;
- per-frame body, wrist, and finger rotations;
- proposal-specific nuisance transforms;
- all with trust regions and staged unlocking.

### 3.9 Binary sign classes discard useful linguistic structure

The local pipeline reduces signs to `0` versus `~0`. SGNify defines richer classes and losses for symmetry and relative pose invariance. A binary switch only decides which hand is active; it does not exploit repeated/symmetric handshape, relative palm orientation, hand location, or movement type.

**Correction:** predict a probability distribution over phonological attributes and apply soft, evidence-gated constraints. Do not assume ground-truth gloss or class metadata at test time unless the comparison explicitly permits it.

### 3.10 Missing frames are avoided instead of inferred

The parser requires available hand detections, and current exception handling can leave sentinel files and continue. These are precisely the high-blur, inter-hand-occlusion, and extreme-pose frames where sequence reconstruction should outperform per-frame regression.

**Correction:** keep the frames, mask unavailable observations, propagate tracks, and use multi-hypothesis temporal completion. Report failure rate rather than silently shrinking the denominator.

### 3.11 Objective-to-metric mismatch

The optimizer is dominated by 2D joints, priors, pose-parameter anchors, and heuristics. None directly matches the region-centred vertex geometry used in the paper table. Axis-angle L1 and Euler-limit losses are also discontinuous around representation boundaries and are not equivalent to mesh error.

**Correction:** use continuous 6D rotations or rotation matrices, geodesic rotation losses, and a differentiable TR-V2V surrogate on pseudo/high-quality 3D supervision.

---

## 4. Public-method landscape and what to reuse

Prioritize components with released code/checkpoints. Treat unreleased 2026 ideas as architectural guidance, not dependencies.

| Method | Public status on audit date | Relevant capability | Recommended role |
|---|---|---|---|
| [WiLoR, CVPR 2025](https://github.com/rolpotamias/WiLoR) | Code and checkpoints | End-to-end hand detection and MANO reconstruction | Primary per-frame hand proposal; retain pose, shape, global wrist rotation, camera and confidence |
| [NLF, NeurIPS 2024](https://virtualhumans.mpi-inf.mpg.de/nlf/) | Public project/code; already present locally | Localizes arbitrary 3D body points and supports model fitting | Body/arm proposal and dense 3D consistency expert |
| [SMPLer-X](https://github.com/caizhongang/SMPLer-X) | Public | Strong whole-body SMPL-X initialization | Keep as a proposal, not an immutable target |
| [SMPLest-X](https://github.com/MotrixLab/SMPLest-X) | Public | Large whole-body estimator | Screen on external validation; latest is not automatically best for this low-resolution signer |
| [DPoser-X, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html) | Paper/code; local branch exists | Masked whole-body pose prior and multi-hypothesis completion | Better initialization/completion prior; condition and fine-tune for signing rather than using an unconditional score alone |
| [Temp-LowRes-Hand, CVPR 2025](https://github.com/NewbieFan/Temp-LowRes-hand) | Public code | Temporal reconstruction for low-resolution hand crops | Hand proposal/refinement expert for small and blurred signer hands |
| [OmniHands / 4DHands](https://github.com/LinDixuan/OmniHands) | Public project/code | Temporally consistent interacting-hand reconstruction | Two-hand occlusion/contact expert; convert outputs carefully to SMPL-X |
| [Dyn-HaMR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html) | Paper/project | 4D interacting-hand motion and generative infilling | Inspiration or optional expert for severe two-hand occlusion |
| [PAD-Hand, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html) | Paper; verify code/checkpoints before dependency | Physics-aware sequence refinement with uncertainty | Design reference for uncertainty-aware hand dynamics |
| [Hand4Whole++, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html) | Paper; verify release | Conditions whole-body features on hand features; rigid hand alignment | Strong evidence for explicit body-hand wrist fusion |
| [FUSION, CVPR 2026 Findings](https://arxiv.org/abs/2601.03959) | Paper; public-code availability must be checked before use | Unified body-and-hands motion diffusion | Closest prior architecture; reimplement/fine-tune only if licensing and data permit |
| [SGNify, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf) | Public research code/data | Symmetry and relative-pose linguistic constraints | Restore richer probabilistic phonological constraints |

### Important competitive lesson

Tamaththul3D's reported ablation changes hands from **18.17/17.47** with SMPLer-X to **10.71/9.03** after WiLoR coordinate conversion; geometric alignment then gives only a small numeric change under its stated PA metric. Therefore:

- a correct WiLoR-to-SMPL-X conversion is the fastest serious baseline;
- a paper whose only contribution is “replace HaMeR with WiLoR” is already obsolete;
- novelty must come from exact TR orientation, uncertainty, sequence inference, missing-frame recovery, and sign-structured body-hand coupling.

---

## 5. Proposed method: SignFusion-TR

`SignFusion-TR` is a working name for an **observation-conditioned joint body-hand sequence posterior**. It is designed around the actual failure modes and metric rather than around a particular initializer.

### 5.1 State representation

For a sign clip of \(T\) frames, estimate:

\[
X = \{\beta, c, R_t^{root}, R_t^{body}, R_t^{lw}, R_t^{rw},
R_t^{lh}, R_t^{rh}, t_t\}_{t=1}^{T},
\]

where \(\beta\) is clip-shared SMPL-X shape, \(c\) contains camera/calibration variables, wrists are explicit, and rotations use a continuous 6D representation internally. SMPL-X forward kinematics produces the final mesh.

### 5.2 Multi-expert observations

Run public estimators once and cache immutable proposals:

- Sapiens: 2D body/hand keypoints and confidence;
- SMPLer-X and NLF: body, arm, root and camera candidates;
- WiLoR: MANO fingers, shape, global wrist orientation, translation/camera;
- OmniHands or Temp-LowRes-Hand: temporal/two-hand proposals where available;
- optical flow or dense point tracks for inter-frame image evidence.

For each joint/vertex observation, estimate uncertainty from detector confidence, temporal inconsistency, crop resolution, left/right ambiguity, occlusion, and cross-estimator disagreement. Do not choose one proposal globally; fuse experts per part and frame.

### 5.3 Exact hand-body integration

Build a unit-tested differentiable converter:

1. convert MANO rotations to matrices;
2. remove/add the correct MANO and SMPL-X rest-pose conventions;
3. reflect left-hand rotations correctly;
4. transform global wrist orientation into the SMPL-X local kinematic chain;
5. solve shoulder/elbow/forearm swing and twist to match the target wrist without breaking arm keypoints;
6. refine shoulder, elbow and wrist jointly in a small trust region.

Synthetic tests should rotate a known SMPL-X hand through random orientations, convert SMPL-X→MANO→SMPL-X, and require sub-millimetre vertex round-trip error before any benchmark experiment.

### 5.4 Bidirectional sequence posterior

Infer complete signs or 32–64-frame overlapping windows. A masked transformer/diffusion/flow prior receives:

- noisy SMPL-X body, wrist and hand rotations;
- observation masks and per-joint covariance;
- time, handedness and optional phonological probabilities;
- contact candidates and relative two-hand geometry.

The model returns multiple plausible clean sequences. This can be initialized from DPoser-X or a FUSION-like architecture, then trained on sign motion. The central novelty is not an unconditional prior; it is **uncertainty-conditioned posterior completion tied to image evidence**.

### 5.5 Soft phonology and contact

Predict rather than manually provide:

- one/two-handedness;
- symmetric/asymmetric movement;
- handshape invariance probability;
- relative palm orientation;
- hand location and movement type;
- hand-hand and hand-body contact probability.

Use these as soft gates. For example, symmetry should constrain relative rotations/trajectories only when its posterior and image support are high. Contact should combine attraction near detected contact with collision avoidance, not force every close approach into contact.

### 5.6 Proposed objective

\[
\mathcal L =
\lambda_{2D}\mathcal L_{2D}^{unc}
+\lambda_{3D}\mathcal L_{handXYZ}^{unc}
+\lambda_{w}\mathcal L_{wrist}^{geo}
+\lambda_{flow}\mathcal L_{track}
+\lambda_{TR}\mathcal L_{TRV2V}
+\lambda_{mot}\mathcal L_{motion}
+\lambda_{prior}\mathcal L_{posterior}
+\lambda_{phon}\mathcal L_{phonology}
+\lambda_{contact}\mathcal L_{contact}
+\lambda_{coll}\mathcal L_{collision}.
\]

Key terms:

- \(\mathcal L_{2D}^{unc}\): robust confidence/covariance-weighted reprojection;
- \(\mathcal L_{handXYZ}^{unc}\): full metric wrist-relative hand XYZ, not standardized Z only;
- \(\mathcal L_{wrist}^{geo}\): geodesic global wrist/palm orientation plus forward-kinematic arm consistency;
- \(\mathcal L_{track}\): optical-flow/dense-track consistency through detector dropouts;
- \(\mathcal L_{TRV2V}\): region-centred differentiable vertex loss matching the evaluation formula on external/pseudo 3D supervision;
- \(\mathcal L_{motion}\): rotation-geodesic velocity and acceleration, with jerk used carefully so fast intentional signing is not oversmoothed;
- \(\mathcal L_{posterior}\): masked sequence denoising/completion prior;
- \(\mathcal L_{phonology}\): soft symmetry/invariance/location constraints;
- \(\mathcal L_{contact}\): probabilistic contact plus nonpenetration.

The SGNify GT must not be used to train or choose these weights. Train the metric-aware term on held-out high-quality or pseudo-labelled external sequences.

### 5.7 Multi-hypothesis selection

Occluded hands are genuinely ambiguous. Sample \(K\) sequence hypotheses from the prior and rank them without benchmark GT using:

\[
S(X_k)=
\mathcal L_{2D}^{unc}+\mathcal L_{track}+\mathcal L_{wrist}^{geo}
+\mathcal L_{contact}+\mathcal L_{prior}.
\]

Keep both the best deterministic output and uncertainty. This is stronger scientifically than hiding uncertainty behind a single mean pose.

---

## 6. Ranked implementation and ablation roadmap

### P0 — Protocol lock (must precede model claims)

1. Implement the 2,872-row sign/frame manifest.
2. Port the author vertex sets and one-hand rules exactly.
3. Assert sign names, frame IDs, topology, region hashes and coverage.
4. Reproduce DexAvatar within ±0.2 mm for all three regions.
5. Publish an evaluator unit test showing translation invariance but rotation/scale sensitivity.

**Go/no-go:** no SOTA statement until P0 passes.

### P1 — Strong public proposal baseline

1. Reconstruct all official 30-fps frames with SMPLer-X + WiLoR.
2. Preserve the optimized hand latent rather than resetting it.
3. Implement verified MANO↔SMPL-X coordinate conversion.
4. Compare HaMeR, WiLoR, OmniHands and Temp-LowRes-Hand on a separate validation set.
5. Never delete a frame; use nearest/linear rotation interpolation only as an explicit baseline.

**Expected research value:** fastest path to a strong hand baseline; likely much larger hand gain than changing the VAE family alone.

### P2 — Full 3D and wrist/forearm fusion

1. Replace normalized Z-only loss with covariance-weighted XYZ.
2. Add global wrist geodesic loss and differentiable swing-twist arm alignment.
3. Optimize root orientation, shoulders, elbows and wrists with trust regions.
4. Estimate one clip-level shape and camera calibration.

**Primary metric expectation:** lower L/R TR-V2V because rotation is retained by TR alignment; improved UBody-F through coherent arms/hands.

### P3 — Whole-sign bidirectional optimization

1. Build deterministic windowed optimization with velocity/acceleration terms.
2. Add optical-flow/dense-track support for missing detections.
3. Use forward-backward proposal fusion and overlap blending.
4. Evaluate framewise error, jitter and high-motion preservation.

**Primary metric expectation:** largest gain on blur, occlusion and detector-failure strata without reducing coverage.

### P4 — Joint masked body-hand motion prior

1. Establish DPoser-X as a frozen prior baseline.
2. Train a FUSION-like body/two-hand sequence model on external sign data.
3. Corrupt training sequences with realistic crop blur, occlusion, time masks, left/right swaps and proposal noise.
4. Produce multiple completions and calibrated uncertainty.

**Publication contribution:** joint posterior completion, not merely a larger pose autoencoder.

### P5 — Probabilistic phonology/contact and metric-aware learning

1. Restore SGNify's richer symmetry/invariance ideas as predicted soft attributes.
2. Add handshape, location, movement, relative orientation and contact heads.
3. Fine-tune with region-centred TR loss on external 3D/pseudo-3D data.
4. Demonstrate gains particularly on high-occlusion and two-hand signs.

**Publication contribution:** sign-language structure improves geometric reconstruction without oracle gloss/class input.

### Required factorized ablation table

| ID | Public proposals | Full XYZ + wrist | Sequence inference | Joint prior | Phonology/contact | Metric-aware | Coverage |
|---|---|---|---|---|---|---|---|
| B0 | DexAvatar | No | Causal body only | Separate VAEs | Binary class | No | Report |
| B1 | Strong estimator bank | No | No | No | No | No | 100% |
| B2 | B1 | Yes | No | No | No | No | 100% |
| B3 | B2 | Yes | Deterministic bidirectional | No | No | No | 100% |
| B4 | B3 | Yes | Multi-hypothesis | Yes | No | No | 100% |
| B5 | B4 | Yes | Multi-hypothesis | Yes | Yes | No | 100% |
| Full | B5 | Yes | Multi-hypothesis | Yes | Yes | Yes | 100% |

Every row must use identical expected frames. Also report proposal-only variants to separate estimator gains from method gains.

---

## 7. Publication-grade experimental design

### 7.1 Data hygiene

- Do not train, fine-tune, select hyperparameters, or choose checkpoints on SGNify test GT.
- Freeze the final config using an external validation set with signer disjointness.
- Record exact pretrained model revisions, licenses, input resolutions, crops and coordinate conversions.
- Cache input proposals and use the same proposals for all relevant ablations.
- Declare any pseudo-label source and prevent benchmark video leakage from pretraining/fine-tuning where controllable.

### 7.2 Metrics

Keep the official table, but add:

- official micro TR-V2V for UBody-F/L/R;
- macro per-sign TR-V2V;
- median and p90 frame error;
- pelvis-aligned body and wrist-aligned hand MPVPE;
- PA-MPVPE in a separate, clearly labelled table;
- acceleration/jerk and motion-amplitude preservation;
- penetration volume and contact precision/recall where annotations are available;
- frame coverage, missing rate, NaN rate and catastrophic-failure rate;
- runtime, peak memory and proposal/preprocessing cost.

Use paired sign/frame bootstrap 95% confidence intervals and a paired non-parametric test. A 0.1 mm mean change without a stable interval is not convincing.

### 7.3 Required strata

Report results by:

- one-handed vs two-handed signs;
- left vs right and dominant vs non-dominant hand;
- hand crop size;
- occlusion and inter-hand overlap;
- blur and motion speed;
- frontal vs lateral palm orientation;
- contact vs non-contact;
- low vs high cross-estimator disagreement.

This identifies whether an improvement actually solves sign reconstruction or only easy static frames.

### 7.4 Generalization evidence

The mocap benchmark has only one signer and one isolated-sign setting. An A*-level submission needs broader evidence:

- signer-disjoint quantitative validation on another 3D or carefully curated pseudo-3D sign dataset;
- qualitative and 2D/temporal evaluation on continuous signing;
- at least two additional sign languages/signing styles;
- robustness to resolution, blur, occlusion, clothing and handedness;
- a Deaf/sign-language-user perceptual study measuring handshape correctness, intelligibility and naturalness, designed with appropriate community involvement.

TR-V2V is necessary for comparison but insufficient for semantic correctness. The SGNify paper itself notes this limitation.

### 7.5 Reproducibility package

Release:

- canonical evaluator and manifest;
- conversion/round-trip tests;
- environment lockfiles and one-command evaluation;
- proposal caches where licenses allow;
- per-frame/per-sign CSV, failure manifest and confidence intervals;
- ablation configs and random seeds;
- model cards documenting sign-language and demographic coverage.

---

## 8. Paper thesis and novelty test

### Proposed central claim

> Accurate sign reconstruction is a structured, uncertain sequence-inference problem. Jointly fusing metric hand observations, global wrist-body kinematics, temporal evidence and probabilistic phonology recovers complete SMPL-X signing motion more accurately than per-frame independent priors.

### Contributions that can support that claim

1. **Metric-consistent body-hand fusion:** complete MANO/SMPL-X conversion and wrist/forearm alignment optimized for translation-only evaluation.
2. **Observation-conditioned joint sequence posterior:** body and two hands reconstructed together with uncertainty and masked-frame completion.
3. **Predicted soft sign structure:** phonological/contact constraints without oracle test labels.
4. **Protocol-clean evaluation:** exact TR-V2V, full coverage, confidence intervals and broader cross-signer/language validation.

### What is not enough for an A* paper

- swapping HaMeR for WiLoR;
- replacing a VAE with VQ-VAE/DPoser-X without a sign-specific sequence formulation;
- reporting half-rate or method-dependent frame sets;
- tuning weights on SGNify test GT;
- claiming SOTA from PA-MPVPE against TR-V2V baselines;
- improving plausibility while reducing sign intelligibility or oversmoothing fast motion.

---

## 9. Risk register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Imperfect mocap/retargeted GT | Plausible fingers can score worse than collapsed GT | Report robust per-sign analysis and perceptual/semantic metrics; do not train on benchmark artifacts |
| WiLoR coordinate mismatch | A small convention error can dominate hand TR-V2V | Synthetic round-trip and global-wrist tests before benchmark use |
| Oversmoothing | Removes meaningful fast transitions | Confidence-aware motion prior; measure amplitude and sign intelligibility, not jerk alone |
| Prior domain bias | Small ASL/Auslan fingerspelling data may not cover DGS/continuous signing | Multi-source training, signer/language-balanced sampling and cross-language validation |
| Test leakage | Tiny public benchmark encourages manual weight tuning | External dev set, preregistered config, hash/checkpoint freeze |
| Proposal license/release changes | Can prevent reproducibility or commercial use | Audit licenses and pin revisions; include an all-public fallback |
| Compute-heavy optimization | Limits dataset-scale deployment | Distill posterior into an amortized model after establishing accuracy; report end-to-end runtime |
| Preprint comparison ambiguity | Wrong SOTA bar or mixed metric | Obtain/reconstruct predictions and score all methods with the same released evaluator |

---

## 10. Concrete next actions

1. **Stop interpreting current `outputs/` as paper-comparable.** Preserve them as development diagnostics.
2. **Implement and test the canonical 2,872-frame evaluator.** This is the highest-priority code change.
3. **Create an untouched released-DexAvatar reproduction config** separate from current NLF/DPoser-X/VQ-VAE experiments.
4. **Build a unit-tested WiLoR→SMPL-X conversion baseline** and rescore every official frame.
5. **Replace the Z-only standardized hand term with full uncertainty-weighted XYZ plus wrist orientation.**
6. **Eliminate frame deletion** and add masked temporal interpolation as the simplest 100%-coverage baseline.
7. **Add bidirectional whole-sign fitting** before training a new prior; this isolates the value of sequence evidence.
8. **Then train the joint masked body-hand posterior** and add soft phonology/contact one factor at a time.
9. **Recompute Tamaththul3D under exact TR-V2V** if predictions/code become available.
10. **Freeze the benchmark configuration and run statistical/generalization studies** only after the method is stable on external validation.

The most likely short path to a strong number is P0→P1→P2. The most likely path to a strong paper is P0→P1→P2→P3→P4, with P5 supplying sign-language-specific novelty and interpretability.

---

## Primary sources

- Kundu et al., [DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Prior](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html), WACV 2026; [official supplement](https://openaccess.thecvf.com/content/WACV2026/supplemental/Kundu_DexAvatar_3D_Sign_WACV_2026_supplemental.zip).
- Forte et al., [Reconstructing Signing Avatars From Video Using Linguistic Priors](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf), CVPR 2023.
- Potamias et al., [WiLoR repository](https://github.com/rolpotamias/WiLoR), CVPR 2025.
- Fan et al., [Pose-Guided Temporal Enhancement for Robust Low-Resolution Hand Reconstruction](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Pose-Guided_Temporal_Enhancement_for_Robust_Low-Resolution_Hand_Reconstruction_CVPR_2025_paper.html), CVPR 2025.
- Lu et al., [DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html), ICCV 2025.
- Yu et al., [Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html), CVPR 2025.
- Moon, [Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html), CVPR 2026.
- Ismayilzada et al., [PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html), CVPR 2026.
- Duran et al., [FUSION: Full-body Unified Motion Prior for Body and Hands Via Diffusion](https://arxiv.org/abs/2601.03959), CVPR 2026 Findings.
- Alghamdi et al., [Tamaththul3D v2](https://arxiv.org/html/2605.05367v2), arXiv, 2026. Its SGNify metric compatibility remains to be independently verified as explained above.
