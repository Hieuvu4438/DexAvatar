# Preliminary Method Specification

This document is a **testable design**, not an implemented or validated contribution. Every predicted benefit is [HYPOTHESIS].

## Primary method: Reliability-Gated Kinematic Residual Fusion (RGKRF)

### Unified hypothesis

[HYPOTHESIS] DexAvatar’s body and hand proposals fail in different frames. Uniformly trusting a hand estimator and a fixed previous-frame prior propagates local errors and creates wrist/forearm/hand incompatibility. A gate driven by proposal reliability, combined with a bounded rotation-space residual and confidence-weighted temporal acceleration, should improve hand orientation/articulation while preserving the upper body.

### Components retained from baseline

- SMPLer-X body/camera/shape initialisation.
- Sapiens/whole-body 2D keypoints and confidence.
- SMPL-X topology and camera projection.
- SignBPoser and SignHPoser priors, initially frozen.
- Existing collision term only after a collision-only ablation justifies it.

### Borrowed evidence, not claimed novelty

- A stronger independent hand proposal (initially WiLoR; HaMeR remains the controlled baseline).
- Forearm/wrist geometric alignment and temporal smoothing as a strong deterministic comparison, motivated by Tamaththul3D.
- Separate body/left/right residual streams and visibility-aware sequence reasoning, motivated by DanceHMR.

### Proposed task-specific contribution

1. A per-frame, per-hand reliability vector based on detector confidence, 2D reprojection, body/hand proposal agreement, crop truncation, forearm consistency, and temporal jerk.
2. A bounded hand rotation residual coupled to elbow–wrist–metacarpal geometry, rather than overwriting SMPL-X hands with an independent MANO estimate.
3. A visibility-weighted second-order temporal objective with reliable frames acting as anchors; low-confidence frames receive stronger interpolation but weaker observation loss.
4. A completeness-aware training/evaluation path that never drops failed frames silently.

### Interface contracts

Let sequence length be `T`, SMPL-X have 10,475 vertices, body pose have 21 axis-angle joints, and each hand have 15 axis-angle joints.

| Interface | Shape / type | Semantics | Trainability |
|---|---|---|---|
| RGB frames | `T × H × W × 3`, float | Same frames for every comparison | frozen input |
| Body proposal | `theta_b: T×21×3`, `beta: T×10`, camera, confidence | SMPLer-X initial state | frozen in first ablation; later latent residual only |
| Hand proposal A | `theta_h^A: T×2×15×3`, confidence | Baseline HaMeR/SignHPoser-compatible estimate | frozen |
| Hand proposal B | `theta_h^B: T×2×15×3`, confidence | WiLoR MANO rotations mapped to SMPL-X order | frozen; mapping unit-tested |
| 2D observations | `k: T×133×2`, `c: T×133` | Sapiens/hand keypoints and confidence | frozen |
| Reliability features | `r: T×2×D` | confidence, reprojection, agreement, visibility, jerk | deterministic features |
| Gate | `g: T×2×15` in `[0,1]` | proposal interpolation strength per joint | small MLP, trainable in primary |
| Residual | `delta: T×2×15×3` | bounded tangent-space correction | trainable/predicted; norm-clipped |
| Output | SMPL-X params and `T×10475×3` vertices | exact topology required by evaluator | saved with manifest/hash |

**Mapping risk:** MANO/SMPL-X hand joint order, left-hand mirroring, global wrist frame, and axis-angle conventions must be verified with synthetic forward-kinematics tests. No “seamless” compatibility is assumed.

### Rotation fusion

For joint `j`, frame `t`, and hand `h`, convert proposal rotations to matrices and interpolate in the tangent space:

\[
R^{f}_{t,h,j} = \exp\!\left(g_{t,h,j}\,\log(R^{B}_{t,h,j}(R^{A}_{t,h,j})^{-1}) + \Delta_{t,h,j}\right) R^{A}_{t,h,j},
\]

where `||Delta||` is capped. This makes the proposal disagreement explicit and independently ablatable.

### Objective

\[
\mathcal L =
\lambda_{2d}\mathcal L_{2d}+
\lambda_{prior}\mathcal L_{prior}+
\lambda_{kin}\mathcal L_{kin}+
\lambda_{temp}\mathcal L_{temp}+
\lambda_{res}\mathcal L_{res}.
\]

- `L_2d`: confidence-weighted robust reprojection loss using the same observation source as baseline.
- `L_prior`: frozen SignBPoser/SignHPoser likelihood or latent norm, separated for body and hands.
- `L_kin`: elbow–wrist direction, wrist-frame, and metacarpal continuity between body and hand streams.
- `L_temp`: robust second difference of body/hand rotations weighted by inverse observation reliability; anchor frames retain higher observation weight.
- `L_res`: bounded residual and gate regularisation to prevent a larger model from winning by unconstrained correction.

Collision/contact loss is **not** part of the minimum method. It enters only if the contact/penetration slice establishes a separate residual failure.

### Training stages

1. **Stage 0 — interface verification:** map MANO ↔ SMPL-X, render toy poses, test left/right symmetry, gradients, and checkpoint round-trip.
2. **Stage 1 — proposal-only baseline:** HaMeR versus WiLoR under unchanged DexAvatar fitting and equal frames.
3. **Stage 2 — deterministic gate:** use fixed reliability thresholds to test the mechanism without learned capacity.
4. **Stage 3 — learned gate:** train a small MLP on non-SGNify training data (preferred: licensed SignAvatars SMPL-X sequences with controlled blur/occlusion/crop corruption). Freeze proposal networks.
5. **Stage 4 — sequence refinement:** optimise/fine-tune only bounded residuals and the gate; select on validation.

If suitable training data are unavailable, Stage 3 is skipped and the deterministic fallback becomes the primary executable method; claims are narrowed accordingly.

### Inference pseudocode

```text
for each video sequence:
    body, camera, keypoints = run_body_pipeline(frames)
    hand_A = run_baseline_hand_pipeline(frames)
    hand_B = run_wilor(frames)
    assert all frame IDs and handedness mappings match manifest
    features = reliability_features(body, hand_A, hand_B, keypoints)
    gate, residual = fusion_module(features, hand_A, hand_B)
    fused_hand = lie_interpolate(hand_A, hand_B, gate, residual)
    params = windowed_refine(body, fused_hand, keypoints,
                             kinematic_loss, visibility_temporal_loss)
    save SMPL-X params, meshes, confidences, completeness, and hashes
```

### Ablation map

| ID | Variant | Question answered |
|---|---|---|
| A0 | DexAvatar reproduced | Local reference |
| A1 | A0 + WiLoR proposal only | Is the hand estimator itself stronger? |
| A2 | A0 + deterministic IK only | Does body–hand geometry matter independently? |
| A3 | A0 + visibility temporal loss only | Does reliability-aware time help independently? |
| A4 | A0 + deterministic gate, no residual | Is selection/fusion sufficient? |
| A5 | A0 + learned gate, no kinematic term | Does capacity alone explain gain? |
| A6 | A0 + kinematic residual, uniform gate | Does coupling matter without uncertainty? |
| A7 | Full RGKRF | Combined result |
| A8 | Full with parameter-matched MLP/no reliability features | Does extra capacity explain gain? |
| A9 | Full with shuffled reliability | Does the gate use meaningful evidence? |

### Complexity and failure modes

- [INFERENCE] The gate is small relative to the hand/body proposal networks; its cost is linear in `T × 2 × 15`. Exact parameters, FLOPs, latency, peak memory, and energy must be measured, not estimated in the paper.
- Full runtime may still be dominated by external proposal networks and SMPL-X optimisation.
- Expected failure signatures: left/right mapping error; over-smoothing fast fingers; gate collapse to one proposal; metric improvement caused only by frame loss; plausible hand corrections penalised by imperfect SGNify GT; gains explained by extra data/compute; non-commercial/no-derivatives licence conflict.

## Fallback method: Visibility-Weighted IK Refinement (VW-IK)

Use WiLoR hand rotations, closed-form forearm/wrist orientation alignment, and robust windowed smoothing weighted by 2D confidence. No learned gate and no extra training data. This is easier to reproduce and likely informative, but its novelty is insufficient by itself for a strong conference paper unless the work contributes a new protocol analysis or task-specific coupling mechanism.

