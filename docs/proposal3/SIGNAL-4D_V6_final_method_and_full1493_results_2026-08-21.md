# SIGNAL-4D V6 UQ-DiffPrior: final method and 1,493-frame result

Date: 2026-08-21  
Parent method: frozen SIGNAL-4D V5 branch `signal4d-v5-frozen-20260821` at `ee519aa`  
Implementation: `signal4d.extensions.v6_uqdiff`  
Official evaluator adapter: `signal4d/evaluate_author_protocol.py`

## 1. What V6 is—and is not

V6 does **not** replace SMPLer-X, WiLoR, HaMeR, SignBPoser, SignHPoser, or any
other DexAvatar/V5 expert. It starts from the immutable V5 SMPL-X sequence and
adds a post-fitting, whole-body-prior factor over a deliberately small set of
upper-limb rotations. The expert observations, V5 shape, global orientation,
translation, legs, head, face, and both local hand-pose blocks remain frozen.

The final implemented method name is `SIGNAL4D_V6_UQDiff`. “V6” denotes a new
research stage after the frozen V5 release, not the sixth replacement image
expert. Its novelty is the interaction of four factors:

1. a DPoser-X whole-body diffusion denoising target;
2. a geodesic pullback of that target to the product of SO(3) rotations;
3. uncertainty- and change-aware factor weighting plus wrist--MCP seam
   preservation;
4. an exact-fallback, GT-free objective-margin gate.

## 2. State and open parameter contract

For frame `t` and an open joint `j`, V6 optimizes a tangent update

`R_t,j = R_t,j^V5 Exp(delta_t,j)`.

Only these eight named SMPL-X body joints are open:

- left/right collar;
- left/right shoulder;
- left/right elbow;
- left/right wrist.

The implementation maps these names through a tested SMPL-X joint registry and
asserts after refinement that every closed rotation is unchanged. The core V6
configuration sets both `optimize_left_hand` and `optimize_right_hand` to
`false`. Consequently, improved hand metrics come from better global arm/wrist
placement of the already fitted V5 hands—not from replacing or retraining the
hand expert.

## 3. DPoser-X factor

The bridge is pinned to DPoser-X commit
`c373fce3d364a4a0946e8445fdea5cbfd490e837` and verifies the upstream files,
four official checkpoints, and five official normalizer files by SHA-256 before
use. The published 256-D whole-body order is retained:

`body(63) + mirrored-left-hand(45) + right-hand(45) + jaw(3) + expression(100)`.

At each target refresh, the frozen diffusion model receives the current
whole-body pose, a fixed seeded noise sample, and diffusion time decreasing
linearly from `0.12` to `0.08`. It produces a stop-gradient denoised target.
The target is refreshed every five optimizer steps. V6 does not minimize the
raw normalized Euclidean distance in the final model. It converts the target
back to valid rotation matrices and minimizes

`L_diff = sum_t,j w_t,j * SNR_t * d_SO(3)(R_t,j, Rhat_t,j)^2`,

only over the eight open upper-limb joints. This avoids treating axis-angle as
a globally Euclidean representation and guarantees that updates remain on
SO(3).

## 4. Calibrated uncertainty and change-aware weighting

For each frame/joint, expert uncertainty is normalized by its temporal median
and clipped to `[0.25, 4.0]`. The change detector produces `c_t in [0,1]`.
The final prior weight is

`w_t,j = clip(sigma_t,j / median_t(sigma_t,j), .25, 4) * [0.1 + 0.9(1-c_t)^2]`.

Thus the learned prior receives more authority when the image experts are
uncertain, but its strength is suppressed at likely sign transitions. The same
change signal controls temporal regularization. This is the concrete meaning of
“calibrated uncertainty-aware SO(3) refinement”: uncertainty does not merely
appear in a confidence plot; it changes the diffusion and temporal factor
strength inside optimization.

## 5. Wrist--MCP seam

For each side, the implementation composes the kinematic chain from pelvis to
wrist and then to the five MCP joints. It penalizes the geodesic discrepancy
between candidate and V5 global MCP rotations. This allows collar, shoulder,
elbow, and wrist corrections while discouraging an arm update that invalidates
the frozen WiLoR/SignHPoser hand evidence. The seam is also down-weighted at
detected rapid transitions.

## 6. Final objective and optimizer

The final W3 candidate minimizes

`L = L_obs + 0.1 L_rot + 0.1 L_V5-anchor`

`    + 0.001 L_temporal-position + 0.001 L_temporal-SO3`

`    + 0.05 L_DPoser-geodesic + 0.01 L_wrist-MCP-seam`.

It uses Adam for 30 steps, learning rate `2e-4`, gradient clipping at `5.0`,
seed `12345`, and records every factor at every step. The full raw run required
478.95 seconds and peaked at 352,342,528 CUDA bytes.

## 7. Frozen GT-free safety gate

The W3 candidate is accepted per frame only if:

1. `candidate_objective + 0.0075 < V5_objective`;
2. maximum open-joint rotation displacement is at most `0.35 rad`;
3. upper-limb uncertainty ratio is at most `1.25`.

Otherwise the complete V5 prediction tensor is copied exactly for that frame.
The gate reads no GT geometry, author region, author metric, sign label, or
evaluation output at inference. Its three thresholds were frozen on the 12-sign
calibration partition before the full run. On the full manifest it selected
271/1,493 candidate frames and fell back to V5 on 1,222 frames.

Temporal coherence is imposed inside the W3 optimizer. The final gate uses no
rejection dilation (`transition_radius=0`), because every tested dilation
reduced the calibration primary endpoint. This detail must be stated instead of
describing the final gate itself as a Viterbi/temporally dilated selector.

## 8. Ablation evidence

On the frozen 260-frame calibration partition, author vertex-micro UBody(-F)
deltas relative to V5 were:

| Stage | Added component | Delta (mm) |
|---|---|---:|
| D0 | constrained arm refinement, no diffusion | -0.062644 |
| D1 | Euclidean DPoser target | -0.063932 |
| D2 | SO(3) geodesic pullback | -0.062681 |
| D3 | uncertainty-aware weighting | -0.062693 |
| D4 | change-aware suppression | -0.094069 |
| D5 | seam + initial safe gate | -0.123507 |
| Final W3 + frozen objective-margin gate | full factor set | **-0.1719996** |

The stronger-optimizer search was rejected because it improved UBody(-F) only
about `0.045--0.050 mm` while regressing TR-all by about `+0.07 mm`. A learned
ExtraTrees gate was also rejected: its OOF clip-bootstrap interval crossed zero.
These negative controls prevent attributing the result to simply taking larger
optimization steps or selecting frames with a GT-trained black box.

## 9. Exact author-protocol result on strict OBJ

All rows below use the functions imported from the attached author evaluator,
the author's vertex regions and class-0 left-hand exclusion rule, and author
vertex-micro aggregation. Coverage is 57/57 signs and 1,493/1,493 frames.

| Method | TR all | UBody | UBody(-F) | LHand | RHand |
|---|---:|---:|---:|---:|---:|
| DexAvatar HaMeR baseline | 42.586721 | 26.455999 | 29.907413 | 13.573462 | 12.927137 |
| SIGNAL-4D V5 | 42.143356 | 26.193547 | 29.593199 | 11.665080 | 11.832928 |
| **SIGNAL-4D V6 UQ-DiffPrior** | **42.111624** | **26.139411** | **29.519683** | **11.633895** | **11.805624** |

Relative to DexAvatar, V6 improves TR-all by `0.475097 mm`, UBody by
`0.316588 mm`, UBody(-F) by `0.387730 mm`, LHand by `1.939567 mm`, and RHand
by `1.121513 mm`. Relative to V5, it improves the same endpoints by
`0.031732`, `0.054136`, `0.073516`, `0.031185`, and `0.027304 mm`.

The paired equal-sign bootstrap for V6 minus V5 gives:

| Endpoint | Sign-macro delta | 95% percentile CI |
|---|---:|---:|
| UBody(-F) | -0.066544 | [-0.116719, -0.019938] |
| TR-all | -0.028319 | [-0.055669, -0.002830] |
| LHand | -0.030024 | [-0.064603, -0.004313] |
| RHand | -0.029857 | [-0.063891, -0.000146] |

## 10. Split audit and claim boundary

The 1,493 frames consist of calibration 260, development 578, and test 655.
The V6-minus-V5 tensor UBody(-F) deltas are `-0.172000`, `-0.086507`, and
`-0.023595 mm`, respectively. On the test partition the sign-bootstrap interval
is `[-0.104529, +0.039004] mm`, so an independent test-only superiority claim is
not supported. Test RHand changes by `+0.016036 mm` vertex-micro and remains
well inside the registered `+0.25 mm` non-inferiority margin.

The full result is the best measured row among the three methods in the local
1,493-frame author-protocol comparison, but it is **not evidence of global
literature SOTA**. Moreover, V6 misses the preregistered V5-relative effect-size
target of `-0.15 mm` on full UBody(-F), although the sign-bootstrap upper bound,
all-body, hand, coverage, and dynamics gates pass. Under the written execution
contract V5 therefore remains the formally promoted release, while V6 is kept
as the current measured-best research artifact.

## 11. Temporal diagnostics

Compared with V5, V6 changes the registered clip-macro dynamics errors as
follows (lower is better):

| Metric | V5 | V6 | Delta | Relative |
|---|---:|---:|---:|---:|
| Velocity | 6.547138 | 6.542203 | -0.004934 | -0.075% |
| Acceleration | 143.094171 | 142.894706 | -0.199465 | -0.139% |
| Jerk | 3816.577984 | 3811.576867 | -5.001118 | -0.131% |

Thus the sparse fallback gate does not buy geometry by degrading the registered
motion diagnostics.

## 12. Reproducibility and outputs

| Artifact | Path |
|---|---|
| Frozen final config | `signal4d/configs/v6_uqdiff/final_v6_uqdiff_w3_safe_gate.yaml` |
| Raw W3 run | `signal4d/runs/signal4d_v6_w3_raw_full1493_20260821` |
| Final gated predictions | `signal4d/runs/signal4d_v6_final_full1493_20260821` |
| Strict 1,493 OBJ | `signal4d/outputs/strict_dexavatar_obj_20260821/full_1493/SIGNAL4D_v6` |
| 1,493 fitting overlays | `signal4d/outputs/reconstruction_signal4d_v6_full1493_20260821` |
| Strict author report | `signal4d/reports/author_evaluator_strict_obj_20260821/full_1493_v6` |
| Split reports | `signal4d/reports/author_evaluator_v6_tensor_20260821/{development,test}` |
| Dynamics reports | `signal4d/reports/v6_dynamics_20260821/{V5,V6}` |

Critical SHA-256 values:

- final config file: `555daf084b9cfb5bb4975e8eeb0a22d27917f38e75679cc06f6b24d3002ead23`;
- full manifest: `02e06c946f9400d8eb2b238c0297b07e188912121748db68ee1d66d12ea7c362`;
- raw-run record: `22a6305eceb94226ee3c7c52899f22fc3eaffa62e4217d821e22604b7413cc53`;
- final gate record: `5e181113109d43a4401f4745ef8324a3e3b0f87a262f2f926f3649e8d51e7867`;
- strict OBJ manifest: `60773fd48e733fcb556d72d80f8fc45c139f8a7f92b6b752f1287aa3aaf72ef9`;
- render manifest: `733e1288178f4b2889a3f1691dc3df522bedc8b4c40756be0d2b464c94840c6c`;
- strict comparison JSON: `dc20d454d870186032695d9b62835f5da1fcbf509dfb94a18931e615de801c42`.

Every OBJ has 10,475 vertices and 20,908 faces. Maximum export round-trip
error is `0.000005 mm`. The fitting folder contains exactly 1,493 PNG overlays
and 1,493 mesh symlinks in the same `<sign>/smplifyx/{images,meshes}` layout as
DexAvatar.
