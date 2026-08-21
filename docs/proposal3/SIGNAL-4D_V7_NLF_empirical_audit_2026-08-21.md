# SIGNAL-4D V7 + NLF: empirical audit and current candidate

**Audit date:** 2026-08-21

**Primary target:** TR-V2V UBody-F, then UBody, LHand and RHand

**Status:** exploratory candidate; frozen SIGNAL-4D V6 remains the defensible reference method

## 1. Executive decision

NLF should **not** replace SMPLer-X/V6 end to end. Direct NLF is slightly worse
on upper body and catastrophically worse on both hands. NLF is useful as a
complementary upper-body articulation expert: it makes errors on different
frames from V6, provides dense non-parametric observations and provides
per-query uncertainty.

The current V7 research candidate routes selected frames to NLF body
articulation, retains V6 identity and hand articulation, and enforces V6's
global wrist orientation through the SMPL-X kinematic tree. With alpha selected
on calibration only, its full 1,493-frame diagnostic is:

| Method | All ↓ | UBody ↓ | UBody-F ↓ | LHand ↓ | RHand ↓ |
|---|---:|---:|---:|---:|---:|
| Frozen V6 | 42.111 | 26.139 | 29.519 | **11.634** | 11.806 |
| V7 NLF Global-Wrist, α=0.75 | **41.438** | **23.829** | **26.886** | 11.640 | **11.752** |
| Δ V7−V6 | **−0.673** | **−2.310** | **−2.633** | +0.006 | **−0.053** |

All values are millimetres; lower is better. Coverage is 57/57 signs and
1,493/1,493 frames.

This is a material UBody improvement, but it is **not yet a publishable SOTA
claim**. The router learned error differences from the frozen development
partition; test target values are excluded from feature materialization and are
not used for fitting or selection. The test output has nevertheless been
inspected during this research cycle. Section 10 defines the clean confirmation
required before promotion.

## 2. What was and was not changed

All implementation is additive under:

```text
/home/haipd/DexAvatar/signal4d_v7_nlf_fusion/
```

No tracked DexAvatar, V5 or V6 source/prediction was overwritten. V6 remains at:

```text
signal4d/runs/signal4d_v6_final_full1493_20260821/predictions
```

New components:

1. `extract_nlf_observations.py`: deterministic export of parametric and
   non-parametric NLF observations and uncertainties.
2. `evaluate_nlf_direct.py`: direct NLF audit using the author's
   translation-relative region definitions.
3. `nlf_body_router.py`: observable-feature router, SO(3) blending, coherent
   SMPL-X materialization and global-wrist compensation.
4. Unit tests for coordinate conversion, NLF record validation, SO(3) blending
   and global-wrist invariance.

Heavy generated outputs remain local and are ignored by Git; compact
provenance and exact metrics are tracked in:

```text
signal4d_v7_nlf_fusion/artifacts/results_summary.json
signal4d_v7_nlf_fusion/artifacts/sources.lock.json
```

## 3. NLF version and provenance

The experiment uses the official [NLF repository](https://github.com/isarandi/nlf)
and the [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd23a1f3bc89e042d70960b466dc20e8-Abstract-Conference.html).

| Item | Locked value |
|---|---|
| NLF source commit | `f8611fc76ff60f262eb0ab2c6abc3947e42a954a` |
| TorchScript release | `nlf_l_multi_0.3.2.torchscript` |
| Model SHA-256 | `52bee28edb6ea9148691331df87cfc238d7e3d9134dc60104a5aaed282a9ddad` |
| Code license | MIT |
| Official model terms | non-commercial research use |
| NLF detections | 1,493/1,493 |

NLF is not merely another SMPL-X regressor. It can query arbitrary canonical
body points, predicts their 2D/3D positions and uncertainties, and then fits a
parametric model to those non-parametric observations. The official code uses
weights proportional to uncertainty raised to `−1.5` during fitting. The old
DexAvatar NLF adapter discarded these dense observations and uncertainties;
the V7 exporter preserves them.

## 4. Strict evaluation protocol

The authoritative wrapper is:

```text
/home/haipd/DexAvatar/signal4d/evaluate_author_protocol.py
```

V7 was evaluated through:

```text
PYTHONPATH=signal4d/src python -m signal4d.cli.main evaluate-author-sgnify
```

Protocol invariants:

- exact author region arrays and class-0 one-hand rule;
- translation-relative V2V for All, UBody, UBody-F, LHand and RHand;
- author vertex-micro aggregation;
- `frame-policy=manifest` on the locked 1,493-frame manifest;
- strict `PredictionArtifact` hash, frame-id, SMPL-X model and coordinate checks;
- SMPL-X topology of exactly 10,475 vertices;
- no missing or extra frames.

Important hashes:

| Material | SHA-256 |
|---|---|
| Author evaluator source | `e43e12a6659f0604752f0adb8b3c06cfb6ff8d910ed29137036351ee8fc44513` |
| Full manifest | `02e06c946f9400d8eb2b238c0297b07e188912121748db68ee1d66d12ea7c362` |
| SMPL-X model | `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` |
| Full comparison JSON | `52c897aedd1aba47990bd2e2e5217736c71e1d673acc52094eccf3920200e419` |

## 5. Direct replacement experiment

Direct NLF output was converted by the explicit camera transform
`Rx(π): [x, −y, −z]` and evaluated without mixing with V6.

| Method | All ↓ | UBody ↓ | UBody-F ↓ | LHand ↓ | RHand ↓ |
|---|---:|---:|---:|---:|---:|
| V6 | 42.111 | **26.139** | **29.519** | **11.634** | **11.806** |
| NLF v0.3.2 direct | 55.412 | 26.856 | 30.022 | 32.129 | 33.774 |
| Δ NLF−V6 | +13.301 | +0.716 | +0.503 | +20.495 | +21.968 |

Decision: reject wholesale replacement. NLF's body pose diversity is valuable;
its hand pose is not suitable for this pipeline.

Despite its worse aggregate, direct NLF beats V6 on UBody-F for approximately
48.5% of calibration frames, 41.7% of development frames and 45.2% of test
frames. The corresponding oracle headroom is approximately 3.40, 2.50 and
3.20 mm. This justifies selection/fusion rather than replacement.

## 6. V7 candidate mechanics

### 6.1 Observable router

At inference, the router consumes only quantities available without GT:

- NLF mean upper-body and whole-body uncertainty;
- NLF parametric-versus-non-parametric fit residual;
- NLF–V6 disagreement on upper-body vertices and joints;
- NLF and V6 temporal velocity;
- V6 uncertainty and three risk channels;
- detector box score and area.

The exploratory regressor is trained on the 578 development frames to predict
`NLF_error − V6_error`. The routing margin is selected on 260 calibration
frames. Margin 1.0 mm chooses NLF articulation on 603/1,493 frames (40.39%).

The largest feature importances are joint disagreement (0.215), V6 uncertainty
(0.179), UBody disagreement (0.173) and detector area (0.104). These values are
diagnostic, not a claim of causal importance.

### 6.2 Coherent SO(3) materialization

For selected frames, local joint rotations move from V6 toward NLF along an
SO(3) geodesic:

```math
R_j(\alpha)=\exp\!\left(\alpha\log(R^{NLF}_j(R^{V6}_j)^\top)\right)R^{V6}_j.
```

There is no raw vertex splicing. The final 10,475-vertex mesh is generated once
by the licensed SMPL-X layer using a coherent pose vector.

NLF's frame-varying shape is rejected. V6 betas, expression and translation
are retained to preserve signer identity and hand geometry.

### 6.3 Why local-wrist copying failed

The first artifact copied V6 local rotations at joints 20:55. This did not
preserve the hand in camera coordinates: changing shoulder/elbow rotations
changes the global wrist rotation through forward kinematics. At alpha 1.0 it
improved UBody-F by 2.894 mm but regressed LHand/RHand by 1.499/0.892 mm.

The corrected candidate computes the reference global wrist rotations through
the SMPL-X parent tree and solves the new local wrist rotations:

```math
R^{local,new}_{w}=(R^{global,new}_{parent(w)})^\top R^{global,V6}_{w}.
```

All finger local rotations then remain V6. This preserves global hand
orientation even while the upstream arm changes. At alpha 1.0, LHand changes
only +0.011 mm and RHand improves 0.068 mm.

## 7. Calibration-only alpha selection

The selection rule was lexicographic: reject any configuration with a hand
regression above 0.1 mm, then minimize primary UBody-F. All candidates satisfy
strict coverage.

| Calibration method | All ↓ | UBody ↓ | UBody-F ↓ | LHand ↓ | RHand ↓ |
|---|---:|---:|---:|---:|---:|
| V6 | 42.865 | 27.132 | 30.718 | **12.577** | 10.891 |
| α=0.50 | **41.991** | 24.737 | 27.933 | 12.588 | 10.840 |
| **α=0.75 selected** | 42.860 | **24.584** | **27.752** | 12.596 | 10.818 |
| α=1.00 | 44.590 | 25.015 | 28.271 | 12.605 | **10.799** |

Alpha 0.75 was frozen before producing the selected full diagnostic.

## 8. Partition results for the frozen alpha

| Split | Frames | Method | All ↓ | UBody ↓ | UBody-F ↓ | LHand ↓ | RHand ↓ |
|---|---:|---|---:|---:|---:|---:|---:|
| Calibration | 260 | V6 | 42.865 | 27.132 | 30.718 | 12.577 | 10.891 |
|  |  | V7 | 42.860 | 24.584 | 27.752 | 12.596 | 10.818 |
| Development | 578 | V6 | 41.722 | 25.190 | 28.525 | 10.996 | 11.890 |
|  |  | V7 | 40.742 | 22.839 | 25.870 | 11.002 | 11.851 |
| Test | 655 | V6 | 42.162 | 26.597 | 29.940 | 11.985 | 12.094 |
|  |  | V7 | 41.501 | 24.415 | 27.456 | 11.988 | 12.036 |
| Full | 1,493 | V6 | 42.111 | 26.139 | 29.519 | 11.634 | 11.806 |
|  |  | V7 | **41.438** | **23.829** | **26.886** | 11.640 | **11.752** |

The UBody-F reduction remains 2.485 mm on the frozen test partition. This is
evidence that NLF contains complementary signal, but the test has now been
seen and must be labelled exploratory.

## 9. Reproduction paths

```text
NLF observations:
signal4d_v7_nlf_fusion/outputs/nlf_v032_full1493

Selected prediction artifact:
signal4d_v7_nlf_fusion/runs/v7_nlf_body_router_globalwrist_alpha075/predictions

Calibration alpha report:
signal4d_v7_nlf_fusion/reports/author_v7_globalwrist_alpha_selection_calibration

Full official report:
signal4d_v7_nlf_fusion/reports/author_v7_globalwrist_alpha075_full1493

Development/test reports:
signal4d_v7_nlf_fusion/reports/author_v7_globalwrist_alpha075_development
signal4d_v7_nlf_fusion/reports/author_v7_globalwrist_alpha075_test
```

The output directory contains one strict safetensors artifact and metadata file
per sign. Each metadata file records artifact hash, SMPL-X hash, coordinate
convention, frame IDs, alpha and wrist/shape policy.

## 10. Promotion gate and scientific limitations

V7 must not replace frozen V6 in the paper table until all of these hold:

1. Train the routing/risk model without SGNify evaluation GT, preferably on
   SignAvatars plus synthetic corruption, or generate sign-level out-of-fold
   predictions for all 57 signs.
2. Fix feature set, margin, alpha and loss weights before a new confirmatory
   evaluation.
3. Use a new untouched holdout if available. If no new holdout exists, report
   nested sign-level cross-validation and explicitly state that the original
   split was used during development.
4. Require 1,493/1,493 coverage, no protocol fallback and exact hashes.
5. Require at least 2.0 mm UBody-F improvement, no hand regression larger than
   0.1 mm for the body-only stage, and a paired sign-cluster bootstrap interval
   below zero for UBody-F.
6. For a hand contribution, require at least 0.5 mm improvement separately on
   both LHand and RHand; +0.006/−0.053 mm is preservation, not a hand result.
7. Evaluate temporal acceleration/jitter and inspect worst-sign renders; TR-V2V
   alone does not prove temporal or perceptual quality.

Current conclusion: V7 proves a strong, geometrically coherent UBody direction.
V6 remains frozen; V7 is the research candidate to be converted into a
GT-independent method and extended with a dedicated hand module.
