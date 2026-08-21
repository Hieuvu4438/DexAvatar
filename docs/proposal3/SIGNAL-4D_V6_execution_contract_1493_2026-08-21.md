# SIGNAL-4D V6: execution contract on 1,493 frames

Date: 2026-08-21  
Frozen parent: `signal4d-v5-frozen-20260821` at `ee519aa`  
Implementation branch: `main`  
Primary evaluator: `signal4d/evaluate_author_protocol.py`

## 1. Immutable boundary

V6 is a post-V5 refinement. It does not replace SMPLer-X, WiLoR, HaMeR,
SignBPoser or SignHPoser, and it does not modify the frozen V5 solver or its
artifacts. Every V6 experiment starts from the exact V5 prediction with the
same 57 clip IDs, 1,493 frame IDs, SMPL-X topology, shape and translation.

The new code lives only in `signal4d.extensions.v6_uqdiff`. Configurations live
in `signal4d/configs/v6_uqdiff`. A V5 regression test remains part of every
stage gate.

## 2. Method contract

The core contribution is **UQ-DiffPrior refinement**:

1. DPoser-X produces a stop-gradient whole-body denoised pose target. This is a
   learned cross-part plausibility factor, not a replacement image expert.
2. The target is pulled back from axis-angle space to a geodesic SO(3) loss.
3. Only named bilateral upper-body SMPL-X joints are open. Global orientation,
   translation, shape, legs, head and face remain frozen.
4. Calibrated expert uncertainty controls prior strength; detected motion
   changes suppress smoothing instead of erasing sign transitions.
5. A body-wrist-MCP seam factor couples arm orientation to the frozen hand
   evidence. Hand pose is opened only in the designated hand ablation.
6. A GT-free safety gate retains V5 wherever the V6 objective, uncertainty or
   rotation trust region does not improve. Ground truth is never read during
   fitting or gating.

## 3. Controlled stages and commit boundaries

| Stage | New component | Required evidence before next commit |
|---|---|---|
| V6-C0 | config schema, named joint map, provenance registry | unit tests and frozen V5 tests pass |
| V6-C1 | checkpoint bridge and exact DPoser normalizer | checksum validation, CPU shape test, CUDA one-step smoke |
| D0 | bilateral upper body, no diffusion | one-clip deterministic smoke and no frozen-parameter drift |
| D1 | upstream Euclidean DPoser factor | reproduce upstream one-step formula at `t=0.12..0.08` |
| D2 | geodesic SO(3) pullback | zero/known-angle tests, finite gradients near identity/pi |
| D3 | calibrated uncertainty weighting | monotonic weight tests and ablation table |
| D4 | change-aware temporal/prior suppression | transition preservation tests and dynamics report |
| D5 | wrist-MCP seam plus GT-free safe gate | fallback identity test and full candidate run |
| V6-R | 1,493-frame release | strict OBJ, author metrics, visualizations, hashes and report |

Each row is committed separately and pushed to `main`. Experimental outputs,
third-party checkpoints, caches and logs remain ignored; compact configuration,
hash registry, aggregate metrics and release metadata are committed.

## 4. Experiment matrix

`H0` is the immutable V5 parent. `D0` opens the bilateral arm chain without a
diffusion factor. `D1` uses DPoser-X's Euclidean normalized-coordinate loss.
`D2` replaces only that distance with SO(3) geodesics. `D3` adds uncertainty
weighting, `D4` adds change awareness, and `D5` adds the seam and gate. Optional
hand optimization is reported separately and cannot silently enter the core
method.

For every stage, the experiment record contains config hash, source commit,
checkpoint hashes, seed, device, clip/frame keys, runtime, peak memory and exact
parent-artifact hashes.

## 5. Full-run acceptance matrix

Coverage is a hard gate: exactly 57 signs and exactly 1,493 keys, one prediction
and one strict-topology OBJ per key, with no duplicate or missing frame. The
primary score is vertex-micro `tr_upper_body_minus_face_mm` from the attached
author evaluator. Secondary scores are `tr_v2v_mm`, upper body, left hand and
right hand; temporal velocity/acceleration and per-sign paired deltas are
reported as diagnostics.

The promotion target relative to frozen V5 is:

- UBody(-F) mean improvement of at least 0.15 mm;
- paired sign-bootstrap 95% upper bound below zero;
- neither hand has a paired 95% upper bound above +0.25 mm;
- all-body regression no larger than +0.10 mm;
- temporal diagnostic regression no larger than 2%.

If no candidate clears all gates, V5 remains the release method and the V6
result is reported honestly as an ablation, not relabeled as an improvement.

