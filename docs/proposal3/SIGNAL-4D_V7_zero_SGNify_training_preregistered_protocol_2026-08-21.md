# SIGNAL-4D V7: preregistered zero-SGNify-training protocol

Date locked: 2026-08-21 (Asia/Ho_Chi_Minh)

## Purpose and claim boundary

This document locks the next V7 evaluation before the author-protocol ground
truth is accessed. The earlier `V7_NLFBodyRouter` remains an exploratory
diagnostic because it trained a router on SGNify development targets and chose
its margin/alpha using SGNify calibration targets. It must not be reported as
the paper's clean primary result.

The method locked here is `SIGNAL4D_V7_GTFree2DTemporalGate`. It has zero
trained parameters and never reads `data/smplx_gt`, author segment labels, or
author metric outputs during inference. SGNify ground truth is reserved for a
single evaluation after the implementation and configuration are committed.

Historical benchmark inspection cannot literally be undone. Therefore this
run removes training/tuning leakage, but the paper must still disclose that V6
and exploratory V7 development had previously inspected this benchmark. A new
unseen dataset remains the strongest final confirmation.

## Frozen candidate

For every one of the 1,493 frames, construct one candidate by moving exactly
halfway (`alpha = 0.5`) on each SO(3) geodesic from frozen V6 rotations toward
official NLF v0.3.2 rotations. The midpoint is a symmetry choice, not a value
selected on SGNify metrics.

The candidate preserves:

- frozen V6/DexAvatar betas and translation;
- frozen V6 expression, jaw, eyes, and local finger rotations;
- the exact V6 global orientations of both wrists, compensated after changing
  their upstream shoulder/elbow chain.

This isolates the intended contribution to torso/arm articulation and avoids
using NLF as a replacement body or hand expert.

## Frozen GT-free gate

The only observed joints are the six detector-native upper-body joints:
left/right shoulder, elbow, and wrist. Synthetic torso fallback joints are
excluded. Each state is projected with the per-frame perspective intrinsics.

For each sign, a two-state dynamic program chooses either frozen V6 or the
fixed candidate. Its unary term is confidence-weighted 2D reprojection error,
normalized by image diagonal. Its transition term, with fixed weight 1.0, is
the error between predicted and observed inter-frame 2D displacement in the
same units. Exact ties select V6. At least two valid arm joints are required.

The NLF candidate is admissible only when at least one third of its six arm
joints have official NLF uncertainty below 250 mm. No SGNify target, class,
segment label, V2V error, fitted regressor, learned weight, calibration margin,
or evaluation result enters the gate.

The machine-readable frozen configuration is:

`signal4d_v7_nlf_fusion/configs/v7_gtfree_2d_temporal_gate_v1.json`

The inference implementation is:

`signal4d_v7_nlf_fusion/nlf_gtfree_2d_temporal_gate.py`

## Evaluation procedure locked before execution

1. Commit this protocol, configuration, implementation, and unit tests.
2. Generate exactly 1,493 prediction frames in a new append-only run folder.
3. Verify frame-set identity against the frozen full manifest and V6.
4. Invoke the unmodified author-protocol adapter exactly once on V7 clean.
5. Report All, UBody, UBody-F, LHand, and RHand for V6 and clean V7, including
   selection rate and artifact hashes. Do not choose a new alpha, threshold, or
   gate variant after seeing the result.

Primary success criterion is lower UBody-F than frozen V6 (29.5194 mm).
Secondary constraints are lower UBody and no material hand regression. A
failure is retained and reported as a valid negative result; it is not tuned
away on this benchmark.
