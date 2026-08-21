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

## Post-lock outcome

The algorithm/configuration lock is commit
`d2d57476c89acc2ce961cfe4b2f876b1b8bc982b`. Commit
`354d438aa57c54a5486f7c88c60389a52017cd4b` subsequently added only the
metadata fields required by the strict evaluator. Selection and every tensor
were verified identical before and after that format-only repair.

The frozen gate selected 1/1,493 frames (`Regen/150`). The unmodified author
protocol with `frame-policy=manifest` produced:

| Method | All | UBody | UBody-F | LHand | RHand |
|---|---:|---:|---:|---:|---:|
| V6 | 42.111111 | 26.139380 | 29.519389 | 11.633903 | 11.805594 |
| V7 zero-training | 42.124616 | 26.144515 | 29.524480 | 11.633841 | 11.805506 |
| V7 − V6 | +0.013505 | +0.005134 | +0.005092 | -0.000062 | -0.000088 |

Thus the preregistered primary success criterion failed. This clean V7 variant
must not be claimed as an accuracy improvement over V6. It is a valid negative
result showing that 2D reprojection plus temporal agreement is insufficient to
identify translation-relative 3D upper-body gains in this monocular setting.
Frozen V6 remains the paper's best leakage-safe author-protocol result. The
earlier label-trained V7 remains exploratory only despite its lower numbers.

Strict outputs:

- Predictions: `signal4d_v7_nlf_fusion/runs/v7_gtfree_2d_temporal_gate_v1_full1493_formatfix_20260821/predictions`
- Gate audit: `signal4d_v7_nlf_fusion/runs/v7_gtfree_2d_temporal_gate_v1_full1493_formatfix_20260821/selection.csv`
- Evaluation: `signal4d_v7_nlf_fusion/reports/author_v7_gtfree_2d_temporal_gate_v1_full1493_formatfix/comparison.json`
- Evaluation SHA-256: `6a8e1886bc9f389143be586fa5dadbd82fe9b9250f2333a75363b6bc0f561651`
