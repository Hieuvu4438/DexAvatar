## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED ON CALIBRATION; FULL-1493 PENDING
- Data scope: frozen 12-sign calibration split, 260/260 manifest frames
- Primary endpoint: author vertex-micro TR UBody(-F), millimetres; lower is better
- Inference GT access: none

# SIGNAL4D V6 frozen calibration decision

## Decision

Freeze `final_v6_uqdiff_w3_safe_gate.yaml` for the one-time 1,493-frame run.
The selected candidate is W3: geodesic DPoser-X whole-body diffusion pullback
with uncertainty/change-aware weighting and the wrist--MCP seam. It keeps every
V5 expert fixed and opens only the eight named upper-limb SMPL-X body joints.

The inference gate accepts a W3 frame only when all of the following GT-free
conditions hold:

1. candidate internal observation objective plus `0.0075` is lower than the V5
   objective;
2. maximum open-joint SO(3) displacement is at most `0.35 rad`;
3. upper-limb uncertainty ratio is at most `1.25`.

No temporal rejection dilation is used (`transition_radius=0`). Temporal
coherence remains inside the candidate optimizer through its translation and
SO(3) temporal factors; the gate independently falls back to the exact V5
artifact at rejected frames.

## Evidence used to freeze the choice

| Candidate | DPoser geodesic weight | Seam weight | UBody(-F) delta vs V5 (mm) | TR-all delta | LHand delta | RHand delta |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 0.01 | 0.00 | -0.093853 | -0.018325 | -0.071878 | -0.071114 |
| W2 | 0.05 | 0.00 | -0.094027 | -0.018735 | -0.074893 | -0.071685 |
| W3 | 0.05 | 0.01 | -0.094052 | -0.018741 | -0.074935 | -0.071694 |
| W3 + frozen rule gate | 0.05 | 0.01 | **-0.1719996** | **-0.073578** | **-0.140883** | **-0.093198** |

The final gated row is an actual rerun through `evaluate_author_protocol.py`,
not a metric interpolation. It selects 76/260 frames (29.23%) and improves all
four reported endpoints on calibration.

## Rejected alternatives

- Stronger optimizer settings regressed TR-all by about `+0.07 mm`; therefore
  raw optimizer strength was rejected as the explanation or remedy.
- The learned ExtraTrees gate achieved OOF clip-macro `-0.047746 mm` and OOF
  frame-micro `-0.064748 mm`, with clip-bootstrap 95% CI
  `[-0.118600, +0.023091] mm`. Because its interval crosses zero and its point
  estimate is below the registered `-0.15 mm` promotion target, it is retained
  as a negative-control artifact rather than the final gate.
- The objective-margin threshold was selected on calibration and is now frozen.
  It must not be changed after examining the full-1,493 author metrics.

## Claim boundary

Calibration establishes only a hyperparameter-selection result. It is not an
independent test estimate and must not be called SOTA. Promotion requires the
frozen full-manifest run, strict OBJ round-trip evaluation, coverage/topology
checks, and a sign-level uncertainty interval.
