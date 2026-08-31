# External-only hand V2 preregistration

Frozen before any V2H target-label evaluation on 2026-08-24.

## Objective and hypothesis

External-only V1 is better than the frozen WiLoR initializer but its absolute
benefit probabilities shift sharply between How2Sign and the unlabeled target
cache. The target-side probability distribution was inspected before this
preregistration, but no SGNify target pose, mesh, vertex error, or per-frame
success label was read for V2H design or selection.

The preregistered hypothesis is narrower than probability calibration: the
within-dataset *rank* of the V1 benefit score transports better than its
absolute scale. Applying a smaller SO(3) residual to a coverage fixed entirely
on external validation may therefore retain helpful hand corrections without
the near-total abstention caused by V1's absolute threshold.

This lane refines finger rotations only. The NLF V2 lane remains responsible
for upper-body changes. The eventual combined result must preserve the V2H
hands when NLF body rotations are fused.

## Frozen inputs

| Artifact | SHA-256 |
|---|---|
| V1 checkpoint | `09ca34e42e88b550af3b046f1f35bff30ae0c66aff65b9cfba430594ccdb90bf` |
| V1 config | `ef67067e73690d7c39005c042211c8373f0ecaa9d08e7df364ffa33896226e35` |
| signer-disjoint validation manifest | `017bfaa3511666017f9797c16406a7ecb249bd269137e9afd3a26c57ba3a46f3` |
| signer-disjoint calibration manifest | `8d07eb536a8f1520f817c45c1416e30fe37510c1d820a1426280ac670d2149c2` |
| frozen external-only V1 run manifest | `37a0e54ef0e1ecbc6337292d03cf1674407ea449f82ae54a74861426e421a682` |

The checkpoint was trained only on How2Sign signers 03, 05, and 08. V2H uses
How2Sign signers 01 and 02 for policy selection and signers 04, 09, and 11 for
the final external gate. Source groups and signer identities are disjoint.

## Frozen protocol

1. For every manifest clip, choose one contiguous window of at most 64 frames.
   The window maximizes summed cached hand activity; this feature is observable
   and target-free. Earliest start wins a tie.
2. Run the frozen V1 checkpoint without its absolute abstention thresholds.
   Cache initializer matrices, raw candidate matrices, benefit probabilities,
   and external pseudo-target rotations. The cache is append-only/resumable and
   records hashes for every source and prediction file.
3. Independently for left and right hands, select on validation over the exact
   grid:

   - residual scale alpha: `{0.25, 0.50, 0.75, 1.00}`;
   - global frame coverage: `{0.10, 0.25, 0.50, 0.75, 1.00}`;
   - centered probability-smoothing half-window in seconds:
     `{0, 2/15, 4/15}`.

4. Selection uses macro-average clip geodesic rotation error over the 15 local
   finger joints. Frames are ranked globally after time-domain smoothing and
   exactly `round_half_up(coverage * frame_count)` are selected. Stable global
   frame index resolves score ties. Hyperparameter ties prefer lower coverage,
   lower alpha, then a shorter smoothing window.
   Coverage is measured only over target-free eligible frames: at least 50% of
   the hand keypoints must be valid and mean cached reliability must be at
   least 0.20. Ineligible frames always retain V1 exactly.
5. Apply the selected policy unchanged on calibration. A hand passes only if:

   - mean calibration gain is strictly positive;
   - at least 10% of valid frames are selected; and
   - the worst calibration signer gain is at least -0.25 degrees.

   Both hands must pass. Calibration is gate-only and must not retune the grid.
6. If the external gate fails, do not materialize or evaluate V2H on SGNify.
   If it passes, run target inference once. Target inference may rank the V1
   benefit probabilities from all 1,493 unlabeled target frames to enforce the
   externally frozen coverage. It must not read SGNify targets or evaluation
   meshes. Unselected frames remain bit-for-bit equal to external-only V1.
7. After materialization, freeze hashes and perform one target evaluation. Do
   not change V2H after seeing its SGNify metrics; a later method must receive a
   new name and preregistration.

## Target-use disclosure

This is an external-only training and selection protocol, but it is
transductive at inference: it uses the empirical rank of **unlabeled target
benefit scores**. It does not use target ground truth, errors, labels, author
fits, or evaluator-owned meshes. The run manifest must report this distinction.

## Frozen implementation

| File | SHA-256 |
|---|---|
| `hand_v2_core.py` | `93a1c47616a2eb9e230a69a8199ff2a05857d306a189d5d3ccf8190cb9eddbb0` |
| `extract_hand_v2_predictions.py` | `7434bfea041bbe0c36a110b202d31a85dc017788cf240e1a10fd3144b85b081f` |
| `calibrate_hand_v2.py` | `5023908e657cc4bca5fd01d307ac72787f39907613e7d206767982ca327357ca` |
| `materialize_hand_v2.py` | `7307612f7783f6925659d5cc8efa1362e324d706731ac2af9a03ca6cfd681dea` |

Tests after the pre-execution temporal/eligibility audit: `19 passed`; Ruff:
`All checks passed`; compileall: passed.

### Pre-execution temporal-contract amendment

Before prediction-cache extraction or V2H target-label evaluation, static audit
found that the selected How2Sign caches can be temporally sparse while the
target cache has a fixed 0.08-second spacing. Index-radius smoothing would therefore
represent different physical durations across domains. The frozen policy grid
above uses cache timestamps and smoothing windows in seconds instead. The
alpha and coverage grids, split roles, gate, and one-evaluation rule are
unchanged.

The same unlabeled audit found a large visibility shift specifically for the
left hand: 99.9% of external validation frames but only 76.0% of target frames
meet the frozen validity/reliability rule (right hand: 100% versus 97.9%). This
motivates the fail-closed eligibility mask above; no target error or ground
truth was inspected.

## Research rationale

The lane follows the literature's recurring separation of a strong parametric
initializer from constrained refinement: ScoreHMR uses observation-guided
optimization around an image-conditioned prior; TokenHMR limits implausible
pose regression through a learned discrete pose prior; WiLoR supplies the
strong in-the-wild hand initializer; and Dyn-HaMR emphasizes temporal reasoning
for interacting hands. V2H makes the conservative adaptation available here:
retain WiLoR/V1 as the fallback, move only along the learned SO(3) residual, and
let independent external signers determine intervention strength and coverage.
