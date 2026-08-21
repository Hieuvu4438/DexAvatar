# Prospective extended-post preregistration v5

Status at authorship: prospective ground-truth values have not been cached,
decoded, evaluated, summarized, or used for model selection. The manifest builder
checked only file names/existence. Its recorded contract is 56 clips, 769 frames,
SHA-256 `33825a3f1ac8aa6d063f90bc12c8061ed60680267615b6d76cbe1e8cee625b32`,
with `ground_truth_values_read: false`.

## Scope and limitation

This is a temporal post-segment holdout: every selected frame is strictly after
the central segment used for historical development. The 56 clip/sign identities
overlap historical data and signer IDs are unavailable. It is therefore a
prospective temporal endpoint, not an independent-signer or published 2,872-frame
leaderboard endpoint.

## Frozen analysis population

- Manifest: `sgnify_extended_post_test_v1.jsonl`.
- All 769 declared frames are required for both methods; no intersection-only
  evaluation or post-hoc frame removal is permitted.
- Primary aggregation unit: equal-weight clip macro over the 56 clips.
- Alignment and indices: pelvis-root translation alignment (`TR-V2V`), pinned
  SMPL-X neutral model, and the repository's frozen upper/left/right vertex sets.
- Paired uncertainty: 10,000 clip-level bootstrap replicates, seed 12345.

## Comparator and candidate

The primary comparator is the pre-existing balanced A1 stack: the original
DexAvatar ensemble fit, with an original-HaMeR A0 fit fallback where the ensemble
produces no valid frame. An availability-only audit before any prospective GT
read found that HaMeR can also lack a usable hand detection on tail frames; raw
SMPLer-X A0 is therefore the terminal fallback only where neither fitted source
exists. The exact per-frame counts must be reported. Legacy code and configurations
are read-only. Every parameter file is decoded per frame with the same pinned
SMPL-X asset.

The candidate is SIGNAL-4D M1-v5 with the corrected left-hand coordinate
conversion and A1 initializer, followed by a GT-free multi-hypothesis gate over
geodesic scales 1.0, 1.5 and 3.0. Gate features contain only predictions,
observations, uncertainty and factor diagnostics. Gate forests are trained on the
already revealed central development/test partitions using grouped out-of-fold
selection; the within-clip transition cost is fixed at 8 mm. No prospective
labels may be supplied to fitting or gate inference.

## Confirmatory endpoint and gates

The sole superiority endpoint is paired clip-macro left-hand TR-V2V in mm,
candidate minus A1 (lower is better). The result qualifies only if all conditions
hold:

1. Both methods cover exactly 56 clips and 769 frames with finite artifacts.
2. Left-hand delta is at most -0.5 mm and its paired 95% bootstrap CI upper bound
   is below zero.
3. Upper-body and right-hand delta 95% CI upper bounds are each below +0.5 mm.
4. Velocity, acceleration and jerk point estimates do not regress by more than
   2% relative to A1; improvement is negative error delta.
5. A frozen smoke subset reproduces within 5%; bit equality is the target.
6. Source, configs, manifests, calibration, gate artifacts, baseline/candidate
   predictions and model hashes are frozen before the GT cache is created.

If any condition fails, the prospective claim fails. The endpoint must not be
retuned or re-tested on the same labels. Contact/semantic claims are out of scope
because no trustworthy contact or semantic annotations exist for this endpoint.

If every gate passes, the maximum wording is: “new best result on the prospective
SIGNAL-4D extended-post SGNify endpoint against the strongest pre-frozen,
recomputed same-protocol A1 baseline.” It is not a claim against a published
external leaderboard or unseen signers.
