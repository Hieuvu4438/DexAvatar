# Corrected full-rerun preregistration v3

Written after test-v1 reveal but before any corrected test-v3 prediction or metric.
This document invokes the pre-existing software-bug clause; it does not define a
new hypothesis or tune a method from test-v1 results.

## Sole allowed change

Apply RL-S4D-011's uniquely determined WiLoR canonical-right to SMPL-X-left
conversion to left-hand 3D points and rotations. Rebuild cache and uncertainty
calibration from scratch. All other code/configuration behavior, splits, manifest,
source priority, seed, optimizer, factors, bootstrap, metrics, stress slices,
minimum effect and claims remain those in `preregistration_before_test.md`.

The new M1 and legacy/fallback configuration files have been mechanically checked
to be identical to their v1 counterparts after removing only the calibration
artifact path. M0 uses the exact original config file.

## Pre-test corrected development evidence

- Corrected cache: 57 clips; exact frame contract and all tensor contracts match
  v2. Sources 0/2 and right-WiLoR are byte-identical; only left-WiLoR values change.
- Corrected calibration: identical eight model-fit and four conformal clips;
  empirical coverage is 90.0–91.1% in all nine source-region groups.
- Corrected development M1 versus strongest legacy/fallback left-hand TR-V2V:
  -0.7134 mm, paired 95% CI [-1.2571, -0.1946].
- Development body/right changes are +0.0006/+0.0030 mm, far inside the original
  +0.5 mm non-inferiority margin. Velocity, acceleration and jerk all improve.

No threshold or solver choice was selected from these numbers; they verify that
the fixed pipeline remains eligible for the mandatory full rerun.

## Unchanged gates

1. Baseline and candidate coverage must be 100%.
2. Left-hand delta point estimate must be at most -0.5 mm and paired 95% CI upper
   bound below zero.
3. Upper-body/right-hand paired CI upper bounds must be below +0.5 mm.
4. Velocity, acceleration and jerk point estimates must not regress by more than 2%.
5. Corrected smoke predictions must reproduce within 5%; bit equality is targeted.
6. Every test method (M0, legacy-full/fallback, M1) must be regenerated and evaluated
   from scratch under the same corrected release freeze and GT-cache integrity tree.

If all gates pass, the maximum claim remains “new best result on the frozen
SIGNAL-4D clean SGNify protocol against the strongest recomputed same-protocol
baseline.” It is still not a published 2,872-frame leaderboard claim.
