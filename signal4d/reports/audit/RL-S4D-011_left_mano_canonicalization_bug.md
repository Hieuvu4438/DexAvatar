# RL-S4D-011 — left-MANO canonicalization bug

Date: 2026-08-19. Discovered after the first clean-test reveal while auditing why
the left-hand effect was heterogeneous. This is a software/convention bug, not a
test-driven hyperparameter change.

## Independent evidence

The read-only legacy parser maps WiLoR/HaMeR's canonical-right representation to
the SMPL-X left-hand convention in two places:

- 3D MANO points: flip the x coordinate before camera placement;
- MANO axis-angle: `[x, y, z] -> [x, -y, -z]`, equivalently conjugating each
  rotation matrix by `diag(-1, 1, 1)`.

The SIGNAL-4D cache v2 mirrored only left-hand 2D crop x coordinates. It inserted
canonical-right 3D points and rotations directly into the SMPL-X left hand. This
violates the adapter coordinate contract and affects M0 observations and M1 soft
factors. The code-level fix is uniquely determined by the coordinate convention.

## Validation before full rerun

- Added analytical unit tests for point reflection and SO(3) conjugation.
- Ruff passes and 28/28 tests pass.
- On the quarantined `Ablehnen` smoke clip, body TR-V2V is unchanged
  (22.8566 mm), while raw M0 left-hand TR-V2V changes from 31.1254 to 20.2439 mm.
  This smoke clip is development-only and was already excluded from final
  reporting.

## Mandatory response

The original preregistration states that a software-only bug requires a complete
rerun of every test method. Therefore:

1. Preserve v1 cache, calibration, freeze, predictions, and report as an invalidated
   historical audit trail; never overwrite them.
2. Build a new three-source cache and retrain calibration with the identical split,
   seed, epochs, learning rate, conformal split, and sigma bounds.
3. Keep every solver weight, source priority, optimizer setting, metric, gate,
   manifest, bootstrap seed, and stress slice unchanged.
4. Run M0, strongest legacy/fallback, and M1 from scratch under one new release
   freeze. No per-method reuse of v1 predictions is allowed.
5. Recompute all paired bootstrap intervals. Claim SOTA only if the original gates
   pass on the complete corrected rerun.

No test v1 metric is used to choose a new threshold or method configuration.
