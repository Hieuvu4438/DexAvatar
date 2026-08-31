# External-only sign-DPoser-X body V3 preregistration

## Scope and leakage boundary

This experiment tests whether the sign-trained DPoser-X score field can make a
small, conservative correction to the frozen SMPLer-X body initializer.  The
DPoser-X checkpoint was trained for 30,000 steps on 1,449 How2Sign `train`
poses.  SGNify contributes zero training, selection, or gate observations.

Hyperparameters are selected only on the 70 official How2Sign `val` clips from
signers 01 and 02 contained in the already frozen 192-clip signer manifest.
The official-split filter uses cache metadata before any target array is read;
the other 122 `train` clips are excluded.  After selection, the policy is
frozen and tested once on 220 official How2Sign `test` clips from held-out
signer 10.  The test split does not choose any parameter.

## Frozen candidate family

For each initializer pose, the sign prior evaluates a zero-noise sub-VP state
`x_t = alpha_t * x`.  Tweedie's formula produces a deterministic projected
pose.  The correction is blended on SO(3), not linearly in axis-angle space.
Only the 21 SMPL-X body joints are eligible; global orientation, identity,
camera, face, and hands remain untouched.

The complete frozen grid is:

- diffusion time: `0.01, 0.025, 0.05, 0.075, 0.1, 0.125`;
- SO(3) blend: `0.05, 0.1, 0.25, 0.5, 1.0`;
- observable movement coverage: `10%, 25%, 50%, 100%`;
- movement selection direction: lowest or highest correction magnitude.

The validation objective is mean arm-weighted upper-body local-rotation gain.
An eligible policy must have positive mean gain, at least 9% coverage,
non-positive median delta, no more than 0.5 degrees at the 95th regression
percentile, and no validation signer worse than -0.1 degrees.

## Promotion gate

The frozen policy passes only if the How2Sign test gate has positive mean gain,
at least 5% coverage, non-positive median delta, at most 0.5 degrees 95th
percentile regression, and no signer worse than -0.1 degrees.  A failed policy
must not be applied to or tuned on SGNify.
