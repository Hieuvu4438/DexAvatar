# External-only arm bundle-adjustment V4 preregistration

## Rationale and scope

The frozen V1 initializer already supplies strong WiLoR local finger poses.
V4 therefore keeps both 45-D hand poses bit-exact and adjusts only the six
SMPL-X body joints for shoulders, elbows, and wrists.  The loss uses observed
2D joints from both arms and both complete hands, allowing arm rotations to
correct global hand placement and orientation without replacing finger pose.

No SGNify target, author mesh, or evaluation metric is read during selection.
The candidate is selected on 70 official How2Sign `val` clips from signers
01/02 and frozen before a one-time gate on 220 official How2Sign `test` clips
from signer 10.

## Frozen optimizer and policy grid

The optimizer uses 30 Adam iterations, learning rate 0.03, a 12-degree bounded
correction, anchor weight 0.02, velocity weight 0.10, and acceleration weight
0.05.  The fixed post-fit grid is:

- SO(3) blend: `0.1, 0.25, 0.5, 0.75, 1.0`;
- observable arm-movement coverage: `10%, 25%, 50%, 100%`;
- movement direction: lowest or highest mean arm correction.

Validation eligibility requires positive mean arm-weighted upper-body
rotation gain, at least 9% coverage, non-positive median delta, no more than
0.5 degrees at the 95th regression percentile, and no signer below -0.1
degrees.

## Promotion gate

The frozen policy passes only with positive How2Sign-test gain, at least 5%
coverage, non-positive median delta, at most 0.5 degrees 95th-percentile
regression, and no held-out signer loss below -0.1 degrees.  Only a passing
policy may be run on target images.  At target inference, each clip must reduce
the observed 2D arm/hand reprojection loss by at least 0.5%; otherwise every
frame in that clip falls back bit-exactly to V1 before rendering.
