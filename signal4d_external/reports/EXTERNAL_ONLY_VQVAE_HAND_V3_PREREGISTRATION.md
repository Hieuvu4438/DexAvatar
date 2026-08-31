# External-only sign-VQ-VAE hand V3 preregistration

## Scope

This experiment tests a sign-hand manifold projection without SGNify training,
selection, or gate reads.  The frozen VQ-VAE checkpoint was trained from 378
How2Sign SMPLer-X hand poses, with 43 external validation poses used by its
original trainer.  The downstream policy is selected on 70 official How2Sign
`val` clips from signers 01/02 and gated once on 220 official How2Sign `test`
clips from held-out signer 10.

## Candidate family

Each 15-joint hand is repeated across the VQ-VAE temporal receptive field,
quantized and reconstructed.  The reconstruction is blended with the frozen
initializer on SO(3).  Left and right hands are calibrated independently; body,
global orientation, wrists, identity, camera, and face are not changed.

The frozen grid is:

- SO(3) blend: `0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0`;
- observable movement coverage: `5%, 10%, 25%, 50%, 100%`;
- movement direction: lowest or highest reconstruction displacement.

Validation eligibility requires positive mean local-rotation gain, at least
4.5% coverage, non-positive median delta, no more than 0.5 degrees at the 95th
regression percentile, and no validation signer below -0.1 degrees.

## Promotion gate

A hand region passes only if the frozen policy remains positive on How2Sign
test, covers at least 2.5% of valid frames, has non-positive median delta, keeps
the 95th regression percentile within 0.5 degrees, and does not regress the
held-out signer by more than 0.1 degrees.  Only individually passing regions
may be materialized; a failed region remains exactly V1.
