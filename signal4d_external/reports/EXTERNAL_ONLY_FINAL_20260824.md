# SIGNAL4D external-only final report (2026-08-24)

## Decision

Keep `full1493_wilor_clipnorm_v1` as the final external-only result.  It is the
best leakage-clean candidate measured by the frozen author evaluator.  None of
the later candidates is promoted, and no policy was retuned after reading a
target metric.

| Method | All | Upper body | UBody-F | Left hand | Right hand | Decision |
|---|---:|---:|---:|---:|---:|---|
| DexAvatar | 42.5867 | 26.4560 | 29.9074 | 13.5735 | 12.9271 | baseline |
| External-only V1 | **42.2423** | **26.2236** | **29.6196** | **12.8102** | **12.1148** | **keep** |
| Hand V2 | 42.2445 | 26.2260 | 29.6215 | 12.8323 | 12.2831 | reject |
| Arm BA V4 | 42.4021 | 26.4136 | 29.8672 | 13.2348 | 12.3247 | reject |
| Historical SIGNAL4D V6 | 42.1116 | 26.1394 | 29.5197 | 11.6339 | 11.8056 | contaminated reference only |

Values are millimetres over the frozen 57-clip/1,493-frame manifest using the
repository wrapper around the original author evaluator.  Historical V6 is
not eligible because its lineage includes SGNify-based selection/calibration.
V1 remains 0.1307 mm behind V6 on All, while improving DexAvatar by 0.3444 mm.

## Strategies tested without SGNify training or selection

1. **Hand V2 rank/soft residual.** It passed its external pose-space gate, but
   the one-time target reveal was 0.0022 mm worse than V1 on All and worse on
   both hands.  It was frozen and rejected without target retuning.
2. **NLF V2 uncertainty router.** 28,672 external observations were extracted
   on GPU.  The router selected no beneficial NLF replacements on held-out
   How2Sign and failed before target materialization.
3. **Sign DPoser-X body prior.** The checkpoint was trained for 30,000 steps on
   1,449 H2S training poses, not SGNify.  Direct Tweedie projection reduced
   neither the selection nor held-out gate error: gains were -0.00536 and
   -0.00083 degrees.  It was rejected before target inference.
4. **Sign VQ-VAE hand prior.** The checkpoint used 378 H2S training poses and
   no SGNify data.  The 64-entry codebook was effectively collapsed
   (preflight perplexity about 2).  Both hand gates failed with negative gains,
   so it was rejected before target inference.
5. **Arm-only temporal bundle adjustment V4.** This preserved both 45-D hand
   poses bit-exactly and optimized only shoulder, elbow, and wrist rotations.
   Its preregistered policy passed held-out How2Sign strongly (1.3114-degree
   pose gain and 40.56% reprojection gain) and all 57 target clips passed the
   target-free reprojection acceptance gate.  Nevertheless, its frozen target
   result was worse than V1 by 0.1598 mm on All, so it was rejected.

The V4 failure is informative: the external pseudo-targets were themselves
produced using 2D temporal fitting.  A candidate from the same optimization
family can therefore look excellent on external pose/reprojection gates while
moving away from true target-domain 3D geometry.  This makes further tuning of
2D fitting strength after the reveal methodologically invalid and unlikely to
close the remaining V6 gap reliably.

## Leakage and freeze audit

- SGNify target reads during training, model selection, and policy selection:
  zero for every eligible candidate.
- Target ground-truth metrics were used only for one-time developmental
  reveals after each artifact and policy had been frozen.
- V4 audit: 57 clips, 1,493 frames, 2,986 hand arrays exact to V1, 22,395
  non-arm body joints exact, 16,423 other fields exact, and zero target reads.
- V4 result tree SHA-256:
  `689f3e634493ff78116b573117476438399b26d15b6ce5b67505943cd92425c1`.
- V4 mesh tree SHA-256:
  `5241df5a71361074dd61a3f6c76bb05e3fdebeeed4215caed23112e56e59756d`.

## Frozen artifacts

- Final V1 root: `outputs/signal4d_external/full1493_wilor_clipnorm_v1`
- V1 run manifest SHA-256:
  `37a0e54ef0e1ecbc6337292d03cf1674407ea449f82ae54a74861426e421a682`
- V1 OBJ registry SHA-256:
  `da2878d01047890dfbe167ab12fda3c2c7c2cf3b83dfd82709aa6f7dacd92748`
- V1 evaluation report SHA-256:
  `2037d545f7851097375c1ffa967ad2147856ee4fe223c810ffa056faf330c3fb`
- V4 run manifest SHA-256:
  `eac3ea065d044402940b8aef94032ec8cad34787d6f2b960fbdb9504034748ea`
- V4 render manifest SHA-256:
  `e1fed3b3dfae2916606432b0f216e31f71aedbca488ca909b84b488f37baea41`
- V4 freeze audit SHA-256:
  `01f35a9bc98d97b70b681d21a035b652ff7c54d3964fc97ea1fd6dc110036b20`
- V4 OBJ registry SHA-256:
  `7ba3014bc6a1c682f61404110effdb32c180e1f0f77bbef5108389c4da912212`
- V4 evaluation report SHA-256:
  `2d9116ef198338b42cb1d69d13726ac883fcb7df3ae17d5204a2923875fb2018`

The V4 evaluation was rerun into a separate directory.  The two
`comparison.json` files are byte-identical.  The external test suite passes
22/22 tests.  GPU was used for observation extraction, fitting, and mesh
rendering; all owned processes were pinned to CPUs 0--4, keeping aggregate
owned CPU capacity below 500% and therefore below the requested 600% limit.

## Implementation corrections

The DPoser-X adapter's repository-root calculation was corrected to resolve
the in-tree `DPoser-X` checkout, and the sign fitting launcher now defaults to
the sign-specific `sign_timefc.py` configuration.  These fixes remain useful
even though the tested DPoser-X projection policy was not promoted.
