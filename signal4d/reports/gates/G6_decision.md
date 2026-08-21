# G6 reproducibility decision

Decision: **pass**.

- Two independent M1 smoke fits are bit-identical: 1/1 exact artifact hashes,
  maximum tensor absolute error 0, and 0% relative error for every primary metric.
- CPU quality gate at code freeze: Ruff clean and 26/26 tests passed, including
  synthetic end-to-end, completeness regression, evaluator isolation, and frozen
  artifact mutation rejection.
- Release freeze SHA-256:
  `351f36aa64f7615c40a6d0c8f8cfacf219ffb0c3d88dc3a59a1aa922db5748d7`.
- All three confirmatory records report `release_integrity_verified=true`, the
  same release hash, and GT-cache tree SHA-256
  `2fb36eeb8a9b1e40cded516df76d0538ca380fdbc57b34024f8ac02be2237f71`.
- Confirmatory coverage is 24/24 clips and 655/655 frames for M0, legacy/fallback,
  and M1.

The environment, source, tests, configs, manifests, calibration artifact,
canonical observation cache, SMPL-X model, region indices, and preregistration
are included in the release integrity tree. Licensed external bytes are not
redistributed.

## Prospective v5 extension

G6 also passes for the extended-post confirmation. Two independent gate
applications are byte-identical over 112 files. The prospective release freeze
SHA-256 is
`0c5808308b6de3f965fa50f0b05bad21cfe04e9e9161e211f2b03740f825975d`;
an independent post-run verification rehashed 5,550 frozen files with zero
mismatch. The release timestamp precedes GT-cache creation by six seconds.
CPU quality gates at final audit are Ruff clean, 38/38 pytest cases passed, and
all shell entry points pass `bash -n`.
