# SIGNAL-4D frozen confirmatory report

## Protocol

- Clean untouched test split: 24 clips / 655 explicit frames at 15 fps.
- Primary aggregation: equal-weight clip macro.
- Primary metric: per-frame, per-region translation-aligned V2V (mm).
- Release freeze SHA-256: `351f36aa64f7615c40a6d0c8f8cfacf219ffb0c3d88dc3a59a1aa922db5748d7`.
- Paired clip bootstrap: 10,000 replicates, seed 20260819.
- Coverage: 100% for every method; no frames were dropped.

## Frozen results

| Method | Body TR-V2V | Left TR-V2V | Right TR-V2V | Velocity | Acceleration | Jerk |
|---|---:|---:|---:|---:|---:|---:|
| Raw M0 hybrid | 24.0083 | 36.7081 | 21.5020 | 6.4468 | 141.7560 | 3780.8072 |
| Legacy-full + M0 fallback | 25.6275 | 23.2527 | 12.5523 | 6.4592 | 142.0646 | 3793.9400 |
| SIGNAL-4D M1 | 25.6208 | 23.0484 | 12.5512 | 6.4214 | 140.4951 | 3754.9889 |

M1 versus the preregistered strongest legacy/fallback control:

| Endpoint | Delta (M1-baseline) | Paired 95% CI | Gate |
|---|---:|---:|---|
| Upper-body TR-V2V (mm) | -0.0068 | [-0.0191, +0.0045] | non-inferiority pass |
| Left-hand TR-V2V (mm) | -0.2043 | [-0.6908, +0.2692] | superiority/effect fail |
| Right-hand TR-V2V (mm) | -0.0011 | [-0.0040, +0.0015] | non-inferiority pass |
| Velocity error | -0.0378 | [-0.0463, -0.0297] | pass |
| Acceleration error | -1.5695 | [-1.8924, -1.2581] | pass |
| Jerk error | -38.9511 | [-49.4342, -28.9007] | pass |
| Left-hand AURC | -0.2248 | [-0.6725, +0.2069] | inconclusive |

The left-hand point improvement is 0.88%, below the preregistered 0.5 mm
practical threshold, and its interval crosses zero. Against raw M0, M1 greatly
improves left/right hands (-13.6597/-8.9508 mm with intervals below zero) but has
an inconclusive +1.6124 mm body difference. Thus no one method dominates every
region across all controls.

The method-independent length stress slice shows heterogeneity: on 10 short
clips M1 improves left hand by 0.6930 mm versus legacy/fallback, whereas on 14
long clips it regresses by 0.1448 mm. This is secondary and was not used to
alter the primary conclusion.

## Runtime and uncertainty

On the recorded RTX 5880 Ada system, fitting 655 frames took 46.42 s for M0,
59.07 s for legacy/fallback, and 562.18 s for M1 (0.858 s/frame), with about
236 MB peak PyTorch CUDA allocation. Preprocessing time is separate; this is
not a real-time claim.

The calibration artifact attained approximately 90–91% held conformal coverage
for all nine source-region groups. On frozen test M1 abstains on 14.01% body,
5.44% left-hand, and 1.80% right-hand frame-region outputs while always retaining
a complete pose. Selective left-hand AURC improvement is not statistically
resolved on test.

## Claim decision

Coverage, reproducibility, body/right non-inferiority, and dynamics gates pass.
The preregistered hand-geometry superiority gate fails. Therefore the permitted
conclusion is:

> SIGNAL-4D M1 improves temporal reconstruction dynamics on the frozen clean
> SGNify protocol while preserving geometry within the registered
> non-inferiority margins; hand-geometry superiority is not demonstrated.

No overall geometry SOTA claim is permitted. Contact correctness, semantic
fidelity, broad generalization, biomechanical accuracy, and real-time performance
remain unclaimed. The August 19 novelty check found
[DexAvatar (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html)
and [Tamaththul3D (arXiv:2605.05367)](https://arxiv.org/abs/2605.05367);
their published numbers use incompatible or ambiguous endpoints/metric labels
and are reference-only, not entries in this clean leaderboard.
