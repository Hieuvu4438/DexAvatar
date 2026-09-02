# PA-MPVPE Evaluation of Selective Canonical Hand Refinement

## Evaluation question

Does selective canonical finger refinement improve a frozen monocular sign-language reconstruction after removing global similarity ambiguity? We answer this question on the complete attached protocol of 57 signs and 1,493 frames. The comparison is paired: the frozen reconstruction and its refined counterpart are evaluated on exactly the same frame identities and ground-truth meshes.

## Hand4Whole-style PA-MPVPE protocol

The implementation follows the PA-MPVPE code path used by the local SMPLer-X/Hand4Whole evaluation source in [`SMPLer-X/data/EHF/EHF.py`](../../SMPLer-X/data/EHF/EHF.py) and its [`rigid_align`](../../SMPLer-X/common/utils/transforms.py) primitive. For a predicted regional mesh $X\in\mathbb{R}^{N\times3}$ and its ground truth $Y\in\mathbb{R}^{N\times3}$, the evaluator estimates

\[
(s^*,R^*,t^*)=
\arg\min_{s>0,\,R\in SO(3),\,t\in\mathbb{R}^3}
\lVert sXR^\top+\mathbf{1}t^\top-Y\rVert_F^2,
\]

using the SVD-based Umeyama solution, and then reports

\[
\operatorname{PA\text{-}MPVPE}(X,Y)=
\frac{1}{N}\sum_{i=1}^{N}
\left\lVert s^*R^*x_i+t^*-y_i\right\rVert_2,
\]

in millimetres. As in the reference implementation, the whole mesh, left hand, right hand, and face are aligned **independently**. The extended sign-language regions use the same operation on fixed SMPL-X vertex subsets. This is a similarity-Procrustes metric: it removes scale, rotation, and translation. It must not be confused with the repository's translation-aligned V2V metric or its separate rotation-and-translation-only PA-V2V implementation.

The evaluated vertex subsets are:

| Region | SMPL-X vertices |
|---|---:|
| All | 10,475 |
| Body and hands, excluding face | 5,452 |
| Body only, excluding face and hands | 3,896 |
| Left hand | 778 |
| Right hand | 778 |
| Face | 5,023 |
| Upper body | 8,888 |
| Upper body excluding face | 7,279 |
| Upper body excluding head | 3,865 |

The reported main values are frame-micro averages. We additionally perform a paired non-parametric bootstrap over the 57 sign identities with 100,000 replicates and seed 20260902. Positive gain means that the refined reconstruction has lower error.

## Results on 57 signs and 1,493 frames

| Region | Frozen reconstruction | + Selective canonical hand refinement | Gain | Relative gain |
|---|---:|---:|---:|---:|
| All | 36.4719 | **36.4496** | +0.0223 | +0.06% |
| Body + hands, no face | 54.4394 | **54.3853** | +0.0541 | +0.10% |
| Body only | **39.6338** | 39.6338 | −0.0000 | −0.00% |
| Left hand | 8.5291 | **8.3063** | +0.2228 | +2.61% |
| Right hand | 9.3834 | **9.2366** | +0.1467 | +1.56% |
| Mean of both hands | 8.9562 | **8.7715** | +0.1848 | +2.06% |
| Active hand(s), sign-aware | 9.2023 | **8.9983** | +0.2040 | +2.22% |
| Face | **3.5230** | 3.5230 | −0.0000 | −0.00% |
| Upper body | 26.4790 | **26.4453** | +0.0336 | +0.13% |
| Upper body excluding face | 30.2293 | **30.1889** | +0.0404 | +0.13% |
| Upper body excluding head | 39.3759 | **39.3223** | +0.0536 | +0.14% |

All values are PA-MPVPE in millimetres; lower is better. “Active hand(s)” uses the right hand for one-handed signs and the average of both hands for two-handed signs. It is a sign-language diagnostic added to the standard PA-MPVPE regions, not a Hand4Whole leaderboard field.

The strongest effect is on local handshape: the mean hand error falls by 0.1848 mm (2.06%), and the sign-aware active-hand error falls by 0.2040 mm (2.22%). The left- and right-hand gains remain positive when signs, rather than individual frames, are treated as the resampling unit. Body-only and face errors are unchanged to numerical precision, which is consistent with the method changing only accepted finger rotations while fixing body pose, wrist orientation, shape, facial expression, camera, and translation.

## Paired sign-level uncertainty

| Region | Mean sign gain | 95% sign-bootstrap interval | Improved / worse / unchanged signs |
|---|---:|---:|---:|
| All | +0.0215 | [+0.0107, +0.0337] | 33 / 11 / 13 |
| Body + hands, no face | +0.0502 | [+0.0286, +0.0745] | 37 / 7 / 13 |
| Body only | −0.0000 | [−0.0000012, +0.0000009] | 22 / 22 / 13 |
| Left hand | +0.2022 | [+0.0990, +0.3186] | 28 / 16 / 13 |
| Right hand | +0.1403 | [+0.0726, +0.2168] | 33 / 11 / 13 |
| Mean of both hands | +0.1713 | [+0.0983, +0.2536] | 31 / 13 / 13 |
| Active hand(s) | +0.1920 | [+0.1137, +0.2794] | 31 / 13 / 13 |
| Upper body | +0.0298 | [+0.0158, +0.0458] | 35 / 9 / 13 |
| Upper body excluding face | +0.0359 | [+0.0190, +0.0552] | 34 / 10 / 13 |
| Upper body excluding head | +0.0496 | [+0.0214, +0.0807] | 31 / 13 / 13 |

The intervals support a reproducible reduction in PA-MPVPE for both hands and the upper-body surface. They do not support a change in body-only or face geometry, nor should one be expected from a finger-only intervention.

## Integrity and interpretation limits

The evaluator matched all 57 signs and all 1,493 frames in both runs. Per-sign frame counts and sign identities are identical. A synthetic similarity-transform recovery test produced a maximum coordinate error of $4.9\times10^{-15}$, verifying the scale–rotation–translation convention. The fit path did not read ground-truth meshes, evaluation errors, temporal pose, SignHPoser, or SignBPoser. WiLoR and the 2D pose estimator were frozen.

This result nevertheless **must not be described as a clean held-out generalization result**. Earlier method selection used a 12-sign subset drawn from this benchmark. No network was trained on those target meshes, but the method design and thresholds were benchmark-informed. Consequently, the 1,493-frame table is a complete descriptive benchmark evaluation; a publishable confirmatory claim requires a new sign-disjoint dataset or a prospectively frozen external test set.

The result is also not directly comparable to a paper that reports a rigid-only alignment, a different frame set, or a differently defined hand region. In particular, literature values should be combined with this table only after confirming that they use region-wise similarity alignment with scale and the same 1,493 frame identities.

## Reproducibility record

The paper-facing names above map internally to the raw canonical baseline and its gated WiLoR finger refinement. These internal labels are included only to locate artifacts; they should not appear as method names in a manuscript.

| Artifact | Path | SHA-256 |
|---|---|---|
| PA-MPVPE evaluator | [`signal4d/evaluate_pampvpe.py`](../../signal4d/evaluate_pampvpe.py) | `cda6aa1856022407930669ec80e11854f78a90d3f3af51948305e20fb4202f0e` |
| Reference similarity transform | [`SMPLer-X/common/utils/transforms.py`](../../SMPLer-X/common/utils/transforms.py) | `8cc645f49c39bc96bde7769eeae3db4d23e5ba2f794267f6f1480eace0b40b29` |
| Frozen reconstruction report | [`a3f_full1493_pa_mpvpe.json`](../../SignEFT-X/reports/a3f_full1493_pa_mpvpe.json) | `64400ce2ee5ee7ba2c2d1b93fec7765be49ef01caa73b2542846014083cdea63` |
| Refined reconstruction report | [`a3f_h1_full1493_pa_mpvpe.json`](../../SignEFT-X/reports/a3f_h1_full1493_pa_mpvpe.json) | `f5486f01a6e276e0c9944f847ddf92a812a32eb32478bfda9f4b26d606341fb3` |
| Paired comparison | [`a3f_h1_full1493_pa_mpvpe_comparison.json`](../../SignEFT-X/reports/a3f_h1_full1493_pa_mpvpe_comparison.json) | `1a4c6a8431e5f4e766f91fbab3d8ff86bbc6c10f7a21c4d5b5a9d70b64c2a31e` |

The two PA reports preserve per-sign measurements, enabling independent recomputation of every aggregate and confidence interval in this document.
