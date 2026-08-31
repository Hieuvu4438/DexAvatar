# External-only hand V2 frozen result

Frozen on 2026-08-24 before opening SGNify ground truth.

## External gate

The preregistered signer-disjoint How2Sign gate passed for both hands. The
frozen policy is full V1 residual (`alpha=1.0`) on every target-free eligible
frame, with no probability smoothing. Eligibility requires at least 50% valid
hand keypoints and mean cached reliability of at least 0.20.

| Region | Validation gain | Calibration gain | Worst calibration signer |
|---|---:|---:|---:|
| Left hand | 3.8508 deg | 4.7226 deg | 4.2866 deg |
| Right hand | 3.5134 deg | 4.0092 deg | 3.7920 deg |

Calibration artifact SHA-256:
`00a717d9d6369e8cf7453e5846cf004439319da4713781d13cbf305208011960`.

## Target-free materialization

The method materialized 57 clips / 1,493 frames without reading target labels,
meshes, author errors, or evaluator assets. The unlabeled observation gate
selected 1,134 left-hand frames (75.95%) and 1,462 right-hand frames (97.92%).
There were zero safety fallbacks. All 14,930 checked non-hand arrays are exact
copies of external-only V1, and all 390 unselected hand arrays are also exact
copies of V1.

| Artifact | SHA-256 |
|---|---|
| run manifest | `bb8a040f8493a7ea749bee8de802886a4c15b4eb8877495ce1f1b4923934a5f1` |
| freeze audit | `364c0824c551c3b562b3879055c71add37a329e10faeb97dc4e66db68e8af681` |
| canonical result-tree hash | `4609d943b0e9cd1d0bc99b43177bdf9f568344bc412e54317e20ed2408c829e2` |

## Frozen render verifier

The CPU-only renderer refuses non-passing or target-reading freeze audits,
checks every V2 result against its V1 source anchor, and requires exact OBJ
filename coverage before atomically publishing the render manifest. Its
SHA-256 is
`d2b00de6991c79106545b1c4da6db905879c21a81a77607bb0df1e3ddb948ad0`.
The external-only test suite passes 22 tests, including three isolated render
integrity tests.

The output root is
`outputs/signal4d_external/full1493_hand_v2_seed42`. Target metrics remain
sealed at the time of this report. After the one-time reveal, this V2H method
must not be changed; any follow-up hand method requires a new name and
preregistration.
