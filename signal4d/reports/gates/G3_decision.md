# G3 M1 core decision

Development split, equal-weight clip macro; both methods have 100% coverage.

| Endpoint | Legacy-full + M0 fallback | SIGNAL-4D M1 | Delta (M1-baseline) | Paired 95% CI |
|---|---:|---:|---:|---:|
| Upper-body TR-V2V (mm) | 24.4858 | 24.4866 | +0.0009 | [-0.0081, +0.0093] |
| Left-hand TR-V2V (mm) | 17.5609 | 16.8659 | -0.6950 | [-1.2096, -0.1985] |
| Right-hand TR-V2V (mm) | 12.9322 | 12.9350 | +0.0028 | [+0.0007, +0.0050] |
| Velocity error | 6.2944 | 6.2544 | -0.0401 | [-0.0500, -0.0306] |
| Acceleration error | 148.5957 | 147.1339 | -1.4617 | [-1.7881, -1.1476] |
| Jerk error | 3962.7396 | 3926.6188 | -36.1209 | [-45.1980, -27.4207] |
| Left-hand AURC | 16.3478 | 15.6750 | -0.6728 | [-1.1895, -0.1772] |

Bootstrap uses 10,000 paired clip replicates. Unknown signer IDs are treated as independent clip clusters; no cross-signer claim is made.

Decision: **G3 pass**. M1 gives a statistically supported left-hand and dynamics improvement while body/right changes are far below the preregistered 0.5 mm non-inferiority margin. Right-hand improvement was not targeted because development showed the frozen legacy right chain was already the stronger control.

M1 refinement contract selected before confirmatory test:

- coherent legacy-full hypothesis where available; raw M0 fallback otherwise;
- calibrated observation weights and frozen abstention thresholds;
- change-point adaptive temporal factors;
- optimize only left elbow, left wrist, and left-hand rotations;
- keep global pose, torso, translation, right arm, and right hand frozen.
