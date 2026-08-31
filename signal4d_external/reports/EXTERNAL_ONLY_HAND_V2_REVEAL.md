# External-only hand V2 one-time reveal

Revealed once on 2026-08-24 after the V2H result tree and policy were frozen.
No SGNify target data was used for training, policy selection, calibration, or
materialization. This report is diagnostic only: V2H will not be retuned from
these target results.

## Integrity

| Artifact | SHA-256 |
|---|---|
| frozen render manifest | `e852af12a70c99af4e523b6f299a22ae7a6918298e4f072d079721df10721279` |
| OBJ export manifest | `15acd222806ddcba98d2e60a9ddf2d2007c59bf12d65d35f8eeb469122be3b7e` |
| evaluator comparison JSON | `48633da414432d05aa43a38f01b254fa6441a234817fcae723c53804429ed032` |

Coverage is 57 clips / 1,493 frames for every method. Rendering used CPU and
the external SMPL-X model; the frozen render manifest records zero SGNify
target reads before evaluation.

## Author metrics (mm, lower is better)

| Method | All | Upper body | Upper body - face | Left hand | Right hand |
|---|---:|---:|---:|---:|---:|
| DexAvatar | 42.586721 | 26.455999 | 29.907413 | 13.573462 | 12.927137 |
| External V1 | **42.242307** | **26.223591** | **29.619596** | **12.810229** | **12.114835** |
| External V2H | 42.244501 | 26.226020 | 29.621510 | 12.832343 | 12.283111 |
| Historical SIGNAL4D V6 | 42.111624 | 26.139411 | 29.519683 | 11.633895 | 11.805624 |

V2H remains better than DexAvatar on every endpoint, but it is worse than the
frozen external V1 by 0.002194 mm All, 0.002429 mm upper body, 0.001915 mm
upper-body-minus-face, 0.022114 mm left hand, and 0.168276 mm right hand.
Therefore V2H is **not promoted** over V1.

## Diagnostic interpretation

The external signer-disjoint pose-space gate did not transfer to the author's
mesh-space target metric. For left hand, 702 of 1,163 evaluable frames improve
over V1 and the median delta is -0.000781 mm, but a smaller set of regressions
has larger magnitude (maximum +0.984126 mm), making the aggregate worse. Right
hand is a systematic mismatch: only 313 of 1,493 frames improve, while 1,149
regress and the median delta is +0.027811 mm (maximum +1.730990 mm).

The next hand experiment, if pursued, must be separately preregistered and use
an external-only validation objective closer to the final mesh-space endpoint
rather than pose-geodesic gain alone. No V2H threshold, alpha, eligibility
rule, or temporal setting may be selected from this reveal.
