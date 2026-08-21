# G1 reproduction decision

Decision: **pass for the recomputed same-protocol controls; published-number
equivalence remains blocked**.

On the 21-clip / 578-frame development split, both the raw M0 hybrid and the
read-only legacy-full + M0 fallback control reach 100% coverage. Equal-weight
clip-macro TR-V2V is:

| Method | Body (mm) | Left hand (mm) | Right hand (mm) |
|---|---:|---:|---:|
| Raw M0 hybrid | 25.4663 | 37.9528 | 19.8187 |
| Legacy-full + M0 fallback | 24.4858 | 17.5609 | 12.9322 |

The large discrepancy from historical/published values is explained by the
explicit every-two-frame endpoint, corrected camera-X export convention, corrected
WiLoR MANO non-tip joint mapping, per-region translation alignment, and clip-macro
aggregation. Therefore the legacy artifact is a valid same-protocol baseline but
historical values are reference-only and cannot support a direct superiority claim.
