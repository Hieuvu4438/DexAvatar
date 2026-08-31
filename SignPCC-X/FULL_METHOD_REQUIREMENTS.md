# Full SignPCC-X requirement matrix

This matrix is derived from `docs/proposal12/SignPCC_X_Implementation_Blueprint.md`.
An item is complete only when both implementation and run evidence exist. Earlier
reduced A3f/A4/A5 artifacts are explicitly not accepted as full-method evidence.

| Requirement | Implementation | Required run evidence |
|---|---|---|
| M0 official baseline/evaluator lock | Existing wrapper, preflight, audited parity | A0 official JSON/stdout and evaluator SHA-256 |
| M1 frozen H4W++/WiLoR teacher | Existing full 1,493-frame cache and strict observation loader | one-frame coordinate/overlay gate; cache completeness |
| M2 shared signer beta and camera | `calibrate-full`, four alternating phases `[60,80,30,20]` | `S1_full.npz/json/jsonl` from 20 diversity frames |
| S0 camera/root | 60 Adam steps, translation/global, signed shared camera from calibration | per-step JSONL |
| S1 upper body | 100 Adam steps, anatomical upper-body gradient mask | per-step JSONL and finite output |
| M3/K0 candidates | H4W CHAM+WiLoR, SMPLer-X wrist+WiLoR, twist ±30°, HaMeR fingers | candidate source/score records |
| M3/K1 | 25 Adam steps per retained one-hand candidate, top-2 per side | per-candidate JSONL |
| M3/K2/S3 | at most top-2×top-2 pairs, 100 Adam steps | pair score and selected names |
| M4 intended contact | gated hand-hand, fingertip-face, hand-torso proposals, threshold 0.70 | proposal records and contact distances |
| M4 non-penetration | symmetric oriented point-to-triangle signed distances | depth/count logs and gradient test |
| S4 deterministic LBFGS | 20 steps, deterministic closure, reject/rollback on worse/nonfinite | decision record per frame |
| S5 canonical refit | fixed beta, body/upper/hand/seam/face vertex weights, joints×10, low 2D/contact | residuals and exact canonical output |
| No temporal pose loss | config and objective contain no previous/next-frame pose | config test and sidecar flag |
| A1–A5 fixed panel | separate immutable configs/run roots | official + audited metrics for all five |
| Full-57 selected run | resume-safe per-frame NPZ/sidecar/log then canonical OBJ | 57 signs/1,493 frames, preflight, official/audited/bootstrap |

Current gate status is intentionally not marked complete here. It must be updated
from artifact hashes after the corresponding commands finish successfully.

