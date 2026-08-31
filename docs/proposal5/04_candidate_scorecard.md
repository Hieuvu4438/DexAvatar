# Candidate Scorecard — Pre-test Ranking

**Locked on:** 2026-08-21, before any SGNify model result was observed.  
**Scale:** 1 (weak/high risk) to 5 (strong/low risk).  
**Weighted score:** `0.20 relevance + 0.15 compatibility + 0.15 evidence + 0.10 code readiness + 0.15 expected gain + 0.10 novelty + 0.05 compute efficiency + 0.05 reproducibility/licence + 0.05 integration-risk score`.

The scores rank which hypothesis to test first; they do not estimate millimetre gains.

| Candidate | Rel. | Compat. | Evidence | Code | Gain | Novelty | Compute | Repro/lic. | Risk score | Weighted | Uncertainty | Tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Uncertainty-gated kinematic hand–body residual fusion (proposed primary) | 5 | 4 | 3 | 3 | 4 | 5 | 3 | 3 | 3 | 3.90 | High until baseline slices exist | A — scientific primary |
| WiLoR proposals + closed-form forearm/wrist IK + visibility-weighted temporal refinement | 5 | 4 | 3 | 5 | 4 | 2 | 4 | 2 | 4 | 3.85 | Medium; external evidence uses other alignment | A — execution fallback |
| Tamaththul3D-style deterministic pipeline reproduced under locked TR-V2V | 5 | 5 | 2 | 1 | 4 | 2 | 5 | 2 | 4 | 3.50 | High; no public code found and paper metric differs | B — strong comparison/reimplementation |
| DanceHMR-style body/hand residual temporal fusion and visibility curriculum | 5 | 4 | 3 | 1 | 4 | 4 | 2 | 2 | 2 | 3.45 | High; adjacent dataset and no code link verified | B — scientific opportunity |
| Hamba hand encoder replacing HaMeR proposal | 3 | 3 | 4 | 5 | 3 | 3 | 3 | 2 | 3 | 3.30 | Medium; FreiHAND PA metric, CC BY-NC | B — single-module probe |
| SAM 3D Body as a stronger full-body proposal | 3 | 1 | 4 | 5 | 3 | 4 | 1 | 3 | 1 | 2.95 | High; MHR-to-SMPL-X conversion and 631–840M model | C — contingency |
| MaskHand-style masked/occlusion robustness pretraining | 4 | 3 | 3 | 1 | 3 | 3 | 2 | 3 | 2 | 2.90 | High; code/checkpoint not verified | B/C — opportunity |
| SOKE DETO body/left/right discrete tokens as a sign-motion prior | 3 | 3 | 3 | 2 | 2 | 4 | 2 | 3 | 2 | 2.75 | High; generation-to-reconstruction transfer untested | C — representation study |
| A2P collision-guided bimanual diffusion refinement | 4 | 2 | 3 | 1 | 3 | 4 | 1 | 2 | 1 | 2.70 | High; two-hand focus, no code link verified | C — high-risk novelty |
| HySUP-V anchor-guided kinematic fusion | 4 | 3 | 2 | 1 | 3 | 4 | 3 | 2 | 2 | 2.85 | High; full details/code unavailable in current audit | C — high-value no-code lead |

## Tier decisions

### Tier A — implement or test first

1. **Primary: uncertainty-gated kinematic residual fusion.** It directly connects the observed failure mechanism (independent, sometimes inconsistent body/hand evidence) to a measurable intervention. Its novelty must come from task-specific reliability gating and coupled rotation-space residuals, not from placing WiLoR after DexAvatar.
2. **Fallback: deterministic WiLoR + IK + temporal refinement.** Official code and checkpoints are available, and the mechanism is easy to ablate. [VERIFIED] The public WiLoR repository labels its checked-in version “demo only” and uses CC-BY-NC-ND model terms plus MANO/Ultralytics terms, so use must remain non-commercial research and licence review is mandatory.

### Tier B — run only after Tier A single-module evidence

- A locked-evaluator reimplementation of Tamaththul3D is important as a strong current comparison, but not as the claimed novel method.
- Hamba and MaskHand may improve corrupted/occluded hand proposals; they must be evaluated as proposal swaps under identical downstream fitting.
- DanceHMR’s residual temporal coupling is scientifically close, but its no-code status and different benchmark make full reproduction expensive.

### Tier C — contingency/scientific opportunity

- SOKE is a text-to-sign generation/tokenisation method, not a hand-reconstruction baseline. Use only if a discrete sign-motion prior is independently justified.
- A2P is attractive for inter-hand contact but could improve plausibility while worsening TR-V2V against imperfect SGNify ground truth.
- SAM 3D Body uses MHR rather than SMPL-X and has a large representation/compute mismatch.

## Rejected directions

| Direction | Reason |
|---|---|
| Claim Tamaththul3D as a verified TR-V2V SOTA from its table | [REJECTED] Its text defines PA-MPVPE with Procrustes alignment; the locked evaluator is translation-only. Re-evaluate raw meshes first. |
| Naive `DexAvatar + WiLoR + smoother` as the paper contribution | [REJECTED] This is module stacking without a new coupling mechanism; keep only as a strong baseline/fallback. |
| Select modules/hyperparameters on SGNify test | [REJECTED] The attached supplement’s DEV+TEST selection is not acceptable for the follow-up protocol. |
| Optimise the TR-V2V evaluator or exploit missing-frame/NaN behaviour | [REJECTED] This would be leakage or evaluator gaming. |
| Add collision diffusion before proving contact/occlusion is a material error slice | [REJECTED] High cost and likely plausibility/metric conflict without evidence. |
