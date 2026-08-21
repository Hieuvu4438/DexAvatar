## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Method: SIGNAL4D_V6_UQDiff
- Coverage: 57 signs, 1,493/1,493 frames
- Evaluation: attached-author TR/V2V functions on strict DexAvatar-layout OBJ

# SIGNAL4D V6 full-1,493 release report

SIGNAL4D V6 completed the frozen full manifest without GT access during fitting
or gating. It selected the W3 refinement on 271 frames and copied the immutable
V5 parent on 1,222 frames. Full inference took 478.95 seconds with peak allocated
CUDA memory 352,342,528 bytes.

## Strict author metrics (mm; lower is better)

| Method | TR all | UBody | UBody(-F) | LHand | RHand |
|---|---:|---:|---:|---:|---:|
| DexAvatar HaMeR | 42.586721 | 26.455999 | 29.907413 | 13.573462 | 12.927137 |
| SIGNAL4D V5 | 42.143356 | 26.193547 | 29.593199 | 11.665080 | 11.832928 |
| **SIGNAL4D V6** | **42.111624** | **26.139411** | **29.519683** | **11.633895** | **11.805624** |

V6 is the lowest measured row for every endpoint. Versus V5, paired sign-macro
UBody(-F) changes by `-0.066544 mm`, bootstrap 95% CI
`[-0.116719, -0.019938]`. Versus DexAvatar, the vertex-micro improvement is
`-0.475097 mm` for TR-all and `-0.387730 mm` for UBody(-F).

## Acceptance audit

| Gate | Result |
|---|---|
| 57 signs / 1,493 unique frames / full coverage | PASS |
| Strict 10,475-vertex / 20,908-face topology | PASS |
| UBody(-F) paired sign-bootstrap upper bound below zero | PASS |
| Neither hand sign-bootstrap upper bound above +0.25 mm | PASS |
| TR-all regression no larger than +0.10 mm | PASS; improves by 0.031732 mm |
| Dynamics regression no larger than 2% | PASS; all three improve |
| Full UBody(-F) effect at least -0.15 mm vs V5 | **FAIL; -0.073516 mm** |

According to the frozen execution contract, missing the last effect-size gate
means V5 remains the formally promoted release. V6 is retained as the current
measured-best research artifact. It must not be described as externally proven
literature SOTA or as test-only significant: the 24-sign test UBody(-F)
bootstrap interval crosses zero.

## Outputs

- Predictions: `signal4d/runs/signal4d_v6_final_full1493_20260821`
- Strict OBJ: `signal4d/outputs/strict_dexavatar_obj_20260821/full_1493/SIGNAL4D_v6`
- Fitting visualization: `signal4d/outputs/reconstruction_signal4d_v6_full1493_20260821`
- Strict metrics: `signal4d/reports/author_evaluator_strict_obj_20260821/full_1493_v6`
- Detailed method: `docs/proposal3/SIGNAL-4D_V6_final_method_and_full1493_results_2026-08-21.md`
