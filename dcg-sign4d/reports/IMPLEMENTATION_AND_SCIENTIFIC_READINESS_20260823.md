# DCG-Sign4D implementation and scientific-readiness audit — 2026-08-23

## Decision

**Development integration: PASS. Scientific SGNify experiment: BLOCKED.**

The isolated implementation now has a real-component SGNify run. It compiled all 57 clips and
1,493 frames, trained development contact and diffusion checkpoints, exercised the official
DPoser-X and selfcontact backends, reconstructed one 15-frame clip, exported meshes, and evaluated
the identical frames with the author-asset evaluator. This supersedes the earlier statement that
the licensed runtime had not been exercised.

This is not a final scientific DCG-Sign4D result. Its calibration is synthetic, contact labels
and patch map are provisional, ranker data are synthetic, training is only 100 steps, inference is
one step/one hypothesis, and signer-aware statistical evaluation is impossible with the available
metadata. The full evidence and proposal-by-proposal ledger are in
`reports/DCG_SIGN4D_DEVELOPMENT_RUN_20260823.md`.

## Current evidence

| Area | Evidence | Status |
|---|---|---|
| Isolated source and official pins | `src/dcg_sign4d`, `third_party/manifest.yaml` | PASS |
| Complete initialization | `artifacts/initialization/sgnify_full1493_complete_camera_user_assets_v2` | PASS development |
| Calibrated observation contract | `artifacts/observations/sgnify_full1493_synthetic_calibrated_development_v2` | PASS schema; synthetic calibration |
| Official selfcontact | `artifacts/audits/selfcontact_sgnify_tisch_frame0_real_v1.json` | PASS with segment test disabled |
| Official DPoser-X | `artifacts/audits/dposer_x_sgnify_tisch_real_v1.json` | PASS, backbone frozen |
| Contact training | `artifacts/training/contact_sgnify_provisional_development_v1` | PASS development, 100 steps |
| Diffusion training | `artifacts/training/diffusion_sgnify_dposerx_development_v1` | PASS development, 100 steps |
| Real reconstruction | `artifacts/runtime_smoke/muell_dcg_real_components_cpu_v2` | PASS, 15 frames |
| Author evaluator | `reports/dcg/muell_real_components_cpu_v2` | PASS engineering, coverage 1.0 |
| Test suite | 111 tests and ruff | PASS |
| Scientific labels/split/ranker | No gold contact, signer IDs, real calibration labels or real validation candidates | BLOCKED |

## Development score, not a method claim

| Output on `Muell` / 15 frames | Root hand PVE mm | Wrist hand PVE mm | Body MPJPE mm | Hand velocity mm/s |
|---|---:|---:|---:|---:|
| DCG development smoke | 93.2422 | 21.3754 | 57.1136 | 274.1976 |
| DexAvatar reference | 92.9340 | 22.0581 | 57.1258 | 268.3707 |
| Signal4D v5 reference | 89.2168 | 21.2451 | 54.8619 | 320.8183 |

Only the right hand is evaluated for this clip. No confidence interval or general comparison is
valid from this sample. Exact values and deltas are recorded in
`reports/dcg/muell_development_comparison_v1.json`.

## Gates

| Gate / run | Status |
|---|---|
| G0 evaluator | PASS engineering / BLOCKED scientific freeze |
| G1 labels | BLOCKED: no real independent double-annotated gold set |
| B0 | Existing full-set baseline evidence only |
| B1–B7 | Development path exercised; matched scientific runs not run |
| A-INF0/A-INF1/A-K | Not run at frozen scientific budget |
| G2–G5 | Not testable |
| Signer bootstrap | BLOCKED: signer IDs unavailable |

## Remaining scientific inputs

1. Independent correctness labels and detector checkpoint provenance for reliability calibration.
2. Author-reviewed geodesic patch map and a double-annotated gold contact subset.
3. Signer IDs plus signer-disjoint train/calibration/validation/test manifests.
4. Frozen source, data, hyperparameter and compute decisions.
5. Full training and real validation-candidate ranker fitting without test leakage.
6. Matched B0–B7 and A-INF0/A-INF1/A-K runs with signer-cluster bootstrap.

The production configuration remains fail-closed. Development readiness being `READY` means only
that the engineering assembly can execute with explicitly development-labelled inputs.
