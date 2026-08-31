# CURRENT_STATE

## CURRENT_STATE

Milestone 1 completed: attachments inventoried; DexAvatar paper/repo/evaluator audited; evaluator metric extracted and sanity-tested; initial primary/fallback hypotheses and pre-test decision rules written. Work is **research/design + static verification only**.

## COMPLETED

- Attached PDF checksum, text extraction, targeted page rendering, reported-result extraction.
- Attached evaluator checksum, full source audit, formal metric spec, 8/8 dependency-light tests.
- DexAvatar clone at `a0dfd427f60f5811aadb35c8657b3856d47f56b5`; clean status.
- README/config/entry/data-loader/optimiser/loss/render/install-script forensics.
- Shell syntax and Python AST static smoke; pipeline CLI smoke.
- Initial primary-source literature map and code-readiness checks for WiLoR, Hamba, SAM 3D Body, Tamaththul3D, DanceHMR, A2P, SOKE, and direct lineage.
- Pre-registered metric vector, reproduction tolerance, keep/kill rules, and experiment ladder.

## DECISIONS

- Original evaluator is immutable; wrapper may only make paths/completeness explicit and must dual-check.
- No Tamaththul3D direct comparison until same evaluator is used.
- Primary hypothesis: reliability-gated kinematic hand/body residual fusion.
- Fallback: deterministic WiLoR proposal + IK + visibility-weighted temporal refinement.
- No test tuning and no SOTA wording without a clean same-protocol final rerun.

## BEST_RESULT

No empirical model result. [REPORTED] DexAvatar paper baseline remains `30.13 / 13.53 / 13.08 mm`; [VERIFIED] only the evaluator sanity suite (8/8 PASS) and static smoke tests.

## BLOCKERS

1. SGNify evaluation frames, ground-truth SMPL-X meshes, exact sign/segment protocol, and dataset terms.
2. Official DexAvatar component checkpoints and licensed SMPL-X/MANO assets.
3. CUDA GPU and compatible isolated legacy environments; current host has none visible.
4. Compute budget and target submission window.
5. A fair validation split independent of held-out SGNify test.

## NEXT_EXACT_ACTION

After assets/compute arrive: hash each asset; build and freeze the exact `(sign_id, frame_id)` manifest; implement a minimal evaluator wrapper that fails on missing/duplicate/NaN frames; prove wrapper/original agreement on a complete fixture; evaluate the official DexAvatar checkpoint; then generate per-sign confidence/occlusion/temporal error slices before implementing candidate modules.

## ARTIFACT_PATHS

- `MILESTONE_1_REPORT.md`
- `research/00_project_charter.md` through `research/08_decision_log.md`
- `research/03_literature_evidence.csv`
- `research/07_results_ledger.csv`
- `audit/evaluator_sanity.py`
- `audit/evaluator_sanity_report.json`
- `README_REPRODUCE.md`
- `paper/main.tex`, `paper/references.bib`, `paper/supplementary.tex`

