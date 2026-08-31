# DexAvatar Research Milestone 1

**Date:** 2026-08-21  
**Verdict:** [BLOCKED] No empirical SOTA claim is currently allowed. The baseline cannot yet run end-to-end in this workspace, but its paper, released execution path, and attached metric have been audited far enough to lock the immediate scientific risks.

## Outcome

1. [VERIFIED] DexAvatar reports `30.13 / 13.53 / 13.08 mm` on SGNify `UBody(-F) / LHand / RHand` (attached paper, p. 6, Table 1).
2. [VERIFIED] The attached evaluator is region-wise translation-aligned V2V, not PA-MPVPE: it subtracts independent region centroids, applies no rotation/scale alignment, averages all included vertex-frame errors, and converts metres to millimetres.
3. [VERIFIED] Eight toy metric tests pass against functions AST-extracted from the exact hashed evaluator.
4. [VERIFIED] The released repo is clean at commit `a0dfd427f60f5811aadb35c8657b3856d47f56b5`; 1,274 Python files parse and shell/CLI smoke checks pass.
5. [VERIFIED] The released pipeline is not self-contained: no SGNify evaluation payload, checkpoint, SMPL-X/MANO model, environment lock, or evaluator is bundled.
6. [VERIFIED] High-impact risks include silent frame dropping, test-set hyperparameter selection stated in the supplement, hard-coded evaluator paths, positional GT/prediction pairing, NaN skipping, asymmetric one-handed metric populations, and several paper–code objective differences.
7. [REPORTED] Tamaththul3D lists lower numbers than DexAvatar, but its paper defines PA-MPVPE/Procrustes alignment. Those values are **not directly comparable** with the locked translation-only evaluator and are not yet evidence of a TR-V2V SOTA.

## Decision

- Preserve DexAvatar and the supplied evaluator unchanged as provenance snapshots.
- Before any method fusion, first evaluate the official checkpoint and produce an exact frame-coverage manifest.
- Treat a deterministic WiLoR/forearm-IK temporal refinement as the strongest execution-ready fallback, but do not claim novelty from simple module stacking.
- Investigate a primary research method based on uncertainty-gated, kinematically coupled hand/body residuals with visibility-weighted temporal refinement. This remains [HYPOTHESIS] until error slices and a minimal single-module validation support it.
- Keep SOKE’s decoupled body/left/right tokenisation as an adjacent representation/pretraining idea, not as a direct hand-reconstruction baseline.

## Evidence and artifact map

- Project charter: `research/00_project_charter.md`
- Input inventory: `research/01_input_inventory.md`
- Paper/code/evaluator audit: `research/02_baseline_forensics.md`
- Literature evidence ledger: `research/03_literature_evidence.csv`
- Candidate scorecard: `research/04_candidate_scorecard.md`
- Preliminary method/interface specification: `research/05_method_spec.md`
- Pre-registered experiment plan: `research/06_experiment_plan.md`
- Results ledger: `research/07_results_ledger.csv`
- Decision log: `research/08_decision_log.md`
- Resume checkpoint: `research/CURRENT_STATE.md`
- Evaluator test/report: `audit/evaluator_sanity.py`, `audit/evaluator_sanity_report.json`
- Reproduction instructions: `README_REPRODUCE.md`

## Blockers requiring user-provided access or compute

1. SGNify evaluation frames, GT meshes, exact segment/sign files, and their licence/access instructions.
2. DexAvatar official checkpoints and licensed SMPL-X/MANO model files, or permission/instructions to download them under the user’s account.
3. A CUDA-capable machine compatible with the required legacy environments; expected minimum follows the paper’s RTX 4090 24 GB setup until measured otherwise.
4. Compute budget and intended submission window, because they materially determine whether to prioritise deterministic refinement or train a gated fusion model.

## Next exact action

On receipt of the missing assets/GPU environment: hash all inputs; create an immutable frame manifest; make a path-only evaluator wrapper that fails on missing/NaN/duplicate frames; dual-check it against the original; then evaluate the official DexAvatar checkpoint before changing model code.

