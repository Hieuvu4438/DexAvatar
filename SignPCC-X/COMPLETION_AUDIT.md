# SignPCC-X completion audit

Audit date: 2026-08-31. Blueprint:
`docs/proposal12/SignPCC_X_Implementation_Blueprint.md`.

## Outcome

The selected final is `signpccx_a3f_external_v1_identity200`. It was promoted
from a frozen 12-sign/298-frame panel, then run once over all 57 signs/1,493
frames without GT or evaluator-region inputs in fitting. Palm-hypothesis and
contact components were implemented and officially ablated, but excluded from
the final because they failed their predefined gates.

| Metric (mm) | DexAvatar A0 | SignPCC-X A3f | Delta |
|---|---:|---:|---:|
| TR All | 42.5867 | **42.0936** | **-0.4931** |
| TR UBody | 26.4560 | **25.8311** | **-0.6249** |
| TR UBody minus face | 29.9074 | **29.1458** | **-0.7616** |
| TR UBody minus head | 40.7960 | **39.6963** | **-1.0997** |
| TR LHand | 13.5735 | **12.8466** | **-0.7269** |
| TR RHand | 12.9271 | **12.1275** | **-0.7996** |

The independent audit reproduced the official rounded values with maximum
absolute discrepancy below `4.9e-5 mm`. A 10,000-replicate paired sign
bootstrap gives a mean-sign All delta of `-0.5753 mm`, 95% CI
`[-0.8088, -0.3488]`; the UBody and both-hand CIs are also wholly negative.

## Requirement-by-requirement matrix

| Blueprint requirement | Status | Evidence |
|---|---|---|
| New isolated implementation directory | PASS | `SignPCC-X/`; no source changes were made outside it. |
| Reuse local repositories; no duplicate clone | PASS | `third_party.lock.yaml`; H4W++, WiLoR, SMPLer-X and NLF paths are local. InterWild remains optional/not installed. |
| Exact source locks and H4W++ patch | PASS | DexAvatar `8401491c...`, H4W++ `f81d35d...`, WiLoR `fcb9113...`, NLF `f8611fc...` with nested SMPLFitter `ea9e632...`; exact two-file patch SHA `37752c66...` and `patches/h4wpp_observation_export.patch`. |
| Unchanged official evaluator | PASS | Read-only mode `0444`; SHA-256 `2722b5cd...`; invoked only as a subprocess. |
| Exact positional manifest | PASS | 57 JSONL manifests, 1,493 records; signs hash `bc5b0da7...`, segment hash `e5d9bd50...`, manifest summary hash `d0efa2a5...`. |
| No frame drop/copy | PASS | Manifest preparation hard-fails count mismatch; full materialization and preflight both report 57/1,493. |
| Canonical topology and finite export | PASS | Every OBJ has 10,475 vertices and 20,908 faces; int64 face hash `2cb81d8e...`; full preflight status `ok`. |
| Coordinate/handedness contracts | PASS | Exactly one x180 boundary is logged; crop round-trip, projection, unmirror involution and export tests pass. |
| One shared beta per signer | PASS | 200-frame pose-diverse robust calibration; identity SHA `544ed9f1...`; each final NPZ stores the same 10-vector beta for every frame. |
| One shared K per camera | PASS | 43,297 valid H4W++/DWPose anchors, no GT; `camera/C1.npz` SHA `7d5b2efc...`; median residual 10.31 px. |
| No temporal pose smoothing | PASS | Final config disables smoothing, velocity and acceleration losses; fit report records `temporal_pose_loss=false`. |
| Canonical parameter-complete state | PASS | All 57 NPZ files retain beta, root/body/hands/face/expression/translation and final canonical vertices. |
| Canonical teacher residual bounded | PASS | 1,493 finite states; mean/max LHand 1.080/2.332 mm and RHand 1.104/1.704 mm. |
| Palm best-of-K ablation | PASS/REJECTED | A4 generated and logged `-30/0/+30` local-axis candidates for every panel frame in `candidate_scores.csv`; official hand errors rose to 19.68/19.77 mm, so A4 is not in final. |
| Confidence-gated contact ablation | PASS/REJECTED | A5 activated 12/298 frames at confidence >=0.70; target error fell 5.04→1.57 mm, but official metrics regressed slightly, so A5 is not in final. |
| DexAvatar same-protocol baseline | PASS | A0 full official result and audited per-frame/per-sign reports under `runs/a0_dexavatar_hamer_baseline/`. |
| A1–A5 fixed-panel comparison | PASS | Same 12 signs/298 frames and evaluator for all rows in `RESULTS.md`; retained official logs under each run's `metrics/dev12/`. |
| Final all-57 frozen run | PASS | A3f official command, stdout/stderr, return code, preflight and fit hashes are retained under its run root. |
| Per-sign statistics and failure cases | PASS | `metrics/audited/per_sign.csv`, `paired_bootstrap.json`, and `failure_cases.json`. |
| Environment/runtime provenance | PASS | `environment.yml`; final `provenance.json`; exact pip/conda exports in the run's `environment/` directory. |
| Automated tests | PASS | 30 tests pass, covering protocol invariance, manifest pairing, geometry, topology/export, identity, camera, hypotheses, contact gradients, evaluator parsing and audit parity. |

## Definition of done audit

1. **Commands:** manifest, teacher export, identity/camera calibration,
   fit/resume, materialization, preflight, official evaluation, audited
   evaluation and provenance are exposed by `python -m signpccx.cli` and
   documented in `README.md`.
2. **100% canonical output:** full preflight passes all 1,493 finite OBJ files;
   per-sign materialization sidecars bind each OBJ set to a parameter-state
   SHA-256.
3. **Baseline:** DexAvatar/HaMeR A0 was run by the same checksum-locked evaluator
   and independently audited.
4. **Ablations:** A1–A5 were all evaluated on the identical fixed panel. Only
   the A3f identity change passed; A4/A5 negative results are retained rather
   than silently omitted.
5. **Final protocol:** all 57 signs use the frozen A3f fit, no GT/evaluator mask
   enters the objective, and the evaluator file was not changed.
6. **Claims/failures:** `RESULTS.md` limits claims to this protocol. Worst
   absolute signs and the largest A0 regressions are recorded in
   `metrics/failure_cases.json`.

## Known, non-blocking scope limits

- The strongest initializer is the existing leakage-clean External V1 artifact;
  this package hashes all 2,986 selected source mesh/parameter files but does
  not rerun its upstream training/inference.
- The shared camera is an observation calibration. A3f does not optimize K in
  its final vertex objective because the frozen initializer already owns the
  camera frame and the official metric is translation-aligned.
- A4 used the targetless K0 ranker. Its large negative result stopped the more
  expensive K1/K2 search under the blueprint's sequential-ablation rule.
- A5 implements high-confidence hand--hand sampled-region attraction and a
  separation margin. Hand--face/torso and optional InterWild were not promoted
  after A5 failed the official no-regression gate.
