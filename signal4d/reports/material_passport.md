# SIGNAL-4D material passport

## Identity and isolation

- Package: `signal4d` 0.1.0, implemented entirely under `signal4d/`.
- Legacy repository branch/revision at environment capture: `main`,
  `e7404e661862dba86f7b52b6cfc38bf7fe523713`.
- Legacy method source and outputs are read-only inputs; no legacy code or output
  file was modified.
- Runtime: isolated Conda environment `signal4d`; Python 3.10.19,
  PyTorch 2.1.1+cu121, CUDA 12.1, NVIDIA driver 570.195.03.
- Hardware: NVIDIA RTX 5880 Ada Generation, 49,140 MiB.

## Data and model materials

| Material | Role | Integrity/provenance |
|---|---|---|
| SGNify explicit endpoint | 57 clips / 1,493 frames | every frame ID frozen in manifests |
| Calibration split | 12 clips / 260 frames | SHA-256 `6e5267…5036` |
| Development split | 21 clips / 578 frames | SHA-256 `fe43a1…c5f9` |
| Test split | 24 clips / 655 frames | SHA-256 `a18084…cccc` |
| SMPL-X neutral model | geometry layer/evaluator regressor | SHA-256 `376021…992` |
| SMPLer-X estimates | body hypothesis | per-file hashes in cache metadata |
| WiLoR estimates | hand hypothesis and 2D evidence | per-file hashes in cache metadata |
| Legacy fitted output | strongest read-only hypothesis/control | per-file hashes in cache metadata |
| Canonical observation cache | three-source frozen tensor cache | full tree in release freeze |
| Test GT cache | evaluator-only vertices after release freeze | tree SHA-256 `2fb36e…f71` |

External SGNify, SMPL-X, estimator, and legacy assets retain their original
licenses and are not redistributed. Placement instructions are in `ASSETS.md`.

## Research stages and decisions

1. G0 protocol/evaluator pass: explicit frame contract and fail-closed coverage.
2. G1 same-protocol reproduction pass; published-number equivalence blocked.
3. G2 full-tree cheap smoother rejected for body regression.
4. M1 calibrated uncertainty/change-point left-chain refinement passes G3 on
   development.
5. G4 real contact-label gate fails; M2 contact claims restricted to synthetic
   and proximity/collision diagnostics.
6. Correctly warm-started M2 fails G5 incremental value and is excluded from
   confirmatory test before test reveal.
7. M1 reproducibility is bit-exact; G6 passes.
8. Confirmatory M1 passes dynamics and non-inferiority gates but fails the
   preregistered hand-superiority/effect gate.

Invalid/intermediate artifacts are retained under clearly named directories,
including the stopped pre-decoder-fix M2 run and under-covered calibration, and
are excluded from claims.

## Immutable evidence

- Release freeze: `artifacts/release/freeze_20260819.json`, SHA-256
  `351f36aa64f7615c40a6d0c8f8cfacf219ffb0c3d88dc3a59a1aa922db5748d7`.
- Confirmatory records SHA-256: M0 `6745b7…a67`, legacy/fallback `82e223…6eb`,
  M1 `3ae698…709`.
- Generated result artifacts: `reports/final_20260819/` reads raw evaluator and
  bootstrap files; it contains CSV, Markdown, SVG, comparisons, and a source
  manifest.
- Release/test command logs and per-window JSONL factor diagnostics are stored
  under `signal4d/logs/` and each run's `logs/` directory.

### Prospective v5 release

- Extended-post manifest: 56 clips/769 frames, SHA-256
  `33825a3f1ac8aa6d063f90bc12c8061ed60680267615b6d76cbe1e8cee625b32`.
- A1 source hierarchy: 607 Ensemble, 145 HaMeR A0, 17 raw SMPLer-X terminal;
  100% availability without evaluator-score selection.
- Frozen GT-free gate states: A1 127, M1×1.0 442, M1×1.5 7, M1×3.0 193;
  zero switches and `gt_used_for_selection: false`.
- Release freeze: `artifacts/releases/extended_post_v5_release_freeze.json`,
  SHA-256 `0c5808308b6de3f965fa50f0b05bad21cfe04e9e9161e211f2b03740f825975d`.
  It predates the prospective GT cache and hashes 24 roots including source,
  scripts, environment lock, calibration, gates, cache, baseline/candidate
  predictions, model and vertex indices.
- Repeated gate output is byte-identical over 112 files. Confirmatory decision:
  PASS, with 56 clips/769 frames and coverage 1.0 for both methods.
- Result pack: `reports/final_extended_post_v5/`; full gate/CI report:
  `reports/confirmatory_extended_post_v5.md`.

## Known limitations

- Signer IDs, language, and sign-type metadata are unavailable; clips are the
  independent bootstrap unit and no cross-signer/language claim is made.
- There is no reliable independent real contact annotation or frozen semantic
  evaluator, so contact correctness and semantic preservation cannot be tested.
- Calibration/model development uses one small benchmark; external OOD evidence
  is absent.
- The earlier central-test hand gain failed its registered effect/significance
  gate. The later prospective temporal endpoint passed, but shares clip/sign
  identities with historical data and cannot establish unseen-signer or
  external-dataset generalization.
