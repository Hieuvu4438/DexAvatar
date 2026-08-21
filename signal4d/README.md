# SIGNAL-4D

An isolated, evaluator-first implementation of the SIGNAL-4D specification. It reads local DexAvatar assets but does not import or modify the legacy fitting method. All new source, manifests, caches, predictions, logs, and reports live under `signal4d/`.

## Reproduce the CPU checks

```bash
conda run -n signal4d ruff check signal4d
conda run -n signal4d pytest -q signal4d/tests
```

## Frozen clean protocol

The explicit SGNify endpoint contains 57 clips and 1,493 available 15-fps frames. The v2 split quarantines the evaluator-debug clip and leaves 12 calibration clips, 21 development clips, and 24 untouched confirmatory-test clips. See `reports/protocol_audit.md` and `artifacts/manifests/frozen_seed_20260819_v2/split_freeze.json`.

The canonical real-data pipeline is:

1. `preprocess`: SMPLer-X body, WiLoR 3D/2D hands, and optional read-only legacy fitted hypothesis into a hashed canonical cache.
2. `calibrate`: clip-disjoint model-fit and split-conformal uncertainty calibration.
3. `fit-smplx`: M0/M1/M2 windowed optimization with SO(3) consensus.
4. `evaluate-sgnify`: fail-closed 100%-coverage TR-V2V, PA-MPVPE, dynamics, uncertainty, and proxy physical diagnostics.
5. `compare`: paired clip-bootstrap confidence intervals.
6. `freeze-release` then `confirmatory`: immutable one-command final fit/evaluation.
7. `report-final`: deterministic CSV/Markdown/SVG tables and figures generated from raw
   evaluator and paired-bootstrap artifacts.

The v5 prospective workflow additionally provides `compose-legacy`,
`materialize-legacy`, `extrapolate`, `apply-multigate`, `verify-tree`, and
`assess-confirmatory`. The orchestration entry point is
`scripts/run_extended_post_confirmatory_v5.sh`. It builds every prediction and
an immutable release freeze before `cache-gt` is allowed to decode the
extended-post OBJ values. The frozen population and pass/fail rules are in
`reports/preregistration_extended_post_v5.md`.

M2 is incremental by contract: pass `--warm-start-root` pointing at the frozen M1
`predictions/` directory. The fitter validates exact frame IDs and records every
warm-start artifact hash in `run.json`.

Use `conda run -n signal4d signal4d --help` for the complete command surface. Real runs require locally licensed SMPL-X and dataset assets; their bytes are never copied into this package.
Exact runtime versions are recorded in `environment.json` and
`environment.lock.txt`; external asset placement and license boundaries are in
`ASSETS.md`.

## Claim boundaries

- Published values with inconsistent frame/alignment labels are references, not direct comparators.
- Contact output is a proximity/collision proxy because independent contact labels are unavailable.
- Semantic fidelity is not claimed because no frozen sign-recognition evaluator with a measured GT ceiling is available for this endpoint.
- A SOTA statement is emitted only if the frozen confirmatory paired comparison passes the preregistered geometry, coverage, and reproducibility gates.
- The extended-post population is temporally disjoint but reuses known
  clip/sign identities; signer IDs are unavailable. Even a passing result is
  scoped to this same-protocol endpoint, not an external leaderboard or unseen
  signer generalization.

## Frozen prospective result

The v5 extended-post run passed every preregistered confirmatory gate on 56
clips/769 frames at 100% coverage. Against the recomputed same-protocol balanced
A1 comparator, SIGNAL-4D reduced equal-weight clip-macro left-hand TR-V2V by
2.1411 mm (paired 95% clip-bootstrap CI 1.4191–2.9547 mm improvement), while
upper-body error and all three dynamics endpoints also improved and right-hand
geometry remained within the registered non-inferiority margin. The permitted
claim is a new best result on the prospective SIGNAL-4D extended-post SGNify
endpoint; it is not a published-leaderboard or unseen-signer SOTA claim. See
`reports/confirmatory_extended_post_v5.md` and
`reports/final_extended_post_v5/`.

## Strict DexAvatar OBJ delivery

The final A1 and SIGNAL-4D meshes are also exported in the native DexAvatar
layout and OBJ dialect under
`outputs/strict_dexavatar_obj_20260820/`. Evaluation reads those files through
the attached author's OBJ parser and metric functions. The full material
passport, direct-original-main validation, and strict comparison tables are in
`reports/author_evaluator_strict_obj_20260820/REPORT.md`.

The matching RGB reconstruction overlays are under
`outputs/reconstruction_signal4d_v5_20260820/`, using the familiar DexAvatar
layout `<sign>/smplifyx/images/low_<frame>.png`. Each corresponding
`smplifyx/meshes/low_<frame>.obj` is linked to the hash-verified strict OBJ
delivery, and `render_manifest.json` records all source and rendered hashes.
