# SignPCC-X

Implementation of `docs/proposal12/SignPCC_X_Implementation_Blueprint.md` in a
new directory. It reuses the repositories and model assets already present in
the DexAvatar workspace; it does not clone duplicates or modify the official
evaluator.

## Best verified run

The best leakage-clean canonical run is `signpccx_a3f_external_v1_identity200`.
It uses one signer-wide beta, frame-independent pose fitting, External V1 as a
frozen geometric initializer, and exact neutral SMPL-X forward/export. Neither
ground truth nor the evaluator upper-body mask is used by its objective.

Official unchanged-evaluator metrics over 57 signs / 1,493 frames:

| Run | TR All | TR UBody | TR UBody (-F) | TR LHand | TR RHand |
|---|---:|---:|---:|---:|---:|
| DexAvatar/HaMeR A0 (same evaluator) | 42.5867 | 26.4560 | 29.9074 | 13.5735 | 12.9271 |
| H4W++ A1 | 84.4572 | 33.8142 | 38.0129 | 15.3293 | 16.8033 |
| H4W++ shared-beta A2 | 84.3778 | 33.4182 | 37.5591 | 15.3296 | 16.8179 |
| External V1, leakage-clean initializer | 42.2423 | 26.2236 | 29.6196 | 12.8102 | 12.1148 |
| SignPCC-X A3d canonical | 42.1966 | 26.1253 | 29.5175 | 12.9332 | 12.1936 |
| **SignPCC-X A3e canonical** | **42.0947** | **25.8721** | **29.1902** | **12.8458** | **12.1245** |
| **SignPCC-X A3f canonical** | **42.0936** | **25.8311** | **29.1458** | **12.8466** | **12.1275** |

All values are millimetres; lower is better. A3f is selected for its best All
and UBody values; A3e remains lower by 0.0008/0.0030 mm on LHand/RHand. The
official A3f record, raw
stdout/stderr, preflight report, manifests and per-sign fit hashes are retained
under `runs/signpccx_a3f_external_v1_identity200/`.

Historical Signal4D V6 is not used: its own provenance report marks it as a
contaminated reference because SGNify influenced selection/calibration.

## Reproduce the best run

From this directory, use a fresh `paths.run_root` if reproducing alongside the
retained result:

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -m pytest -q
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml prepare-manifests
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml calibrate-external-identity --device cpu
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml calibrate-camera
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml canonical-refit-external --device cpu
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml materialize-fitted
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml preflight
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml evaluate-official
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml evaluate-audited
python -m signpccx.cli --config configs/ablations/a3f_external_v1_identity200.yaml record-provenance
```

The evaluator wrapper verifies SHA-256
`2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300`
before launching the evaluator as an unchanged subprocess.

## Implemented blueprint components

- exact positional manifests with no dropped or copied frames;
- pinned local H4W++, WiLoR, SMPLer-X, NLF/SMPLFitter assets;
- strict canonical topology, coordinate-boundary and OBJ preflight checks;
- robust pose-diverse signer identity calibration and one shared beta;
- canonical SMPL-X refit with best-state rollback and per-sign resume hashes;
- local-axis wrist-twist hypotheses, chirality scoring and deterministic ranking;
- reproducible contact-region construction, confidence-gated intended-contact
  attraction, and a differentiable penetration barrier;
- no temporal pose smoothing; and
- checksum-locked official evaluation with preserved logs.

M3/M4 hypothesis/contact refiners are implemented, logged, tested and evaluated
on the fixed 12-sign panel. Both remain disabled in the best run because they
failed their official-metric gates. InterWild remains an optional, uninstalled M5
ablation, as prescribed by the blueprint.

## Protocol invariants

- Never drop or copy a neighbouring frame.
- Never infer filename pairing; manifests store the positional mapping.
- Never evaluate NaN/Inf or mismatched topology.
- Apply the internal-camera to evaluator `x180` transform exactly once.
- Never edit, import, or monkeypatch `evaluate_new_fitting.py`.
- Do not use GT or evaluator region masks inside fitting objectives.
- Do not use temporal continuity to select pose hypotheses.
