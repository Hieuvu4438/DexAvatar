# Phase 3: Relational Diffusion Posterior

This package is an additive implementation of the Phase 3 proposal. It imports stable
Phase 2 cache, rotation, rendering, and evaluation utilities read-only. It does not
modify DexAvatar fitting, Phase 1, or Phase 2 behavior.

## Safety state

The code is training-capable, but the current repository is **not paper-grade GO**:

- the official DPoser-X whole-body mixed checkpoint and exact 51-joint/6D
  normalization contract are not installed;
- the local DPoser sign checkpoint is body-only and is deliberately rejected as a
  whole-body prior;
- SignAvatars and repaired Motion-X assets are not ready; and
- the 300-clip manual quality/contact audit and local license verification remain
  required for P3-G0.

The default model therefore uses an explicitly named from-scratch spatial prior.
It never silently labels that route as pretrained. Supply a validated frozen prior
through `FrozenSpatialPrior` only after its contract audit passes.

## Ordered execution

Run each stage only after the preceding numerical gate passes:

```bash
python -m phase3_posterior.models.dposer_adapter \
  --contract phase3_posterior/configs/dposer_wholebody_contract.template.json \
  --output outputs/phase3_gates/g1/dposer_contract_audit.json

python -m phase3_posterior.data.build_phase3_index \
  --sources phase3_posterior/configs/data_sources_v1.yaml \
  --output cache/phase3/v1

python -m phase3_posterior.data.audit_phase3_cache \
  --index cache/phase3/v1/splits/train.json \
  --index cache/phase3/v1/splits/val.json \
  --manual-quality cache/phase3/v1/manual_quality_300.json \
  --output outputs/phase3_gates/g0/cache_audit.json
```

Long training commands must be launched under tmux with `OMP_NUM_THREADS=4`,
`MKL_NUM_THREADS=4`, and `OPENBLAS_NUM_THREADS=4`, and append logs under
`logs/phase3/`. The stage commands are `train_relation`, `train_diffusion`, and
`train_selector`. Configs R2 through R7 preserve the proposal's frozen starting
hyperparameters and use append-only output directories.

Inference always includes candidate zero (the unchanged initializer), uses shared
overlap noise and quaternion hemisphere-aligned blending for long sequences, and
fails closed when source PKLs or frame contracts are missing. Render and evaluate
with `phase3_posterior.render` and `phase3_posterior.evaluate`; decide gates using
`phase3_posterior.gates`.

## Validation

```bash
ruff check phase2_refiner phase3_posterior
pytest -q phase2_refiner/tests phase3_posterior/tests
python -m compileall -q phase2_refiner phase3_posterior
git diff --check
```
