# Phase 2 refiner

This package is the isolated implementation of the uncertainty-aware whole-sequence refiner described in `docs/proposal/DEXAVATAR_PHASE2_UNCERTAINTY_AWARE_WHOLE_SEQUENCE_REFINER.md`.

It does not modify or overwrite existing DexAvatar methods. Cache and prediction commands refuse to overwrite by default, and source-anchored rendering writes only below the requested new output directory.

## Implemented vertical slice

- versioned NPZ observation cache;
- explicit per-joint observation confidence, presence, missingness, crop, innovation, and duplicate-detection features;
- 51-joint body/two-hand SO(3) state;
- factorized bidirectional spatial-temporal Transformer;
- zero-initialized bounded residual head and uncertainty head option;
- target-motion, anchor, and heteroscedastic losses;
- synthetic contiguous burst corruption;
- padded training and overlapping-window inference;
- DexAvatar-compatible result PKL export; and
- source-anchored mesh rendering that is exactly identity-preserving.

## Cache one existing method

```bash
python -m phase2_refiner.data.build_observation_cache \
  --frames-root data/frames \
  --initializer-root outputs/method_hamer \
  --output cache/phase2/method_hamer_v1
```

Use `--target-root` only for a separately approved training target. SGNify evaluation ground truth must never be supplied as a training target.

## Identity smoke inference

```bash
python -m phase2_refiner.infer \
  --config phase2_refiner/configs/uawsr_u0.yaml \
  --cache-root cache/phase2/method_hamer_v1 \
  --output outputs/phase2_identity_smoke \
  --identity --render
```

`--identity` is explicit because running an untrained model must never be confused with an evaluated Phase 2 method.

## Train

Update the cache globs in a copied YAML config, then run:

```bash
python -m phase2_refiner.train \
  --config phase2_refiner/configs/uawsr_u0.yaml
```

`--identity-target` exists only for plumbing smoke tests. An accepted training run requires independent clean or quality-filtered targets and must pass the proposal's data gate.

## Strict evaluation

After rendering a complete method on the locked manifest:

```bash
python -m phase2_refiner.evaluate \
  --manifest probes/results/phase0/frame_manifest.csv \
  --prediction outputs/<phase2_method> \
  --baseline outputs/<phase1_method> \
  --output outputs/<phase2_method>/evaluation
```

The command fails on missing, duplicate/stale, or topology-incompatible meshes instead of truncating the comparison.

## Calibrate U1

Prepare a disjoint validation NPZ containing `error` and `log_variance`, plus an optional `group`, then run:

```bash
python -m phase2_refiner.calibrate \
  --residuals outputs/<experiment>/calibration_residuals.npz \
  --output outputs/<experiment>/calibration.json
```

## Tests

```bash
pytest -q phase2_refiner/tests
```
