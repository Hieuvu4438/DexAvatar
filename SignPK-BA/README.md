# SignPK-BA

Implementation of the method specified in
`docs/proposal11/SignPK-BA_End-to-End_Method_and_Implementation.md`.

The project keeps Hand4Whole++ and OmniHands frozen and communicates through
versioned observer caches. The final export is always standard-topology SMPL-X;
MANO output is used only as an observation.

See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) for the proposal-to-code
traceability matrix, verified smoke results, and the intentionally pending
runtime artifacts (real OmniHands cache and trained PKC checkpoint).

## Current data integration

- SGNify RGB: `../data/frames/<sign>/low_<video_id>.png`
- SGNify GT: `../data/smplx_gt/<sign>/<gt_id>.obj`
- official segmentation/classes: `../data/evaluation_from_author/{segment.json,signs.txt}`
- existing H4W++ cache: `../SignCAST/data/cache/v3/h4wpp`
- OmniHands source/checkpoints: `./OmniHands`
- Hand4Whole++ source: `../signalign-tr/third_party/hand4whole_pp`

Pinned revisions/assets used by the checked configuration:

- Hand4Whole++ commit `f81d35ddd2b74206c40142243eb62b6d64ce0d65`
- OmniHands commit `935e1f580975263be799ebf56932e27ab18e1a01`
- OmniHands video checkpoint SHA-256 `0d09f681a94c81be9bd544107102306cf47a54990c6d180b8371baffd901ad4f`
- official SGNify evaluator SHA-256 `2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300`

Restricted SMPL-X/MANO assets are referenced in place and are not copied.

## Setup and smoke tests

```bash
cd /home/haipd/DexAvatar/SignPK-BA
python -m pip install -e .
pytest -q
```

Build and validate the exact benchmark manifests:

```bash
python scripts/build_sgnify_manifest.py --config configs/data/sgnify.yaml
python scripts/validate_observers.py --config configs/data/sgnify.yaml
```

Cache OmniHands after its environment/checkpoints are active:

```bash
python scripts/cache_omnihands.py \
  --config configs/data/sgnify.yaml \
  --output-root cache/omni
```

The cache command exports model outputs before rendering and records temporal
window indices, padding ratio, checkpoint hash, git revision, coordinates, and
units. It never reads benchmark GT.

Run the three inference modes:

```bash
python scripts/fit_sgnify.py --config configs/fit/signpk_ba.yaml --mode h4w_init
python scripts/fit_sgnify.py --config configs/fit/signpk_ba.yaml \
  --mode pkc_feedforward --checkpoint checkpoints/pkc_stage_b.pt
python scripts/fit_sgnify.py --config configs/fit/signpk_ba.yaml \
  --mode signpk_ba --checkpoint checkpoints/pkc_stage_b.pt
```

`h4w_init` is runnable from the existing cache now. The learned modes require a
PKC checkpoint and OmniHands cache; they fail explicitly when either is absent.
Training is intentionally separate from observer extraction:

```bash
python scripts/train_pkc.py --config configs/train/stage_a.yaml
python scripts/train_pkc.py --config configs/train/stage_b.yaml \
  --init-from runs/pkc_stage_a/latest.pt
python scripts/train_pkc.py --config configs/train/stage_c.yaml \
  --init-from runs/pkc_stage_b/latest.pt
```

Stage C freezes every parameter except the log-variance heads. Stage A/B use a
separate uncertainty-head learning rate. Observer backbones are never part of
the training graph. The training defaults include no SGNify frame, manifest,
segment, or GT root. `--seed`, `--output-dir`, and `--device` provide explicit
multi-seed and resource control.

## Post-extraction training bridge

When SignAvatars extraction finishes, serialize each sequence as a
`DualObserverBundle` plus a `signpk-pseudogt-sequence-v1` target archive, then
create the gap-augmented training windows:

```bash
python scripts/prepare_training_windows.py \
  --sequence-index data/signavatars_sequences.jsonl \
  --output-root cache/training_windows \
  --output-index data/signavatars_windows.jsonl \
  --window-size 9 --gaps 1 2 3 5
```

The converter checks every sequence length, finite tensor, signer/sequence
identity, quality weight, and rejects any source/path containing SGNify,
`smplx_gt`, or `evaluation_from_author`. It writes detached cache tensors only;
it does not run or train an observer.

## Optional H4W++ feature hook

The existing H4W++ cache contains parameters and geometry. New extractions may
also capture body-pose, WiLoR, and HandControl features without modifying the
upstream repository:

```python
from signpk.observers.h4w_feature_hook import H4WFeatureCapture

feature_batches = []
with H4WFeatureCapture(model) as capture:
    for batch in loader:
        output = model(*batch)
        feature_batches.append(capture.pop_batch())
```

Save with `save_h4w_feature_cache(..., sign_root / "features.pt")`, then set
`h4w_hand_feature_dim` and `body_observer_feature_dim` in the model config to
the pooled feature widths. The wrapper verifies exact IDs and automatically
maps the 25 body task tokens to the 14 PKC upper-body tokens. Parameter-only
caches remain valid with both dimensions set to zero.

Evaluate by explicit manifest IDs:

```bash
python scripts/evaluate_sgnify.py --config configs/eval/trv2v.yaml \
  --prediction-root outputs/h4w_init --strict-frame-ids
```

The evaluator reports the audited metrics and can optionally reproduce the
official evaluator's class-0 left-hand exclusion. It never pairs meshes by
ordinal position.

The JSON explicitly labels results as `audited_strict`, stores the original
official evaluator path/hash, and states that the original ordinal-pairing
script was not executed. It additionally reports per-sign metrics, available
one/two-hand, interaction, velocity, disagreement and segment subgroups, plus
regional velocity/acceleration diagnostics. These diagnostics never replace
the three reconstruction metrics.

Render upright benchmark-space front/side checks with:

```bash
python scripts/render_diagnostics.py \
  --prediction-root outputs/signpk_ba \
  --output-root outputs/diagnostics --stride 10
```

## Cache contracts

Canonical observer caches use meters and camera coordinates `+x right, +y down,
+z forward`. `manifest.json` is the source of truth for frame identity. Every
cache includes a schema version and provenance metadata; incompatible caches are
rejected rather than silently converted.

## Important protocol constraints

- Only central `start:end` RGB frames enter benchmark inference.
- For this release, `gt_id = 2 * video_id` is validated against actual files.
- Reflection padding is index-based and logged for each nine-frame window.
- Left/right mirroring and coordinate transforms are explicit metadata.
- PKC residuals and BA updates are composed on SO(3), never subtracted as raw
  axis-angle vectors.
- BA keeps the best valid checkpoint from each stage and falls back to PKC/H4W.
- One-hand metadata softly downweights the non-dominant arm; it never disables it.
- Internal geometry is `+x right, +y down, +z forward`; the verified SGNify OBJ
  boundary applies `diag(1,-1,-1)` once, only during export.

## Licensing boundary

Hand4Whole++ and the parent DexAvatar repository include MIT licenses. The
checked OmniHands clone has no license file, so redistribution/use terms must be
confirmed with its authors before release. SMPL-X and MANO assets remain under
their separate licenses and are referenced in place; this project does not copy
or redistribute them.
