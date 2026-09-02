# SignEFT-X

SignEFT-X reconstructs a signer-consistent SMPL-X sequence from monocular RGB
frames and then refines only the finger articulations using palm-canonical 3D
hand evidence. The released method has no Transformer, temporal network,
SignHPoser, SignBPoser, NLF, or ground-truth-dependent fitting stage.

## Method

The inference path has three operations:

1. **Frozen monocular initialization.** Each frame starts from the locked
   WiLoR-based reconstruction, with deterministic HaMeR coverage fallback.
2. **Signer-consistent canonicalization.** A robust shared SMPL-X shape is
   estimated from pose-diverse initializer frames. The sequence is re-fitted
   to this single identity while retaining the initializer's global placement
   and allowing only the upper-limb chain and hands to absorb the shape change.
3. **Palm-canonical hand refinement.** Available WiLoR hand joints are
   root-centered, rotated into a local palm frame, and scale-normalized. Only
   the 15 local finger rotations of the corresponding SMPL-X hand are fitted.
   Each joint update is bounded to 12 degrees; wrists, body, identity, face,
   translation, and camera remain fixed. If a hand observation is unavailable,
   that side falls back exactly to the canonical reconstruction.

No evaluation mesh, evaluator asset, or reference annotation can be passed to
the inference API. Evaluation is a separate command that operates only after
prediction meshes have been frozen.

## Repository layout

```text
SignEFT-X/
├── configs/inference.yaml       # complete 57-sign / 1,493-frame run
├── inputs/                      # local target-free frontend caches
├── scripts/verify_refactor.py   # exact equivalence check against frozen output
├── src/signeft/
│   ├── frontend/               # initializer and WiLoR cache adapters
│   ├── canonical/              # shared identity and canonical re-fitting
│   ├── hand/                   # bounded palm-canonical finger refinement
│   ├── evaluation.py           # isolated post-hoc evaluator adapter
│   └── pipeline.py             # target-free end-to-end inference
├── tests/
├── reference/                  # frozen reports and audit evidence
├── outputs/                    # new runs
└── _archive/research_history/  # rejected branches and historical artifacts
```

The archive is not imported by the package. It preserves prior experiments for
traceability without exposing them as part of the method.

## Prepare frozen frontend inputs

The repository includes the adapters that define the initializer and WiLoR
input contracts. Third-party WiLoR/HaMeR/SMPL-X repositories and checkpoints
are not vendored.

Build the full-coverage initializer view from already frozen primary and
fallback reconstructions:

```bash
signeft build-initializer \
  --manifest /path/to/frame_manifest.csv \
  --primary /path/to/wilor_reconstructions \
  --fallback /path/to/hamer_reconstructions \
  --output /path/to/locked_initializer
```

To regenerate the hand-observation cache, first create an RGB-hashed manifest,
run the bundled WiLoR adapter with the external repository, then import and
validate its sidecar:

```bash
signeft prepare-wilor \
  --manifests SignEFT-X/outputs/full1493/manifests \
  --output SignEFT-X/inputs/wilor_frames.json

python SignEFT-X/scripts/extract_wilor.py \
  --frame-manifest SignEFT-X/inputs/wilor_frames.json \
  --repo /path/to/WiLoR \
  --checkpoint /path/to/wilor_final.ckpt \
  --detector /path/to/detector.pt \
  --model-config /path/to/model_config.yaml \
  --out SignEFT-X/inputs/wilor_raw.pkl

signeft import-wilor \
  --manifests SignEFT-X/outputs/full1493/manifests \
  --sidecar SignEFT-X/inputs/wilor_raw.pkl \
  --output SignEFT-X/inputs/wilor_full1493_v1

signeft validate-wilor \
  --manifests SignEFT-X/outputs/full1493/manifests \
  --cache SignEFT-X/inputs/wilor_full1493_v1 \
  --output SignEFT-X/inputs/wilor_validation.json
```

## Run inference

From the DexAvatar workspace root:

```bash
python -m pip install -e SignEFT-X
signeft infer --config SignEFT-X/configs/inference.yaml
```

The default configuration checks that exactly 57 signs and 1,493 RGB frames
are present. Predictions are written to
`SignEFT-X/outputs/full1493/predictions`.

After inference, audit every frozen artifact:

```bash
PYTHONPATH=SignEFT-X/src python SignEFT-X/scripts/audit_release.py \
  --config SignEFT-X/configs/inference.yaml \
  --output SignEFT-X/outputs/full1493/release_audit.json
```

To verify that the extracted hand implementation is numerically identical to
the frozen pre-refactor implementation:

```bash
PYTHONPATH=SignEFT-X/src python SignEFT-X/scripts/verify_refactor.py
```

## Post-hoc evaluation

First freeze the evaluator layout, then invoke the official evaluator. These
commands are intentionally outside `signeft infer`:

```bash
signeft export \
  --manifest SignEFT-X/outputs/full1493/hand_manifest.jsonl \
  --predictions SignEFT-X/outputs/full1493/predictions \
  --output SignEFT-X/outputs/full1493/evaluation_layout

signeft evaluate \
  --evaluator data/evaluation_from_author/evaluate_new_fitting.py \
  --evaluator-sha256 2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300 \
  --predictions SignEFT-X/outputs/full1493/evaluation_layout \
  --reference data/smplx_gt \
  --signs data/evaluation_from_author/data/data/signs.txt \
  --segments data/evaluation_from_author/data/data/segment.json \
  --output SignEFT-X/outputs/full1493/evaluation
```

The copied Hand4Whole++-style PA-MPVPE protocol can be run directly on the
source-frame-named meshes:

```bash
PYTHONPATH=SignEFT-X/src python SignEFT-X/scripts/evaluate_pa_mpvpe.py \
  --evaluate_folder SignEFT-X/outputs/full1493/predictions/meshes \
  --gt_folder data/smplx_gt \
  --sign_file data/evaluation_from_author/data/data/signs.txt \
  --sign_seg data/evaluation_from_author/data/data/segment.json \
  --data_base_dir data/evaluation_from_author/data/data \
  --face_indices SMPLer-X/common/utils/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy \
  --output_json SignEFT-X/outputs/full1493/evaluation/pa_mpvpe.json \
  --output_csv SignEFT-X/outputs/full1493/evaluation/pa_mpvpe_per_sign.csv
```

SMPL-X, WiLoR, and HaMeR models/checkpoints are third-party dependencies and
remain subject to their original licenses.

## Verified full-sequence result

The clean pipeline was rerun on all 57 signs / 1,493 frames. The release audit
passed with 1,493 meshes, states, and decisions. The official author evaluator
and the copied PA-MPVPE protocol were invoked only after these predictions were
frozen.

| Metric | mm |
|---|---:|
| Official TR upper body | 25.7755 |
| Official TR upper body minus face | 29.0791 |
| Official TR left hand | 12.2806 |
| Official TR right hand | 11.4150 |
| PA-MPVPE upper body | 26.4008 |
| PA-MPVPE upper body minus face | 30.1391 |
| PA-MPVPE left hand | 8.1493 |
| PA-MPVPE right hand | 8.7999 |

The outputs differ from the former Transformer-based run only at the
thousandth-of-a-millimeter level, supporting the decision to omit the
Transformer rather than claim it as a contribution.
