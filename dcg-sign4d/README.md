# DCG-Sign4D

Isolated implementation candidate for **Dynamic-Contact-Guided Monocular 4D Sign
Reconstruction**, following
`../docs/proposal6/05_DCG_Sign4D_official_method_end_to_end.md`. This directory does not
modify the existing DexAvatar or Signal4D implementations.

## Claim boundary

The real-component development pipeline has now run end to end on SGNify: 57 clips were compiled,
development contact/diffusion checkpoints were trained, the official DPoser-X and selfcontact
backends ran, one 15-frame clip was reconstructed, and its meshes were scored with the strict
author-asset evaluator. This is an **engineering result, not a scientific DCG-Sign4D result**.
The run uses a synthetic reliability calibrator, provisional proximity labels, a provisional
patch map, a two-row bootstrap ranker, 100-step training, and a one-step/one-hypothesis inference
budget. Signer IDs, independent calibration labels, a double-annotated contact gold set, and
author-frozen protocol decisions remain unavailable.

Development defaults are accepted only by configurations that explicitly set
`experiment.development_only: true`, including `configs/smoke.yaml` and the named development
run configurations. `configs/inference/dcg_sign4d_v1.yaml` intentionally fails its production
audit while any `AUTHOR_REQUIRED` value remains unresolved.

The full status and gate evidence are in
[`reports/IMPLEMENTATION_AND_SCIENTIFIC_READINESS_20260823.md`](reports/IMPLEMENTATION_AND_SCIENTIFIC_READINESS_20260823.md).

## Implemented method path

```text
calibrated observations
  -> DexAvatar trajectory initialization
  -> shape-aware contact geometry
  -> HACO-style temporal contact proposal
  -> exact semi-Markov decoding
  -> one holistic DPoser-X-backed graph-conditioned trajectory denoiser
  -> DPS observation/contact guidance
  -> fixed-round alternating inference
  -> K independent hypotheses
  -> validation-fitted, GT-free ranking
```

Implemented contracts include immutable manifests/caches, 6D/SMPL-X trajectory encoding,
root/body/hands/face part conditioning, contact event states (`off/onset/hold/release`),
rotation-aware overlapping-window stitching, per-round artifacts, numerical failure fallback,
contact/temporal/uncertainty metrics, staged training objectives, immutable checkpoints, and
paired cluster bootstrap. Missing optional cues yield zero gradients; missing signed penetration
is fail-closed unless an explicitly development-only caller opts in. Numerical sampling failure
may retry once with the same seed and a frozen lower-guidance factor; configuration errors are
never hidden as reconstruction fallbacks.

The official selfcontact adapter verifies a hash-frozen local asset registry and aggregates the
upstream detached inside/outside mask plus differentiable closest-point distances into per-edge
penetration depth/area. A real SGNify-frame audit passed with the official generalized
winding-number signed method. The upstream optional segment test is disabled because its Python
constructor did not finish in ten minutes; this limitation is explicit in the run configuration.
The production `reconstruct` command remains fail-closed, while the explicitly development-only
configuration passes readiness and was exercised through the real component runtime.

Official source pins are recorded in `third_party/manifest.yaml`:

- DexAvatar `a0dfd427f60f5811aadb35c8657b3856d47f56b5`
- TUCH `e15732bdc6bf3f214d305f47b30bfd9ef4a85f20`
- selfcontact `08da422526419c24736c0616bca49623e442c26a`
- DPoser-X `c373fce3d364a4a0946e8445fdea5cbfd490e837`

These are reproducible engineering pins, not author-approved scientific freezes. See
`LICENSES.md` before redistributing code, weights, body models, or data.

## Install and verify

From this directory:

```bash
python -m pip install -e '.[dev]'
python -m dcg_sign4d.cli.audit_environment --config configs/smoke.yaml
python -m dcg_sign4d.cli.validate_manifest --manifest manifests/smoke.jsonl
ruff check src tests
pytest -q
```

Run the development-only synthetic path (use a new output directory because artifacts are
immutable):

```bash
python -m dcg_sign4d.cli.smoke_pipeline \
  --config configs/smoke.yaml \
  --output artifacts/smoke/synthetic_e2e_NEW
```

The verified synthetic run from this implementation is at
`artifacts/smoke/synthetic_e2e_20260823_v4/synthetic_smoke`. It used one hypothesis, one
alternating round, four diffusion steps, and had zero failures. It is a wiring test only.

Validate the complete prediction contract—including trajectory/graph tensors, ranking,
selection, source/config/manifest identities and completion marker:

```bash
python -m dcg_sign4d.cli.validate_prediction \
  --artifact artifacts/smoke/synthetic_e2e_20260823_v4/synthetic_smoke
```

Run the deterministic Stage 2/3 tiny-overfit and checkpoint lifecycle test:

```bash
python -m dcg_sign4d.cli.tiny_overfit \
  --config configs/smoke.yaml \
  --manifest manifests/smoke.jsonl \
  --third-party-manifest third_party/manifest.yaml \
  --learning-rate 0.01 \
  --device cpu \
  --output artifacts/training/tiny_overfit_NEW \
  --development-only
```

The verified fixture reduced contact loss from 4.2997 to 0.5873 and diffusion loss from 1.4276
to 0.5500 in ten steps. Its two hash-verified checkpoints are development-only and cannot be
loaded by a production call.

Audit the official sources, licenses, and reused DPoser-X weights:

```bash
python -m dcg_sign4d.cli.audit_licenses \
  --third-party third_party \
  --manifest third_party/manifest.yaml \
  --dposer-runtime-root ../DPoser-X \
  --dposer-registry configs/diffusion/dposer_x_registry.json
```

The DPoser-X bridge uses the pinned official whole-body mixed model, official normalizers and
sub-VP score function. Legacy Lightning checkpoints are loaded with a restricted
`weights_only=True` path. The current real-SGNify audit at
`artifacts/audits/dposer_x_sgnify_tisch_real_v1.json` produced finite `[1, 13, 337]` output and
reduced a two-step adapter loss from 1.3064 to 1.0786 while the official backbone remained frozen.
The subsequent 100-step development diffusion training improved validation loss from 1.9076 at
step 10 to 1.5761 at step 100. These checks establish optimization wiring; they do not constitute
scientific training under the proposal protocol.

## Reused SGNify artifacts

No DexAvatar fitting or Sapiens extraction was rerun. Existing outputs and the user-provided
models were hashed and converted into immutable, development-only artifacts:

- `artifacts/initialization/sgnify_full1493_complete_camera_user_assets_v2`: 57 clips / 1,493
  frames; exact trajectory replay, SMPL-X forward, shape and camera. The neutral model SHA-256 is
  `37602144...992` and the SMPLer-X checkpoint SHA-256 is `3d405111...33b`.
- `artifacts/observations/sgnify_full1493_synthetic_calibrated_development_v2`: 57 clips / 1,493
  frames under the strict calibrated observation schema. It is explicitly marked incomplete
  extractor provenance and uses a synthetic scalar calibrator, so it is development-only.
- `artifacts/contact/sgnify_full1493_provisional_development_v1`: 60 provisional edges; no
  gold annotation; thresholds and Euclidean/FPS patch selection are development defaults, not
  author-reviewed DCG labels.
- `artifacts/registries/selfcontact_essentials_smplx_v1.json`: exact registry for the supplied
  official SMPL-X selfcontact assets, SHA-256 `14886f1e...a09`.

Proposal-name reuse commands verified all 57 clips / 1,493 frames without copying or recomputing
tensors. Their immutable indexes are under `artifacts/indexes/`:

```bash
python -m dcg_sign4d.cli.run_initialization \
  --config configs/initialization/reuse_existing_dexavatar.yaml \
  --manifest ../signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl
python -m dcg_sign4d.cli.extract_observations \
  --config configs/observation/reuse_existing_sapiens.yaml \
  --manifest ../signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl
python -m dcg_sign4d.cli.generate_pseudo_contacts \
  --config configs/contact/reuse_existing_provisional.yaml \
  --manifest ../signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl \
  --split development
```

Production readiness evidence remains
`reports/gates/RECONSTRUCT_READINESS_current_20260823.json` and correctly blocks unresolved
author decisions. Development readiness is `reports/readiness_sgnify_development_v1.json` and is
READY for its five validation clips; that status is not transferable to a scientific run.

The old saved DexAvatar OBJs do not exactly replay their PKLs (up to 15.8891 mm on an audited
vertex); details and the safe reuse rule are in `reports/SMPLX_REPLAY_FINDING.md`.

## Strict SGNify evaluator

The evaluator enforces exact clip/frame coverage and SMPL-X topology. Its primary endpoint is
clip-macro pelvis/root-aligned hand PVE, which preserves hand placement relative to the body.
Wrist-aligned PVE is reported only for local articulation; the attached author's region-mean
translation metric is retained as a clearly labelled legacy endpoint. See
`reports/G0_EVALUATOR_AUDIT.md`.

Example baseline evaluation using the existing meshes (choose a new output directory):

```bash
python -m dcg_sign4d.cli.evaluate_sgnify \
  --manifest ../signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl \
  --predictions ../signal4d/outputs/strict_dexavatar_obj_20260820/full_1493/DexAvatar_HaMeR \
  --gt-root ../data/smplx_gt \
  --author-asset-root ../data/evaluation_from_author/data/data \
  --author-sign-file ../data/evaluation_from_author/data/data/signs.txt \
  --trusted-author-assets \
  --output reports/baselines/NEW_OUTPUT
```

Verified baseline-only results (57 clips / 1,493 frames):

| Existing output | Root-aligned hand PVE (mm) | Wrist-aligned hand PVE (mm) | Hand velocity error (mm/s) |
|---|---:|---:|---:|
| DexAvatar + HaMeR | 67.6893 | 22.3698 | 223.3767 |
| Signal4D v5 | 65.9737 | 20.1553 | 265.2477 |

The paired clip sensitivity for Signal4D v5 minus DexAvatar is -1.7156 mm for the primary
endpoint (95% interval [-2.5244, -0.9196]) but +41.8710 mm/s for hand velocity error
([22.0539, 63.4159]). This is not a signer-cluster confidence interval and neither row is a
DCG-Sign4D result.

## Real-component development run

The `Muell` smoke used the actual trained development checkpoints, complete SMPL-X/camera input,
official DPoser-X, official selfcontact signed geometry, calibrated-observation contract, contact
decoder, guided trajectory sampler and ranker. CPU inference took 239.67 s for 15 frames with one
diffusion step and one hypothesis; it completed without retry. The identical 15 frames were then
scored against the author GT:

| Output | Root hand PVE (mm) | Wrist hand PVE (mm) | Body MPJPE (mm) | Hand velocity (mm/s) |
|---|---:|---:|---:|---:|
| DCG development smoke | 93.2422 | 21.3754 | 57.1136 | 274.1976 |
| DexAvatar reference | 92.9340 | 22.0581 | 57.1258 | 268.3707 |
| Signal4D v5 reference | 89.2168 | 21.2451 | 54.8619 | 320.8183 |

The development DCG smoke is 0.3082 mm worse than DexAvatar on the primary root-aligned hand
endpoint, 0.6827 mm better on wrist-aligned articulation, and 5.8269 mm/s worse on hand velocity.
One right-hand clip cannot support a method claim, confidence interval, or ablation conclusion.
See `reports/DCG_SIGN4D_DEVELOPMENT_RUN_20260823.md` for the proposal compliance ledger and exact
limitations.

## Inputs required before a scientific DCG run

1. Written freezes for every `AUTHOR_REQUIRED` value and the official data/split/license
   manifest.
2. Detector checkpoint provenance plus independent keypoint correctness labels for calibration;
   frozen masks/tracks/depth policies if those cues remain enabled.
3. Author-reviewed SMPL-X geodesic patch map. The local signed selfcontact assets now pass a real
   frame audit, but their exact registry still needs protocol approval for scientific use.
4. A double-annotated gold contact subset with agreement and pseudo-label audit (G1).
5. Scientifically trained contact proposal and holistic DCG diffusion checkpoints, followed by
   ranker fitting on real validation candidates. The present 100-step/provisional checkpoints and
   two-row bootstrap ranker are development fixtures only.
6. Signer IDs for the required signer-cluster bootstrap.
7. Matched B0-B7 and A-INF0/A-INF1/A-K runs before any G2-G5 or method claim.

Until those are supplied, attempting to present the development caches, synthetic output, or
existing Signal4D results as DCG-Sign4D evidence is invalid.

## Calibration and contact-label gates

The corresponding proposal CLIs now exist and require explicit, preregistered thresholds:

```bash
python -m dcg_sign4d.cli.fit_calibrators --help
python -m dcg_sign4d.cli.audit_contact_labels --help
python -m dcg_sign4d.cli.audit_pseudo_contacts --help
python -m dcg_sign4d.cli.audit_g1 --help
```

Synthetic fixtures verify each code path. The composite G1 report at
`reports/gates/G1_current_20260823/G1_report.json` remains `BLOCKED`: passing synthetic agreement
and pseudo-label tests cannot substitute for real independent SGNify annotations or a frozen
patch map.
