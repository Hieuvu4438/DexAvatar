# Phase 2 uncertainty-aware whole-sequence refiner

This isolated package implements the deterministic Phase 2 system described in:

- `docs/proposal/DEXAVATAR_PHASE2_UNCERTAINTY_AWARE_WHOLE_SEQUENCE_REFINER.md`; and
- the Phase 2 stage of `docs/proposal/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md`.

It never imports training targets from SGNify evaluation artifacts. Existing DexAvatar and Phase 1 outputs are read-only initializers. Cache construction is append-only, prediction commands refuse overwrites by default, and inference rejects an output path that would replace source initializer PKLs.

## Implemented system

- backward-compatible cache schema v2 with frame IDs/numbers/timestamps/FPS, image and source hashes, coordinate transforms, shared shape, 2D/3D observations with explicit masks, torso/wrist geometry, palm normals, U0 reliability, target geometry hooks, and provenance;
- provider-neutral interfaces for future external observation and target datasets;
- explicit source-/signer-disjoint split-index builder;
- 43-D joint tokens containing SO(3) state, target-motion features, reliability, masked 2D/3D geometry, palm context, and frame-gap timing;
- reliability-weighted spatial, bidirectional temporal, and body/arm/hand group attention with learned relative temporal bias;
- zero-initialized, gated, bounded SO(3) residuals, auxiliary joint positions, palm normals, U0 reliability, and calibration-gated U1 variance;
- realistic contiguous upper-body, hand, both-hand, finger-chain, wrist, crop, missing-keypoint, and handedness-swap corruption;
- rotation, balanced regional vertex hook, joints, fingertips, palms, observation likelihood, target motion, anchor, biomechanical, and heteroscedastic losses;
- BF16, EMA, no-decay norm/bias groups, gradient accumulation/clipping, early stopping, periodic recovery, complete optimizer/scheduler/RNG checkpoints, safe resume, and optional compatible spatial-prior initialization;
- whole-clip/sliding-window inference with quaternion hemisphere blending, calibrated uncertainty, groupwise fallback, exact coverage checks, checkpoint/cache hashes, standard PKLs, and source-anchored meshes; and
- strict common-manifest regional TR-V2V with topology/coverage/stale-output checks and sign-clustered statistics.
- executable 50/25/25 T2 residual mixing, fail-closed independent-target audits,
  matched U0/U1 calibration export, and an exact eight-condition G6 decision.

Dataset-specific readers, licensed targets, DPoser-X state translation, and external target-quality filters intentionally remain provider additions. The neutral contracts are in `phase2_refiner/data/providers.py` and `phase2_refiner/models/pretrained.py`.

## Available local training adapters

InterHand2.6M is integrated as official train/validation, hands-only Tier-B
supervision. Its official test annotations are never read by the builder, and
SGNify ground truth is forbidden as a training target:

```bash
python -m phase2_refiner.data.build_interhand_cache \
  --annotations data/InterHand2.6M/annotations \
  --output cache/phase2/interhand_t1_v1 \
  --splits train val --min-frames 8 --max-frames 64

python -m phase2_refiner.data.audit_training_cache \
  --train-manifest cache/phase2/interhand_t1_v1/splits/train.json \
  --val-manifest cache/phase2/interhand_t1_v1/splits/val.json \
  --output cache/phase2/interhand_t1_v1/readiness_report.json
```

The supplied SignBPoser data are a shuffled body-pose bank, not a temporal
dataset. The adapter therefore emits one-frame, body-only spatial warm-up
samples and explicitly marks temporal supervision false:

```bash
python -m phase2_refiner.data.build_signbposer_cache \
  --input data/signbposer_data/train \
  --output cache/phase2/signbposer_spatial_v1
```

Partial rotation masks are enforced by the cache, corruption curriculum, and
all target losses. Hand-only samples cannot supervise identity body rotations,
and body-only samples cannot supervise identity hands.

The completed seed-42 T1 feasibility run was launched non-interactively with an external
append-only text log so the checkpoint directory is still protected by the
non-overwrite preflight:

```bash
tmux new-session -d -s phase2_t1_interhand_run2_20260724 \
  "cd /home/haipd/DexAvatar && set -o pipefail && \
   PYTHONUNBUFFERED=1 python -u -m phase2_refiner.train \
   --config phase2_refiner/configs/uawsr_t1_interhand.yaml \
   --output-dir outputs/phase2_training/t1_interhand_seed42_run2 \
   2>&1 | tee -a logs/phase2/t1_interhand_seed42_run2_20260724.txt"
```

This is T1 synthetic hand-corruption feasibility only. The readiness audit
must report `G2_main_training: true` before this can be called complete Phase 2
training. G2 also requires the threshold volume to be explicitly tagged as
sign-domain motion; a sufficiently large generic motion cache cannot pass it.

The best EMA checkpoint recovered `50.94%`, `50.70%`, and `48.71%` of injected
rotation error for fixed 4-, 8-, and 16-frame bursts, respectively. These pass
the T1 rotation proxy but do not close G3 without decoded regional vertex and
clean-preservation measurements.

After training, evaluate the exact 4/8/16-frame recovery buckets. The command
refuses to mark G3 as passed until decoded regional vertex preservation is also
available:

```bash
python -m phase2_refiner.evaluate_t1_recovery \
  --config phase2_refiner/configs/uawsr_t1_interhand.yaml \
  --checkpoint outputs/phase2_training/t1_interhand_seed42_run2/best.pt \
  --output outputs/phase2_training/t1_interhand_seed42_run2/t1_recovery.json
```

For decoded regional vertex recovery, use the SMPL-X geometry evaluator:

```bash
python -m phase2_refiner.evaluate_t1_vertices \
  --config phase2_refiner/configs/uawsr_t1_interhand.yaml \
  --checkpoint outputs/phase2_training/t1_interhand_seed42_run2/best.pt \
  --output outputs/phase2_training/t1_interhand_seed42_run2/t1_vertex_recovery.json \
  --model-folder SMPLer-X/common/utils/human_model_files \
  --vertex-ids SMPLer-X/common/utils/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl
```

Both formal evaluators default to FP32 inference. Pass
`--eval-precision training` only to reproduce BF16/FP16 training-time
validation; reduced-precision quantization is not used for the formal clean
preservation gate.

Build the subject-disjoint complete ARCTIC cache directly from its raw ZIP:

```bash
python -m phase2_refiner.data.build_arctic_cache \
  --archive data/ARCTIC/downloads/raw_seqs.zip \
  --output cache/phase2/arctic_t1_v1
```

## 1. Build a locked cache

Create a new versioned directory each time. `--overwrite` is intentionally rejected.

```bash
python -m phase2_refiner.data.build_observation_cache \
  --frames data/frames \
  --initializer outputs/<complete_phase1_method> \
  --frame-manifest probes/results/phase0/frame_manifest.csv \
  --provenance-json configs/<initializer_provenance>.json \
  --fps 25 \
  --out cache/phase2/v2
```

The locked manifest makes a missing initializer frame fatal instead of silently shortening a sequence. Do not pass SGNify meshes, author-evaluation data, or any benchmark derivative as `--target-root`.

## 2. Build explicit splits

Prepare a CSV with `clip_id,split,source,signer`. `source` identifies the original video/source group, not merely the dataset name.

```bash
python -m phase2_refiner.data.build_sequence_index \
  --cache-root cache/phase2/v2 \
  --assignments configs/phase2_split_assignments.csv \
  --output cache/phase2/v2/splits
```

The command rejects signer or source overlap between train, validation, test, and calibration.

## 3. Identity contract

```bash
python -m phase2_refiner.infer \
  --config phase2_refiner/configs/uawsr_u0.yaml \
  --cache cache/phase2/v2 \
  --output outputs/phase2_t0_identity \
  --identity --render
```

`--identity` is explicit because an untrained identity model is not a Phase 2 quality result.

## 4. Train U0

Only start after independent sequence targets and the proposal's G2 gate pass.

```bash
python -m phase2_refiner.train \
  --config phase2_refiner/configs/uawsr_u0.yaml
```

Resume without losing optimizer, scheduler, EMA, or RNG state:

```bash
python -m phase2_refiner.train \
  --config phase2_refiner/configs/uawsr_u0.yaml \
  --resume outputs/phase2_training/uawsr_u0/last.pt
```

`--identity-target` is restricted to plumbing/T0 smoke tests. `--spatial-init` accepts an adapter-produced checkpoint with identically named and shaped tensors and reports every loaded tensor.

## 5. Calibrate and run U1

T2 must use independently produced targets. Audit them before training; an
identity/pseudo-teacher cache cannot pass this command:

```bash
python -m phase2_refiner.data.audit_real_residual_cache \
  --train-manifest cache/phase2/t2_real_v1/splits/train.json \
  --val-manifest cache/phase2/t2_real_v1/splits/val.json \
  --calibration-manifest cache/phase2/t2_real_v1/splits/calibration.json \
  --output cache/phase2/t2_real_v1/real_residual_audit.json

python -m phase2_refiner.train \
  --config phase2_refiner/configs/uawsr_t2_real_residual.yaml \
  --spatial-init outputs/phase2_training/t1_how2sign_geometry_seed42/best.pt
```

For the available How2Sign H32 cache, the reproducible Tier-C proxy is built by
optimizing the frozen per-frame poses against the independently supplied ordered
133-point tracks with temporal and bounded-pose constraints:

```bash
python -m phase2_refiner.data.refine_how2sign_targets \
  --train-manifest cache/phase2/how2sign_t1_v1/splits/train.json \
  --val-manifest cache/phase2/how2sign_t1_v1/splits/val.json \
  --output cache/phase2/t2_how2sign_2d_temporal_v1 \
  --train-clips 11000 --validation-clips 500 --calibration-clips 500 \
  --batch-size 64 --iterations 30 --device cuda
```

The builder corrects COCO-WholeBody versus SMPL-X hand-joint ordering, rejects
clips that do not improve the independent signal, and writes source-disjoint
manifests. This is valid proxy/pretraining evidence, but H32 is not the exact
Lane-L ensemble A1. Formal G4 must therefore add `--require-locked-initializer`
to the audit; the command intentionally fails until exact-A1 Tier-C caches exist.

```bash
python -m phase2_refiner.data.audit_real_residual_cache \
  --train-manifest cache/phase2/t2_how2sign_2d_temporal_v1/splits/train.json \
  --val-manifest cache/phase2/t2_how2sign_2d_temporal_v1/splits/val.json \
  --calibration-manifest cache/phase2/t2_how2sign_2d_temporal_v1/splits/calibration.json \
  --output cache/phase2/t2_how2sign_2d_temporal_v1/proxy_residual_audit.json

# Formal audit: expected to fail closed until the exact ensemble A1 is cached.
python -m phase2_refiner.data.audit_real_residual_cache \
  --train-manifest cache/phase2/t2_how2sign_2d_temporal_v1/splits/train.json \
  --val-manifest cache/phase2/t2_how2sign_2d_temporal_v1/splits/val.json \
  --calibration-manifest cache/phase2/t2_how2sign_2d_temporal_v1/splits/calibration.json \
  --require-locked-initializer \
  --output cache/phase2/t2_how2sign_2d_temporal_v1/formal_exact_a1_audit.json
```

After proxy T2 training, `evaluate_residual_checkpoint` reports untouched
regional SO(3) gains, the frozen observation-difficulty subset, clip-bootstrap
intervals, and fallback. Its formal G4 evidence bit remains false for a proxy
audit even when the numerical experiment improves.

The delegated catastrophic-target audit is reproducible rather than an
undocumented checkbox:

```bash
python -m phase2_refiner.data.render_how2sign_audit \
  --queue outputs/phase2_gates/g2/how2sign_manual_audit_100.csv \
  --output outputs/phase2_gates/g2/how2sign_visual_audit_100_v1 \
  --device cuda
python -m phase2_refiner.data.complete_visual_audit \
  --queue outputs/phase2_gates/g2/how2sign_manual_audit_100.csv \
  --evidence-manifest outputs/phase2_gates/g2/how2sign_visual_audit_100_v1/manifest.json \
  --output-csv outputs/phase2_gates/g2/how2sign_ai_visual_audit_100_completed.csv \
  --output-report outputs/phase2_gates/g2/how2sign_ai_visual_audit_100_report.json \
  --reviewer "Codex (OpenAI), delegated by project owner"
```

The completion command requires a rendered image for every queue row, records
the review modality, and never overwrites the original pending queue.

The T2 config implements the proposal's 50% untouched real residual, 25%
synthetic burst from the target, and 25% clean identity mixture. Cached
reprojection residuals remain in physical normalized-image coordinates; the
dataset applies the explicit `data.reprojection_residual_scale` only when
forming model tokens. Non-real mixture rows receive zero reprojection residual
because their initializer has been replaced and the cached H32 residual would
otherwise be geometrically stale. Checkpoint selection uses the proposal's
clip-balanced, equal-region external-validation score on EMA weights and logs
the raw-weight score alongside it. The 45-feature model also has a
zero-initialized cross-joint reprojection skip: it is exactly identity-safe at
initialization, while giving the per-frame residual a direct bounded SO(3)
correction path in parallel with the whole-sequence transformer.

Train U1 only after T2/G4 passes, then export matched errors against U0 on the
same disjoint calibration split:

```bash
python -m phase2_refiner.evaluate_uncertainty \
  --manifest cache/phase2/t2_real_v1/splits/calibration.json \
  --u1-config phase2_refiner/configs/uawsr_u1_real_residual.yaml \
  --u1-checkpoint outputs/<u1_experiment>/best.pt \
  --u0-config phase2_refiner/configs/uawsr_t2_real_residual.yaml \
  --u0-checkpoint outputs/<t2_experiment>/best.pt \
  --real-residual-audit cache/phase2/t2_real_v1/real_residual_audit.json \
  --output outputs/<u1_experiment>/calibration_residuals.npz
```

```bash
python -m phase2_refiner.calibrate \
  --residuals outputs/<u1_experiment>/calibration_residuals.npz \
  --output outputs/<u1_experiment>/calibration.json

python -m phase2_refiner.infer \
  --config phase2_refiner/configs/uawsr_u1.yaml \
  --cache cache/phase2/v2 \
  --checkpoint outputs/<u1_experiment>/best.pt \
  --calibration outputs/<u1_experiment>/calibration.json \
  --output outputs/phase2_u1 --render
```

U1 inference refuses a missing or failed calibration report. Use U0 when the calibration gate fails.

## 6. Strict evaluation

```bash
python -m phase2_refiner.evaluate \
  --manifest probes/results/phase0/frame_manifest.csv \
  --prediction outputs/<phase2_method> \
  --baseline outputs/<phase1_method> \
  --output outputs/<phase2_method>/evaluation
```

For incomplete Phase-1 candidates, create a full-coverage symlink view without
copying or modifying either source method:

```bash
python -m phase2_refiner.data.build_locked_fallback_view \
  --manifest probes/results/phase0/frame_manifest.csv \
  --primary outputs/<candidate> --fallback outputs/method_hamer \
  --output outputs/phase2_gates/g1_views/<candidate>
```

`phase2_refiner.gates --g6 ...` rejects fewer than three seeds and evaluates
all numerical G6 requirements, including hard/clean subsets and fallback rate.

## Validation

```bash
ruff check phase2_refiner
pytest -q phase2_refiner/tests
python -m compileall -q phase2_refiner
```
