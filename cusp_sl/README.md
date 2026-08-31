# CUSP-SL implementation

This directory is an append-only implementation of the revised CUSP-SL Methods. It imports immutable cache/schema utilities from the repository, but writes all new runs below `outputs/cusp_sl/` and never overwrites `outputs/method_hamer` or legacy checkpoints.

## Executed v1 configuration

- 51 local SMPL-X rotations: 21 body, 15 left hand, 15 right hand.
- Reliability target: probability that geodesic error is at most 15 degrees for body or 20 degrees for hands; scalar temperature is fitted on How2Sign validation data.
- Training mixture: 50% real frozen-initializer/pseudo-target pairs, 35% target-origin synthetic observation bursts, 15% clean identity examples.
- Residual convention: `R_edit = R_base Exp(g * delta)`; residuals and the reliability gate are each applied exactly once.
- Factorized temporal/joint rectified flow, 16-frame windows, velocity blending at every one of three Euler steps, four sampled candidates plus an exact base candidate.
- Candidate evidence: original-image 2D reprojection, visible 2D motion, and deterministic ROM validity. SGNify projection uses the per-frame camera matrix stored in the frozen initializer PKL. No SGNify GT is read by preprocessing, training, inference, or selection.
- The optional form scorer is disabled (`w_s=0`): no official SignDINO code/checkpoint with the required preprocessing and license was found locally or in the primary public sources. The code contains a trainable video-pose scorer, but it is not represented as an executed component.
- Exact mesh penetration is not enabled in the Python 3.13 core run. The repository's CUDA BVH binary was built for Python 3.10 only. The executed physical term is therefore ROM validity, not a mislabeled penetration metric.

## Locked protocol

`prepare_sgnify.py` reproduces the author's positional pairing and freezes exactly 1,493 rows across 57 signs. The generated input clips are targetless. GT paths occur only in the audit CSV so the evaluator pairing can be reproduced; pose/mesh GT is not loaded into the model cache.

Two evaluation tracks are deliberately separate:

1. `evaluate_author.py` invokes the original evaluator without modifying it and archives its command, hash, stdout, and stderr.
2. `evaluate_audited.py` uses explicit frame IDs, refuses incomplete or invalid coverage, verifies topology, writes per-frame errors, and calculates paired sign-cluster bootstrap intervals.

The locally reproduced frozen baseline on 1,493 frames is recorded in `outputs/cusp_sl/baseline_author_eval_v1/` and `outputs/cusp_sl/baseline_audited_eval_v2/`. The original evaluator reports several upper-body variants; do not compare its `Tr Above Pelvis Upper Body` line to a result labeled `Minus Face`.

## Executed v1 outcome and corrected diagnosis

The geometry-only flow-500 pilot did not beat the local baseline. Author-track `upper-body-minus-face / left / right` was `31.0461 / 13.5715 / 12.9212` mm versus `29.9074 / 13.5735 / 12.9271` mm. Audited coverage was 1,493/1,493 with no invalid frames. The tiny right-hand decrease is not practically meaningful, the left-hand sign-cluster interval includes zero, and upper-body error worsened materially.

A later scope audit found that this v1 execution omitted multiplication of the inference gate by the cache `refine_mask`. Among 13,437 saved out-of-scope tokens, mean gate was 0.6280 and 66.11% were nonzero; unsupported root/spine edits can move broad mesh regions. The v1 score is therefore an honest executed result but not a clean A7 Methods ablation. v2 enforces identity outside the 42 supervised joints. The deviation is not retrospectively tuned on SGNify.

The first development analysis appeared to show no oracle headroom because it measured candidates *after* the deployed Q gate. A gate-split audit corrected that interpretation. At flow step 1,000, gated best-generated error was 6.03067° versus the 6.02714° base and beat base on 3.91% of clips. The exact same sampled residuals with the gate removed achieved 5.63091° and beat base on 82.03% of clips. The generator therefore has development headroom; v1 Q/gate suppresses it. See `outputs/cusp_sl/diagnostics/development_headroom_flow1000_gate_split_v1/`.

The reason is measurable: the v1 15° body / 20° hand correctness target is 98.76% positive on inspected development tokens, Q is nearly saturated around 0.995, and its correlation with actual error is approximately 0.03. The low Brier/ECE values were dominated by class prevalence and do not establish useful discrimination.

The SGNify flow-500 result is the first test exposure. A later hands-only control is explicitly exploratory/test-exposed. The flow-1,000 checkpoint was not evaluated on SGNify.

## Development-only v2 redesign

`cusp_sl_v2_normalized.yaml` leaves v1 artifacts untouched and declares three changes before any new benchmark run:

1. Fit per-joint tangent-residual mean/std from the training split and declared real/synthetic/clean mixture only. The official HandFlow implementation likewise flow-matches normalized pose/translation targets and denormalizes predictions.
2. Retrain Q with 3° body / 5° hand tolerances and report prevalence, AUROC, average precision, balanced accuracy, Brier, ECE and risk–coverage.
3. Fit gate thresholds using fixed-seed K=1 on a hash-defined development fit fold and report once on a disjoint development audit fold.

The train-only residual statistics and full-budget Q v2 run are now executed. Statistics used 10,822 train clips, 5,288,623 weighted tokens and matched the declared mixture to 0.4981/0.3516/0.1503. The selected step-2,500 Q checkpoint has mixture-validation AUROC 0.98642, balanced accuracy 0.95202, Brier 0.04074 and ECE-15 0.03001. Its SHA-256 is `704c5dcc00b30c3e77ba427d1573f455e1624b5dcd167cb459a157b9e8ddae4e`.

These Q metrics include the declared real/synthetic/clean validation mixture and therefore do not by themselves prove reliability on natural frontend errors. The separate `evaluate_reliability_development.py` audit now passes on 334,656 uncorrupted tokens: overall/body/hands AUROC is 0.97221/0.90734/0.96918 and Pearson(q,error) is −0.88401/−0.79494/−0.90810. Overall natural Brier is 0.05522 and ECE-15 is 0.02978. The full-budget deterministic A3 checkpoint at step 10,000 reduces development overall/body/hands geodesic error from 6.0271/2.5456/7.4198° to 1.9018/1.3027/2.1414° and beats base on all 128 clips; ungated overall error is 1.7788°. Its append-only Q-bound checkpoint has SHA-256 `e10b5848610aee586a26ad4b22cd88828092a3b3fd03a0a90f8f275938e01fd1`. A separate inference run on the physically target-free 4,096-frame artifact reproduced 1.901800/1.302729/2.141429° at 24.92 CPU fps; body/hands selected-minus-base cluster CIs are [−1.2870,−1.1962]° and [−5.3901,−5.1601]°.

The normalized flow also completed all 10,000 steps. Its source checkpoint has SHA-256 `78b8ecbc919d493a37812b1524f684c4790bd756e39523996a1f23577b9cf08a` and validation flow MSE 0.83194; the bit-identical Q-bound artifact is `best_bound_v1.pt`, SHA-256 `44aef8ad6c1afc195f3f6c3f0d49a6208dd7916d08b5715e413cf7ebff6da9f5`. Source-group gate fitting selected `(tau_low,tau_high)=(0.9,1.0)` on 23 fit groups. On 29 held-out groups it improved K1 overall/body/hands from 2.3850/1.4035/2.7776° to 2.3213/1.2144/2.7640°. Targetless A4 K=1, A5 random-K4 and A7 geometry selection use bit-identical candidate seeds and exact manifest/Q/G/gate hashes. Their overall errors are 2.3032°, 2.3117° and 2.2997°, while the same K4 candidate set has A6 oracle error 2.2368°. A7's selector regret is 0.0629°. Thus the flow distribution has clear headroom, but geometry selection improves only 0.0035° over K1 and 0.0120° over the fixed random control on this pseudo-target task. These results remain source-disjoint pseudo-target evidence from a SMPLer-X-only initializer, not mocap or strong-A1 evidence. The complete gap ledger is in `IMPLEMENTATION_AUDIT.md`.

The original SMPLer-X-only How2Sign caches are not A1: their metadata records `initializer_matches_locked_lane_a1=false`, and they do not retain WiLoR's global wrist rotation. The local WiLoR exporter therefore uses a versioned record containing `global_orient`, MANO betas, detector confidence/box, focal length, camera translation, image size, and chirality metadata; detector dropout remains an explicit empty-hand frame record. An upstream-demo audit corrected the full-camera focal calculation to use the official config's `EXTRA.FOCAL_LENGTH=5000` before the accepted v3 runs. `prepare_wilor_frame_manifest.py` locks both How2Sign video seeks and protocol PNG hashes. The 128-clip development frame manifest contains 4,096 frames and has SHA-256 `a8e0a208d3070f7560dbcdef8cfeee35cb0b0a8dd8ae7e9b2520991528dc2d79`; the valid targetless protocol manifest is `outputs/cusp_sl/wilor_frame_manifest_protocol1493_v3.json` (57 signs, 1,493 unique frames, SHA-256 `8856785041c5186b25be68fde2cc375391ffa6ca44cd6f7b78d5fe1d255bc4bf`). All source PNG hashes were rechecked and the decoder smoke test returns `(height,width,channels)=(300,514,3)`. Earlier manifest versions remain negative provenance records: v1 selected fitting visualizations and v2 reversed SGNify's `height,width` tuple. Fresh v3 inference, overlays and direct-control metrics have now executed as reported below; the older 1,450-frame output is not promoted. Because v2 Q/G training caches remain SMPLer-X-only, their application to derived A1 caches is explicitly frontend-domain transfer rather than A1-conditioned retraining.

## Executed strong-A1 and test-exposed protocol

Fresh WiLoR raw-v3 inference is now complete rather than inferred from the older 1,450-frame output. The development and protocol pickles have SHA-256 `af51afbb89eaf3bb8b8fe28d2da0c45f1d9cbd9758a179b094723f62d857417a` and `0d07490dde320673404c3d41ff988f5d63c7006fcbcf86f7b74cc78838adcb59`, respectively. The protocol artifact covers exactly 1,493/1,493 frames, retains 27 explicit full-frame detector dropouts, and passed the spread-sampled original-image chirality/camera overlay audit. The source-bound strong-A1 development and targetless manifests have SHA-256 `78a02d1cdc05277d9a5722dfbbc76902451a0594f5e7a7d8140678394cedb8e6` and `2815ae708cb51496a9d3bf8e8317d662f79dc73d31a4d96099d9452557f5a1d6`.

Development model selection used no pose targets. Relative to strong A1, all frozen predictions reduce target-free observation and visible-motion evidence with 10,000-replicate source-group CIs entirely below zero. A3 changes observation/motion from 0.021338/0.025983 to 0.012458/0.017515; A4 to 0.014652/0.020342; A5 to 0.014741/0.020447; A7 to 0.013437/0.019942; and A10 to 0.013649/0.019974. The independent pseudo-target join is reported only as a mechanism audit. The strong-A1-specific gate artifacts for flow and deterministic A3 have SHA-256 `3ad2ec3f42258876cb38c5260a7a378e5c2f77a0fb995bcc4144c38fc94afb58` and `5fef659b54b1a7f33b077e339513590511f9a213c842045a5d556bbb48fedee8`; both select `(tau_low,tau_high)=(0.9,1.0)` on 23 fit source groups before a 29-group audit. Frozen energy statistics have SHA-256 `0037101436fffb251df2ebadc013fb4efdecf00808d18a75fe775b4a949fb705`.

The complete 1,493-frame author-comparability results below are test-exposed descriptive ablations. Lower is better; the local DexAvatar reproduction is 29.9074/13.5735/12.9271 mm.

| Variant | UBody(-F) | Left hand | Right hand | Interpretation |
|---|---:|---:|---:|---|
| A1 fresh SMPLer-X + WiLoR | 29.9030 | 13.1492 | 12.7304 | Frozen direct-fusion control |
| A2 support-only fixed filter | 29.8965 | 13.1306 | 12.7116 | Development lock remains A1; A2 is a control |
| A3 deterministic domain transfer | 36.0381 | 15.7783 | 15.6325 | Fails all three TR-V2V regions |
| A4 flow K=1 | 35.9999 | 16.7500 | 16.0900 | Fails all three regions |
| A5 flow K=4 random | 35.8666 | 16.7339 | 16.0395 | Fails all three regions |
| A7 geometry select | 29.9030 | 13.1492 | 12.7304 | Selects exact base on 57/57 signs |
| A10 always-on | 29.9030 | 13.1492 | 12.7304 | Selected rotations are bit-identical to A7/base |

All audited tracks cover 1,493/1,493 frames with zero failures. A3/A4/A5 method-minus-DexAvatar paired sign CIs are strictly positive for body and both hands. A7/A10 inherit A1 exactly: body CI [−0.0631, 0.0505] mm, left [−0.8189, −0.1255], and right [−0.4786, 0.0267]. Thus the proposed learned residuals improve admitted How2Sign image evidence but do not transfer to SGNify TR-V2V; the frozen geometry selector correctly abstains on every protocol sign. A8/A9 remain disabled because no validated form checkpoint/annotations passed the feasibility rule.

A11 is complete on equal-budget deterministic seeds 42/43/44. Every checkpoint has 2,431,491 parameters, 10,000 steps, and identical config/Q/residual-statistics/architecture provenance. Checkpoint SHA-256 values are `e10b5848...1fd1`, `b1410d84...2b9e`, and `a8ed7fdb...f709`; validation residual MSE is 0.21981/0.26502/0.24668. All three target-free strong-A1 runs improve observation and motion evidence: selected observation is 0.012458/0.014981/0.013925 and selected motion is 0.017515/0.018228/0.018130 versus common bases 0.021338/0.025983. Their pseudo-target overall errors are 29.8549/29.2355/29.8689° versus 30.9010°, but these are mechanism diagnostics. The provenance-checked A11 summary has SHA-256 `cad400161707b1ef408cdd5fefb5395df42b856c5989d594475bb7bbfbc8339f`; it explicitly records `target_reads=0`, `development_only=true`, and no additional protocol evaluation. Matched restarts therefore support repeatable development evidence, but do not rescue the already-observed cross-domain protocol failure. No confirmatory superiority claim is supported.

## Commands

The block below is the declared full-budget template. The executed falsification pilot used `--steps 500` for both Q and G, wrote to `outputs/cusp_sl/pilot_seed42/`, and stopped continuation at the validated step-1,000 flow checkpoint after the then-available post-gate oracle criterion failed. The later gate-split audit corrected the causal diagnosis; it did not retroactively turn the short run into a 2,500/10,000-step run.

Before v2 training, fit the release residual statistics once, then train Q and the two matched models into distinct append-only directories:

```bash
python -m cusp_sl.fit_residual_statistics \
  --config cusp_sl/configs/cusp_sl_v2_normalized.yaml \
  --output outputs/cusp_sl/v2_normalized_seed42/residual_statistics_train.npz

python -m cusp_sl.train_reliability \
  --config cusp_sl/configs/cusp_sl_v2_normalized.yaml \
  --output outputs/cusp_sl/v2_normalized_seed42/reliability

python -m cusp_sl.train_deterministic \
  --config cusp_sl/configs/cusp_sl_v2_normalized.yaml \
  --reliability-checkpoint outputs/cusp_sl/v2_normalized_seed42/reliability/best.pt \
  --output outputs/cusp_sl/v2_normalized_seed42/deterministic

python -m cusp_sl.train_flow \
  --config cusp_sl/configs/cusp_sl_v2_normalized.yaml \
  --reliability-checkpoint outputs/cusp_sl/v2_normalized_seed42/reliability/best.pt \
  --output outputs/cusp_sl/v2_normalized_seed42/flow

python -m cusp_sl.calibrate_gate \
  --config cusp_sl/configs/cusp_sl_v2_normalized.yaml \
  --manifest outputs/cusp_sl/development_how2sign_val128_v1.json \
  --reliability-checkpoint outputs/cusp_sl/v2_normalized_seed42/reliability/best.pt \
  --flow-checkpoint outputs/cusp_sl/v2_normalized_seed42/flow/best_bound_v1.pt \
  --output outputs/cusp_sl/v2_normalized_seed42/gate_calibration_source_v1

python -m cusp_sl.evaluate_development \
  --config cusp_sl/configs/cusp_sl_v2_normalized.yaml \
  --manifest outputs/cusp_sl/development_how2sign_val128_v1.json \
  --reliability-checkpoint outputs/cusp_sl/v2_normalized_seed42/reliability/best.pt \
  --flow-checkpoint outputs/cusp_sl/v2_normalized_seed42/flow/best_bound_v1.pt \
  --gate-calibration outputs/cusp_sl/v2_normalized_seed42/gate_calibration_source_v1/gate_calibration.json \
  --generator-kind flow \
  --output outputs/cusp_sl/v2_normalized_seed42/development_flow_calibrated_v1
```

```bash
python -m cusp_sl.prepare_sgnify \
  --config cusp_sl/configs/cusp_sl_v1.yaml \
  --output outputs/cusp_sl/protocol_inputs_1493_v1

python -m cusp_sl.train_reliability \
  --config cusp_sl/configs/cusp_sl_v1.yaml \
  --output outputs/cusp_sl/v1_seed42/reliability

python -m cusp_sl.train_flow \
  --config cusp_sl/configs/cusp_sl_v1.yaml \
  --reliability-checkpoint outputs/cusp_sl/v1_seed42/reliability/best.pt \
  --output outputs/cusp_sl/v1_seed42/flow

python -m cusp_sl.prepare_development \
  --source-manifest cache/phase2/t2_how2sign_2d_temporal_reprojection_v2/splits/val.json \
  --clips 128 --seed 42 \
  --output outputs/cusp_sl/development_how2sign_val128_v1.json

python -m cusp_sl.prepare_targetless_development \
  --manifest outputs/cusp_sl/development_how2sign_val128_v1.json \
  --output outputs/cusp_sl/development_how2sign_val128_targetless_v1

python -m cusp_sl.inference \
  --config cusp_sl/configs/cusp_sl_v1.yaml \
  --input-manifest outputs/cusp_sl/development_how2sign_val128_targetless_v1/manifest.json \
  --reliability-checkpoint outputs/cusp_sl/v1_seed42/reliability/best.pt \
  --flow-checkpoint outputs/cusp_sl/v1_seed42/flow/best.pt \
  --output outputs/cusp_sl/v1_seed42/development_candidates \
  --protocol-role development_validation --variant a7_geometry

python -m cusp_sl.calibrate_energy \
  --candidate-root outputs/cusp_sl/v1_seed42/development_candidates \
  --output outputs/cusp_sl/v1_seed42/energy_statistics.npz

python -m cusp_sl.inference \
  --config cusp_sl/configs/cusp_sl_v1.yaml \
  --input-manifest outputs/cusp_sl/protocol_inputs_1493_v1/manifest.json \
  --reliability-checkpoint outputs/cusp_sl/v1_seed42/reliability/best.pt \
  --flow-checkpoint outputs/cusp_sl/v1_seed42/flow/best.pt \
  --energy-statistics outputs/cusp_sl/v1_seed42/energy_statistics.npz \
  --output outputs/cusp_sl/v1_seed42/inference_a7 \
  --variant a7_geometry

python -m cusp_sl.render_predictions \
  --config cusp_sl/configs/cusp_sl_v1.yaml \
  --predictions outputs/cusp_sl/v1_seed42/inference_a7 \
  --output outputs/cusp_sl/v1_seed42/render_a7

python -m cusp_sl.evaluate_author \
  --config cusp_sl/configs/cusp_sl_v1.yaml \
  --prediction-root outputs/cusp_sl/v1_seed42/render_a7 \
  --output outputs/cusp_sl/v1_seed42/eval_author_a7
```

Use a new versioned output path for every run. Training checkpoints contain the config path/hash, software versions, optimizer state, step, and validation metrics; inference manifests additionally hash both checkpoints and the locked input manifest.

## Tests

```bash
python -m compileall -q cusp_sl
pytest -q cusp_sl/tests
```

The tests cover the SO(3) residual convention, exact identity path, gate direction and temporal dilation, model shapes/gradients, candidate validity/selection, and release configuration.

## Public-source provenance

The source/checkpoint/license decisions are tracked in
[`THIRD_PARTY_AUDIT.md`](THIRD_PARTY_AUDIT.md). In particular, a similarly
named repository or a paper project page is not treated as released model code.
Local dataset admission and the explicit rejection of the test-derived,
body-only PHOENIX aggregate are recorded in [`DATA_AUDIT.md`](DATA_AUDIT.md).

- HandFlow official repository: <https://github.com/mxxu00/HandFlow> (local clone commit `67fa7df536db233408fe6270ca5d2de28d5959c3`). Its public v1 supplies inference/configuration mechanisms, not the missing training pipeline; CUSP-SL reimplements training around repository-native SMPL-X caches.
- HandFlow project/checkpoints: <https://mxxu00.github.io/HandFlow/> and <https://huggingface.co/mxxu00/HandFlow>.
- SignDINO paper: <https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html>. No checkpoint is claimed here.
- MaskHand paper: <https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html>. Used as corruption/masking motivation only.
- HandFlow paper: <https://arxiv.org/abs/2607.11221>.

Local PDF extraction preflight sidecars are under `docs/proposal4/pdf_preflight/`. Because the configured ARS preflight could not import `pypdf`, those PDFs have no trusted page anchors in this run; implementation claims above use source-level code inspection and primary web pages rather than invented page citations.
