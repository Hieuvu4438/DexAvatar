# Training-Free Finite-Branch Evaluation Protocol

**Frozen:** 2026-09-02, before running the selector or evaluator described here.

## Scope and correction

The previously reported learned selector is invalid under the study constraint because its candidate-benefit labels and decision threshold were derived from 298 SGNify benchmark frames. Its 1,195-frame result must not be used as evidence for the training-free method.

The present protocol uses all 1,493 benchmark frames only as inference inputs and, after predictions are locked, as evaluation targets. No SGNify frame, mesh, metric, sign identity, or region index is used to fit a model, choose a threshold, choose a feature, or choose a method variant.

Because results from this benchmark have already been inspected during earlier project work, the final 1,493-frame result is descriptive and exploratory. It is not a pristine confirmatory result. A new sealed dataset is required for a confirmatory generalization claim.

## Frozen input reconstruction

The method starts from the raw framewise A3F reconstruction stored before the benchmark-specific SignEFT hand-refinement ladder. It does not start from the hand-refined incumbent selected using the 12-sign panel. The reconstruction is used at inference in the same way that an image-to-mesh method uses any initial prediction; no SGNify target is read.

## Frozen candidate construction

For each arm, projection-equivalent collar-to-shoulder, shoulder-to-elbow, and elbow-to-wrist depth branches are enumerated from the initial mesh, its calibrated camera, and its bone lengths. Each branch is realized by state-consistent SMPL-X rotations. The global wrist orientation is retained, and alternatives whose centered hand-surface change exceeds 0.5 mm are removed. The exact initial reconstruction is always present as the abstention hypothesis.

## Frozen training-free selector

The selector has zero fitted parameters. It compares each finite branch with the parametric and non-parametric 3D joint predictions emitted by the frozen NLF checkpoint.

For each NLF output and each arm bone, endpoint uncertainty is propagated to angular uncertainty as

\[
\sigma_b = \operatorname{atan2}\left(\sqrt{u_p^2+u_c^2},\lVert q_c-q_p\rVert\right).
\]

A non-incumbent branch is accepted only when all conditions hold:

1. the parametric and non-parametric NLF outputs independently prefer the same finite branch;
2. that branch has a smaller angular residual than the incumbent on every bone it changes, under both NLF outputs;
3. the two NLF outputs agree within their propagated 95% uncertainty intervals;
4. the selected branch lies within the corresponding 95% uncertainty interval under both outputs; and
5. the reduction in uncertainty-normalized squared angular residual exceeds the fixed 95% chi-square quantile for the number of changed bones under both outputs.

The fixed constants are the standard normal 95% quantile (1.9599639845) and chi-square 95% quantiles for one, two, and three degrees of freedom (3.8414588207, 5.9914645471, and 7.8147279033). They are mathematical reference constants and are not estimated from SGNify.

Failure of any condition returns the exact input reconstruction for that arm. Left and right decisions are made independently and composed before one unmodified SMPL-X forward pass.

## Process isolation

The inference configuration contains no path to SGNify ground truth, the official frame pairing, or evaluator region assets. The inference executable rejects a configuration that exposes any of those paths. Prediction artifacts and their hashes are written before the evaluator is launched. The evaluator then reads the locked predictions and computes the six official centered vertex errors.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| Full 1,493-frame manifest | `ef54791decc8ff8df44277173c24b834848ffe64c822fe5cf7011b42749eea78` |
| Inference configuration | `86b7a9a9d9cba90401352d91b2118110320cb6f4607640dfa3eca49858dc9afd` |
| Candidate generator | `1316ef3cd72e7236e66d284e2ab356e7c37355c6348a10225374e5d9a322ee7a` |
| Training-free selector core | `58451c9382e5c58a400645fcb123784f53e9026bd10d866ebac07986dd8f76d8` |
| Training-free inference executable | `6cd8ebab5bb4b75c634007ee9c15190a1c1d88d3b41b4a1ee5196617537661de` |
| Frozen NLF observation metadata | `ac302836156863febb95e7280244d02375dad2870ef0f1eaf0dc8177acef4cfb` |
| Frozen NLF checkpoint | `52bee28edb6ea9148691331df87cfc238d7e3d9134dc60104a5aaed282a9ddad` |

## Commands frozen for execution

```bash
PYTHONPATH=. python -m signdart.audit.generate_candidates \
  --config configs/training_free_inference_full1493.yaml

PYTHONPATH=. python -m signdart.audit.apply_training_free_consensus \
  --config configs/training_free_inference_full1493.yaml \
  --nlf-root ../signal4d_v7_nlf_fusion/outputs/nlf_v032_full1493 \
  --output-root runs/training_free_consensus_full1493

PYTHONPATH=. python -m signdart.audit.evaluate_selection \
  --config configs/training_free_evaluation_full1493.yaml \
  --selection-root runs/training_free_consensus_full1493 \
  --output reports/training_free_full1493/evaluation.json
```

The first two commands are inference-only. The third command is the first process permitted to read SGNify targets and evaluator assets.
