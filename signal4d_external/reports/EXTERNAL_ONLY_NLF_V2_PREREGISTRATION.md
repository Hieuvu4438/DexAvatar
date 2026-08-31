# SIGNAL4D external-only NLF V2 preregistration

Date locked: 2026-08-24

## Objective and immutable baselines

The objective is to improve the frozen external-only V1 result without using
SGNify parameters, meshes, labels, author errors, or partitions to train a
model or select a hyperparameter. SGNify RGB frames may be read once the model
and all selection rules are frozen, because they are inference inputs.

Frozen V1 artifacts remain append-only:

| Artifact | SHA-256 |
|---|---|
| V1 checkpoint `best.pt` | `09ca34e42e88b550af3b046f1f35bff30ae0c66aff65b9cfba430594ccdb90bf` |
| V1 calibration | `a3180aa4c203af61892d74ea297193bc5ae40d3f8230a966252b36823909785e` |
| V1 final author comparison | `2037d545dd9dc0c7f61ab889161171532556398e85289903ed58708bd36af94ab` |

The already revealed author-protocol numbers are:

| Method | All | UBody | UBody-F | LH | RH |
|---|---:|---:|---:|---:|---:|
| external-only V1 | 42.2423 | 26.2236 | 29.6196 | 12.8102 | 12.1148 |
| historical SIGNAL4D V6 | 42.1116 | 26.1394 | 29.5197 | 11.6339 | 11.8056 |

These values define the comparison but cannot be used to tune V2.

## Failure diagnosis

V1 trained on a How2Sign `SMPLer-X H32` initializer, while target inference
uses a WiLoR/HaMeR initializer. Its externally calibrated benefit gate then
accepted only 6 of 4,479 region-frames on SGNify. The learned refiner therefore
contributed almost none of V1's gain. ARCTIC supplies accurate generic
hand-object motion but no local RGB frames in the current workspace;
InterHand2.6M supplies partial hand targets; the local SignAvatars download is
incomplete. Increasing V1 model capacity does not address this observation and
initializer mismatch.

## Research basis

- [ScoreHMR (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Stathopoulos_Score-Guided_Diffusion_for_3D_Human_Recovery_CVPR_2024_paper.html)
  supports fixed-prior, observation-guided refinement rather than retraining on
  the target domain.
- [TokenHMR (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Dwivedi_TokenHMR_Advancing_Human_Mesh_Recovery_with_a_Tokenized_Pose_Representation_CVPR_2024_paper.pdf)
  motivates tolerance-aware loss: ignore small noisy 2D deviations and retain
  large alignment errors.
- [NLF (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd23a1f3bc89e042d70960b466dc20e8-Abstract-Conference.html)
  provides non-parametric image evidence and per-point uncertainty complementary
  to a parametric initializer.
- [Dyn-HaMR (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html)
  supports hierarchical initialization followed by temporal, multi-objective
  refinement.
- [WiLoR (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html)
  supports retaining the strong in-the-wild hand expert rather than replacing
  its articulation wholesale.

The local Phase-3 v5 experiment independently established the same structural
lesson: residual completion centred on a strong corrupted initializer passed
its external geometry gate, whereas Gaussian/unconditional completion failed.
Its documented checkpoint is no longer present, so V2 does not reuse that
model or claim its result as a V2 artifact.

## Locked V2 method

1. Extract NLF 0.3.2 observations from deterministic How2Sign subsets and from
   SGNify inference images. Preserve NLF parametric/non-parametric 2D and 3D
   outputs and uncertainty.
2. Construct one upper-body candidate by SO(3) geodesic interpolation between
   the frozen initializer and NLF. Copy face and both hand poses exactly from
   the initializer. Compensate local wrist rotations so both global wrist
   orientations remain exactly those of WiLoR. The kinematic tree is loaded
   from the external ARCTIC copy of the neutral SMPL-X model, not from an
   SGNify/author-evaluation asset.
3. Fit a random-forest benefit regressor using only external How2Sign train
   signers. Observable features are NLF uncertainty, parametric/non-parametric
   fit, tolerance-aware 2D residual, initializer reprojection residual,
   initializer--NLF disagreement, temporal velocities, visibility, detection
   score/area, and torso scale.
4. Candidate alpha and temporal transition penalty are selected only on
   signer-disjoint How2Sign validation. The conservative selection margin is
   selected only on a third signer-disjoint How2Sign calibration partition.
5. Calibration passes only if external mean gain is positive, at least 1% of
   frames are selected, and no calibration signer regresses by more than 0.25
   degrees in the weighted upper-body pseudo-target metric.
6. If calibration fails, do not run SGNify inference. If it passes, materialize
   all 1,493 predictions without SGNify targets, freeze hashes, and run the
   author evaluator exactly once.

Candidate grids are fixed before execution:

- alpha: `{0.25, 0.50, 0.75, 1.00}`;
- temporal transition penalty: `{0, 0.25, 0.50, 1.00}` degrees;
- selection margin: `{0, 0.25, 0.50, 1.00, 2.00}` degrees;
- random forest: 600 trees, minimum leaf 12, feature fraction 0.8, seed 4202.

### Pre-execution temporal-contract amendment

The final static audit, still before NLF extraction, external router fitting,
or any V2 target-label evaluation, found that the 32-frame How2Sign caches are
temporally subsampled while the target cache has fixed 0.08-second spacing. Raw
per-index rotation differences would therefore encode sampling density rather
than motion speed. The locked implementation normalizes both initializer and
NLF angular velocities to a `1/15 s` reference interval and exposes the actual
time gap as `time_gap_reference_units`. Viterbi switching cost is multiplied by
`min(1, reference_interval / elapsed_interval)`, so distant samples are not
incorrectly forced to share a state. The alpha, transition, margin, forest, and
decision grids above are unchanged.

The audit also locks a cross-domain NLF observation contract. External and
target observations must have identical TorchScript model SHA-256, NLF source
commit, augmentation count, detector threshold, and person-selection rule.
Device and manifest paths may differ. Materialization fails closed on a
contract mismatch.

Frozen post-audit implementation hashes:

| Artifact | SHA-256 |
|---|---|
| `nlf_v2_core.py` | `654a7de5e04b1a2116d8f5662337a1a264d11991458c9dc67ff2519dcda9a9a2` |
| `extract_external_nlf_v2.py` | `825c1e67b6310fc198366128eac6a89dd0ae0034df2671267c9e552828a97e0d` |
| `train_nlf_router_v2.py` | `0598a3bdceae0d4ab3fc16a11ea26574df587517074b6501cda0f619f4a51583` |
| `materialize_nlf_v2.py` | `a8b4aefb9462bea87a4837687600ad365c2c421b6eae12b57adf180cfc4df54c` |
| protocol manifest | `182f436802f2d36e3c4791f9048267782ae75efc58a980d9e270ec493dee1b58` |

## Data split and execution budget

Deterministic seed 42 sampling uses 512 train clips, 192 validation clips, and
192 calibration clips. Sampling round-robins signer identities and first
maximizes unique source videos. Expected maximum observation volume is 28,672
frames. The split must remain signer-disjoint and every cache metadata record
must declare zero SGNify training reads.

## Decision and reporting rules

Primary target after the one-time reveal is `All <= 42.1116 mm`, with the
historical V6 regional numbers reported descriptively. V2 is considered a
useful scoped improvement if it beats V1 on All and both upper-body metrics
without worsening either hand by more than 0.10 mm. Any other result is
reported as a negative or scoped result; thresholds will not be loosened after
viewing SGNify errors.

Because the research program has already inspected this benchmark repeatedly,
even a successful run is developmental evidence. A pristine confirmatory claim
requires a newly sealed dataset.
