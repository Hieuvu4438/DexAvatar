# SIGNAL-4D protocol audit

Date: 2026-08-19 (Asia/Ho_Chi_Minh)

## Isolation and assets

- Implementation root: `signal4d/`; no file under the legacy fitting method was edited.
- Runtime environment: cloned `signal4d` Conda environment, Python 3.10, PyTorch 2.1.1+cu121.
- SMPL-X neutral model SHA-256: `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.
- Source estimators are read-only local artifacts: SMPLer-X, WiLoR, and an optional legacy DexAvatar-Biomech fitted hypothesis.

## Endpoint and completeness

- The released segment table sums to 2,872 under an incompatible exclusive 30-fps interpretation.
- Actual locally available frames are sampled every two source frames. Explicit inclusive selection gives 1,493 evaluable frames in 57 clips at 15 fps.
- The clean endpoint enumerates every available frame ID rather than assuming a contiguous interval.
- All new caches contain 57/57 clips and 1,493/1,493 frames. Missing legacy hypotheses remain masked observations; they never remove an output frame.

## Frozen split v2

- Calibration: 12 clips / 260 frames, SHA-256 `6e5267f23d15e43cdb0691727abc8f94ce0aead535777297437c0ece2f685036`.
- Development: 21 clips / 578 frames, SHA-256 `fe43a102dfe36340d50015a910bf45fd4141d8df0cb9fbb83da006f10334c5f9`.
- Confirmatory test: 24 clips / 655 frames, SHA-256 `a18084794ae654eaf11cd3f57f6c231c16bdf48ad3cb656edf639fcab601cccc`.
- `Ablehnen` is quarantined to development because it was used for evaluator/coordinate smoke debugging before the first split freeze.
- SGNify lacks usable signer IDs in this release. Splits are clip-disjoint, but signer-disjointness cannot be verified; inference therefore treats each unknown signer as its clip and does not claim cross-signer generalization.

## Corrections found during implementation

1. The legacy exporter applies a 180-degree camera-X transform; failing to reproduce it caused a false 344 mm body error.
2. WiLoR's 21-joint layout contains five fingertips. Sequentially assigning its first 16 entries to the SMPL-X wrist plus 15 articulated joints misroutes four finger chains. The canonical adapter now uses indices `0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19`.
3. Observation uncertainty must be fit after the same per-frame translation alignment as the primary endpoint. An absolute-camera residual was invalid because source translation can be about 19 m.
4. A clipped-softplus sigma parameterization initialized above `sigma_max` and produced zero gradients. It was replaced by a bounded sigmoid parameterization.
5. Calibration model fitting and conformal scaling now use disjoint clip subsets (8 and 4 clips) within the calibration split.

## Invalidated artifacts retained for audit

- `artifacts/calibration/invalid_absolute_translation_seed12345`: wrong absolute-camera residual and zero-gradient sigma.
- `artifacts/calibration/invalid_same_fit_conformal_seed12345`: model fit and conformal scaling reused the same samples.
- `artifacts/calibration/invalid_pre_mano_mapping_split_conformal_seed12345`: valid split-conformal procedure on the pre-fix hand-joint mapping.
- Interrupted tmux logs explicitly state why their partial outputs are not used.

## Claim restrictions fixed before confirmatory evaluation

- No contact correctness claim: no independent contact labels or inter-rater reliability are available.
- No semantic-fidelity claim: no frozen sign recognition/retrieval evaluator with a measured GT ceiling is available for these 57 clips.
- Published DexAvatar and later-paper numbers are references only; metric labels and frame policies are inconsistent. A SOTA claim requires a same-manifest, same-evaluator paired comparison and 100% coverage.

## Prospective extended-post confirmation (v5)

- Filename-only construction froze 56 clips/769 temporally post-central frames,
  manifest SHA-256 `33825a3f1ac8aa6d063f90bc12c8061ed60680267615b6d76cbe1e8cee625b32`.
  GT vertex values were not decoded during manifest construction, fitting,
  gating, or model selection.
- The availability-only A1 hierarchy resolved to 607 balanced Ensemble frames,
  145 original-HaMeR A0 frames, and 17 terminal raw SMPLer-X A0 frames; no
  declared frame was dropped.
- The GT-free multiscale gate selected A1 for 127 frames, M1 scale 1.0 for 442,
  scale 1.5 for 7, and scale 3.0 for 193, with zero within-clip switches.
- Release freeze SHA-256
  `0c5808308b6de3f965fa50f0b05bad21cfe04e9e9161e211f2b03740f825975d`
  was written at 2026-08-20 11:43:31 +07; the prospective GT cache was created
  at 11:43:37 +07, after source/config/model/input/prediction hashes were fixed.
- Candidate and repeated gate inference were byte-identical across 112 files.
  Both evaluated methods cover exactly 56 clips/769 frames with finite metrics.
- The sole superiority endpoint passed: left-hand TR-V2V changed by -2.1411 mm
  (paired 95% clip-bootstrap CI [-2.9547, -1.4191]). Upper body improved by
  -1.0745 mm; right hand changed by +0.0020 mm and passed non-inferiority.
  Velocity, acceleration, and jerk all improved. All preregistered gates pass.
- The allowed SOTA wording is restricted to the prospective SIGNAL-4D
  extended-post SGNify endpoint versus the pre-frozen recomputed same-protocol
  A1 baseline. Sign identities overlap historical data and signer IDs remain
  unavailable, so no external-leaderboard or unseen-signer claim is made.
