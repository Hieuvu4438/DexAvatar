# SIGNAL-4D extended-post confirmatory report

**Decision: PASS**

This report is generated from the frozen evaluator, paired-bootstrap, gate, and release artifacts. The endpoint contains 56 clips and 769 declared frames.

## Confirmatory results

All geometry values are equal-weight clip-macro TR-V2V in mm. Dynamics use the evaluator's joint-error units. Deltas are candidate minus A1; lower is better.

| Metric | A1 baseline | SIGNAL-4D | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Left hand | 24.9751 | 22.8340 | -2.1411 | [-2.9547, -1.4191] |
| Upper body | 39.8024 | 38.7279 | -1.0745 | [-1.5975, -0.6002] |
| Right hand | 13.1358 | 13.1378 | +0.0020 | [-0.0002, +0.0042] |
| Velocity error | 6.5377 | 6.5234 | -0.0143 | [-0.0265, +0.0013] |
| Acceleration error | 137.7112 | 136.8182 | -0.8929 | [-1.2831, -0.4353] |
| Jerk error | 3678.6309 | 3657.0457 | -21.5851 | [-34.8538, -7.1724] |

## Preregistered gates

- PASS — `byte_exact_gate_reproducibility`
- PASS — `complete_56_clips_769_frames`
- PASS — `dynamics_no_more_than_2pct_regression`
- PASS — `left_superiority_effect_and_ci`
- PASS — `right_hand_noninferiority`
- PASS — `upper_body_noninferiority`

## Label-blind selection evidence

The grouped historical OOF gate delta was -0.9872 mm (95% CI [-1.4238, -0.5696]), with 0 within-clip switches. Prospective GT was not available to the gate.

## Integrity and claim boundary

Release status: `frozen_before_confirmatory_test`. The release file hashes source, configs, manifest, calibration, gate artifacts, inputs, baseline and candidate predictions before the prospective GT cache is created.

This is a temporally disjoint post-segment endpoint with overlapping clip/sign identities and unavailable signer IDs. It is not an external published leaderboard or unseen-signer evaluation. Contact and semantic claims remain blocked because trustworthy annotations/evaluators are absent.
