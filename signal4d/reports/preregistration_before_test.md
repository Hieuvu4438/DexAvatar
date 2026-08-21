# SIGNAL-4D confirmatory preregistration

This decision record is written before loading or evaluating the clean 24-clip test targets.

## Frozen endpoint

- Test manifest: 24 clips / 655 frames.
- Manifest SHA-256: `a18084794ae654eaf11cd3f57f6c231c16bdf48ad3cb656edf639fcab601cccc`.
- Primary aggregation: equal-weight clip macro.
- Primary metric: per-frame, per-region translation-aligned V2V in millimeters.
- Missing-frame policy: any missing/reordered prediction invalidates the entire run.

## Methods

- Strongest baseline: complete legacy-Biomech pose when available, deterministic M0 fallback otherwise.
- Candidate: SIGNAL-4D M1 with the same initializer/fallback, calibrated soft observation weights, change-point adaptive temporal factors, and left elbow/wrist/hand refinement.
- M0 raw hybrid is reported as a secondary cheap baseline.
- M2 is excluded from the primary claim because G4 real contact-label reliability failed before test evaluation.

Method-independent stress slices are frozen in
`configs/protocol/stress_slices.json`: short clips have at most 24 frames and
long clips have more than 24 frames. Signer/language/sign-type slices are not
reported because the corresponding metadata is unknown. Slice results are
secondary diagnostics and do not alter the primary gate.

## Confirmatory gates

Paired clip bootstrap: 10,000 replicates, seed 20260819. Because signer IDs are unavailable, each clip is its own cluster.

1. Coverage must equal 100% for baseline and candidate.
2. Left-hand TR-V2V candidate-minus-baseline must be below zero with a 95% CI upper bound below zero. The point estimate must improve by at least 0.5 mm to count as practically meaningful.
3. Upper-body and right-hand TR-V2V are non-inferior when the paired 95% CI upper bound is below +0.5 mm.
4. Velocity, acceleration, and jerk error must not regress by more than 2% in point estimate; all are reported regardless of gate outcome.
5. No configuration, weight, threshold, source mapping, or stopping rule may change after the first clean test metric is revealed. A software-only bug would require an audit entry and complete rerun of every test method.
6. The full smoke run must reproduce within 5% on every primary metric; deterministic tensor equality is the target.

## Claim rule

If gates 1–6 pass, the permitted wording is: “new best result on the frozen SIGNAL-4D clean SGNify protocol against the strongest recomputed same-protocol baseline.” This is not interchangeable with the published 2,872-frame DexAvatar number because endpoint and evaluator definitions differ. Contact correctness, semantic fidelity, cross-signer generalization, and real-time claims remain prohibited.
