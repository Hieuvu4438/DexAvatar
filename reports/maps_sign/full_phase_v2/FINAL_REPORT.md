# MAPS-Sign learned full-phase report

## Outcome

The frozen MAPS candidate does not improve all three local reference point estimates.
 This is a comparison against the strongest locally reproduced full-coverage DexAvatar ensemble, not a general SOTA claim. How2Sign targets are independent same-view 2D-refined pseudo-SMPL-X and are not certified 3D ground truth.

| Method/control | UBody-F (mm) | LHand (mm) | RHand (mm) | Delta vs reference (U/L/R) |
|---|---:|---:|---:|---:|
| Frozen ensemble reference | 29.534720 | 12.824893 | 12.112852 | +0.000000 / +0.000000 / +0.000000 |
| candidate | 31.788012 | 15.052314 | 16.289365 | +2.253292 / +2.227422 / +4.176513 |
| global | 31.799559 | 15.074832 | 16.306729 | +2.264839 / +2.249939 / +4.193877 |
| all_stable | 31.781586 | 15.071102 | 16.303204 | +2.246866 / +2.246209 / +4.190352 |
| all_transition | 31.891945 | 15.091480 | 16.355174 | +2.357225 / +2.266587 / +4.242322 |
| shifted | 31.791333 | 15.052890 | 16.283689 | +2.256613 / +2.227997 / +4.170837 |
| shuffled | 31.787624 | 15.050315 | 16.286136 | +2.252904 / +2.225422 / +4.173283 |
| independent | 31.788035 | 15.052025 | 16.288726 | +2.253315 / +2.227132 / +4.175874 |
| post-failure input-only safety guard | 29.534720 | 12.824893 | 12.112852 | +0.000000 / +0.000000 / +0.000000 |

All rows use the reconstructed author protocol, translation registration, the authors' vertex subsets and one-hand exclusion rule, with 1,493/1,493 frames.
The safety-guard row was designed only after the frozen candidate failed and is therefore a deployment no-harm result, not a pre-registered benchmark improvement.

## Paired sign-cluster inference

Differences are candidate minus reference; negative is better. Intervals use 10,000 sign-cluster bootstrap samples and p-values use 100,000 sign-flip permutations with Holm correction across the three co-primary regions.

| Region | Paired signs | Mean delta (mm) | 95% CI (mm) | Holm p | Improved / worsened |
|---|---:|---:|---:|---:|---:|
| UBody-F | 57 | +2.116780 | [+1.744037, +2.487510] | 2.99997e-05 | 6 / 51 |
| LHand | 42 | +2.120434 | [+1.581406, +2.663572] | 2.99997e-05 | 5 / 37 |
| RHand | 57 | +4.042631 | [+3.555625, +4.527906] | 2.99997e-05 | 1 / 56 |

## Validation selection and parser evidence

Selection used only the frozen How2Sign validation split. The chosen candidate is `pair42_314_coupled` with `coupled` decoding and an equal-region mean target-rotation error of 2.300669 degrees.

Selected checkpoint(s):<br>/home/haipd/DexAvatar/outputs/maps_sign_full/runs_v2/seed_42/checkpoint_epoch_020.pt<br>/home/haipd/DexAvatar/outputs/maps_sign_full/runs_v2/seed_314/checkpoint_epoch_017.pt

Pseudo-state accuracy is 0.7912, macro-F1 is 0.7843, and tolerance-one boundary F1 is 0.7289. These labels are motion-quantile weak labels, so the parser metrics are mechanism diagnostics rather than human phonological annotation accuracy.

Held-out signer-10 transfer (220 clips, evaluated once after selection): 2.503704 degrees versus 5.480230 degrees for its initializer.

Held-out regional fallback scales: lhand=1.00, rhand=1.00, ubody=1.00.

## Training runs

| Seed | Epochs | Best epoch | Best validation mean (deg) | Seconds | Status |
|---:|---:|---:|---:|---:|---|
| 42 | 20 | 20 | 2.325011 | 2073.8 | complete |
| 314 | 20 | 17 | 2.321409 | 2383.8 | complete |
| 2026 | 20 | 20 | 2.342853 | 2330.5 | complete |

The model has 642,952 parameters. Training used 10,822 clips / 346,304 frames; validation used 498 / 15,936; calibration used 497 / 15,904. Numerical libraries were capped at two threads and data loading at two workers.

## Phase status

| Phase | Result |
|---|---|
| Isolation and protocol freeze | PASS: standalone branch/worktree; original methods unchanged |
| Cache and state calibration | PASS: source-disjoint manifests; 335,482 train motion steps |
| Learned unary parser | COMPLETE: metrics reported above |
| Semi-Markov selection | COMPLETE: `coupled` selected; opposite decoder retained as control |
| Reliability/safety | COMPLETE: input-only calibration-support guard preserves the initializer outside support |
| Three-seed/generalization | COMPLETE: three seeds plus unseen signer-10 transfer |
| Frozen benchmark | COMPLETE: candidate and causal controls, 1,493/1,493 coverage |

## Root-cause audit

The earlier oracle joint/skinning proxy remains a recorded negative result. The redesign adds an externally learned correction direction, composes bounded rotations on SO(3), calibrates states from training motion, uses the complete strong initializer, and normalizes observation residuals per clip. It never imports or overwrites the workspace's Phase2/Phase3 method implementations.
The frozen SGNify failure persists across every state control, while all 57 official clips fall below the held-out calibration reliability support. This supports cross-domain residual-transfer failure as the dominant cause rather than a boundary-decoder failure. The post-failure guard responds by applying zero residual outside calibration support; it does not establish an improvement.

## Reproducibility and retained artifacts

- Branch revision: `0dee0a00c35b0e0a1b77d49d14cadcbfad442228`
- Selection: `/home/haipd/DexAvatar/outputs/maps_sign_full/selection_v2/selection.json`
- Runs: `/home/haipd/DexAvatar/outputs/maps_sign_full/runs_v2`
- Official records and statistics: `/home/haipd/DexAvatar/reports/maps_sign/full_phase_v2`
- Prediction roots: `outputs/maps_sign_full/official_*_v2`
- Durable tmux logs: `reports/maps_sign/full_phase_v2/*.log`

No overall SOTA claim is made without unified reruns of every relevant external comparator. The defensible claim is limited to the exact local reference, coverage, protocol and statistics reported here.
