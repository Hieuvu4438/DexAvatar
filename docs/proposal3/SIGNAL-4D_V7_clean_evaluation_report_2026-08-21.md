# SIGNAL-4D V7 clean evaluation report

## Decision

`SIGNAL4D_V7_GTFree2DTemporalGate` is protocol-clean with respect to SGNify
training: zero frames and zero targets were used for training, calibration, or
selection. It does not improve the frozen V6 result and is therefore rejected
as the paper's main accuracy method. V6 remains the defensible primary result.

Do not report the exploratory V7 number (26.886 mm UBody-F) as a clean SOTA
result. That router learned from SGNify development target errors and selected
hyperparameters on SGNify calibration errors after the benchmark had been
inspected.

## Strict author-protocol results (1,493 frames, 57 signs)

All values are the author's translation-relative vertex-micro V2V means in mm;
lower is better.

| Method | All | UBody | UBody-F | LHand | RHand |
|---|---:|---:|---:|---:|---:|
| Frozen V6 | **42.111111** | **26.139380** | **29.519389** | 11.633903 | 11.805594 |
| V7 clean | 42.124616 | 26.144515 | 29.524480 | **11.633841** | **11.805506** |
| Δ V7−V6 | +0.013505 | +0.005134 | +0.005092 | -0.000062 | -0.000088 |

Coverage is 1.0 for both methods. Frame IDs match exactly. The clean gate chose
only `Regen/150`; on that frame UBody-F changed from 25.304101 to 32.726269 mm,
which explains the small full-set regression. No post-evaluation retuning was
performed.

## Reproducibility and material passport

| Material | Location / identity |
|---|---|
| Algorithm/config lock | commit `d2d57476c89acc2ce961cfe4b2f876b1b8bc982b` |
| Format-only metadata repair | commit `354d438aa57c54a5486f7c88c60389a52017cd4b` |
| Frozen config | `signal4d_v7_nlf_fusion/configs/v7_gtfree_2d_temporal_gate_v1.json` |
| Config SHA-256 | `5ba31d8513fb58c8cf5c5d833d6ff1c4d6e0d3a40d8eedbb1e55b9370452254a` |
| NLF observations | `signal4d_v7_nlf_fusion/outputs/nlf_v032_full1493` |
| Frozen V6 | `signal4d/runs/signal4d_v6_final_full1493_20260821/predictions` |
| 2D observation cache | `cache/phase2/lane_l_a1_ensemble_v1` |
| V7 clean run | `signal4d_v7_nlf_fusion/runs/v7_gtfree_2d_temporal_gate_v1_full1493_formatfix_20260821` |
| Gate selection SHA-256 | `b6af256c5e0cd6d4c2d88e9413b9fd67bc8bc51d8e2d8bdebc948dfde4b8043e` |
| Official evaluator wrapper | `signal4d/evaluate_author_protocol.py` |
| Evaluator SHA-256 | `e43e12a6659f0604752f0adb8b3c06cfb6ff8d910ed29137036351ee8fc44513` |
| Full manifest SHA-256 | `02e06c946f9400d8eb2b238c0297b07e188912121748db68ee1d66d12ea7c362` |
| Comparison SHA-256 | `6a8e1886bc9f389143be586fa5dadbd82fe9b9250f2333a75363b6bc0f561651` |

The predictions and evaluation reports are intentionally local ignored
artifacts; the code, locked configuration, tests, and this report are tracked.

## Honest paper wording

Safe claim: “On the complete 1,493-frame author protocol, frozen SIGNAL-4D V6
obtains 29.519 mm UBody-F. A preregistered zero-training NLF/2D temporal gate
did not improve V6 (29.524 mm), demonstrating that image-space agreement alone
does not reliably select translation-relative 3D refinements.”

Unsafe claim: “V7 achieves 26.886 mm UBody-F as a clean held-out SOTA.”

Because this benchmark had already been inspected during exploratory V7 work,
any future V7 accuracy claim needs a newly sealed evaluation set or an external
dataset protocol fixed before seeing its test metrics.
