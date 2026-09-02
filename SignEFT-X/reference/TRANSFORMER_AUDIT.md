# Transformer exclusion audit

## Decision

The Transformer is excluded from SignEFT-X. It is neither imported nor invoked
by `src/signeft`, and `configs/inference.yaml` consumes the direct frozen
WiLoR/HaMeR initializer rather than the Transformer-refined cache.

## Evidence

The frozen full-sequence diagnostic reported only **6 accepted region-frames
out of 4,479** (three regions for each of 1,493 frames). A parameter-level
comparison found learned pose changes in exactly six frames, all confined to
the right-hand pose:

| Parameter group | Changed frames |
|---|---:|
| Body pose | 0 / 1,493 |
| Left-hand pose | 0 / 1,493 |
| Right-hand pose | 6 / 1,493 |
| Per-frame shape coefficients | 30 / 1,493 |

The changed frames were `Frisch/143`, `Jahr/172`, `Jahr/176`, `Jahr/178`,
`Schnee/243`, and `Schwer/167`. The 30 shape-array differences arise from the
historical cache materializer's clip-level beta consolidation, not from an
accepted learned pose region; the new method estimates one robust signer shape
directly and does not need that cache transformation. Relative to the direct initializer, the
reported metric deltas were −0.0015 mm for upper-body-minus-face,
+0.000001 mm for the left hand, and +0.0007 mm for the right hand. These
differences are practically zero, and two hand directions are marginally
worse. The result therefore does not support a Transformer contribution.

## Refactor verification

The selected palm-canonical hand refinement was extracted from the historical
implementation and checked on the same eight-frame batch, model, inputs, and
hyperparameters. The extracted implementation reproduced both hand-pose
arrays and output vertices exactly (maximum absolute difference: 0).

The new 1,493-frame run starts from
`outputs/phase2_gates/g1_views/output_wilor`, not from
`outputs/signal4d_external/full1493_wilor_clipnorm_v1`.
