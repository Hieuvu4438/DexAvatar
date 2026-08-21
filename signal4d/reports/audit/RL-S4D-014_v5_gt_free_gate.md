# RL-S4D-014 — v5 GT-free multiscale gate

Date: 2026-08-20

The v5 candidate uses four coherent whole-pose hypotheses: A1 and M1 geodesic
updates at scales 1.0, 1.5 and 3.0. ExtraTrees models see only frozen predictions,
observations, calibrated uncertainty and factor diagnostics. They do not accept
ground-truth geometry. Grouped out-of-fold selection on already revealed
historical clips produced a -0.9872 mm clip-macro left-hand delta with a 95%
clip-bootstrap interval of [-1.4238, -0.5696] mm over 45 clips and 1,233 frames.
The fixed 8 mm transition cost selected no within-clip switches.

The forests were exported to JSON plus safetensors and verified against sklearn
to numerical precision before the prospective endpoint was fit. All models,
hypotheses, transition cost and inference code are frozen for the extended-post
test. Historical final-model results on revealed clips are diagnostics only and
are not reported as confirmatory evidence.
