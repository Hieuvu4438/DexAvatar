# RL-S4D-013 — Corrected rerun decision

Date: 2026-08-20

The v3 rerun rebuilt the canonical cache, calibration, predictions and evaluator
artifacts after fixing the left-MANO canonical-right to SMPL-X-left reflection.
No method hyperparameter was changed for that rerun. Against the frozen
legacy/fallback control, corrected M1 changed clip-macro left-hand TR-V2V by
-0.2035 mm with a paired 95% clip-bootstrap interval of [-0.7080, +0.2800] mm.
It improved the registered dynamics endpoints but failed both the -0.5 mm
practical-effect threshold and the CI-upper-below-zero threshold.

The original central-test SOTA claim therefore remains rejected. Because those
labels are revealed, subsequent use of the central population is explicitly
development/diagnostic evidence, not a second confirmation. Any later claim
must come from a separately frozen prospective endpoint.
