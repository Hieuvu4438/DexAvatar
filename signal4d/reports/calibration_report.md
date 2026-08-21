# Observation uncertainty calibration report

Artifact: `artifacts/calibration/sgnify_smplerx_wilor_split_conformal_seed12345`

- Calibration manifest: 12 clips / 260 frames.
- Model fitting: 8 clips.
- Split-conformal scaling: 4 disjoint clips (`Blitz`, `EinstellenJob`, `Frech`, `Schwer`).
- Residual: per-frame source-0-pelvis translation-aligned 3D joint distance.
- Distributional training loss: Student-t NLL.
- Bounds: 0.002–0.5 m.
- Initial/final model-fit loss: -1.2752 / -1.8848.

Held-out conformal coverage at nominal 90%:

| Source | Body | Left hand | Right hand |
|---|---:|---:|---:|
| SMPLer-X | 90.04% | 90.06% | 90.06% |
| WiLoR | 91.10% | 90.20% | 90.03% |
| Legacy-Biomech | 90.02% | 90.10% | 90.10% |

Frozen 90th-percentile abstention thresholds are 0.2411 m (body), 0.3872 m (left), and 0.1368 m (right). These calibrate observation residuals, not final mesh error; final-output risk utility is evaluated separately by AURC/selective error.
