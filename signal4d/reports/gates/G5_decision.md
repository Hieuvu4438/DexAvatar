# G5 M2 incremental-value decision

Development split, equal-weight clip macro, 100% coverage for M1 and M2.
M2 was correctly warm-started from each frozen M1 prediction; all warm-start
artifact hashes are recorded in `runs/dev_v2_m2_warm_v2/run.json`.

| Endpoint | M1 | M2 | Delta (M2-M1) | Paired 95% CI |
|---|---:|---:|---:|---:|
| Upper-body TR-V2V (mm) | 24.4866 | 24.4852 | -0.0014 | [-0.0063, +0.0029] |
| Left-hand TR-V2V (mm) | 16.8659 | 16.9771 | +0.1112 | [-0.1175, +0.3436] |
| Right-hand TR-V2V (mm) | 12.9350 | 12.9364 | +0.0014 | [-0.0000, +0.0027] |
| Velocity error | 6.2544 | 6.2384 | -0.0160 | [-0.0203, -0.0119] |
| Acceleration error | 147.1339 | 146.5629 | -0.5710 | [-0.7138, -0.4379] |
| Jerk error | 3926.6188 | 3912.0210 | -14.5978 | [-18.6660, -10.7636] |

M2 contact active fraction and collision-proxy penetration are both zero on this
development endpoint; mean contact probability is 0.000539. Runtime is 285.81 s
for 578 frames with 239,507,968 peak allocated CUDA bytes.

Decision: **G5 fail**. M2 slightly improves dynamics but supplies no observed real
contact/collision value and does not improve the target left-hand geometry. Keep
M2 as a complete exploratory module and use preregistered M1 for confirmatory test.
