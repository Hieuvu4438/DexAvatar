# SIGNAL4D v5 full-1493 release report

Date: 2026-08-20

## Coverage and provenance

- Full manifest: `signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl`
- Manifest SHA-256: `02e06c946f9400d8eb2b238c0297b07e188912121748db68ee1d66d12ea7c362`
- Coverage: 57 signs, 1,493 frames, with no duplicate frame keys.
- Calibration/development/test split coverage: 260 + 578 + 655 = 1,493 frames.
- Pairwise overlap between the three splits: 0 frames.
- DexAvatar baseline: the native `outputs/method_hamer` SMPL-X fitting output
  (HaMeR initialization followed by SignBPoser/SignHPoser fitting).
- DexAvatar baseline coverage: exactly the same 57 signs and 1,493 frame keys.
- SMPL-X model SHA-256: `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.

The previously missing SIGNAL4D calibration partition was fitted with the frozen
`m1_a1_v5.yaml` configuration.  Its run completed successfully for 12 signs and
260 frames.  The frozen GT-free multiscale gate then selected 28 baseline, 137
alpha-1.0, 0 alpha-1.5, and 95 alpha-3.0 frame states.  Together with the
existing development and test predictions, this produces the full 1,493-frame
release without refitting or modifying the older method outputs.

## Author SGNify evaluation

The comparison uses the attached, unmodified
`data/evaluation_from_author/evaluate_new_fitting.py` implementation, the
author's central-frame policy, native SMPL-X OBJ topology, and the same frame
manifest for both methods.  Values are the author's vertex-micro TR-V2V means
in millimetres; lower is better.

| Method | TR all | TR upper body | TR left hand | TR right hand |
|---|---:|---:|---:|---:|
| DexAvatar HaMeR + SignBPoser/SignHPoser | 42.5867 | 26.4560 | 13.5735 | 12.9271 |
| SIGNAL4D v5 | **42.1434** | **26.1935** | **11.6651** | **11.8329** |
| SIGNAL4D minus DexAvatar (mm) | **-0.4434** | **-0.2625** | **-1.9084** | **-1.0942** |
| Relative error reduction | **1.04%** | **0.99%** | **14.06%** | **8.46%** |

Coverage is 57/57 signs and 1,493/1,493 frames for both methods.  Calling the
original author's `main()` directly reproduced the structured evaluator values
to the four decimals printed by the original code.

## Output locations

- Full SIGNAL4D prediction release:
  `signal4d/runs/signal4d_v5_full1493_20260820`
- Strict SIGNAL4D OBJ files:
  `signal4d/outputs/strict_dexavatar_obj_20260820/full_1493/SIGNAL4D_v5`
- Validated native DexAvatar HaMeR OBJ registry:
  `signal4d/outputs/strict_dexavatar_obj_20260820/full_1493/DexAvatar_HaMeR`
- SIGNAL4D fitting overlays (1,493 PNG plus 1,493 mesh links):
  `signal4d/outputs/reconstruction_signal4d_v5_full1493_20260820`
- Structured author evaluation:
  `signal4d/reports/author_evaluator_strict_obj_20260820/full_1493`
- Original-author evaluator logs:
  `signal4d/logs/author_original_main_full1493_DexAvatar_HaMeR.log`
  and `signal4d/logs/author_original_main_full1493_SIGNAL4D_v5.log`
- Calibration fitting and gate logs:
  `signal4d/logs/full1493_calibration_fit_20260820.log` and
  `signal4d/logs/full1493_calibration_gate_20260820.log`

## Integrity notes

- The older DexAvatar method folders were read and symlinked only; no baseline
  OBJ, PKL, image, or source-code file was modified.
- The strict OBJ manifests record the source manifest/model hashes and every OBJ
  hash.  All meshes passed the 10,475-vertex/20,908-face SMPL-X topology check.
- The release assembly uses clip-disjoint symlinks to the frozen split outputs,
  so it does not duplicate or alter prior SIGNAL4D predictions.
