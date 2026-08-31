# Experiment record

Date: 2026-08-31. Dataset: the 57-sign, 1,493-frame central-frame protocol
defined by the author-provided `signs.txt` and `segment.json`.

## Gates and decisions

1. A1 established the exact evaluator protocol and H4W++ coordinate boundary.
   The reused cache had already received the camera-to-evaluator x180 transform,
   so export was identity. Applying x180 again was rejected by a geometry test.
2. A2 replaced per-frame H4W++ beta with a pose-diverse Huber signer beta. It
   improved UBody by 0.3960 mm but did not improve hands, so H4W++ was not used
   as the final initializer.
3. The leakage-clean External V1 artifact was canonicalized through the exact
   neutral SMPL-X model. A direct absolute-coordinate refit and a regularized
   centered-hand refit failed the one-sign gate.
4. A3d removed the pose anchor and reduced hand-to-initializer residual from
   roughly 25–30 mm to at most 1.92 mm. It improved official All and UBody.
5. A3e refined the single shared beta without an identity anchor. On a fixed
   12-sign/298-frame panel it beat A3d on All, UBody, LHand and RHand, so the
   setting was frozen and run on the full protocol. Full residual was at most
   1.44 mm and official metrics improved again.
6. A3f increased the pose-diverse identity calibration set from 50 to 200
   frames. It won the frozen 12-sign panel on All and UBody with effectively
   tied hand metrics, so it was promoted to the all-57 run. The full run reached
   42.0936 All / 25.8311 UBody / 12.8466 LHand / 12.1275 RHand mm.
7. A4 evaluated structured `-30/0/+30` degree local-axis wrist hypotheses using
   only H4W++ 2-D/3-D anchors, chirality and a fixed twist prior. It increased
   chirality mismatches and official hand errors, so it was rejected.
8. A5 enabled confidence-gated hand--hand attraction on 12/298 panel frames.
   Contact target error improved from 5.04 to 1.57 mm, but every official
   primary metric regressed slightly, so contact was rejected for the final.

No frame-level choice was made using GT. GT was accessed only by the unchanged
official evaluator after candidates/settings had been fixed at the stated
sign-level gates.

## Fixed 12-sign ablation panel

The panel contains 298 frames and is stored in
`runs/signpccx_a3e_external_v1_free_identity/dev_signs_12.txt`. Every row below
was evaluated by the checksum-locked author script on the identical panel.

| Run | Component changed | TR All | TR UBody | TR UBody (-F) | TR LHand | TR RHand | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| A0 DexAvatar/HaMeR | same-protocol baseline | 41.5498 | 25.0369 | 28.4327 | 12.7495 | 12.1387 | reference |
| A1 H4W++ | frozen teacher export | 83.5794 | 32.3897 | 36.3158 | 15.3049 | 16.3241 | reject |
| A2 H4W++ + shared beta | signer identity | 83.4253 | 31.9342 | 35.8029 | 15.2843 | 16.3348 | reject |
| A3e External canonical | 50-frame shared beta | 41.1614 | 24.5185 | 27.7585 | 12.3309 | **11.9140** | keep |
| **A3f External canonical** | **200-frame shared beta** | **41.1539** | **24.4695** | **27.7074** | **12.3310** | 11.9162 | **promote** |
| A4 palm hypotheses | local twist best-of-K | 41.5707 | 25.1852 | 28.4581 | 19.6773 | 19.7694 | reject |
| A5 contact | gated attraction + barrier | 41.1548 | 24.4709 | 27.7091 | 12.3413 | 11.9217 | reject |

All figures are millimetres; lower is better. A3f versus A3e changes are
`-0.0075` All, `-0.0490` UBody, `-0.0511` UBody (-F), `+0.0001` LHand and
`+0.0022` RHand. A5 is not
retained despite better contact diagnostics because the preregistered gate
requires no hand regression.

## Best artifact locations

- Config: `configs/ablations/a3f_external_v1_identity200.yaml`
- Identity: `runs/signpccx_a3f_external_v1_identity200/identity/S1.npz`
- Shared camera: `runs/signpccx_a3f_external_v1_identity200/camera/C1.npz`
- Fit provenance: `runs/signpccx_a3f_external_v1_identity200/fit_sequences/run_manifest.json`
- Preflight: `runs/signpccx_a3f_external_v1_identity200/preflight.json`
- Official result: `runs/signpccx_a3f_external_v1_identity200/metrics/official_result.json`
- Raw evaluator output: `runs/signpccx_a3f_external_v1_identity200/metrics/official_stdout.txt`
- Audited per-sign metrics: `runs/signpccx_a3f_external_v1_identity200/metrics/audited/per_sign.csv`
- Paired bootstrap: `runs/signpccx_a3f_external_v1_identity200/metrics/paired_bootstrap.json`
- Failure cases: `runs/signpccx_a3f_external_v1_identity200/metrics/failure_cases.json`

## Known limitations

- The strongest final initializer is the existing frozen External V1 artifact,
  not a fresh end-to-end teacher extraction inside this package.
- Camera focal calibration is represented by the blueprint/configuration but
  the shared full-image K is used for observation diagnostics rather than the
  final canonical objective because External V1 already fixes the camera frame
  and the official metrics are translation-aligned.
- Contact/hypothesis modules are implemented, unit-tested and officially
  ablated. Both are gated off because they failed their fixed-panel criteria.
- InterWild is deliberately not installed; its expected direct gain on the
  separately centred hand metrics is low and no duplicate repository is needed.
