# RL-S4D-016 — Prospective v5 confirmation

Date: 2026-08-20

## Firewall and population

The extended-post manifest contains 56 clips and 769 temporally post-central
frames. Its SHA-256 is
`33825a3f1ac8aa6d063f90bc12c8061ed60680267615b6d76cbe1e8cee625b32`.
Manifest construction inspected filenames/existence only. Availability was
resolved without GT as 607 balanced Ensemble A1 frames, 145 original-HaMeR A0
frames, and 17 raw SMPLer-X terminal frames. No frame was dropped.

The candidate gate records `gt_used_for_selection: false` and selected A1 for
127 frames, M1 scale 1.0 for 442, scale 1.5 for 7, and scale 3.0 for 193, with
zero within-clip switches. Its repeated output is byte-identical over 112 files.
Release SHA-256
`0c5808308b6de3f965fa50f0b05bad21cfe04e9e9161e211f2b03740f825975d`
was written six seconds before the prospective GT cache was created.

## Confirmatory result

Both methods cover all 56 clips/769 frames. Values below are candidate minus A1;
lower is better, and intervals are paired 10,000-replicate clip bootstraps.

| Endpoint | Delta | 95% CI | Gate |
|---|---:|---:|---|
| Left-hand TR-V2V | -2.1411 mm | [-2.9547, -1.4191] | PASS superiority/effect |
| Upper-body TR-V2V | -1.0745 mm | [-1.5975, -0.6002] | PASS non-inferiority |
| Right-hand TR-V2V | +0.0020 mm | [-0.0002, +0.0042] | PASS non-inferiority |
| Velocity error | -0.0143 | [-0.0265, +0.0013] | PASS dynamics margin |
| Acceleration error | -0.8929 | [-1.2831, -0.4353] | PASS dynamics margin |
| Jerk error | -21.5851 | [-34.8538, -7.1724] | PASS dynamics margin |

All preregistered checks pass. The permitted wording is: new best result on the
prospective SIGNAL-4D extended-post SGNify endpoint against the strongest
pre-frozen, recomputed same-protocol A1 baseline. This endpoint reuses known
clip/sign identities, lacks signer IDs, and is not an external published
leaderboard. Contact correctness and semantic fidelity remain unclaimed because
trustworthy independent evaluators are unavailable.
