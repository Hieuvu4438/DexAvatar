# RL-S4D-012 — Baseline protocol and decode audit

Date: 2026-08-19

## Trigger

The prospective extended-post run was initially launched with the historical
Biomech fitting configuration. A cross-check against the completed Phase-2
artifacts showed that the pre-existing balanced A1 baseline was instead the
atomic `method_ensemble + method_hamer fallback` view. The Biomech run was
stopped, marked invalid, and replaced before any extended-post GT vertex value
was loaded.

## Protocol dependence

Phase-2 selected A1 with its own locked left-hand availability mask and
aggregation. SIGNAL-4D uses all enumerated frames and all 57 signs, 100%
coverage, equal-weight clip macro aggregation, and a pinned SMPL-X decode.
Consequently, the Phase-2 ordering cannot simply be copied into the new table.

All locally available historical candidates were rematerialized from trusted
parameter files using the same SMPL-X model hash as SIGNAL-4D. Missing primary
files use the already-defined HaMeR atomic fallback. Results below are on the
full 57-clip/1,493-frame development population under the SIGNAL-4D evaluator:

| Pinned decode | Upper TR-V2V | Left TR-V2V | Right TR-V2V | Velocity error |
|---|---:|---:|---:|---:|
| HaMeR A0 | 34.7211 | 22.8072 | 12.9322 | 6.5402 |
| Ensemble + A0 fallback | **34.3846** | 22.0902 | **12.1486** | 6.5389 |
| Biomech + A0 fallback | 34.4142 | **22.0282** | 12.1707 | 6.5384 |
| Hand2D + A0 fallback | 34.4982 | 22.1264 | 12.1951 | 6.5381 |
| WiLoR + A0 fallback | 34.4607 | 22.0738 | 12.1529 | **6.5292** |

Geometry is in millimeters. The Direct candidate was fail-closed because its
locked parameter population contains non-finite pose values; its historical
saved-mesh result was already rejected for upper-body regression.

## Decision before prospective evaluation

- Retain Ensemble+A0 as the primary A1 comparison because it was selected
  before SIGNAL-4D development and is strongest on the balanced body/right-hand
  geometry endpoints.
- Retain Biomech+A0 as the strongest-left endpoint-sensitivity control; its
  full-population left advantage over A1 is 0.0620 mm.
- Require any practical SOTA decision to remain unchanged against both controls
  or explicitly narrow the claim.
- Use per-frame shapes and the pinned model when materializing controls. The
  earlier one-step solver baseline used a clip-mean shape and is retained only
  as an exploratory diagnostic, not the final strongest baseline.

## Prospective firewall

The extended-post manifest was constructed from filename intersections only.
At the time of this decision, no extended-post OBJ contents or cached GT
vertices had been read. The running A1 fitting jobs write only under
`signal4d/artifacts/legacy_a1_extended_post_v1`; legacy sources remain read-only.
