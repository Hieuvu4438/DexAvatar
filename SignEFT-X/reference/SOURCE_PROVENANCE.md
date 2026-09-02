# Source provenance

The active package is self-contained: it does not import `SignPCC-X`,
`signal4d_external`, `signal4d`, or the historical SignEFT-X source tree.

No implementation from `signal4d_external` is used by the final method. Its
learned Transformer was audited and excluded, so copying that model or its
training/inference stack would incorrectly enlarge the released method. The
direct frozen initializer is consumed through the local frontend contract.

Two validated implementations were brought into this folder and then reduced
to the selected method:

| Local component | Source | Frozen source SHA-256 | Adaptation |
|---|---|---|---|
| `src/signeft/canonical/refinement.py` | `SignPCC-X/src/signpccx/model/canonicalizer.py` | `526cb20122dcebe16a0b4e57e361b16f58ce7e906c566edd432485c9c8c5f3f4` | Renamed around the frozen initializer; removed cross-project imports; retained shared-identity and canonical refit only. |
| `scripts/evaluate_pa_mpvpe.py` | `signal4d/evaluate_pampvpe.py` | `cda6aa1856022407930669ec80e11854f78a90d3f3af51948305e20fb4202f0e` | Made all reference assets explicit CLI inputs and relabeled the report. |
| `src/signeft/frontend/initializer.py` | `phase2_refiner/data/build_locked_fallback_view.py` | `cca44c4c0f6edfdc9fc6d133b91042a1e126dc586bc482fc46fb65080115d922` | Removed cross-project provenance imports and retained deterministic whole-frame primary/fallback selection. |
| `scripts/extract_wilor.py` | historical `export_wilor_v3.py` | `2f1015870dc02587feb2d64c477d789286e2e3d2bca6102a16955d5fd9241126` | Copied as the external WiLoR/checkpoint adapter. |
| `src/signeft/frontend/wilor.py` | historical `observations/wilor.py` | `e5d331cfb3b14cfb3f8a958a460da22f48d4056174137e424cd926b7ff2c8bf4` | Adapted to target-free per-sign manifests and removed historical package dependencies. |

The palm-canonical hand refiner was extracted from the former SignEFT-X
research tree and independently reproduced a frozen eight-frame batch exactly.
Rejected branches remain under `_archive/research_history` and are not on the
package import path.
