# SGNify 56-sign TR-V2V recap

Date: 2026-08-20

## Protocol identity

- Population: frozen prospective extended-post manifest, 56 signs and 769 frames.
- Coverage: 100% for both methods; no intersection-only frame removal.
- Prediction transport: strict DexAvatar OBJ, parsed with the author's
  `read_verts_and_faces` implementation.
- Metric kernel: functions and region assets from
  `data/evaluation_from_author/evaluate_new_fitting.py`.
- Author evaluator source SHA-256:
  `400bfbd736fc59fcc1867af7650188b61772136982f64b623df31494e6116877`.
- Aggregation below: author's vertex-micro mean in millimetres; lower is better.
- Frame policy: the frozen 56-sign prospective manifest, not the author's fixed
  central-frame selector.

## Aggregate results

| Author metric | A1 composite | SIGNAL-4D v5 | Delta | Relative delta |
|---|---:|---:|---:|---:|
| V2V left wrist | 14.4918 | **14.0353** | **-0.4565** | **-3.1500%** |
| V2V right wrist | **12.8414** | 12.8425 | +0.0012 | +0.0092% |
| TR all | 43.0164 | **43.0071** | **-0.0093** | **-0.0216%** |
| TR left hand | 14.4918 | **14.0353** | **-0.4565** | **-3.1500%** |
| TR right hand | **12.8414** | 12.8425 | +0.0012 | +0.0092% |
| TR above-pelvis upper body | 29.2932 | **29.2763** | **-0.0169** | **-0.0577%** |
| TR above-pelvis minus head | 45.5682 | **45.4137** | **-0.1545** | **-0.3391%** |
| TR above-pelvis minus face | 33.2424 | **33.2035** | **-0.0390** | **-0.1172%** |

Delta is SIGNAL-4D minus A1. Negative is an improvement.

The wrist-centered and translation-aligned hand numbers are equal under this
implementation because both paths independently remove each hand mesh's mean;
adding the same GT wrist center to both meshes does not change their difference.

## Class-0 rule and denominators

The author code excludes left-hand evaluation for class-0 one-handed signs and
removes left-hand vertices from every other evaluated region for those signs.

- Left-hand metrics: 41 signs, 610 frames, 778 vertices per evaluated mesh.
- Excluded from left-hand metrics: 15 signs, 159 frames.
- All other metrics: 56 signs, 769 frames.
- Region sizes before the class-0 subtraction: all 10,475; upper body 8,888;
  upper minus head 3,865; upper minus face 7,279; each hand 778 vertices.

Excluded class-0 signs: Blitz, Boese, Dort, Duenn, Frech, Jahr, Klar,
LiebBitte, Luegen, Muell, Mutter, SauerWuetend, Sonne, Sorry, and Vater.

## Win/tie/loss counts

Counts compare SIGNAL-4D with A1 independently for each unit. A tie is exact;
the 127 recurring frame ties are principally frames for which the frozen gate
selected the A1 hypothesis.

| Metric | Per-frame win/tie/loss | Per-sign win/tie/loss |
|---|---:|---:|
| TR all | 398 / 127 / 244 | 34 / 8 / 14 |
| TR upper body | 398 / 127 / 244 | 30 / 8 / 18 |
| TR upper minus face | 409 / 127 / 233 | 34 / 8 / 14 |
| TR upper minus head | 415 / 127 / 227 | 38 / 8 / 10 |
| TR left hand | 314 / 127 / 169 (610 valid) | 24 / 8 / 9 (41 valid) |
| TR right hand | 275 / 127 / 367 | 17 / 8 / 31 |

## Comparator identity

The A1 column is not the untouched vanilla DexAvatar pipeline. It is the
pre-frozen composite comparator: 607 Ensemble-A1 fitted frames, 145 HaMeR-A0
fitted frames, and 17 terminal raw SMPLer-X frames. It should be reported as
`A1 ensemble composite`, not `official vanilla DexAvatar`.

## Machine-readable artifacts

- `prospective_extended_post/comparison.json`: exact aggregate values and
  hashes.
- `prospective_extended_post/comparison.csv`: compact comparison table.
- `prospective_extended_post/methods/{A1,SIGNAL4D_v5}/per_frame.csv`: all 769
  frame rows.
- `prospective_extended_post/methods/{A1,SIGNAL4D_v5}/per_clip.csv`: all 56
  sign rows.
