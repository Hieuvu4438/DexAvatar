# SIGNAL4D external-only audit

## Outcome

The new lane is leakage-clean and improves the frozen DexAvatar baseline, but
the learned residual does not transfer strongly enough to claim that the old
SIGNAL4D V6 gains were reproduced without SGNify training. The locked gate
refined only 6 of 4,479 region-frames; almost all measured improvement comes
from the clean WiLoR/HaMeR initializer.

| Method | All | Upper body | UBody-F | Left hand | Right hand |
|---|---:|---:|---:|---:|---:|
| DexAvatar | 42.5867 | 26.4560 | 29.9074 | 13.5735 | 12.9271 |
| SIGNAL4D external-only | 42.2423 | 26.2236 | 29.6196 | 12.8102 | 12.1148 |
| Historical SIGNAL4D V6 | 42.1116 | 26.1394 | 29.5197 | 11.6339 | 11.8056 |

All values are millimetres from the repository's strict wrapper around the
author evaluator, using the full frozen 57-clip/1,493-frame manifest. V6 is
shown only as historical context: its lineage includes model selection and
calibration on the 12 SGNify clips and is therefore not external-only.

Compared with DexAvatar, the clean lane improves All by 0.3444 mm, upper body
by 0.2324 mm, UBody-F by 0.2878 mm, left hand by 0.7632 mm, and right hand by
0.8123 mm. This retains 72.5%, 73.4%, 74.2%, 39.4%, and 72.4% respectively of
the historical V6 improvement.

## Protocol and leakage audit

- Training: 10,822 How2Sign clips / 346,304 frames.
- Validation: 498 How2Sign clips / 15,936 frames.
- Calibration: 497 How2Sign clips / 15,904 frames.
- The three splits have zero source-group overlap.
- SGNify reads for training, checkpoint selection, and threshold selection: 0.
- The checkpoint is EMA step 1,500, selected by How2Sign validation score.
- Calibration passed non-inferiority in all regions before final inference.
- SGNify was evaluated only after checkpoint, thresholds, and all 1,493 meshes
  had been frozen. The first evaluator invocation failed before metric
  computation because the baseline lacked an export manifest; the successful
  invocation used validated immutable-style OBJ registries.

How2Sign targets in this cache are 2D-guided pseudo-labels, not exact 3D ground
truth. Therefore the large source-domain geodesic improvements do not by
themselves establish target-domain 3D accuracy.

## Diagnosis

The model itself trained successfully: on held-out How2Sign validation, the
final EMA prediction-to-initializer error ratios were 0.492 (left hand), 0.521
(right hand), and 0.640 (upper body). On the independent How2Sign calibration
split the selected ratios were 0.482, 0.519, and 0.639.

The failure is transfer of the benefit gate, not source optimization. With
thresholds fixed at 0.75/0.70/0.80, SGNify observations were almost always
classified as out of support. Relative to the frozen WiLoR initializer, the
final result changes UBody-F by -0.0015 mm, left hand by +0.000001 mm, and
right hand by +0.0007 mm. The learned residual is thus effectively neutral on
this target set. Loosening thresholds after seeing this result would tune on
the evaluation set and was deliberately not done.

## Integrity artifacts

- Checkpoint SHA-256: `09ca34e42e88b550af3b046f1f35bff30ae0c66aff65b9cfba430594ccdb90bf`
- Calibration SHA-256: `a3180aa4c203af61892d74ea297193bc5ae40d3f8230a966252b36823909785e`
- Frozen run manifest SHA-256: `37a0e54ef0e1ecbc6337292d03cf1674407ea449f82ae54a74861426e421a682`
- OBJ registry SHA-256: `da2878d01047890dfbe167ab12fda3c2c7c2cf3b83dfd82709aa6f7dacd92748`
- Evaluation report SHA-256: `2037d545f7851097375c1ffa967ad2147856ee4fe223c810ffa056faf330c3fb`

The original tracked `signal4d/` and `phase2_refiner/` source trees have no
diffs from this work. New implementation code lives under `signal4d_external/`.
