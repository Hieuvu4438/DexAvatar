# E0 Phase Report — DexAvatar Asset and Protocol Audit

**Date:** 2026-07-22  
**Status:** Completed; mandatory stop gate reached  
**Working branch:** `probes/diagnostics`  
**Repository SHA:** `42561b21f79249e6ea5a34d2d52185dd4d0270d6`

## 1. Objective

E0 audited the assets and evaluation protocol required for the diagnostic experiments in `E1E3_PROBE_AGENT_PROMPT.md`. No metric parity, oracle substitution, anchor sweep, GT rectification, or statistical experiment was run during this phase.

The executable audit and complete machine-readable outputs are:

- `probes/experiments/e0_audit.py`
- `probes/results/phase0/audit.json`
- `probes/results/phase0/frame_manifest.csv`
- `probes/REPORT.md`

## 2. Prediction assets

Per-frame DexAvatar-compatible outputs are available under:

```text
outputs/<method>/<sign>/smplifyx/results/*.pkl
outputs/<method>/<sign>/smplifyx/meshes/*.obj
```

For the audited `method_hamer` output:

- signs: **57**
- parameter files: **1,493**
- mesh files: **1,493**

A representative parameter file is:

```text
outputs/method_hamer/Ablehnen/smplifyx/results/low_149.pkl
```

It contains:

| Field | Shape | Interpretation |
|---|---:|---|
| `body_pose` | `(1, 63)` | 21 body-joint axis-angle rotations |
| `left_hand_pose` | `(1, 45)` | 15 left-hand joint axis-angle rotations |
| `right_hand_pose` | `(1, 45)` | 15 right-hand joint axis-angle rotations |
| `betas` | `(1, 10)` | SMPL-X shape coefficients |
| `global_orient` | `(1, 3)` | root orientation |
| `transl` | `(1, 3)` | global translation |
| `expression` | `(1, 10)` | expression coefficients |

The saved hand poses are full **45-D axis-angle**, not PCA coefficients.

### Configuration inconsistency

The paper fitting YAML contains settings corresponding to PCA hands (`use_pca: True`, `num_pca_comps: 12`, `flat_hand_mean: False`), but the inspected fitting code constructs the effective neutral model with `use_pca=False` and `flat_hand_mean=True`. Future probes must follow the effective saved 45-D representation and log this discrepancy as a threat to validity.

## 3. Ground-truth assets

SGNify ground truth is located at:

```text
data/smplx_gt/<sign>/<frame>.obj
```

Audit totals:

- sign directories: **57**
- OBJ meshes: **4,152**
- GT SMPL-X parameter files: **0**

The released GT is therefore **mesh-only**.

### E1 blocker

E1 requires oracle replacement using GT values for:

- finger articulation;
- wrist rotation;
- elbow rotation;
- shape coefficients;
- complete hand-region SMPL-X parameters.

These values are not available. E1 cannot be run as specified.

A possible fallback would fit SMPL-X parameters to each GT mesh and report the mesh-fitting residual as an error floor. This is a methodological substitution and requires explicit approval before implementation.

## 4. SMPL-X model assets

Loadable SMPL-X assets are available, including:

- `data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz`
- `SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz`
- `SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl`

Male and female variants also exist.

Representative neutral-model metadata:

- template vertices: `(10475, 3)`
- joint regressor: `(55, 10475)`
- skinning weights: `(10475, 55)`

The author evaluator does not instantiate a body model. It loads the neutral model archive to obtain `J_regressor`. The fitting configuration uses the neutral model.

## 5. Frame-pairing protocol

The author evaluator does not join GT and predictions by equal frame ID. Its effective process is:

1. read a sign interval `[start, end]`;
2. select existing GT frame numbers in the inclusive interval `[2 × start, 2 × end]`;
3. independently sort prediction meshes numerically;
4. pair the two lists by ordinal position.

For `method_hamer`:

| Quantity | Count |
|---|---:|
| Selected GT meshes | **1,493** |
| Prediction meshes | **1,493** |
| Ordinal pairs | **1,493** |
| Frames stated in the paper | **2,872** |

The released evaluator and available outputs therefore produce **1,493 pairs, not 2,872**.

This is a material protocol discrepancy. No frames were duplicated, interpolated, or synthesized to force the count to 2,872.

The exact per-sign pair list is stored in:

```text
probes/results/phase0/frame_manifest.csv
```

## 6. Evaluator region masks

The exact author-evaluator masks are:

| Region | Source | Vertex count |
|---|---|---:|
| UBody(-F) | `data/evaluation_from_author/data/data/sgnify_part_segm_above_pelvis_joint/upper_body_minus_face.npy` | **7,279** |
| LHand | `data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl['left_hand']` | **778** |
| RHand | `data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl['right_hand']` | **778** |

Similarly named masks elsewhere in the repository have different cardinalities and cannot be substituted in a parity experiment.

## 7. One-handed evaluation behavior

The sign mapping contains:

- class `0`: **15 signs**, **330 paired frames**
- class `~0`: **42 signs**, **1,163 paired frames**

For class-0 signs, the evaluator:

- skips LHand evaluation;
- removes left-hand vertices from all other evaluated regions;
- continues evaluating RHand.

Effective populations are therefore:

| Region | Signs | Paired frames |
|---|---:|---:|
| LHand | **42** | **1,163** |
| RHand | **57** | **1,493** |

The left- and right-hand aggregate values are not calculated over the same sign/frame population.

## 8. Baseline-output availability

Several local DexAvatar variants have parameter and mesh outputs, although coverage varies by method.

No paired per-frame outputs were found for:

- EVA*;
- SGNify evaluation predictions.

Only their published aggregate values are available locally.

### Statistical blocker

A paired DexAvatar-versus-EVA* bootstrap cannot be performed from aggregate table values. Per-frame outputs aligned to the audited manifest would be required.

## 9. Biomechanical hand rectifier

No reusable implementation was found that transforms a 15-joint hand pose using:

- per-joint bending limits;
- splaying limits;
- twisting limits;
- the paper’s MANO/SMPL-X axis alignment.

The fitting code contains related penalties and heuristics, but these are not the training-data rectifier shown in Fig. 4 and cannot produce `GT_rect`.

### E3 blocker

E3 is blocked by both:

1. missing GT hand-pose parameters;
2. missing reusable biomechanical rectifier.

Any rectifier reimplementation would be probe-authored rather than recovered upstream code and requires explicit approval.

## 10. Runtime audit

The obvious assets required by the existing one-sign launcher are present:

- sign frames;
- NLF/WiLoR inputs;
- SMPL-X model files;
- fitting config;
- launcher script.

Runtime was not measured because the canonical launcher writes into `outputs/`, outside the probe write boundary. A probe-owned output redirect is deferred until a later phase is authorized.

## 11. Guard and repository state

The repository had approved pre-existing changes before the probes began:

```text
 M sapiens
?? docs/dexavatar_diagnosis/E1E3_PROBE_AGENT_PROMPT.md
```

The user authorized a baseline-aware guard rather than cleaning or stashing these changes. The guard hashes the pre-existing superproject and nested `sapiens` states and rejects new unintended changes outside the probe/report paths.

Phase-end result:

```text
OK: no new changes outside probes/ relative to approved baseline
```

This establishes no new upstream source modification during E0; it is not equivalent to a pristine official clone.

## 12. E0 conclusions

1. **Prediction parameters exist** and use full 45-D axis-angle hand poses.
2. **E1 is blocked** because SGNify GT is mesh-only.
3. **The released protocol produces 1,493 ordinal pairs**, not the paper-stated 2,872 frames.
4. **The author masks have exact cardinalities of 7,279 / 778 / 778** for UBody(-F), LHand, and RHand.
5. **LHand and RHand use different evaluation populations** because class-0 signs skip LHand.
6. **Paired EVA*/SGNify outputs are unavailable.**
7. **E3 is blocked** because both GT hand parameters and the reusable rectifier are unavailable.
8. No missing data were synthesized and no blocked measurement was fabricated.

## 13. Required decisions before later phases

- Whether to authorize a mesh-to-SMPL-X fitting fallback for E1.
- Whether to authorize a probe-side rectifier reimplementation for E3.
- Whether Phase 1 parity should use the released **1,493-pair** protocol or await another source aligned to the claimed **2,872 frames**.
- Whether an approved source for paired EVA*/SGNify outputs can be provided.

Per the experiment brief, execution stopped after E0.
