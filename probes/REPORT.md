# DexAvatar Diagnostic Probes — Phase 0 Asset Audit

**Status:** Mandatory Phase 0 stop gate reached. No Phase 1/E1/E2/E3/E4 implementation has been created or run.

## 1. Repository provenance and guard

- Repository: `/home/haipd/DexAvatar`
- Branch: `probes/diagnostics`
- HEAD: `42561b21f79249e6ea5a34d2d52185dd4d0270d6`
- Origin: `https://github.com/Hieuvu4438/DexAvatar.git`
- Audit command:

```bash
python /home/haipd/DexAvatar/probes/experiments/e0_audit.py \
  --repo /home/haipd/DexAvatar \
  --prediction-method method_hamer
```

The original brief required a clean official clone and an exact clean-tree guard. The user subsequently explicitly authorized direct work in this existing repository using a **baseline allowlist**. The approved pre-existing non-`probes/` state was:

```text
 M sapiens
?? docs/dexavatar_diagnosis/E1E3_PROBE_AGENT_PROMPT.md
```

`probes/tools/guard_upstream.sh` hashes both that superproject status and the complete nested `sapiens` status. It fails if either changes, or if any new path outside `probes/` appears.

Phase 0 guard output:

```text
OK: no new changes outside probes/ relative to approved baseline
```

This is not equivalent to proving a pristine official-upstream clone; it proves that this phase introduced no new changes outside `probes/` relative to the user-approved baseline.

## 2. Machine-readable outputs

- `probes/results/phase0/audit.json`
- `probes/results/phase0/frame_manifest.csv`

The result directory is ignored only by `probes/.gitignore`; the repository-root `.gitignore` was not modified.

## 3. Predictions

Per-frame fitted parameter pickles and meshes are available under `outputs/*/<sign>/smplifyx/{results,meshes}/`.

For the audited `method_hamer` output:

- parameter pickles: **1,493**
- meshes: **1,493**
- signs: **57**
- representative parameter file: `outputs/method_hamer/Ablehnen/smplifyx/results/low_149.pkl`

Required fields and shapes in that sample:

| Field | Shape | Representation |
|---|---:|---|
| `body_pose` | `(1, 63)` | 21 body joints, axis-angle |
| `left_hand_pose` | `(1, 45)` | 15 hand joints, full axis-angle |
| `right_hand_pose` | `(1, 45)` | 15 hand joints, full axis-angle |
| `betas` | `(1, 10)` | SMPL-X shape coefficients |
| `global_orient` | `(1, 3)` | axis-angle |
| `transl` | `(1, 3)` | translation |
| `expression` | `(1, 10)` | expression coefficients |

The saved hand outputs are full 45-D axis-angle, not PCA coefficients.

A configuration/model inconsistency must be retained as a threat to validity: the paper YAML contains `use_pca: True`, `num_pca_comps: 12`, and `flat_hand_mean: False`, while the inspected fitting code constructs the neutral model with `use_pca=False` and `flat_hand_mean=True`. Oracle substitution must follow the effective saved 45-D representation, not assume the YAML is authoritative.

Other output directories exist, but many have incomplete frame coverage. Exact counts are recorded in `audit.json`.

## 4. SGNify ground truth

Ground truth location:

```text
data/smplx_gt/<sign>/<frame>.obj
```

Audit result:

- signs: **57**
- OBJ meshes: **4,152**
- GT `.pkl`/`.npz`/other parameter files: **0**
- representation: **meshes only**

### Blocker: E1

E1 cannot be run as specified because the required GT parameter groups do not exist:

- GT `left_hand_pose` / `right_hand_pose`
- GT wrist rotation inside `body_pose`
- GT elbow rotation
- GT `betas`
- complete GT SMPL-X parameter records

A possible fallback is to fit SMPL-X parameters to every GT mesh, quantify the mesh-fitting residual, and treat that residual as an error floor on V1–V6. This is a methodological substitution and requires explicit user approval before implementation.

## 5. SMPL-X assets and model convention

Loadable SMPL-X assets were found in multiple locations, including:

- `data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz`
- `SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz`
- `SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl`
- male/female variants under `SMPLer-X/common/utils/human_model_files/smplx/`
- additional ARCTIC and local dependency copies recorded in `audit.json`

Representative neutral metadata:

- vertices: `(10475, 3)`
- joint regressor: `(55, 10475)`
- skinning weights: `(10475, 55)`

The author evaluator does not instantiate SMPL-X. It loads the neutral NPZ and uses its `J_regressor` to obtain wrist joints. The paper fitting configuration selects neutral gender.

## 6. Exact frame-pairing audit

The author evaluator does **not** join prediction and GT by filename or frame ID. It:

1. reads `[start, end]` from the sign segment file;
2. selects GT frame numbers in the inclusive interval `[2*start, 2*end]` when the OBJ exists;
3. independently sorts all prediction meshes numerically;
4. pairs each selected GT mesh to the prediction at the same ordinal list position.

Audited `method_hamer` counts:

| Quantity | Count |
|---|---:|
| Selected GT meshes | **1,493** |
| Prediction meshes | **1,493** |
| Ordinal pairs | **1,493** |
| Paper-stated central frames | **2,872** |

The result is therefore **not 2,872**. All per-sign rows and exact paired paths are in `frame_manifest.csv`.

This is a blocking protocol discrepancy. It must not be normalized away by duplicating, interpolating, or changing frame semantics.

The evaluator also contains hardcoded original-author paths for masks/model assets, despite CLI arguments for other paths. The probe audit resolves the repository copies but does not edit the evaluator.

## 7. Region masks

The exact author-evaluator assets are:

| Region | Source | Cardinality |
|---|---|---:|
| UBody(-F) | `data/evaluation_from_author/data/data/sgnify_part_segm_above_pelvis_joint/upper_body_minus_face.npy` | **7,279** |
| LHand | `data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl['left_hand']` | **778** |
| RHand | `data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl['right_hand']` | **778** |

Hashes are recorded in `audit.json`.

Similarly named fallback masks elsewhere in the repository must not be substituted because they have different cardinalities and would change the metric.

## 8. One-handed handling

From the evaluator’s sign mapping:

- class `0`: **15 signs**, **330 paired frames**
- class `~0`: **42 signs**, **1,163 paired frames**

For class-0 signs, the evaluator:

- skips the LHand metric;
- removes left-hand vertices from every other evaluated region, including UBody(-F);
- continues evaluating RHand.

Effective populations:

| Region | Signs | Paired frames |
|---|---:|---:|
| LHand | **42** | **1,163** |
| RHand | **57** | **1,493** |

Thus LHand and RHand are not evaluated on identical populations.

## 9. Baseline-output availability

- DexAvatar/local method outputs: available for several methods, with varying coverage.
- EVA*: **no per-frame parameter or mesh outputs found**.
- SGNify: source/configuration is present, but **no paired per-frame SGNify evaluation outputs found**.

### Blocker: paired E4 comparison

A paired DexAvatar-versus-EVA* bootstrap cannot be performed from published table numbers. It remains blocked unless aligned per-frame EVA* outputs are provided with documented provenance.

## 10. Hand rectifier

No reusable implementation was found that transforms a 15-joint GT hand pose using per-joint bend/splay/twist limits plus the required MANO/SMPL-X axis alignment.

Related fitting code contains heuristics/penalties, but those are not the Fig. 4 data rectifier and cannot generate `GT_rect`.

### Blocker: E3

E3 is blocked by both:

1. missing GT hand-pose parameters; and
2. missing reusable hand rectifier.

Reimplementing the rectifier would be a methodological change with uncertain joint limits and axis conventions. It requires explicit approval and must be labeled as a probe reimplementation, not as recovered upstream code.

## 11. Runtime audit

All obvious assets needed by the existing one-sign launcher were found, including frames, NLF/WiLoR inputs, model files, and paper config.

Runtime was **not measured** in Phase 0 because the canonical launcher writes to `outputs/`, which violates the approved write boundary. A probe-owned redirected launcher/config would be new experiment infrastructure and is intentionally deferred until the user authorizes the next phase.

## 12. Phase 0 blockers and decisions required

1. **E1 is blocked:** GT is mesh-only. Approval is required to fit SMPL-X parameters to GT meshes and quantify the fit residual as an error floor.
2. **E3 is blocked:** GT hand parameters and a reusable rectifier are unavailable. Approval is required for any rectifier reimplementation, and E3 still depends on obtaining/fitting GT parameters.
3. **Parity protocol discrepancy:** the released author-style evaluator/available outputs produce 1,493 ordinal pairs, not 2,872.
4. **EVA*/SGNify paired outputs are absent:** paired ranking/statistics are blocked unless assets are provided.
5. **Repository provenance deviation:** this run uses the user-authorized Hieuvu fork with a baseline-aware guard, not a pristine official clone.

No synthetic data, placeholder parameters, interpolation, copied external outputs, or inferred GT parameters were used.

## 13. Mandatory stop

Per the prompt, work stops here. The independent TR-V2V metric, parity run, E1, E2, E3, and E4 have not been implemented or executed.
