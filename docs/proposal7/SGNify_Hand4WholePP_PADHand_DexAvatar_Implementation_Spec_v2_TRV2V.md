# Implementation Specification
## SGNify SOTA-Oriented 3D Sign Language Reconstruction
### Hand4Whole++ + PAD-Hand + DexAvatar Sign-Specific Fitting

**Status:** Engineering proposal / implementation blueprint  
**Primary benchmark:** SGNify  
**Primary output:** temporally coherent **SMPL-X** sequence  
**Primary goal:** improve SGNify reconstruction accuracy while preserving sign-critical hand articulation, wrist-body consistency, and temporal plausibility.

---

## 0. Executive decision

The proposed main system is:

\[
\boxed{
\text{Hand4Whole++}
\rightarrow
\text{PAD-Hand temporal finger refinement}
\rightarrow
\text{DexAvatar sign-specific fitting}
}
\]

with one new core mechanism:

\[
\boxed{
\text{Confidence-Adaptive Visual–Temporal–Sign Prior Fusion}
}
\]

and one optional second-stage extension:

\[
\boxed{
\text{Sign-Aware Hand–Hand / Hand–Body Contact Refinement}
}
\]

The three upstream methods have deliberately different responsibilities:

| Component | Responsibility in our system | What we do **not** let it own |
|---|---|---|
| **Hand4Whole++** | Frame-level whole-body SMPL-X initialization, body–hand spatial coupling, body-consistent wrist, detailed WiLoR finger evidence | Long-term temporal reasoning |
| **PAD-Hand** | Temporal/physics-aware refinement of **finger articulation**, especially during blur/occlusion | Final body pose; hard overwrite of wrist orientation |
| **DexAvatar** | Sign-specific body/hand priors, biomechanical constraints, fitting, 2D evidence, collision, final SMPL-X optimization | Generic hand/body estimation from scratch |
| **Ours** | Reliability-aware fusion of visual, temporal, and sign priors; optional semantic contact | Replacing the above pretrained backbones unnecessarily |

The most important architectural constraint discovered from the released source code is:

> **Hand4Whole++ should remain the owner of wrist/body-chain consistency. PAD-Hand should primarily refine the 15 MANO finger joints. PAD root/global wrist orientation is a soft cue only.**

This is not cosmetic. Hand4Whole++ intentionally uses the whole-body branch / CHAM context to determine wrist placement/orientation and then rigidly aligns the detailed MANO hand to the SMPL-X wrist region using the wrist + MCP anchors. Hard-overwriting the wrist using an independent temporal MANO sequence would partially undo the exact body–hand coupling that makes Hand4Whole++ useful.

A second critical implementation constraint is:

> The public PAD-Hand demo cannot be used unchanged for two-hand sign-language video.

Its released inference script selects a single detected hand per frame and preferentially selects a right hand. The underlying MANO code supports both LEFT and RIGHT and the PAD Transformer accepts frame masks, therefore our implementation must construct a true two-hand sequence adapter.

---

# 1. Scope

## 1.1 Problem definition

Given a monocular sign-language video

\[
\mathcal V=\{I_t\}_{t=1}^{T},
\]

estimate a temporally coherent SMPL-X sequence

\[
\Theta_{1:T}
=
\{
\theta^{root},
\theta^{body},
\theta^{lh},
\theta^{rh},
\theta^{jaw},
\theta^{face},
\beta,
\psi,
\pi
\}_{1:T},
\]

where the main evaluation focus is:

- upper-body geometry,
- left-hand geometry,
- right-hand geometry,
- correct hand articulation,
- body-consistent wrist orientation,
- temporal stability without over-smoothing meaningful signs.

The final reconstruction must remain in **SMPL-X topology/parameterization** so that it is directly compatible with the DexAvatar/SGNify fitting and evaluation pipeline.

---

## 1.2 Primary benchmark target

The implementation must be evaluated on **SGNify** using the same official or baseline-verified protocol as DexAvatar.

Primary reporting should include, at minimum:

- Upper-body TR-V2V
- Left-hand TR-V2V
- Right-hand TR-V2V

The evaluator must be treated as **locked infrastructure**:

1. reproduce the DexAvatar baseline with the original code,
2. reproduce the published/official metric within a small tolerance,
3. only then evaluate our variants.

No new vertex subsets, alternative alignment, or custom Procrustes alignment may be introduced silently.

---

## 1.3 Non-goals for version 1

Version 1 deliberately does **not**:

- replace SMPL-X with another human representation;
- add SMPLest-X to the main forward path;
- average outputs from multiple whole-body estimators;
- replace PAD-Hand with a custom diffusion/flow model;
- train Hand4Whole++ from scratch;
- modify every DexAvatar prior simultaneously;
- hard-overwrite H4W++ wrist orientation using PAD-Hand;
- use elementwise averaging of rotation matrices;
- tune hyperparameters on final SGNify test ground truth.

The goal is first to build the strongest **clean, reproducible, ablatable** system.

---

# 2. Source audit before architecture design

This specification was written after auditing the execution-critical paths of the three official codebases and their papers. “Execution-critical” means the model forward path, inference scripts, cache interfaces, pose conventions, fitting losses, and model-output composition required by this proposal; it does not mean every unrelated utility or dataset loader in the repositories was read line-by-line.

## 2.1 DexAvatar

**Official repository**

- https://github.com/kaustesseract/DexAvatar

**Execution path audited**

- `Full_running_command.sh`
- `run_dexavatar.py`
- `dexavatar_fitting/`
- `dexavatar_fitting/smplifyx/data_parser.py`
- `dexavatar_fitting/smplifyx/fit_single_frame.py`
- `dexavatar_fitting/smplifyx/fitting.py`

**Verified behavior relevant to us**

The original high-level run sequence is effectively:

```text
Sapiens extraction
    ↓
SMPLer-X extraction
    ↓
HaMeR extraction
    ↓
DexAvatar / SMPLify-X fitting
```

The data parser reads, among other things:

- Sapiens observations;
- per-frame whole-body SMPL-X initialization;
- hand reconstruction outputs;
- 2D keypoints;
- hand boxes / handedness;
- 3D hand information;
- sign labels / active-hand information.

The final fitting code includes:

- 2D joint reprojection;
- 3D hand/depth-related terms;
- SignBPoser;
- SignHPoser;
- shape regularization;
- body biomechanics;
- temporal regularization;
- collision/interpenetration;
- face/hand terms.

**Design implication**

We should **not rewrite DexAvatar from zero**. We should preserve its sign-specific fitting logic and replace/adapt the upstream observations and add our new temporal/fusion loss in a controlled way.

---

## 2.2 Hand4Whole++

**Official repository**

- https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE

**Execution-critical files audited**

- `main/model.py`
- `common/nets/wilor.py`
- `common/nets/module.py`
- configuration / demo / inference path

**Verified architecture relevant to us**

Hand4Whole++ contains:

- a whole-body image backbone;
- DWPose for hand localization/cues;
- WiLoR for specialist hand reconstruction;
- `HandControlNet` / CHAM-style hand feature modulation into the body ViT;
- SMPL-X regression;
- hand-body geometry combination/alignment.

The released model freezes the large pretrained components and primarily trains the hand-control/modulation component in its original training setup.

### Critical wrist finding

Inside the hand-body composition path, detailed MANO hand geometry is canonicalized and rigidly aligned to the body hand/wrist region. The body model remains responsible for the global body-chain placement of the wrist.

Conceptually:

```text
Whole-body branch:
shoulder → elbow → forearm → wrist frame
                            ↑
                            │ body context owns this

WiLoR:
detailed MANO fingers + hand geometry
                            ↓
canonicalize root
                            ↓
rigid alignment by wrist + MCP anchors
                            ↓
insert/aligned detailed hand
```

Therefore:

- WiLoR/PAD finger articulation is valuable;
- PAD global hand orientation is **not** allowed to blindly overwrite the wrist in the final SMPL-X chain.

### WiLoR wrapper finding

The H4W++ WiLoR wrapper already deals with handedness internally:

- left crops are mirrored for the right-hand-oriented hand model;
- outputs are transformed back;
- left/right pose mean conventions are handled;
- rotation outputs are transformed into the conventions expected by H4W++.

**Design implication**

Our integration should extract the WiLoR/H4W++ outputs **after H4W++’s own handedness/convention handling**, rather than implementing a second independent flip pipeline.

---

## 2.3 PAD-Hand

**Official repository**

- https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026

**Execution-critical files audited**

- `wilor_inference.py`
- `demo.py`
- `models/pad_hand.py`
- MANO-related model code
- diffusion sampling path

**Verified method behavior**

PAD-Hand refines a noisy hand-motion sequence using:

- a temporal Transformer;
- MeshCNN-derived geometry features;
- conditional diffusion;
- physics-aware constraints / dynamics observations.

The released model operates on a pose representation containing:

\[
16 \times 6
\]

rotation dimensions:

- one global/root hand rotation,
- 15 MANO finger-joint rotations,
- each represented in 6D inside the model.

The released inference model uses a temporal sequence length of 16 in the demo configuration.

### Critical public-demo limitation

The released WiLoR preprocessing path selects one hand per frame and preferentially chooses a right hand when available.

Therefore the public demo is **not a two-hand sign-language inference pipeline**.

### Important hidden capability

The underlying code supports:

- LEFT and RIGHT MANO models;
- Transformer frame masks (`src_key_padding_mask`-style use);
- direct access to decoded rotation matrices internally.

Therefore our adapter can support:

- both hands;
- missing detections;
- overlapping temporal windows;
- rotation-space merging.

### Another important finding

The paper discusses uncertainty/variance from the physics formulation, but the public execution path audited for the released demo does not expose a clean ready-made per-frame/per-joint variance tensor that we can safely depend on.

**Design implication**

Version 1 will **not require PAD’s theoretical variance output**. Reliability will be estimated from observable signals such as keypoint confidence, visibility, reprojection error, PAD–visual disagreement, and motion statistics.

---

# 3. Final system architecture

## 3.1 High-level pipeline

```text
                           SIGN VIDEO
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Hand4Whole++       │
                    │  frame-level        │
                    └─────────────────────┘
                       │              │
          SMPL-X body  │              │ WiLoR/H4W++ hand evidence
          wrist chain  │              │ L/R MANO finger pose
                       │              │ bbox / exist / features
                       │              ▼
                       │      ┌─────────────────────┐
                       │      │ Two-Hand PAD Adapter │
                       │      └─────────────────────┘
                       │          │            │
                       │          ▼            ▼
                       │      PAD-left     PAD-right
                       │      temporal     temporal
                       │      fingers      fingers
                       │          │            │
                       └──────────┴─────┬──────┘
                                      ▼
                         ┌────────────────────────┐
                         │ Reliability estimation │
                         │ frame × side × joint   │
                         └────────────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ DexAvatar fitting      │
                         │ + SignBPoser           │
                         │ + SignHPoser           │
                         │ + biomechanics         │
                         │ + collision            │
                         │ + original observations│
                         │ + OUR adaptive prior   │
                         └────────────────────────┘
                                      │
                                      ▼
                          FINAL SMPL-X SEQUENCE
                                      │
                                      ▼
                           SGNify TR-V2V eval
```

---

## 3.2 Ownership of each parameter

This table is mandatory for implementation consistency.

| Parameter/group | Frame initializer | Temporal target | Final optimizer owner |
|---|---|---|---|
| global body/root | Hand4Whole++ | DexAvatar original temporal term | DexAvatar |
| torso/body pose | Hand4Whole++ | DexAvatar original temporal term | DexAvatar + SignBPoser |
| shoulder/elbow/forearm | Hand4Whole++ | DexAvatar original temporal term | DexAvatar |
| **wrist global/body-chain orientation** | **Hand4Whole++** | PAD only optional weak cue | **DexAvatar / body chain** |
| 15 L-hand finger joints | H4W++/WiLoR | **PAD-left** | DexAvatar + SignHPoser + ours |
| 15 R-hand finger joints | H4W++/WiLoR | **PAD-right** | DexAvatar + SignHPoser + ours |
| body shape \(\beta\) | Hand4Whole++ | sequence-shared/regularized | DexAvatar |
| MANO \(\beta\) used by PAD | H4W++/WiLoR hand expert | robust sequence statistic | conditioning only; never final body shape |
| camera | Hand4Whole++ / DexAvatar-compatible adapter | optional smoothing | DexAvatar |
| face | H4W++ or existing DexAvatar path | original behavior | DexAvatar |

---

# 4. Repository organization

Recommended project:

```text
sgnify_h4wpp_pad_dex/
├── README.md
├── IMPLEMENTATION_SPEC.md
├── pyproject.toml
├── configs/
│   ├── sgnify.yaml
│   ├── h4wpp.yaml
│   ├── pad_hand.yaml
│   ├── dexavatar.yaml
│   ├── fusion.yaml
│   └── experiments/
│       ├── b0_dex_original.yaml
│       ├── b1_h4wpp_dex.yaml
│       ├── b2_h4wpp_pad_dex.yaml
│       ├── b3_adaptive.yaml
│       └── b4_contact.yaml
├── third_party/
│   ├── Hand4Whole-plus-plus_RELEASE/
│   ├── PAD-Hand-CVPR-2026/
│   └── DexAvatar/
├── assets/
│   ├── README.md
│   └── .gitkeep
├── data/
│   ├── raw/
│   ├── sgnify/
│   └── cache/
│       ├── h4wpp/
│       ├── pad/
│       ├── sapiens/
│       └── fitting/
├── src/
│   ├── common/
│   │   ├── schema.py
│   │   ├── rotations.py
│   │   ├── camera.py
│   │   ├── mesh.py
│   │   ├── paths.py
│   │   └── logging.py
│   ├── h4wpp_adapter/
│   │   ├── patch_outputs.py
│   │   ├── infer_sequence.py
│   │   ├── cache_writer.py
│   │   └── validate_cache.py
│   ├── pad_adapter/
│   │   ├── sequence_builder.py
│   │   ├── two_hand_refiner.py
│   │   ├── windowing.py
│   │   ├── missing_data.py
│   │   ├── overlap_merge.py
│   │   └── validate_pad.py
│   ├── dex_adapter/
│   │   ├── dataset.py
│   │   ├── init_from_h4wpp.py
│   │   ├── adaptive_loss.py
│   │   ├── fit_frame.py
│   │   ├── fit_sequence.py
│   │   └── export_smplx.py
│   ├── fusion/
│   │   ├── reliability_features.py
│   │   ├── deterministic_gate.py
│   │   ├── learned_gate.py
│   │   └── calibration.py
│   └── contact/
│       ├── pseudo_contact.py
│       ├── zones.py
│       └── loss.py
├── scripts/
│   ├── 00_check_assets.py
│   ├── 01_run_dexavatar_baseline.sh
│   ├── 02_extract_h4wpp.sh
│   ├── 03_refine_pad_twohand.sh
│   ├── 04_extract_sapiens.sh
│   ├── 05_fit_ours.sh
│   ├── 06_eval_sgnify.sh
│   └── run_all.sh
├── eval/
│   ├── official_wrapper.py
│   ├── metric_lock.py
│   └── aggregate.py
└── tests/
    ├── test_rotations.py
    ├── test_left_right_identity.py
    ├── test_h4wpp_export.py
    ├── test_pad_windows.py
    ├── test_pad_missing_frames.py
    ├── test_overlap_so3_merge.py
    ├── test_wrist_preservation.py
    ├── test_camera_projection.py
    └── test_metric_lock.py
```

---

# 5. Environment strategy

Do **not** initially merge all dependencies into one Python environment.

Use three isolated environments and communicate through cache files.

## 5.1 `h4wpp` environment

Created exactly according to the Hand4Whole++ release.

Responsibilities:

- DWPose;
- WiLoR internal hand inference;
- H4W++ forward;
- exporting SMPL-X + raw hand observations.

Output only serialized `.npz`/`.pkl`.

---

## 5.2 `pad_hand` environment

Created according to PAD-Hand release.

Responsibilities:

- load PAD checkpoint;
- construct left/right MANO temporal sequences;
- refine temporal hand pose;
- export rotation matrices / MANO meshes.

Do not import DexAvatar here.

---

## 5.3 `dexavatar` environment

Created from the official DexAvatar environment.

Responsibilities:

- Sapiens extraction;
- sign priors;
- SMPL-X fitting;
- our adaptive loss;
- final export;
- official/locked evaluation.

---

## 5.4 Why serialization-first is preferable

Trying to import all three projects into one process creates unnecessary risks:

- incompatible PyTorch/Python versions;
- conflicting PyTorch3D builds;
- duplicate MANO/SMPL-X packages;
- local package-name collisions;
- different CUDA assumptions;
- harder reproducibility.

The cache boundary gives a stable contract.

---

# 6. Canonical cache schema

All stages must use one versioned schema.

## 6.1 `H4WFrameRecord`

Pseudo-dataclass:

```python
@dataclass
class H4WFrameRecord:
    schema_version: str
    frame_idx: int
    frame_name: str
    image_hw: tuple[int, int]

    # SMPL-X frame estimate
    smplx_root_pose_aa: np.ndarray      # [3]
    smplx_body_pose_aa: np.ndarray      # [J_body, 3] or flattened
    smplx_lhand_pose_aa: np.ndarray     # [15, 3]
    smplx_rhand_pose_aa: np.ndarray     # [15, 3]
    smplx_shape: np.ndarray             # [10] or model-specific
    smplx_trans: np.ndarray             # [3]
    camera: dict

    # hand presence / boxes
    lhand_exist: bool
    rhand_exist: bool
    lhand_bbox_xyxy: np.ndarray | None  # [4]
    rhand_bbox_xyxy: np.ndarray | None  # [4]

    # raw H4W++/WiLoR specialist outputs
    l_wilor_global_rotmat: np.ndarray | None  # [3,3]
    r_wilor_global_rotmat: np.ndarray | None
    l_wilor_hand_rotmat: np.ndarray | None    # [15,3,3]
    r_wilor_hand_rotmat: np.ndarray | None
    l_wilor_betas: np.ndarray | None          # [10]
    r_wilor_betas: np.ndarray | None
    l_wilor_vertices: np.ndarray | None       # [778,3]
    r_wilor_vertices: np.ndarray | None
    l_wilor_kpts2d: np.ndarray | None
    r_wilor_kpts2d: np.ndarray | None

    # confidence signals if available
    l_kpt_conf: np.ndarray | None
    r_kpt_conf: np.ndarray | None
    l_bbox_score: float | None
    r_bbox_score: float | None
```

### Rule

Store **rotation matrices** as the canonical interchange representation for hand rotations whenever possible.

Axis-angle is retained only because DexAvatar/SMPL-X may require it at its interface.

---

## 6.2 `PADSequenceRecord`

```python
@dataclass
class PADSequenceRecord:
    schema_version: str
    side: Literal["left", "right"]
    frame_indices: np.ndarray            # [T]
    valid_visual: np.ndarray             # [T] bool

    visual_rotmat: np.ndarray            # [T,16,3,3]
    pad_rotmat: np.ndarray               # [T,16,3,3]
    pad_finger_rotmat: np.ndarray        # [T,15,3,3]

    conditioning_betas: np.ndarray       # [10] robust seq beta
    pad_vertices: np.ndarray | None      # [T,778,3]

    # derived reliability
    geodesic_disagreement: np.ndarray    # [T,15]
    temporal_accel_score: np.ndarray     # [T,15] or [T]
```

---

## 6.3 `FusionRecord`

```python
@dataclass
class FusionRecord:
    side: str
    frame_idx: int

    w_visual: np.ndarray   # [15]
    w_pad: np.ndarray      # [15]
    w_sign: np.ndarray     # [15]

    feature_vector: np.ndarray
```

For version 1:

\[
w_{visual}+w_{pad}+w_{sign}=1
\]

for every finger joint.

---

# 7. Stage A — Reproduce and lock DexAvatar baseline

This stage is non-negotiable.

## 7.1 Goal

Verify:

1. dataset path and SGNify frames are correct;
2. SMPL-X model assets are correct;
3. sign priors are loaded correctly;
4. evaluator is correct;
5. output coordinate convention is correct.

---

## 7.2 Procedure

Run the original repository **without any local modifications**.

```bash
bash scripts/01_run_dexavatar_baseline.sh
```

Store:

```text
results/
  dexavatar_original/
    outputs/
    per_frame_metrics.csv
    aggregate.json
    environment.txt
    git_commits.json
```

---

## 7.3 Metric-lock invariant

The evaluator wrapper must verify:

```python
assert abs(reproduced_metric - expected_baseline_metric) < tolerance
```

Tolerance should reflect deterministic/numerical differences only.

If baseline is not locked:

> stop all SOTA experiments.

Do not tune a new method against an unverified evaluator.

---

# 8. Stage B — Patch Hand4Whole++ output interface

## 8.1 Objective

Run H4W++ only once per frame and expose **all information required by PAD and DexAvatar**.

Avoid a second standalone WiLoR run.

Why:

- H4W++ already runs WiLoR internally;
- crop/handedness conventions stay consistent;
- avoids subtle differences between H4W++ boxes and PAD demo boxes;
- saves compute;
- makes exact provenance of visual hand evidence clear.

---

## 8.2 Minimal patch to `main/model.py`

The original test output already returns useful SMPL-X and mesh outputs.

Add raw tensors to the test dictionary before they go out of scope:

```python
out.update({
    "lhand_exist": lhand_exist,
    "rhand_exist": rhand_exist,

    "l_wilor_global_rotmat": lmano_root_pose_rotmat,
    "r_wilor_global_rotmat": rmano_root_pose_rotmat,

    "l_wilor_hand_rotmat": lmano_hand_pose_rotmat,
    "r_wilor_hand_rotmat": rmano_hand_pose_rotmat,

    "l_wilor_betas": lmano_shape,
    "r_wilor_betas": rmano_shape,

    "l_wilor_vertices": lmano_vert_cam,
    "r_wilor_vertices": rmano_vert_cam,

    # if available before postprocessing
    "l_wilor_kpts2d": ...,
    "r_wilor_kpts2d": ...,
})
```

**Do not alter** H4W++’s own handedness transformations.

---

## 8.3 Preserve body-owned wrist

Add a debug export of:

- shoulder global rotation;
- elbow global rotation;
- H4W++ wrist global orientation;
- WiLoR root/global orientation.

They should not be assumed equal.

The test `test_wrist_preservation.py` will verify that adding PAD later does not alter the chosen body-chain wrist unless explicitly enabled as an experiment.

---

## 8.4 Sequence inference

Pseudo-flow:

```python
for frame in video:
    out = h4wpp(frame)

    record = H4WFrameRecord(
        smplx_* = ...,
        l/r_wilor_* = ...,
        l/r_hand_exist = ...,
        confidence = ...
    )

    save(record)
```

Output:

```text
data/cache/h4wpp/<sequence>/<frame>.npz
```

Also write:

```text
data/cache/h4wpp/<sequence>/manifest.json
```

containing:

- fps;
- number of frames;
- original image paths;
- image dimensions;
- H4W++ git SHA;
- checkpoint hash;
- schema version.

---

# 9. Stage C — Two-hand PAD-Hand adapter

This is the largest required adapter.

---

## 9.1 Do not use released `wilor_inference.py` as the main sequence extractor

The released PAD demo chooses a single hand per frame.

Instead:

\[
\text{H4W++ raw WiLoR outputs}
\rightarrow
\text{our PAD sequence builder}
\]

This avoids duplicated detection and allows simultaneous left/right tracks.

---

## 9.2 Construct left and right temporal tracks independently

For side \(s\in\{L,R\}\):

\[
X^s_t=
\left[
R^s_{root,t},
R^s_{1,t},\dots,R^s_{15,t},
\beta^s_t,
m^s_t
\right].
\]

`m_t` is visual validity.

Track IDs are fixed by handedness, not detection order.

Never allow:

```text
frame t: left = detection 0
frame t+1: left = detection 1
```

without using explicit side identity.

---

## 9.3 Shape conditioning

PAD uses MANO shape to generate geometry.

Use a robust sequence statistic:

\[
\bar\beta_s
=
\operatorname{median}_{t\in \mathcal V_s}
\beta^s_t
\]

or a confidence-weighted mean.

Do **not** let frame-by-frame WiLoR beta jitter enter the PAD sequence if avoidable.

Important:

> PAD MANO beta is conditioning only. It never overwrites final SMPL-X body shape.

---

## 9.4 Missing detections

Public demo skips a 16-frame window if any frame is missing. That is too destructive for sign language.

Instead:

1. create a numeric placeholder pose for missing frames;
2. mark missing frames invalid in the PAD source mask.

Recommended placeholder:

- interpolate between nearest valid rotations using geodesic/Slerp behavior;
- if only one neighbor exists, carry the nearest valid pose;
- if no valid observation exists in an entire region, use neutral/last reliable pose only as numeric input.

Then set:

```python
src_valid[t] = False
```

so the Transformer knows that the frame is not a trusted observation.

---

## 9.5 Rotation interpolation

Never linearly interpolate rotation matrices.

For \(R_0,R_1 \in SO(3)\):

\[
R(\alpha)
=
R_0
\exp
\left(
\alpha
\log(R_0^\top R_1)
\right).
\]

Use a tested library implementation or a numerically stable local utility.

---

# 10. Temporal windowing

## 10.1 Original public demo

The public PAD demo uses length-16 windows and a non-overlapping stride in its simple demonstration path.

For sign video this risks block boundaries.

---

## 10.2 Proposed production setting

Start with:

- window length = 16;
- stride = 8.

Ablate:

- stride = 4;
- stride = 8;
- stride = 16.

Do not change PAD’s trained sequence length unless retraining.

---

## 10.3 Window inference

```python
windows = make_windows(T, length=16, stride=8)

for w in windows:
    x, mask = build_pad_input(track, w)
    result = pad_model(x, src_mask=mask)
    save_window_prediction(result)
```

The adapter must retrieve the model’s decoded **rotation matrices**, not only the demo’s rendered vertices.

Output per window:

```text
window_start
window_end
rotmat [16,16,3,3]
vertices [16,778,3] (optional)
valid_mask [16]
```

---

# 11. Merge overlapping PAD windows on SO(3)

A frame can receive predictions from multiple windows.

Do not:

\[
R = \frac{R_1+R_2}{2}.
\]

That is not generally a valid rotation.

---

## 11.1 Center-weighted merge

For a 16-frame window use triangular/Hann weights with higher confidence at the center.

For each frame/joint, collect:

\[
\{(R_k,w_k)\}_{k=1}^{K}.
\]

Compute a weighted geodesic mean:

\[
R^\star=
\arg\min_{R\in SO(3)}
\sum_k
w_k
d_{SO(3)}(R,R_k)^2.
\]

An iterative Karcher mean is sufficient.

---

## 11.2 Practical implementation

```python
def so3_weighted_mean(rotations, weights, iters=8):
    R = rotations[np.argmax(weights)].copy()

    for _ in range(iters):
        delta = 0
        wsum = 0

        for Ri, wi in zip(rotations, weights):
            delta += wi * so3_log(R.T @ Ri)
            wsum += wi

        delta /= max(wsum, 1e-8)
        R = R @ so3_exp(delta)

    return project_to_so3(R)
```

Unit-test:

\[
R^\top R\approx I,\qquad \det(R)\approx1.
\]

---

# 12. Left-hand PAD validation

Although the underlying MANO implementation supports both sides, do not assume the pretrained PAD checkpoint is perfectly side-symmetric.

Before running SGNify:

## 12.1 Parity test

Take a right-hand sequence and construct a known mirrored left-hand equivalent.

Compare:

1. PAD on original right;
2. PAD on canonicalized/mirrored left;
3. mirrored-back output.

The difference should be small under equivalent geometry.

---

## 12.2 Two implementation branches

### Preferred branch

Run native side-specific MANO:

```python
mano_r = MANO(side="RIGHT")
mano_l = MANO(side="LEFT")
```

### Fallback branch

If checkpoint behavior is not symmetric:

```text
left observations
    ↓
canonical right-hand representation
    ↓
PAD RIGHT
    ↓
inverse canonical transform
    ↓
left output
```

This decision must be made based on the parity test, not intuition.

---

# 13. PAD output used by our system

For every side/frame:

```text
PAD root/global rotation
PAD 15 finger rotations
PAD MANO mesh
```

Usage:

| PAD output | Usage |
|---|---|
| root/global hand rotation | reliability cue / optional very weak prior |
| **15 finger rotations** | **main temporal hand prior** |
| MANO vertices | debugging / optional geometry loss |
| temporal motion behavior | reliability feature |

PAD does **not** directly output the final SMPL-X hand mesh used for SGNify scoring.

---

# 14. Rotation utilities

Create `src/common/rotations.py`.

Required API:

```python
axis_angle_to_matrix(aa)
matrix_to_axis_angle(R)

rot6d_to_matrix(x)
matrix_to_rot6d(R)

so3_log(R)
so3_exp(v)

so3_geodesic(R1, R2)

so3_slerp(R0, R1, alpha)

so3_weighted_mean(Rs, ws)

project_to_so3(R)
```

Tests:

```python
R2 = axis_angle_to_matrix(matrix_to_axis_angle(R))
assert geodesic(R, R2) < eps

assert ||R.T @ R - I|| < eps
assert det(R) ≈ 1
```

No hand pose is allowed to move between modules without passing these invariants.

---

# 15. Stage D — DexAvatar adapter

## 15.1 Preferred strategy

Do **not** fake old HaMeR files if avoidable.

Create a new DexAvatar-compatible dataset/parser class that directly consumes:

- H4W++ SMPL-X initialization;
- H4W++/WiLoR visual hand observations;
- PAD temporal finger targets;
- Sapiens observations;
- sign labels / existing sign metadata.

This is cleaner than generating synthetic `hamer.pkl` files just to satisfy the legacy parser.

---

## 15.2 New parser output

The fitting stage should receive:

```python
{
    # original-like fields
    "fn": ...,
    "img_path": ...,
    "cam_param": ...,
    "smplx_param": ...,
    "label": ...,
    "keypoints": ...,
    "img": ...,

    # our visual hand initialization
    "h4w_lhand_rotmat": [15,3,3],
    "h4w_rhand_rotmat": [15,3,3],

    # temporal priors
    "pad_lhand_rotmat": [15,3,3],
    "pad_rhand_rotmat": [15,3,3],

    # masks
    "lhand_visual_valid": bool,
    "rhand_visual_valid": bool,
    "lhand_pad_valid": bool,
    "rhand_pad_valid": bool,

    # reliability vectors
    "lhand_reliability_features": ...,
    "rhand_reliability_features": ...
}
```

The final `smplx_param` must initialize:

- root/body/shape/camera from H4W++;
- hand finger pose from H4W++’s hand-aware output;
- no direct PAD wrist overwrite.

---

# 16. Preserve DexAvatar’s sign priors

Do not remove SignBPoser or SignHPoser in the first implementation.

They are the main domain-specific reason to keep DexAvatar.

Keep:

\[
L_{SignBody}
\]

and

\[
L_{SignHand}
\]

as existing priors, then add our visual-temporal observation loss.

This enables a clean research story:

```text
generic spatial prior      = Hand4Whole++
generic temporal hand prior= PAD-Hand
sign-domain prior          = DexAvatar
adaptive fusion            = Ours
```

---

# 17. Core proposed contribution
## Confidence-Adaptive Visual–Temporal–Sign Fusion

This is the main method contribution, not the use of pretrained code.

For side \(s\), frame \(t\), finger joint \(j\):

- current optimized joint rotation: \(R_{t,s,j}\);
- H4W++ visual target: \(R^V_{t,s,j}\);
- PAD temporal target: \(R^T_{t,s,j}\);
- sign-specific prior acts through SignHPoser latent \(z^S_{t,s}\).

Define:

\[
L_{fusion}
=
\sum_{t,s,j}
\left[
w^V_{t,s,j} d_{SO(3)}^2(R,R^V)
+
w^T_{t,s,j} d_{SO(3)}^2(R,R^T)
\right]
+
\sum_{t,s}
w^S_{t,s} L_{SignHPoser}.
\]

With:

\[
w^V+w^T+w^S=1.
\]

---

# 18. Reliability features

Do not build the gate from final SGNify GT.

Possible input features per frame/side:

\[
f_{t,s}=
[
c_{2D},
c_{bbox},
a_{bbox},
r_{reproj},
d_{V,T},
v_{motion},
a_{motion},
m_{visual},
m_{pad}
].
\]

Where:

### 18.1 2D keypoint confidence

Mean/min confidence over wrist/fingers from Sapiens/DWPose.

### 18.2 Box confidence / size

Small or truncated hand boxes are less reliable.

Normalize area:

\[
a_{bbox}
=
\frac{w_{box}h_{box}}{WH}.
\]

### 18.3 Reprojection residual

After projecting the initial hand/body:

\[
r_{reproj}
=
\frac1J
\sum_j
\|\Pi(J_j)-u_j\|_2.
\]

High residual:

\[
w^V\downarrow.
\]

### 18.4 Visual–PAD disagreement

\[
d_{V,T}
=
d_{SO(3)}
(R^V,R^T).
\]

Large disagreement means at least one source is unreliable; combine with visual confidence to decide which.

### 18.5 Motion magnitude / acceleration

Very high apparent one-frame acceleration can indicate:

- true fast sign motion;
- visual failure.

Therefore it must not alone force temporal smoothing.

---

# 19. Deterministic gate — version 1

Before training any gate, implement a deterministic baseline.

Example:

\[
q_V
=
\alpha_1 c_{2D}
+
\alpha_2 c_{bbox}
-
\alpha_3 \hat r_{reproj}
\]

\[
q_T
=
\beta_1(1-\hat c_{2D})
+
\beta_2 m_{pad}
-
\beta_3 \hat d_{V,T}
\]

\[
q_S
=
\gamma_0
+
\gamma_1(1-c_{2D}).
\]

Then:

\[
[w_V,w_T,w_S]
=
\operatorname{softmax}
([q_V,q_T,q_S]/\tau).
\]

Start simple. Verify behavior visually.

---

# 20. Learned gate — version 2

Only after deterministic fusion is stable.

Use a tiny MLP:

```python
class ReliabilityGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
```

Potential granularity:

1. frame × hand;
2. frame × finger;
3. frame × joint.

Recommended progression:

```text
v1: frame × hand
v2: frame × finger
v3: frame × joint
```

Avoid a highly expressive gate before verifying that the basic loss helps.

---

# 21. How to train/calibrate the gate

Do not train on SGNify final evaluation ground truth.

Possible supervision:

### Strategy A — held-out development sequences

If a valid train/dev protocol is available:

- train gate on train;
- select hyperparameters on dev;
- evaluate once on test.

### Strategy B — synthetic corruption

Take reliable pose observations and simulate:

- hand occlusion;
- motion blur;
- truncated crops;
- missing fingers;
- dropped frames.

Teach the gate:

- visual weight ↓ under corrupted images;
- temporal/sign weight ↑.

### Strategy C — self-consistency

Minimize:

- 2D reprojection;
- temporal consistency;
- sign prior;
- body-hand alignment;

without using final 3D test GT.

---

# 22. Wrist policy

This deserves its own implementation rule.

## 22.1 Finger joints

For \(j=1,\ldots,15\):

\[
L^j_{fusion}
=
w_V d(R_j,R_j^V)^2
+
w_T d(R_j,R_j^T)^2
+
w_S L^j_{sign}.
\]

## 22.2 Wrist/root

For wrist:

\[
w_T^{wrist}
\ll
w_T^{finger}.
\]

Default v1:

```text
PAD wrist hard overwrite: OFF
PAD wrist soft prior:     OFF initially
H4W++ body wrist:         ON
```

After baseline:

```text
PAD wrist weak prior: optional ablation
```

The final wrist orientation should emerge from the SMPL-X arm chain, sign fitting, 2D evidence, and H4W++ body-hand context.

---

# 23. Avoid double temporal smoothing

DexAvatar already has a temporal term.

PAD-Hand adds a much stronger temporal prior for the hand.

Do not simply maximize both.

Recommended:

- retain DexAvatar body temporal term for torso/arms;
- exclude or reduce generic hand temporal smoothing where PAD is valid;
- use PAD finger target instead.

Conceptually:

\[
L_{temp}
=
L_{temp}^{body}
+
\lambda_{PAD} L_{PAD}^{fingers}.
\]

Not:

\[
L_{temp}^{all}
+
\lambda_{PAD}L_{PAD}^{all}
\]

with large weights.

---

# 24. Proposed full loss

Start from the DexAvatar total objective:

\[
L_{Dex}
=
L_{2D}
+
L_{3D-hand}
+
L_{SignB}
+
L_{SignH}
+
L_{shape}
+
L_{bio}
+
L_{angle}
+
L_{temp}
+
L_{collision}
+\cdots
\]

Our version:

\[
\boxed{
L_{ours}
=
L_{Dex}^{modified}
+
\lambda_F L_{fusion}
+
\lambda_W L_{wrist-consistency}
+
\lambda_C L_{contact}
}
\]

where \(L_C\) is disabled in the first core model.

---

# 25. Wrist consistency loss

Optional lightweight constraint:

Let:

- \(R^{body}_{wrist}\) = wrist global orientation implied by SMPL-X chain;
- \(R^{H4W}_{wrist}\) = H4W++ body-aware wrist target.

\[
L_{wrist}
=
d_{SO(3)}^2
(R^{body}_{wrist},
R^{H4W}_{wrist}).
\]

Use small weight; it is not intended to freeze the optimizer.

---

# 26. Optional geometry consistency loss

PAD provides a temporally refined MANO hand.

If pose-space supervision is insufficient, reconstruct a canonical PAD MANO mesh and rigidly align it to the current SMPL-X wrist/MCP anchors using the same conceptual strategy as H4W++.

Then:

\[
L_{mesh-hand}
=
\frac1{|V_H|}
\sum_i
\rho
(
\|\hat v^{SMPLX}_{i}-v^{PAD\rightarrow body}_{i}\|
).
\]

Important:

- use only after verifying vertex correspondence/mapping;
- never replace the final SMPL-X topology with the MANO mesh;
- do not add this loss before pose-only fusion is stable.

---

# 27. Stage E — Sign-aware contact extension (optional)

This is a later contribution, not a prerequisite for the first strong result.

Sign language contains meaningful:

- hand–hand contact;
- fingertip–face/chin contact;
- hand–chest contact;
- hand–shoulder contact;
- hand–arm contact.

Collision loss alone only discourages penetration.

It does not encourage a required contact.

---

## 27.1 Body contact zones

Define SMPL-X vertex groups:

```text
face
chin
upper_chest
lower_chest
left_shoulder
right_shoulder
left_forearm
right_forearm
left_upper_arm
right_upper_arm
```

Store vertex IDs in:

```text
assets/contact_zones/*.json
```

---

## 27.2 Pseudo-contact probability

Estimate:

\[
p_{contact}(t,z)
\]

from:

- 2D hand/body proximity;
- segmentation overlap/proximity;
- estimated depth agreement if available;
- temporal persistence;
- current 3D geometry.

Only activate attraction if confidence is high.

---

## 27.3 Soft contact loss

\[
L_{contact}
=
p
\cdot
d(H,Z)^2
+
(1-p)
\cdot
[\max(0,\epsilon-d(H,Z))]^2
\]

with a separate penetration term still active.

This prevents the system from “solving” contact by putting the hand several centimeters away from the body.

---

# 28. End-to-end execution

## 28.1 Asset verification

```bash
python scripts/00_check_assets.py \
  --config configs/sgnify.yaml
```

Check:

- SMPL-X assets;
- MANO LEFT/RIGHT;
- H4W++ checkpoint;
- WiLoR assets required by H4W++;
- DWPose checkpoint;
- PAD-Hand checkpoint;
- DexAvatar SignBPoser;
- DexAvatar SignHPoser;
- Sapiens assets;
- SGNify frames/GT/evaluation resources.

Print SHA256 for all checkpoints.

---

## 28.2 Baseline lock

```bash
bash scripts/01_run_dexavatar_baseline.sh \
  --data /path/to/sgnify \
  --out results/b0
```

Then:

```bash
python eval/metric_lock.py \
  --pred results/b0 \
  --gt /path/to/sgnify_gt
```

---

## 28.3 H4W++ extraction

```bash
conda run -n h4wpp \
python -m src.h4wpp_adapter.infer_sequence \
  --input /path/to/sequence \
  --output data/cache/h4wpp/SEQ_ID \
  --checkpoint /path/to/h4wpp.pth
```

Validate:

```bash
python -m src.h4wpp_adapter.validate_cache \
  data/cache/h4wpp/SEQ_ID
```

---

## 28.4 PAD two-hand refinement

```bash
conda run -n pad_hand \
python -m src.pad_adapter.two_hand_refiner \
  --h4w-cache data/cache/h4wpp/SEQ_ID \
  --checkpoint /path/to/pad_hand.pth \
  --window 16 \
  --stride 8 \
  --output data/cache/pad/SEQ_ID
```

Validation:

```bash
python -m src.pad_adapter.validate_pad \
  data/cache/pad/SEQ_ID
```

---

## 28.5 Sapiens observations

Reuse DexAvatar’s official extraction path.

```bash
bash scripts/04_extract_sapiens.sh ...
```

Do not replace Sapiens initially; we want the fitting comparison to isolate H4W++/PAD/fusion changes.

---

## 28.6 Final fitting

```bash
conda run -n dexavatar \
python -m src.dex_adapter.fit_sequence \
  --config configs/experiments/b3_adaptive.yaml \
  --h4w-cache data/cache/h4wpp/SEQ_ID \
  --pad-cache data/cache/pad/SEQ_ID \
  --sapiens-cache data/cache/sapiens/SEQ_ID \
  --sign-meta /path/to/sign_metadata \
  --out results/b3_adaptive/SEQ_ID
```

---

## 28.7 Evaluation

```bash
python eval/official_wrapper.py \
  --pred results/b3_adaptive \
  --gt /path/to/sgnify_gt \
  --out results/b3_adaptive/metrics.json
```

Then aggregate:

```bash
python eval/aggregate.py results/b3_adaptive
```

---

# 29. `run_all.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SEQ=$1

python scripts/00_check_assets.py --config configs/sgnify.yaml

conda run -n h4wpp \
python -m src.h4wpp_adapter.infer_sequence \
  --input "$SGNIFY/$SEQ" \
  --output "data/cache/h4wpp/$SEQ"

conda run -n pad_hand \
python -m src.pad_adapter.two_hand_refiner \
  --h4w-cache "data/cache/h4wpp/$SEQ" \
  --output "data/cache/pad/$SEQ"

bash scripts/04_extract_sapiens.sh "$SEQ"

conda run -n dexavatar \
python -m src.dex_adapter.fit_sequence \
  --config configs/experiments/b3_adaptive.yaml \
  --h4w-cache "data/cache/h4wpp/$SEQ" \
  --pad-cache "data/cache/pad/$SEQ" \
  --out "results/b3_adaptive/$SEQ"
```

Do not add evaluation into the inner optimization loop for final test data.

---

# 30. Suggested config

```yaml
experiment:
  name: h4wpp_pad_dex_adaptive
  seed: 42

h4wpp:
  export_raw_wilor: true
  preserve_body_wrist: true

pad:
  seq_len: 16
  stride: 8
  diffusion_steps: 4
  use_root_as_hard_target: false
  finger_only: true
  missing_frame_mask: true
  overlap_merge: so3_karcher

fusion:
  mode: deterministic   # learned later
  temperature: 1.0

  visual:
    use_kpt_conf: true
    use_bbox_score: true
    use_reproj: true

  temporal:
    use_pad_disagreement: true
    use_motion_features: true

  wrist:
    pad_weight: 0.0

dexavatar:
  keep_signbposer: true
  keep_signhposer: true
  keep_biomechanics: true
  keep_collision: true

  temporal:
    body: true
    fingers_generic: reduced

contact:
  enabled: false
```

---

# 31. Ablation matrix

The experiment order must isolate causes.

| ID | H4W++ | PAD | Adaptive fusion | Sign priors | Contact | Purpose |
|---|---:|---:|---:|---:|---:|---|
| **B0** | ✗ | ✗ | ✗ | ✓ | original | reproduce DexAvatar |
| **B1** | ✓ | ✗ | ✗ | ✓ | original | effect of frame-level H4W++ |
| **B2** | ✓ | ✓ fingers | ✗ fixed weights | ✓ | original | effect of PAD temporal fingers |
| **B3** | ✓ | ✓ | ✓ | ✓ | original | core proposed method |
| **B4** | ✓ | ✓ | ✓ | ✓ | ✓ | optional full model |

Additional diagnostic ablations:

| Test | Change |
|---|---|
| wrist ablation | PAD root weak prior vs zero |
| PAD stride | 4 vs 8 vs 16 |
| missing-frame handling | skip-window vs masked interpolation |
| overlap merge | center select vs SO(3) mean |
| gate | deterministic vs learned |
| sign prior | with vs without SignHPoser |
| temporal | PAD fingers vs generic hand smoothness |
| confidence | no confidence vs frame-level vs joint-level |

---

# 32. Evaluation protocol

## 32.1 Primary

Report:

```text
Upper-body TR-V2V ↓
Left-hand TR-V2V ↓
Right-hand TR-V2V ↓
```

using the evaluator verified in the metric-lock stage.

---

## 32.2 Secondary diagnostic metrics

Recommended:

- hand MPJPE if available under the same GT convention;
- wrist error;
- finger-only vertex error;
- temporal acceleration error;
- frame-to-frame angular velocity/acceleration;
- penetration volume/distance;
- contact distance if contact module is enabled.

These are diagnostic, not substitutes for SGNify primary metrics.

---

## 32.3 Per-condition slices

If labels can be derived without test GT leakage, report:

- high-confidence hand frames;
- low-confidence hand frames;
- strong motion blur;
- one-hand visible;
- both hands visible;
- hand-hand overlap;
- hand-face/body proximity.

This is especially important to prove **why** PAD/adaptive fusion helps.

---

# 33. Required unit and integration tests

## 33.1 Rotation round trip

```python
def test_rotation_roundtrip():
    aa = random_axis_angle()
    R = axis_angle_to_matrix(aa)
    aa2 = matrix_to_axis_angle(R)
    R2 = axis_angle_to_matrix(aa2)
    assert so3_geodesic(R, R2) < 1e-5
```

---

## 33.2 SO(3) validity

For all exported rotations:

\[
\|R^\top R-I\|_F<\epsilon,
\quad
|\det(R)-1|<\epsilon.
\]

---

## 33.3 Left/right identity

Load a sequence where both hands are visible.

Verify:

- left cache always remains left;
- right cache always remains right;
- no side swap occurs after missing detections.

---

## 33.4 PAD missing-window test

Synthetic input:

```text
valid valid valid missing valid ...
```

Expected:

- window is still processed;
- missing frame has `src_valid=False`;
- neighboring valid frames are preserved;
- output contains no NaNs.

---

## 33.5 Overlap merge test

For identical predictions from two windows:

\[
R_{merged}=R.
\]

For slightly different rotations:

- merged matrix remains in SO(3);
- result lies between them geodesically.

---

## 33.6 Wrist preservation test

Run:

```text
H4W++ only
H4W++ + PAD-fingers
```

Verify:

\[
d_{SO(3)}
(
R^{wrist}_{H4W},
R^{wrist}_{PAD-finger-system}
)
\approx 0
\]

before DexAvatar optimization.

If this changes, the adapter is incorrectly writing PAD root into SMPL-X.

---

## 33.7 Camera projection test

Project H4W++ SMPL-X joints with the adapter camera.

Compare with H4W++/DexAvatar expected 2D projection.

Large mismatch indicates:

- focal-length convention error;
- principal-point error;
- crop/full-image mismatch;
- coordinate-axis mismatch.

---

## 33.8 Metric lock

Original DexAvatar prediction run through our evaluator wrapper must reproduce the reference baseline within tolerance.

---

# 34. Debug visualizations

Every sequence should optionally render a 4-panel video:

```text
┌──────────────┬──────────────┐
│ RGB + 2D     │ H4W++        │
│ observations │ SMPL-X       │
├──────────────┼──────────────┤
│ PAD MANO     │ Final ours   │
│ hand overlay │ SMPL-X       │
└──────────────┴──────────────┘
```

Overlay:

- hand boxes;
- keypoint confidence;
- visual/PAD/sign weights;
- visual vs PAD angular disagreement;
- contact state if used.

This will catch errors much faster than aggregate metrics.

---

# 35. Confidence failure diagnostics

Log per frame:

```json
{
  "frame": 37,
  "left": {
    "visual_valid": true,
    "kpt_conf": 0.32,
    "reproj": 14.2,
    "pad_disagreement_deg": 18.6,
    "w_visual": 0.21,
    "w_pad": 0.51,
    "w_sign": 0.28
  }
}
```

Then compare low-confidence frames before/after.

---

# 36. Optimization strategy

Do not enable all losses from iteration zero.

Suggested stage-wise fitting:

### Stage 1 — body/camera stabilization

Optimize:

- camera;
- global orientation;
- body pose;
- body shape.

Hands stay near H4W++ initialization.

### Stage 2 — visual hand fitting

Enable:

- hand 2D evidence;
- H4W++ visual finger targets;
- SignHPoser.

### Stage 3 — temporal adaptive hand fitting

Enable:

- PAD finger prior;
- reliability fusion;
- reduced generic hand smoothness.

### Stage 4 — physical refinement

Enable:

- biomechanics;
- interpenetration;
- optional contact.

This prevents a strong temporal/contact term from dragging a badly initialized body into a local minimum.

---

# 37. Optimizer parameter groups

Recommended grouping:

```python
params_stage1 = [
    global_orient,
    body_pose_latent,
    camera,
    betas,
]

params_stage2 = params_stage1 + [
    lhand_latent,
    rhand_latent,
]

params_stage3 = params_stage2
```

The adaptive gate can be:

- frozen deterministic during fitting;
- pretrained MLP frozen at test-time.

Avoid jointly training the gate inside each test sequence using 3D test GT.

---

# 38. SignHPoser interaction with PAD

Both PAD and SignHPoser can act as priors.

They are not redundant:

- PAD: generic temporal physics/motion plausibility;
- SignHPoser: sign-domain pose manifold.

Potential conflict:

```text
PAD prefers smooth generic trajectory
SignHPoser prefers sign-specific articulation
```

The adaptive gate should arbitrate based on image evidence.

Important case:

- visually clear rapid finger change → visual/sign dominate;
- blurred intermediate frames → PAD/sign dominate.

---

# 39. Why not simply use PAD output as initialization only?

That is an important ablation:

### Option A

```text
PAD pose → initialize DexAvatar → no PAD loss
```

### Option B

```text
H4W pose init + PAD soft loss during fitting
```

### Option C

```text
PAD init + PAD soft loss
```

Expected best candidate is B or C, but this must be measured.

The proposal chooses **B as default** because it preserves H4W++ spatial body-hand consistency and treats temporal reconstruction as a prior rather than an absolute truth.

---

# 40. Why Hand4Whole++ is the main backbone

The key sign-language requirement is not just a low-error isolated hand:

\[
\text{shoulder}
\rightarrow
\text{elbow}
\rightarrow
\text{forearm}
\rightarrow
\text{wrist}
\rightarrow
\text{fingers}.
\]

Hand4Whole++ explicitly integrates specialist hand features into a whole-body model.

Therefore it is better aligned with our task than a design where:

```text
body estimator ─────┐
                    ├─ combine outputs late
hand estimator ─────┘
```

Our main system inherits H4W++’s spatial coupling and extends it temporally/sign-specifically.

---

# 41. Source-code modifications summary

## 41.1 Hand4Whole++

Patch:

```text
main/model.py
```

Purpose:

- export raw internal WiLoR rotations/shape/vertices;
- export hand existence/bboxes/confidence;
- keep original model output unchanged.

Do not alter:

- CHAM/HandControlNet behavior;
- WiLoR handedness restoration;
- body wrist ownership;
- rigid hand-body alignment.

---

## 41.2 PAD-Hand

Do **not** modify the original demo in-place.

Create wrapper under our code:

```text
src/pad_adapter/
```

Reuse:

- pretrained model;
- model architecture;
- diffusion sampler;
- MANO LEFT/RIGHT.

Replace demo assumptions:

- one-hand → two independent hand tracks;
- right-priority selection → explicit side identity;
- non-overlap → overlap;
- skip missing window → padding mask + interpolation;
- demo vertices-only output → export decoded rotations.

---

## 41.3 DexAvatar

Prefer subclass/new parser and a new loss component.

Changes:

```text
src/dex_adapter/dataset.py
src/dex_adapter/adaptive_loss.py
src/dex_adapter/fit_sequence.py
```

Minimal patching of `third_party/DexAvatar` is preferable.

If direct internal patch is unavoidable, maintain a small `.patch` file and document every changed line.

---

# 42. Third-party code policy

Keep third-party repositories as pinned submodules.

Example:

```bash
git submodule add <repo> third_party/Hand4Whole-plus-plus_RELEASE
git submodule add <repo> third_party/PAD-Hand-CVPR-2026
git submodule add <repo> third_party/DexAvatar
```

Record exact SHAs in:

```json
{
  "Hand4Whole++": "<sha>",
  "PAD-Hand": "<sha>",
  "DexAvatar": "<sha>"
}
```

Do not vendor or redistribute MANO/SMPL-X model files if their licenses prohibit it.

Hand4Whole++ code is released under its repository license; DexAvatar contains dependencies/assets with separate terms; MANO/SMPL-X assets retain their original licensing. PAD-Hand’s repository terms should be checked before redistribution of modified code/checkpoints.

---

# 43. Reproducibility metadata

Every experiment must save:

```text
git SHA
checkpoint hashes
config YAML
random seed
CUDA version
PyTorch version
PyTorch3D version
GPU model
sequence IDs
evaluation protocol ID
```

Create:

```text
results/<exp>/run_manifest.json
```

---

# 44. Caching policy

Caches are immutable by default.

Example:

```text
h4wpp_v1/
pad_v1_stride8/
sapiens_original/
```

If code changes, create a new cache version.

Never silently overwrite H4W++ extraction while comparing experiments.

---

# 45. Acceptance gates

## Gate A — baseline

- Original DexAvatar runs.
- Metric lock passes.

## Gate B — H4W++

- all SGNify frames processed;
- left/right tracks stable;
- SMPL-X projects correctly;
- wrist consistency visualization is correct.

## Gate C — PAD

- both hands supported;
- missing frames supported;
- no NaNs;
- overlap merge valid;
- left parity test passes or fallback canonicalization is implemented.

## Gate D — Dex fusion

- H4W++ → DexAvatar improves or at least preserves baseline;
- adding PAD improves low-confidence frames;
- no large degradation on high-confidence rapid-motion frames.

## Gate E — SOTA experiment

Only after A–D pass:

- run complete SGNify evaluation;
- freeze all hyperparameters;
- generate final tables.

---

# 46. Expected failure modes and mitigation

## 46.1 PAD over-smooths semantic motion

**Symptom**

- finger transitions become too slow;
- sign handshape is averaged.

**Mitigation**

- adaptive lower PAD weight on high-confidence fast motion;
- preserve SignHPoser;
- PAD finger-only;
- do not add large generic temporal hand loss.

---

## 46.2 Wrist/body discontinuity

**Symptom**

- isolated hand looks good;
- hand orientation inconsistent with forearm.

**Mitigation**

- H4W++ owns wrist;
- PAD root disabled as hard target;
- wrist preservation test;
- optional weak wrist-consistency loss.

---

## 46.3 Left/right swap

**Symptom**

- sudden jump in one hand trajectory;
- wrong prior assigned.

**Mitigation**

- explicit handedness key;
- one cache namespace per side;
- no detection-order association;
- left/right integration tests.

---

## 46.4 PAD windows create temporal seams

**Symptom**

- discontinuity every 16 frames.

**Mitigation**

- stride 8/4;
- SO(3) overlap merge;
- center weighting.

---

## 46.5 Missing hand kills whole window

**Symptom**

- no PAD output around occlusions.

**Mitigation**

- interpolation for numeric input;
- use PAD source mask;
- do not skip window.

---

## 46.6 Camera mismatch between H4W++ and DexAvatar

**Symptom**

- correct 3D body but poor 2D reprojection.

**Mitigation**

- explicit full-image camera conversion;
- projection unit test before fitting;
- never pass crop-camera parameters as full-image camera.

---

# 47. Research-story boundary

A strong paper should not claim:

> “We combine three SOTA models.”

The research contribution should be written as:

> **A sign-aware reconstruction framework that unifies spatially coupled body–hand evidence, physics-aware temporal finger priors, and sign-domain pose priors through confidence-adaptive optimization.**

The pretrained methods are the foundation.

The novelty is the **cross-source reliability model and optimization behavior for sign-language reconstruction**.

---

# 48. Recommended experiment progression

Do not jump to the full model.

## Experiment 0

Original DexAvatar.

## Experiment 1

H4W++ initialization + original DexAvatar fitting.

Question:

> Does a hand-aware whole-body initializer improve SGNify?

## Experiment 2

Add PAD fingers with constant small weight.

Question:

> Does temporal hand recovery help?

## Experiment 3

Adaptive visual/PAD/sign fusion.

Question:

> Does reliability-aware fusion outperform fixed weighting?

## Experiment 4

Optional sign contact.

Question:

> Does semantic contact improve contact-heavy frames without harming global TR-V2V?

---

# 49. Core pseudocode

```python
def reconstruct_sign_sequence(frames, sign_meta, cfg):
    # ------------------------------------------
    # 1. frame-level spatial reconstruction
    # ------------------------------------------
    h4w_records = h4wpp_infer_sequence(frames)

    # ------------------------------------------
    # 2. build two temporal MANO tracks
    # ------------------------------------------
    left_track = build_pad_track(h4w_records, side="left")
    right_track = build_pad_track(h4w_records, side="right")

    # ------------------------------------------
    # 3. temporal finger refinement
    # ------------------------------------------
    pad_left = pad_refine_track(
        left_track,
        side="left",
        seq_len=16,
        stride=cfg.pad.stride,
    )

    pad_right = pad_refine_track(
        right_track,
        side="right",
        seq_len=16,
        stride=cfg.pad.stride,
    )

    # ------------------------------------------
    # 4. Sapiens / sign observations
    # ------------------------------------------
    sapiens = load_or_extract_sapiens(frames)

    # ------------------------------------------
    # 5. sequential sign-specific fitting
    # ------------------------------------------
    outputs = []

    temporal_state = None

    for t in range(len(frames)):
        init = build_smplx_init(h4w_records[t])

        rel_l = compute_reliability(
            h4w_records[t],
            pad_left[t],
            side="left",
        )

        rel_r = compute_reliability(
            h4w_records[t],
            pad_right[t],
            side="right",
        )

        weights_l = gate(rel_l)
        weights_r = gate(rel_r)

        result, temporal_state = dex_fit_frame(
            frame=frames[t],
            init=init,
            visual_hand=h4w_records[t],
            temporal_hand={
                "left": pad_left[t],
                "right": pad_right[t],
            },
            weights={
                "left": weights_l,
                "right": weights_r,
            },
            sapiens=sapiens[t],
            sign_meta=sign_meta[t],
            temporal_state=temporal_state,
            preserve_body_wrist=True,
        )

        outputs.append(result)

    return outputs
```

---

# 50. `compute_reliability` pseudocode

```python
def compute_reliability(h4w, pad, side):
    vis = side_view(h4w, side)

    keypoint_conf = robust_mean(vis.kpt_conf)
    bbox_area = normalized_bbox_area(vis.bbox, h4w.image_hw)
    visual_valid = float(vis.exist)

    reproj = estimate_reprojection_error(vis)
    disagreement = mean_geodesic(
        vis.finger_rotmat,
        pad.finger_rotmat
    )

    motion = pad.temporal_accel_score

    return np.array([
        keypoint_conf,
        bbox_area,
        visual_valid,
        normalize(reproj),
        normalize(disagreement),
        normalize(motion),
        float(pad.valid),
    ], dtype=np.float32)
```

---

# 51. `adaptive_hand_loss` pseudocode

```python
def adaptive_hand_loss(
    current_R,     # [15,3,3]
    visual_R,      # [15,3,3]
    pad_R,         # [15,3,3]
    w_visual,      # [15]
    w_pad,         # [15]
):
    d_vis = so3_geodesic(current_R, visual_R) ** 2
    d_pad = so3_geodesic(current_R, pad_R) ** 2

    return (
        (w_visual * d_vis).sum()
        + (w_pad * d_pad).sum()
    )
```

SignHPoser remains a separate latent prior term whose coefficient can also be modulated by a frame/hand confidence scalar.

---

# 52. Data leakage rules

Never use SGNify test GT to:

- choose PAD stride;
- choose fusion weights;
- train the gate;
- choose contact thresholds;
- decide which frames get special handling.

Use:

- train/dev partitions;
- separate sign videos;
- synthetic corruptions;
- fixed rules chosen before final test.

The final benchmark should be a single frozen evaluation.

---

# 53. Implementation order checklist

- [ ] Pin official repositories and commit SHAs.
- [ ] Download model assets under their original licenses.
- [ ] Reproduce original DexAvatar.
- [ ] Lock SGNify evaluator.
- [ ] Run H4W++ official demo unchanged.
- [ ] Patch H4W++ test output to export internal WiLoR signals.
- [ ] Build versioned H4W++ frame cache.
- [ ] Validate camera projection.
- [ ] Build explicit L/R hand tracks.
- [ ] Run PAD official demo unchanged on its expected example.
- [ ] Implement PAD two-hand adapter.
- [ ] Implement missing-frame mask.
- [ ] Implement overlapping windows.
- [ ] Implement SO(3) overlap merge.
- [ ] Validate left-hand parity.
- [ ] Confirm PAD does not overwrite body wrist.
- [ ] Create DexAvatar-native parser for H4W++/PAD caches.
- [ ] Reproduce H4W++→DexAvatar baseline without PAD.
- [ ] Add constant PAD finger prior.
- [ ] Add deterministic reliability gate.
- [ ] Add learned gate only if deterministic gate proves useful.
- [ ] Run ablation matrix.
- [ ] Add contact only after core model is stable.
- [ ] Freeze config.
- [ ] Run final SGNify evaluation.
- [ ] Export per-frame diagnostics and failure cases.

---

# 54. Source files to read again while implementing

These are the code paths that should remain open during development.

## Hand4Whole++

```text
main/model.py
common/nets/wilor.py
common/nets/module.py
```

Focus on:

- exact tensor names;
- coordinate conventions;
- hand existence masks;
- output assembly;
- `combine_smplx_mano`;
- rigid alignment anchors;
- hand feature modulation.

## PAD-Hand

```text
wilor_inference.py
demo.py
models/pad_hand.py
models/MANO.py
```

Focus on:

- 16-pose layout;
- rotmat → 6D conversion;
- temporal masks;
- decoded rotmat output;
- sequence length;
- left/right MANO model path.

## DexAvatar

```text
Full_running_command.sh
run_dexavatar.py
dexavatar_fitting/smplifyx/data_parser.py
dexavatar_fitting/smplifyx/fit_single_frame.py
dexavatar_fitting/smplifyx/fitting.py
```

Focus on:

- exact data dictionary;
- hand pose initialization;
- SignBPoser/SignHPoser latent initialization;
- temporal-state update;
- 2D/3D hand loss;
- collision;
- final SMPL-X export.

---

# 55. Final recommended “v1 paper model”

\[
\boxed{
\begin{aligned}
\mathcal V
&\xrightarrow{\text{Hand4Whole++}}
\left(
\Theta^{H4W}_{body},
H^V_L,
H^V_R
\right)
\\
(H^V_L,H^V_R)
&\xrightarrow{\text{two-hand PAD}}
(H^T_L,H^T_R)
\\
(V,T,S)
&\xrightarrow{\text{confidence fusion}}
(w_V,w_T,w_S)
\\
(\Theta^{H4W},H^V,H^T,S)
&\xrightarrow{\text{DexAvatar fitting}}
\Theta^{final}_{1:T}.
\end{aligned}
}
\]

Where:

- \(V\) = visual hand evidence;
- \(T\) = temporal/physics hand evidence;
- \(S\) = sign-specific prior;
- final representation = SMPL-X;
- PAD hard target = fingers only by default;
- wrist = body-aware H4W++ / DexAvatar chain.

---

# 56. Definition of success

The implementation is considered technically successful only if all conditions hold:

1. original DexAvatar baseline is reproduced;
2. H4W++ caches are geometrically/camera consistent;
3. both PAD hands run without side swapping;
4. missing hand observations do not destroy 16-frame windows;
5. overlap merging stays on \(SO(3)\);
6. PAD does not hard overwrite H4W++ wrist;
7. final output remains valid SMPL-X;
8. official/locked SGNify evaluation runs;
9. ablations isolate the effect of H4W++, PAD, and adaptive fusion;
10. improvements are not obtained by tuning on final test GT.

Only after these are satisfied should the implementation be described as a SOTA-oriented full system rather than an experimental integration.

---

# 57. Immediate coding target

The first code milestone should **not** be the full adaptive model.

Implement this minimal chain first:

```text
SGNify frames
   ↓
Hand4Whole++
   ↓
export:
  SMPL-X body
  L/R WiLoR 15-finger pose
  hand validity
   ↓
two-hand PAD
  sequence length 16
  stride 8
  missing-frame mask
   ↓
PAD L/R 15-finger targets
   ↓
DexAvatar original fitting
  H4W init
  PAD constant soft finger loss
  original sign priors
   ↓
SMPL-X
   ↓
SGNify evaluation
```

Once this chain is correct and measurable, add the adaptive gate.

This prevents research novelty from being built on top of an unverified coordinate or temporal-interface bug.

---

## Official sources used for this implementation design

### DexAvatar
- Repository: https://github.com/kaustesseract/DexAvatar

### Hand4Whole++
- Repository: https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE
- CVPR 2026 paper/project should be cross-checked against the repository version used in the experiment.

### PAD-Hand
- Repository: https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026
- CVPR 2026 paper/project should be cross-checked against the released checkpoint/configuration.

---

## Final implementation principle

> **Do not make the model more complicated until the previous interface has been geometrically validated.**

For this project, most catastrophic failures are more likely to come from:

- left/right convention mismatch,
- crop-camera mismatch,
- root/wrist ownership conflict,
- MANO/SMPL-X interface mismatch,
- temporal window seams,
- evaluator mismatch,

than from insufficient neural-network capacity.

The proposed system is therefore deliberately built as a sequence of **validated, versioned interfaces** before adding research novelty.
---

# 58. OFFICIAL SGNify TR-V2V Evaluation Protocol
## This section supersedes any generic interpretation of TR-V2V elsewhere in this document

The **author-provided SGNify evaluator supplied with this project must be treated as the ground-truth evaluation protocol**.

The implementation described below is derived directly from the provided `evaluate_new_fitting.py`.

The primary rule is:

> **Do not replace this evaluator with PA-MPVPE, Procrustes alignment, pelvis alignment, root alignment, or any other “equivalent-looking” implementation.**

For final SGNify reporting, our prediction meshes must be exported into the folder structure expected by the official evaluator and evaluated using the same vertex subsets, central-frame selection, sign-class handling, translation-only centering, and millimeter conversion.

---

## 58.1 Exact TR-V2V definition implemented by the official code

For a selected SMPL-X vertex subset \(V\), let

\[
\hat{\mathbf v}_i \in \mathbb R^3
\]

be the predicted vertex and

\[
\mathbf v_i \in \mathbb R^3
\]

the corresponding ground-truth vertex.

The evaluator independently computes the centroid of the predicted subset and the centroid of the GT subset:

\[
\hat{\boldsymbol\mu}
=
\frac{1}{|V|}
\sum_{i\in V}
\hat{\mathbf v}_i,
\]

\[
\boldsymbol\mu
=
\frac{1}{|V|}
\sum_{i\in V}
\mathbf v_i.
\]

Then it translation-centers both subsets:

\[
\hat{\mathbf v}'_i
=
\hat{\mathbf v}_i-\hat{\boldsymbol\mu},
\]

\[
\mathbf v'_i
=
\mathbf v_i-\boldsymbol\mu.
\]

The per-vertex translation-aligned error is

\[
e_i^{TR}
=
\left\|
\hat{\mathbf v}'_i-\mathbf v'_i
\right\|_2.
\]

The reported TR-V2V for that subset is

\[
\boxed{
E_{TR\text{-}V2V}
=
\frac{1}{|V|}
\sum_{i\in V}
e_i^{TR}
}
\]

and the Python evaluator multiplies the result by \(1000\), therefore the final reported unit is:

\[
\boxed{\text{millimeters}}
\]

### Important interpretation

This is **translation-only alignment by each subset's own centroid**.

It does **not** estimate:

- rotation;
- scale;
- similarity transform;
- Procrustes transform.

Therefore:

\[
\boxed{
TR\text{-}V2V \neq PA\text{-}MPVPE
}
\]

and it must not be reported as such.

The exact primitive used in the official evaluator is:

```python
def point_error(source_points, target_points):
    return np.sqrt(
        np.power(source_points - target_points, 2).sum(axis=-1)
    )


def transl_point_error(source_points, target_points):
    source_center = np.mean(
        source_points, axis=0, keepdims=True
    )
    target_center = np.mean(
        target_points, axis=0, keepdims=True
    )

    aligned_source_points = source_points - source_center
    aligned_target_points = target_points - target_center

    return point_error(
        aligned_source_points,
        aligned_target_points
    )
```

The return value is a vector with one Euclidean error per selected vertex. The final mean is computed only after errors from all evaluated frames/signs have been concatenated.

---

# 59. Official evaluated SMPL-X vertex subsets

The evaluator assumes the standard SMPL-X topology with:

\[
10475
\]

vertices:

```python
vertex_indices = {
    "all": np.arange(0, 10475),
    "left hand": left_hand_ids,
    "right hand": right_hand_ids,
}
```

The exact left/right hand vertex IDs are **not reconstructed manually**.

They are loaded from:

```text
MANO_SMPLX_vertex_ids.pkl
```

using:

```python
with open(mano_path, "rb") as f:
    mano_data = pickle.load(f)

left_hand_ids = mano_data["left_hand"]
right_hand_ids = mano_data["right_hand"]
```

The upper-body subsets are loaded from the official SGNify segmentation assets:

```text
sgnify_part_segm_above_pelvis_joint/
```

Specifically:

```python
vertex_indices["above pelvis upper body"] = np.load(
    "upper_body.npy"
)

vertex_indices["above pelvis minus head"] = np.load(
    "upper_body_minus_head.npy"
)

vertex_indices["above pelvis minus face"] = np.load(
    "upper_body_minus_face.npy"
)
```

Therefore the primary subsets relevant to our paper should come directly from these official assets:

- `left hand`
- `right hand`
- `above pelvis upper body`

Do not regenerate these subsets from anatomical intuition.

---

# 60. Official central-frame selection

The evaluator reads a JSON sign-segmentation file:

```python
with open(sign_seg_file, "r") as file:
    frame_segment = json.load(file)
```

For each sign:

```python
segment_range = frame_segment[file_name]
start_idx, end_idx = (
    segment_range[0] * 2,
    segment_range[1] * 2,
)
```

It then selects GT `.obj` files whose numeric filename lies in the inclusive interval:

```python
range(start_idx, end_idx + 1)
```

Therefore, for this benchmark implementation:

\[
\boxed{
f_{\text{GT,start}}
=
2 f_{\text{segment,start}}
}
\]

\[
\boxed{
f_{\text{GT,end}}
=
2 f_{\text{segment,end}}
}
\]

and the endpoint is inclusive.

### Implementation rule

Our final evaluator wrapper must reproduce this behavior exactly.

Do not independently redefine “central frames” based on:

- sequence midpoint;
- sign temporal center;
- a percentage crop;
- predicted sign boundaries.

The provided `sign_seg` JSON is the authority.

---

# 61. Prediction and GT mesh ordering

## 61.1 Ground truth

For every sign directory:

```python
folder = Path(
    os.path.join(gt_folder_root, file_name)
)
files = sorted(
    folder.glob("*.obj"),
    key=lambda p: int(p.stem)
)
```

The central segment is selected from these numeric GT filenames.

---

## 61.2 Prediction

For every sign:

```python
meshes_dir = Path(
    os.path.join(
        evaluate_folder,
        file_name,
        "smplifyx",
        "meshes"
    )
)
```

Prediction `.obj` files are sorted by the **first integer appearing in the filename stem**:

```python
files = sorted(
    meshes_dir.glob("*.obj"),
    key=lambda p: int(
        re.search(r"\d+", p.stem).group()
    )
)
```

Then the evaluator pairs:

```python
gt_objs[idx][inter_idx]
```

with:

```python
mocap_objs[idx][inter_idx]
```

by list position.

### Critical consequence

The official script does **not** perform an explicit timestamp join between the selected central GT filenames and predicted frame IDs.

Therefore our exported prediction sequence must preserve the ordering expected by the evaluator.

A missing prediction mesh before or inside the evaluated interval can shift all subsequent pairings and invalidate the metric.

### Required pre-evaluation assertion

Before evaluation:

```python
assert len(pred_meshes) >= len(selected_gt_meshes)
```

and we should additionally verify the intended frame-to-frame mapping in our wrapper.

---

# 62. Mesh topology invariant

For each evaluated pair, the official code loads vertices and triangle faces from the two OBJ files and asserts:

```python
np.testing.assert_array_equal(
    soma_faces,
    faces
)
```

Therefore final predictions must have exactly the same SMPL-X topology and face indexing as GT.

This is another reason the main proposal keeps the final representation as:

\[
\boxed{\text{SMPL-X}}
\]

rather than evaluating a MANO or alternative body topology directly.

### Required invariant

For every exported mesh:

```python
assert pred_vertices.shape[0] == 10475
assert np.array_equal(pred_faces, official_smplx_faces)
```

PAD-Hand MANO geometry is only an intermediate prior.

It must never be written as the final SGNify evaluation mesh.

---

# 63. Sign class handling in the official protocol

The sign file is parsed line by line:

```python
tokens = line.strip().split()

soma_to_method[tokens[0]] = args.method
class_sign[tokens[0]] = tokens[1]
```

So every line is assumed to contain at least:

```text
<SIGN_ID> <CLASS_ID>
```

The evaluator contains special handling for:

```python
class_sign[soma_key] == "0"
```

For such signs:

### 63.1 Left hand is excluded from non-left-hand subsets

Before evaluating a subset:

```python
if key != "left hand" and class_sign[soma_key] == "0":
    vertex_index_set = np.setdiff1d(
        vertex_index_set,
        left_hand_ids
    )
```

This means for class-0 signs, the left-hand vertices are removed from:

- `all`;
- `right hand` has no practical overlap with left hand;
- `above pelvis upper body`;
- `above pelvis minus head`;
- `above pelvis minus face`.

### 63.2 Left-hand TR-V2V itself is skipped

The official evaluator explicitly does:

```python
if key == "left hand" and class_sign[soma_key] == "0":
    continue
```

Thus:

\[
\boxed{
\text{class 0 signs do not contribute to the reported left-hand TR-V2V}
}
\]

### Consequence for aggregation

Do not replace this with:

- zero left-hand error;
- NaN converted to zero;
- averaging every sign equally.

The skipped frames simply do not enter the global left-hand error vector.

---

# 64. Exact aggregation used for the final metric

Per selected frame and body subset:

```python
tr_error = transl_point_error(
    curr_method_verts,
    curr_soma_verts
)
```

This returns:

```text
[number_of_subset_vertices]
```

errors.

These vectors are appended:

```python
sign_errors[f"TR {key}"].append(tr_error)
```

After all signs:

```python
full_errors[key] += sign_errors[key]
```

Finally:

```python
full_errors[key] = np.concatenate(
    full_errors[key],
    axis=0
)

mean_value = full_errors[key].mean() * 1000
```

Therefore the final benchmark metric is a **global mean over all participating vertices across all participating evaluated frames**, not an unweighted mean of per-sign means.

Formally, for valid evaluated frame/subset pairs \(\mathcal F\):

\[
\boxed{
E
=
\frac{
\sum_{f\in\mathcal F}
\sum_{i\in V_f}
e_{f,i}
}{
\sum_{f\in\mathcal F}|V_f|
}
}
\]

Because a fixed anatomical subset normally has a fixed vertex count, this is equivalent to a frame-weighted mean for that subset, but the implementation should still preserve the official concatenate-then-mean procedure.

---

# 65. Auxiliary wrist-centered V2V in the official script

The evaluator also computes auxiliary hand errors using the GT wrist.

The joint regressor is loaded from:

```text
SMPLX_NEUTRAL.npz
```

as:

```python
J_regressor = smplx_model_data["J_regressor"]
```

Then:

```python
gt_joints = J_regressor.dot(soma_vertices)

soma_left_wrist = gt_joints[20]
soma_right_wrist = gt_joints[21]
```

with:

```python
LEFT_WRIST_INDEX = 20
RIGHT_WRIST_INDEX = 21
```

The helper:

```python
def point_error_common_center(
    source_points,
    target_points,
    center,
):
    centered_source_points = (
        source_points
        - np.mean(source_points, axis=0, keepdims=True)
        + center
    )

    centered_target_points = (
        target_points
        - np.mean(target_points, axis=0, keepdims=True)
        + center
    )

    return point_error(
        centered_source_points,
        centered_target_points
    )
```

is used for:

- `V2V left wrist`
- `V2V right wrist`

These are useful diagnostics.

However, the primary metric that our method should optimize/report as the SGNify TR-V2V target is the official:

```text
TR left hand
TR right hand
TR above pelvis upper body
```

unless the target paper table explicitly requests another official field.

---

# 66. Exact final units

All loaded OBJ vertex coordinates are treated as meters by the evaluator.

Final log output applies:

```python
mean_value = full_errors[key].mean() * 1000
```

and prints:

```text
(mm)
```

Therefore:

\[
\boxed{
1.0\ \text{evaluator coordinate unit}
\rightarrow
1000\ \text{mm}
}
\]

Our exporter must not pre-scale vertices to millimeters.

Final OBJ predictions must remain in the same coordinate unit as SGNify GT / SMPL-X.

---

# 67. NaN handling

The official implementation checks:

```python
if np.isnan(method_vertices).any():
    continue
```

A prediction mesh containing any NaN is skipped for that evaluated loop iteration.

For our system this must be treated as a **hard failure**, not a strategy.

Our wrapper should instead assert before running the official evaluator:

```python
assert np.isfinite(pred_vertices).all()
```

A method should never improve a benchmark by silently losing difficult frames.

---

# 68. Official-compatible prediction folder contract

Our final result exporter should produce:

```text
<evaluate_folder>/
├── <SIGN_ID_001>/
│   └── smplifyx/
│       └── meshes/
│           ├── 000000.obj
│           ├── 000001.obj
│           ├── 000002.obj
│           └── ...
├── <SIGN_ID_002>/
│   └── smplifyx/
│       └── meshes/
│           └── ...
└── ...
```

The filenames only need a parseable integer stem/substring for the official sorting logic, but using deterministic zero-padded frame IDs is recommended.

Example:

```python
out_path = (
    Path(evaluate_folder)
    / sign_id
    / "smplifyx"
    / "meshes"
    / f"{frame_idx:06d}.obj"
)
```

---

# 69. Official evaluator CLI contract

The provided script requires:

```text
--method
--central
--evaluate_folder
--gt_folder
--sign_file
--sign_seg
```

A canonical call should be stored with every final experiment:

```bash
python evaluate_new_fitting.py \
  --method ours \
  --central true \
  --evaluate_folder /path/to/ours_predictions \
  --gt_folder /path/to/sgnify_gt \
  --sign_file /path/to/sign_file.txt \
  --sign_seg /path/to/sign_segments.json
```

### Note on `--central`

The supplied script parses this argument, but the central-frame loading logic is driven directly by `sign_seg_file` in the current code path.

Therefore our wrapper will still pass:

```text
--central true
```

to match the expected protocol, while preserving the exact `sign_seg` behavior.

---

# 70. Required evaluator wrapper in our project

Add:

```text
eval/
├── official_sgnify_trv2v.py
├── metric_lock.py
├── preflight.py
└── aggregate_official.py
```

The wrapper must not modify the metric.

Its job is only to:

1. validate inputs;
2. copy/import the official primitives;
3. run the same protocol;
4. save machine-readable metrics;
5. detect silent ordering/topology failures.

---

# 71. Reference implementation for our wrapper

```python
from pathlib import Path
import json
import pickle
import re

import numpy as np


LEFT_WRIST_INDEX = 20
RIGHT_WRIST_INDEX = 21
SMPLX_NUM_VERTS = 10475


def load_obj(path):
    verts = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("v "):
                xyz = [
                    float(x)
                    for x in line.split()[1:4]
                ]
                verts.append(xyz)

            elif line.startswith("f "):
                ids = [
                    int(tok.split("/")[0])
                    for tok in line.split()[1:]
                ]
                faces.extend(ids)

    verts = np.asarray(
        verts,
        dtype=np.float64
    )

    faces = (
        np.asarray(faces)
        .reshape(-1, 3)
        - 1
    )

    return verts, faces


def point_error(
    source_points,
    target_points,
):
    return np.sqrt(
        np.power(
            source_points - target_points,
            2,
        ).sum(axis=-1)
    )


def transl_point_error(
    source_points,
    target_points,
):
    """
    Official SGNify TR-V2V primitive:
    translation-only alignment by
    independently subtracting each subset centroid.
    """

    source_center = np.mean(
        source_points,
        axis=0,
        keepdims=True,
    )

    target_center = np.mean(
        target_points,
        axis=0,
        keepdims=True,
    )

    source_aligned = (
        source_points
        - source_center
    )

    target_aligned = (
        target_points
        - target_center
    )

    return point_error(
        source_aligned,
        target_aligned,
    )


def load_vertex_subsets(data_base_dir):
    data_base_dir = Path(data_base_dir)

    with open(
        data_base_dir
        / "MANO_SMPLX_vertex_ids.pkl",
        "rb",
    ) as f:
        mano_ids = pickle.load(f)

    left_hand_ids = mano_ids["left_hand"]
    right_hand_ids = mano_ids["right_hand"]

    segm = (
        data_base_dir
        / "sgnify_part_segm_above_pelvis_joint"
    )

    subsets = {
        "all": np.arange(
            SMPLX_NUM_VERTS
        ),
        "left hand": left_hand_ids,
        "right hand": right_hand_ids,
        "above pelvis upper body": np.load(
            segm / "upper_body.npy"
        ),
        "above pelvis minus head": np.load(
            segm / "upper_body_minus_head.npy"
        ),
        "above pelvis minus face": np.load(
            segm / "upper_body_minus_face.npy"
        ),
    }

    return (
        subsets,
        left_hand_ids,
        right_hand_ids,
    )


def load_sign_file(path):
    sign_class = {}

    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()

            if not tokens:
                continue

            sign_id = tokens[0]
            class_id = tokens[1]

            sign_class[sign_id] = class_id

    return dict(
        sorted(sign_class.items())
    )


def load_central_gt(
    gt_root,
    sign_id,
    sign_segments,
):
    start, end = sign_segments[sign_id]

    start *= 2
    end *= 2

    folder = Path(gt_root) / sign_id

    all_files = sorted(
        folder.glob("*.obj"),
        key=lambda p: int(p.stem),
    )

    by_number = {
        int(p.stem): p
        for p in all_files
    }

    return [
        by_number[i]
        for i in range(
            start,
            end + 1,
        )
        if i in by_number
    ]


def load_prediction_meshes(
    pred_root,
    sign_id,
):
    folder = (
        Path(pred_root)
        / sign_id
        / "smplifyx"
        / "meshes"
    )

    return sorted(
        folder.glob("*.obj"),
        key=lambda p: int(
            re.search(
                r"\d+",
                p.stem,
            ).group()
        ),
    )


def evaluate_trv2v(
    pred_root,
    gt_root,
    sign_file,
    sign_seg_file,
    data_base_dir,
):
    sign_class = load_sign_file(
        sign_file
    )

    with open(
        sign_seg_file,
        "r",
    ) as f:
        sign_segments = json.load(f)

    (
        subsets,
        left_hand_ids,
        _,
    ) = load_vertex_subsets(
        data_base_dir
    )

    full_errors = {
        f"TR {name}": []
        for name in subsets
    }

    for sign_id, class_id in (
        sign_class.items()
    ):
        gt_meshes = load_central_gt(
            gt_root,
            sign_id,
            sign_segments,
        )

        pred_meshes = (
            load_prediction_meshes(
                pred_root,
                sign_id,
            )
        )

        if len(pred_meshes) < len(gt_meshes):
            raise RuntimeError(
                f"{sign_id}: "
                f"{len(pred_meshes)=} "
                f"< {len(gt_meshes)=}"
            )

        for idx, gt_path in enumerate(
            gt_meshes
        ):
            pred_path = pred_meshes[idx]

            pred_v, pred_f = load_obj(
                pred_path
            )

            gt_v, gt_f = load_obj(
                gt_path
            )

            if not np.isfinite(
                pred_v
            ).all():
                raise FloatingPointError(
                    f"Non-finite mesh: "
                    f"{pred_path}"
                )

            if pred_v.shape[0] != (
                SMPLX_NUM_VERTS
            ):
                raise ValueError(
                    pred_v.shape
                )

            np.testing.assert_array_equal(
                pred_f,
                gt_f,
            )

            for name, ids in (
                subsets.items()
            ):
                eval_ids = ids

                # exact official class-0 behavior
                if (
                    name != "left hand"
                    and class_id == "0"
                ):
                    eval_ids = np.setdiff1d(
                        eval_ids,
                        left_hand_ids,
                    )

                # exact official skip
                if (
                    name == "left hand"
                    and class_id == "0"
                ):
                    continue

                err = transl_point_error(
                    pred_v[eval_ids],
                    gt_v[eval_ids],
                )

                full_errors[
                    f"TR {name}"
                ].append(err)

    metrics_mm = {}

    for name, errors in (
        full_errors.items()
    ):
        if not errors:
            continue

        all_errors = np.concatenate(
            errors,
            axis=0,
        )

        metrics_mm[name] = float(
            all_errors.mean()
            * 1000.0
        )

    return metrics_mm
```

This wrapper deliberately preserves the official metric mathematics and class handling while adding stricter preflight failures for:

- missing prediction meshes;
- NaNs;
- wrong topology.

Those checks prevent accidental benchmark inflation due to silent skipping.

---

# 72. Machine-readable metric output

Every final run should save:

```json
{
  "protocol": "SGNify_official_TRV2V",
  "alignment": "subset_centroid_translation_only",
  "unit": "mm",
  "TR left hand": 0.0,
  "TR right hand": 0.0,
  "TR above pelvis upper body": 0.0
}
```

The zeros above are placeholders only.

No result table should be generated from console text manually.

---

# 73. Preflight before official evaluation

Implement:

```bash
python eval/preflight.py \
  --pred /path/to/pred \
  --gt /path/to/gt \
  --sign-file /path/to/sign.txt \
  --sign-seg /path/to/sign_seg.json
```

Checks:

- every sign exists;
- central GT segment is non-empty;
- enough predicted meshes exist;
- prediction mesh ordering is deterministic;
- every prediction has 10475 vertices;
- no NaN/Inf;
- faces exactly equal GT;
- class-0 signs are identified correctly;
- all official vertex-subset files exist.

---

# 74. Metric lock with original DexAvatar

Before any experiment:

```bash
python eval/official_sgnify_trv2v.py \
  --pred /path/to/original_dexavatar \
  --gt /path/to/gt \
  --sign-file ... \
  --sign-seg ... \
  --data-base-dir ...
```

Compare against the output from the provided author evaluator:

```bash
python evaluate_new_fitting.py \
  --method dexavatar \
  --central true \
  --evaluate_folder /path/to/original_dexavatar \
  --gt_folder /path/to/gt \
  --sign_file ... \
  --sign_seg ...
```

Required:

\[
\boxed{
|E_{\text{wrapper}}
-
E_{\text{official-script}}|
<
10^{-6}
}
\]

up to printed precision / floating-point parsing.

Do this independently for:

- `TR left hand`
- `TR right hand`
- `TR above pelvis upper body`

Only when the wrapper is numerically identical do we use it in automated ablations.

---

# 75. How TR-V2V affects our optimization strategy

Because the official metric removes **only the centroid translation of each evaluated subset**, it remains sensitive to:

- finger articulation;
- hand orientation;
- hand shape deformation;
- relative geometry inside the hand;
- torso/arm configuration;
- rotation errors;
- scale errors.

It is insensitive to a pure global translation of the selected subset.

Therefore our proposed method should focus on geometry/orientation rather than trying to optimize absolute subset translation for the benchmark.

However:

> this does **not** mean translation can be ignored during reconstruction.

Correct translation and hand/body placement still matter for:

- image reprojection;
- hand-body contact;
- hand-hand contact;
- physical plausibility;
- downstream sign-avatar use.

The official metric simply removes subset-global translation at scoring time.

---

# 76. Why wrist/body coupling still matters under TR-V2V

A common misconception would be:

> “Hand TR-V2V subtracts the hand centroid, therefore wrist orientation no longer matters.”

This is false.

Suppose the entire hand rotates incorrectly around the wrist.

Centroid translation removal cannot undo this rotation:

\[
R\hat V-\mu(R\hat V)
\neq
V-\mu(V)
\]

when

\[
R\neq I.
\]

Therefore H4W++’s body-aware wrist orientation remains directly relevant to hand TR-V2V.

This supports the architectural decision:

\[
\boxed{
\text{H4W++ owns wrist orientation}
}
\]

while:

\[
\boxed{
\text{PAD primarily refines finger articulation}
}
\]

---

# 77. Why PAD-Hand can directly reduce official hand TR-V2V

For the hand subset, after centering, error primarily reflects:

- palm orientation/shape;
- relative finger positions;
- joint articulation;
- reconstruction noise.

PAD-Hand targets temporal hand-pose corruption.

Therefore, when visual evidence is weak:

\[
R^V_{t,j}
\]

may produce temporally inconsistent fingers, while:

\[
R^T_{t,j}
\]

provides a plausible temporal target.

If PAD reduces finger-angle error without over-smoothing semantic motion, it should reduce:

\[
TR\text{-}V2V_{hand}
\]

even though centroid translation is removed.

---

# 78. Primary table format for our experiments

All ablations should use one table:

| Method | TR Upper Body ↓ | TR Left Hand ↓ | TR Right Hand ↓ |
|---|---:|---:|---:|
| DexAvatar official reproduction | — | — | — |
| + Hand4Whole++ | — | — | — |
| + PAD-Hand fingers | — | — | — |
| + adaptive fusion | — | — | — |
| + sign-aware contact | — | — | — |

Unit:

```text
mm
```

Footnote:

> TR-V2V follows the official SGNify evaluation script: each anatomical vertex subset is independently centered by its own centroid before vertex-to-vertex Euclidean error; no rotational, scale, or Procrustes alignment is performed.

---

# 79. Per-frame diagnostic TR-V2V

The official script additionally constructs per-frame means for:

- right hand;
- left hand;
- above-pelvis upper body.

For debugging, preserve this capability.

For each frame \(f\):

\[
E_f
=
\frac1{|V|}
\sum_{i\in V}
e_{f,i}
\cdot 1000.
\]

Save:

```text
results/<exp>/per_frame_trv2v.csv
```

Columns:

```text
sign_id
frame_order
gt_obj
pred_obj
class_id
tr_upper_body_mm
tr_left_hand_mm
tr_right_hand_mm
visual_conf_left
visual_conf_right
pad_weight_left
pad_weight_right
```

For class-0 signs:

```text
tr_left_hand_mm = NaN
```

not zero.

This file is diagnostic only; final aggregate numbers must still use the official concatenate-then-mean behavior.

---

# 80. Error slicing for the proposed method

Because we specifically propose adaptive temporal hand refinement, the most informative analysis is:

### High visual confidence

Does PAD/adaptive fusion preserve already-correct frames?

### Low visual confidence

Does it reduce TR hand error?

### Large H4W++–PAD disagreement

Which source does the gate choose?

### Fast motion

Does PAD over-smooth?

### Hand overlap / occlusion

Does temporal prior help?

The slicing labels may use model-side signals, but the official primary TR-V2V remains unchanged.

---

# 81. Benchmark hygiene rules

For every reported number:

- use the same `MANO_SMPLX_vertex_ids.pkl`;
- use the same upper-body segmentation `.npy`;
- use the same `sign_file`;
- use the same `sign_seg`;
- use the same GT OBJ files;
- use SMPL-X 10475-vertex topology;
- use the official class-0 rule;
- use subset-centroid translation alignment only;
- multiply meters by 1000 exactly once;
- do not drop failed frames;
- do not use GT to fix frame correspondence after seeing errors.

If any one of these changes, the number must not be called official SGNify TR-V2V.

---

# 82. Final metric-specific acceptance gate

The project is eligible for a SOTA claim only when:

1. the original DexAvatar prediction files reproduce the author evaluator;
2. our wrapper reproduces the same numbers;
3. every method uses identical sign/frame lists;
4. every output mesh has valid SMPL-X topology;
5. no evaluated prediction is silently skipped;
6. final values are reported in mm;
7. the best model improves the target TR-V2V under this exact protocol.

The benchmark claim must be phrased in terms of:

\[
\boxed{
\text{official SGNify TR-V2V protocol}
}
\]

not an approximately equivalent custom metric.
