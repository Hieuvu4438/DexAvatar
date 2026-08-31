# Implementation Specification
## SGNify SOTA-Oriented 3D Sign Language Reconstruction — Reviewed v3
### Hand4Whole++ + Two-Hand PAD-Hand + DexAvatar with Uncertainty-Calibrated Visual–Physics–Sign Fusion

**Status:** reviewed, paper-oriented engineering specification / implementation blueprint  
**Primary benchmark:** SGNify  
**Primary output:** temporally coherent **SMPL-X** sequence  
**Primary goal:** reduce official SGNify TR-V2V while preserving sign-critical finger articulation, body-consistent wrist orientation, temporal plausibility, and strict benchmark reproducibility.

### Revision note for v3

This revision incorporates the post-review architectural corrections that are mandatory before full implementation:

1. **Hand4Whole++ remains the spatial/body-wrist backbone.** PAD-Hand is a temporal/physics finger expert, not a replacement for the SMPL-X arm-chain wrist.
2. **DexAvatar hand visual supervision and SignHPoser latent prior are disentangled.** The legacy DexAvatar hand-prior term combines these roles; our method must not count the visual hand observation twice.
3. **The core contribution is no longer a heuristic three-way softmax gate.** It is an **uncertainty-calibrated energy fusion** over heterogeneous sources.
4. **Visual and temporal/physics uncertainties are calibrated separately.** Their weights are not forced to sum with the SignHPoser energy because the loss terms are not commensurate by construction.
5. **PAD-Hand physics variance is preferred when it can be faithfully reproduced.** If the public execution path/checkpoint does not expose it cleanly, use an empirical diffusion/window ensemble variance; only then fall back to deterministic proxy uncertainty.
6. **DexAvatar temporal regularization is preserved for the body.** The paper formulates its temporal term on body pose. Any reduction of a hand/finger temporal loss is allowed only after source-code audit confirms that such an overlapping hand term actually exists.
7. **Reliability is stop-gradient/frozen inside each fitting stage.** The optimizer must not be able to reduce its own penalty by manipulating the reliability features that set its weights.
8. **A MANO↔SMPL-X semantic hand-bridge test is mandatory.** Valid SO(3) matrices alone do not prove that left/right conventions, local joint axes, pose means, or joint ordering are correct.
9. **A masked-placeholder leakage test is mandatory for PAD.** A Transformer padding mask is insufficient evidence that invalid placeholders cannot leak through geometry/diffusion conditioning branches.

## 0. Executive decision

The proposed main system is:

\[
\boxed{
\text{Hand4Whole++}
\rightarrow
\text{Two-Hand PAD-Hand temporal finger refinement}
\rightarrow
\text{DexAvatar sign-specific SMPL-X fitting}
}
\]

with one new core mechanism:

\[
\boxed{
\textbf{Uncertainty-Calibrated Visual–Physics–Sign Fusion (UC-VPSF)}
}
\]

and one optional second-stage extension:

\[
\boxed{
\text{Sign-Aware Hand–Hand / Hand–Body Contact Refinement}
}
\]

The upstream methods deliberately have different responsibilities:

| Component | Responsibility in our system | What we do **not** let it own |
|---|---|---|
| **Hand4Whole++** | Frame-level whole-body SMPL-X initialization, body–hand spatial coupling, body-consistent wrist orientation, detailed WiLoR finger evidence | Long-term temporal reasoning |
| **PAD-Hand** | Temporal/physics-aware refinement of the **15 finger joints** for each hand; temporal uncertainty/evidence when available | Final body pose; final body shape; hard overwrite of wrist/global hand orientation |
| **DexAvatar** | Sign-domain body/hand priors, 2D/3D evidence, biomechanics, collision, staged fitting, final SMPL-X optimization | Generic body/hand estimation from scratch |
| **UC-VPSF (ours)** | Disentangle visual, temporal/physics, and sign-manifold energies; calibrate source uncertainty; perform reliability-aware MAP-like optimization | Retraining/replacing all pretrained backbones unnecessarily |

The most important architectural constraint is:

> **Hand4Whole++ remains the owner of wrist/body-chain consistency. PAD-Hand primarily refines the 15 MANO/SMPL-X finger joints. PAD root/global orientation is never a hard replacement for the SMPL-X wrist chain.**

Conceptually:

```text
shoulder → elbow → forearm → wrist frame      = body-aware H4W++ / SMPL-X chain
                                   ↓
                         15 local finger joints = H4W++ visual + PAD temporal + sign prior
```

This is essential because Hand4Whole++ uses whole-body context to predict wrist orientation and aligns detailed hand geometry into that body-consistent frame. Replacing the wrist with an independent temporal MANO root would partially undo the spatial coupling that motivated using Hand4Whole++.

A second critical implementation constraint is:

> **The released PAD-Hand demo is not a ready-made two-hand sign-language inference pipeline.**

Our implementation therefore creates explicit left/right tracks from H4W++-processed hand evidence, supports missing observations, overlaps temporal windows, merges rotations on \(SO(3)\), and validates left-hand parity.

A third critical methodological constraint is:

> **Do not treat visual, temporal, and SignHPoser terms as three interchangeable scalar losses whose softmax weights sum to one.**

The visual and PAD terms can be expressed as local rotation residuals, while SignHPoser is a learned sign-manifold/latent energy with a different scale and semantics. UC-VPSF therefore calibrates source precision separately and keeps the sign-manifold coefficient explicit.

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

### Uncertainty finding and v3 policy

PAD-Hand's method includes a physics-consistency uncertainty formulation, but the audited public demo path should not be assumed to expose a stable ready-to-use variance tensor for our integration without additional verification.

Therefore v3 uses a strict uncertainty hierarchy:

**Tier A — native PAD physics variance (preferred)**

Use the paper/source implementation's per-joint/per-time physics variance only if we can reproduce it from the released checkpoint and verify:

- exact tensor meaning;
- exact joint/time indexing;
- left/right handling;
- numerical scale;
- deterministic provenance from the released implementation.

**Tier B — empirical temporal variance (production fallback)**

If Tier A cannot be reproduced reliably, estimate temporal uncertainty from multiple valid PAD predictions, for example:

- multiple diffusion samples;
- overlapping temporal windows;
- optionally both when computationally feasible.

For rotations \(\{R_k\}\), first compute a Karcher mean \(\bar R\), then estimate geodesic variance:

\[
\sigma^2_T
=
\frac{\sum_k w_k\, d_{SO(3)}^2(R_k,\bar R)}
{\sum_k w_k + \epsilon}.
\]

This gives an uncertainty quantity in the same local rotation space used by the PAD finger energy.

**Tier C — deterministic proxy uncertainty (minimum viable fallback)**

Only when neither Tier A nor Tier B is available, construct a proxy from observable quantities such as:

- PAD validity/missingness;
- visual confidence;
- visual–PAD disagreement;
- motion/acceleration outliers;
- window-edge support.

Importantly, large visual–PAD disagreement must **not** automatically imply that PAD is wrong. If visual evidence is already weak, the disagreement itself is ambiguous.

**Design implication**

Version 1 of the end-to-end implementation must work with Tier B or Tier C. Tier A is promoted to the preferred paper model only after a source-level audit proves it is correctly reproduced.

---

# 3. Final system architecture

## 3.1 High-level pipeline

```text
                              SIGN VIDEO
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │     Hand4Whole++       │
                     │ spatial body-hand HMR  │
                     └────────────────────────┘
                         │                │
              SMPL-X body│                │ H4W++-processed WiLoR evidence
              wrist chain│                │ L/R finger rotations + confidence
                         │                ▼
                         │      ┌────────────────────────┐
                         │      │ Two-Hand PAD Adapter   │
                         │      │ L/R temporal tracks    │
                         │      └────────────────────────┘
                         │              │
                         │              ├── PAD finger means
                         │              ├── validity/masks
                         │              └── temporal uncertainty
                         │
                         ├───────────────────────┐
                         │                       │
                         ▼                       ▼
              ┌───────────────────┐   ┌──────────────────────┐
              │ Visual uncertainty│   │ Temporal uncertainty │
              │ σ²_V / precision  │   │ σ²_T / precision     │
              └───────────────────┘   └──────────────────────┘
                         │                       │
                         └───────────┬───────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │ UC-VPSF energy builder   │
                         │ L_visual + L_temporal    │
                         │ + pure SignHPoser energy │
                         └──────────────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────────┐
                         │ DexAvatar fitting        │
                         │ SignBPoser               │
                         │ SignHPoser latent energy │
                         │ biomechanics/collision   │
                         │ 2D/3D observations       │
                         │ body temporal regularizer│
                         └──────────────────────────┘
                                     │
                                     ▼
                             FINAL SMPL-X SEQUENCE
                                     │
                                     ▼
                            OFFICIAL SGNify TR-V2V
```

## 3.2 Ownership of each parameter

| Parameter/group | Frame initializer | Temporal/uncertainty evidence | Final optimizer owner |
|---|---|---|---|
| global body/root | Hand4Whole++ | DexAvatar original body temporal state | DexAvatar |
| torso/body pose | Hand4Whole++ | DexAvatar body temporal term | DexAvatar + SignBPoser |
| shoulder/elbow/forearm | Hand4Whole++ | body temporal state | DexAvatar |
| **wrist global/body-chain orientation** | **Hand4Whole++** | PAD root only diagnostic / optional weak ablation | **DexAvatar / SMPL-X body chain** |
| 15 L-hand finger joints | H4W++/WiLoR | PAD-left mean + \(\sigma^2_{T,L}\) | DexAvatar + UC-VPSF + SignHPoser |
| 15 R-hand finger joints | H4W++/WiLoR | PAD-right mean + \(\sigma^2_{T,R}\) | DexAvatar + UC-VPSF + SignHPoser |
| body shape \(\beta\) | Hand4Whole++ | sequence-shared/regularized | DexAvatar |
| MANO \(\beta\) used by PAD | H4W++/WiLoR | robust sequence statistic | PAD conditioning only |
| camera | H4W++ → Dex-compatible conversion | optional body-only stabilization | DexAvatar |
| face | H4W++ or original DexAvatar path | original behavior | DexAvatar |

## 3.3 Three non-overlapping hand energies

The hand objective must expose three conceptually different roles:

```text
Visual observation energy   E_V  : "what does this frame show?"
Temporal/physics energy     E_T  : "what motion is plausible given nearby frames?"
Sign-manifold energy        E_S  : "is this handshape plausible for the sign domain?"
```

Do not hide all three inside one opaque legacy hand-prior scalar. This decomposition is required for correct ablation and to prevent double counting.

## 3.4 Reliability update policy

Reliability is computed from **frozen evidence** at a fitting stage boundary.

Default:

```text
Stage k input estimates
       ↓
compute σ²_V, σ²_T
       ↓
STOP-GRADIENT / detach
       ↓
optimize Stage k with fixed precisions
```

Optional later ablation:

```text
alternating re-estimation between stages
```

Never allow the current LBFGS/gradient step to directly manipulate a reprojection-derived confidence and thereby lower the weight of its own residual.

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
│   ├── uncertainty.yaml
│   ├── fusion.yaml
│   └── experiments/
│       ├── b0_dex_original.yaml
│       ├── b1_h4wpp_dex.yaml
│       ├── b2_h4wpp_pad_fixed.yaml
│       ├── b3_ucvpsf_proxy.yaml
│       ├── b4_ucvpsf_empirical.yaml
│       ├── b5_ucvpsf_native_padvar.yaml
│       └── b6_contact.yaml
├── third_party/
│   ├── Hand4Whole-plus-plus_RELEASE/
│   ├── PAD-Hand-CVPR-2026/
│   └── DexAvatar/
├── assets/
│   ├── README.md
│   ├── hand_semantic_maps/
│   └── contact_zones/
├── data/
│   ├── raw/
│   ├── sgnify/
│   └── cache/
│       ├── h4wpp/
│       ├── pad/
│       ├── uncertainty/
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
│   │   ├── uncertainty_native.py
│   │   ├── uncertainty_ensemble.py
│   │   └── validate_pad.py
│   ├── uncertainty/
│   │   ├── visual.py
│   │   ├── temporal.py
│   │   ├── calibrate.py
│   │   ├── precision.py
│   │   └── diagnostics.py
│   ├── dex_adapter/
│   │   ├── dataset.py
│   │   ├── init_from_h4wpp.py
│   │   ├── decompose_dex_losses.py
│   │   ├── ucvpsf_loss.py
│   │   ├── fit_frame.py
│   │   ├── fit_sequence.py
│   │   └── export_smplx.py
│   ├── bridge/
│   │   ├── mano_smplx.py
│   │   └── semantic_validation.py
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
│   ├── 05_build_uncertainty.sh
│   ├── 06_fit_ours.sh
│   ├── 07_eval_sgnify.sh
│   └── run_all.sh
├── eval/
│   ├── official_sgnify_trv2v.py
│   ├── metric_lock.py
│   ├── preflight.py
│   └── aggregate_official.py
└── tests/
    ├── test_rotations.py
    ├── test_left_right_identity.py
    ├── test_h4wpp_export.py
    ├── test_pad_windows.py
    ├── test_pad_missing_frames.py
    ├── test_pad_mask_leakage.py
    ├── test_overlap_so3_merge.py
    ├── test_wrist_preservation.py
    ├── test_mano_smplx_semantic_bridge.py
    ├── test_uncertainty_nonnegative.py
    ├── test_precision_clipping.py
    ├── test_dex_hand_prior_decomposition.py
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
    valid_pad: np.ndarray                # [T] bool

    visual_rotmat: np.ndarray            # [T,16,3,3]
    pad_rotmat: np.ndarray               # [T,16,3,3]
    pad_finger_rotmat: np.ndarray        # [T,15,3,3]

    conditioning_betas: np.ndarray       # [10] robust sequence beta
    pad_vertices: np.ndarray | None      # [T,778,3]

    # uncertainty provenance
    uncertainty_mode: str                # native | ensemble | proxy
    pad_var_native: np.ndarray | None    # [T,15] if source-faithful Tier A exists
    pad_var_ensemble: np.ndarray | None  # [T,15] geodesic variance
    pad_var_proxy: np.ndarray | None     # [T,15] or [T,1]
    temporal_var: np.ndarray             # [T,15] selected σ²_T

    # diagnostics
    geodesic_disagreement: np.ndarray    # [T,15]
    temporal_accel_score: np.ndarray     # [T,15] or [T]
    support_count: np.ndarray            # [T,15] number of valid predictions
```

The field `temporal_var` is the **selected production uncertainty** after applying the Tier A → Tier B → Tier C hierarchy.

## 6.3 `UncertaintyRecord`

```python
@dataclass
class UncertaintyRecord:
    side: Literal["left", "right"]
    frame_idx: int

    # visual uncertainty / precision
    visual_var: np.ndarray        # [15] σ²_V
    visual_precision: np.ndarray  # [15] κ_V

    # temporal uncertainty / precision
    temporal_var: np.ndarray        # [15] σ²_T
    temporal_precision: np.ndarray  # [15] κ_T

    # source validity
    visual_valid: np.ndarray      # [15] bool
    temporal_valid: np.ndarray    # [15] bool

    # frozen feature provenance
    feature_vector: np.ndarray
    feature_version: str
    calibration_version: str
```

Required invariant:

\[
\sigma^2 \ge 0,
\qquad
\kappa = \frac{1}{\sigma^2+\epsilon} > 0.
\]

Use bounded precision:

\[
\kappa
\leftarrow
\operatorname{clip}
(\kappa,\kappa_{min},\kappa_{max}).
\]

## 6.4 `FusionRecord`

Do **not** store a three-way simplex weight by default.

```python
@dataclass
class FusionRecord:
    side: str
    frame_idx: int

    visual_precision: np.ndarray     # [15]
    temporal_precision: np.ndarray   # [15]

    lambda_visual: float
    lambda_temporal: float
    lambda_sign: float

    # optional hand-level sign modulation, not a simplex weight
    sign_scale: float

    reliability_frozen: bool
```

Default v3 rule:

```text
visual_precision + temporal_precision + sign_scale != 1  # intentionally
```

The source terms have different semantics and are calibrated through their own precision/energy scales, not a shared softmax.

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

For every side/frame, export:

```text
PAD root/global rotation
PAD 15 finger rotations
PAD MANO mesh (optional/debug)
PAD validity/mask
PAD uncertainty provenance
PAD temporal variance σ²_T
PAD support count / window positions
```

Usage:

| PAD output | Usage |
|---|---|
| root/global hand rotation | diagnostic / optional weak wrist ablation only |
| **15 finger rotations** | **main temporal/physics pose evidence** |
| native physics variance | preferred Tier-A temporal uncertainty if source-faithful |
| diffusion/window geodesic variance | Tier-B temporal uncertainty |
| proxy variance | Tier-C fallback |
| MANO vertices | debugging / optional geometry consistency |
| temporal motion behavior | uncertainty diagnostics |

PAD does **not** directly own the final SMPL-X mesh.

## 13.1 Empirical rotation variance

When multiple predictions exist for one joint/frame:

1. collect \(\{R_k,w_k\}\);
2. compute the weighted Karcher mean \(\bar R\);
3. compute:

\[
\sigma^2_T
=
\frac{\sum_k w_k d_{SO(3)}^2(R_k,\bar R)}
{\sum_k w_k+\epsilon}.
\]

Use center/window quality weights only if they are fixed before evaluating the prediction itself.

## 13.2 Uncertainty is not correctness

A small \(\sigma_T^2\) means the temporal model is internally confident/consistent under the selected estimator. It does **not** guarantee semantic correctness for sign language.

That is why SignHPoser and visual evidence remain independent terms in the final fitting objective.

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
    # original-compatible fields
    "fn": ...,
    "img_path": ...,
    "cam_param": ...,
    "smplx_param": ...,
    "label": ...,
    "keypoints": ...,
    "img": ...,

    # spatial visual hand targets
    "h4w_lhand_rotmat": [15,3,3],
    "h4w_rhand_rotmat": [15,3,3],

    # temporal/physics targets
    "pad_lhand_rotmat": [15,3,3],
    "pad_rhand_rotmat": [15,3,3],

    # selected variances / precisions
    "visual_var_l": [15],
    "visual_var_r": [15],
    "temporal_var_l": [15],
    "temporal_var_r": [15],
    "visual_precision_l": [15],
    "visual_precision_r": [15],
    "temporal_precision_l": [15],
    "temporal_precision_r": [15],

    # masks
    "lhand_visual_valid": [15],
    "rhand_visual_valid": [15],
    "lhand_pad_valid": [15],
    "rhand_pad_valid": [15],

    # provenance
    "uncertainty_mode_l": ...,
    "uncertainty_mode_r": ...,
    "reliability_frozen": True,
}
```

The final `smplx_param` initializes:

- root/body/shape/camera from H4W++;
- finger pose from H4W++'s body-aware hand output;
- **no PAD wrist overwrite**.

## 15.3 DexAvatar hand-loss decomposition requirement

Before adding UC-VPSF, expose the legacy DexAvatar hand prior as separate terms.

Conceptually:

```text
legacy DexAvatar hand prior
    ├── visual hand supervision term
    └── SignHPoser latent regularization term
```

Our implementation must be able to disable/reweight these parts independently.

The required audit output should print, for each hand:

```text
loss_hand_visual_legacy
loss_signh_latent
loss_hand_biomech
loss_hand_2d
loss_hand_3d
```

Only after this decomposition is verified should the new visual H4W++ energy be enabled.

---

# 16. Preserve DexAvatar's sign priors — but disentangle them

Keep **SignBPoser** as the body sign-domain prior.

Keep **SignHPoser** as a sign-domain hand-manifold/latent prior.

However, do **not** reuse a composite DexAvatar hand-prior scalar as the `sign` source if that scalar also contains a visual hand target. Otherwise visual evidence would be counted twice:

```text
H4W++ visual target ──────────────┐
                                  ├── duplicated visual supervision  ✗
legacy hand-prior visual target ──┘
```

The v3 decomposition is:

```text
spatial visual observation = Hand4Whole++ / H4W++-processed WiLoR
physics-temporal observation = PAD-Hand
sign-domain hand manifold = pure SignHPoser latent energy
sign-domain body manifold = SignBPoser
fusion/calibration = Ours (UC-VPSF)
```

If the DexAvatar code cannot expose the pure SignHPoser latent term without a patch, add the smallest possible patch and keep it as a documented `.patch` file.

---

# 17. Core proposed contribution
## Uncertainty-Calibrated Visual–Physics–Sign Fusion (UC-VPSF)

The core contribution is **not** the use of three pretrained methods. It is the calibrated optimization that combines heterogeneous spatial, temporal/physics, and sign-manifold evidence without forcing them into one arbitrary shared loss scale.

For side \(s\), frame \(t\), finger joint \(j\):

- optimized local finger rotation: \(R_{t,s,j}\);
- H4W++ visual target: \(R^V_{t,s,j}\);
- PAD temporal target: \(R^T_{t,s,j}\);
- visual variance: \(\sigma^2_{V,t,s,j}\);
- temporal variance: \(\sigma^2_{T,t,s,j}\);
- SignHPoser latent/manifold energy: \(E^S_{t,s}\).

Define precisions:

\[
\kappa_V
=
\operatorname{clip}
\left(
\frac{1}{\sigma_V^2+\epsilon},
\kappa_V^{min},
\kappa_V^{max}
\right),
\]

\[
\kappa_T
=
\operatorname{clip}
\left(
\frac{1}{\sigma_T^2+\epsilon},
\kappa_T^{min},
\kappa_T^{max}
\right).
\]

The robust visual and temporal energies are:

\[
L_V
=
\sum_{t,s,j}
M^V_{t,s,j}
\,\tilde\kappa^V_{t,s,j}
\,\rho
\left(
 d_{SO(3)}^2
 (R_{t,s,j},R^V_{t,s,j})
\right),
\]

\[
L_T
=
\sum_{t,s,j}
M^T_{t,s,j}
\,\tilde\kappa^T_{t,s,j}
\,\rho
\left(
 d_{SO(3)}^2
 (R_{t,s,j},R^T_{t,s,j})
\right).
\]

`M` denotes validity masks.

The sign-manifold energy is kept separate:

\[
L_S
=
\sum_{t,s}
E^S_{t,s}.
\]

The core hand objective is:

\[
\boxed{
L_{UC-VPSF}
=
\lambda_V L_V
+
\lambda_T L_T
+
\lambda_S L_S
}
\]

Crucially:

\[
\boxed{
\kappa_V + \kappa_T + \lambda_S \neq 1
}
\]

and no such constraint is required.

## 17.1 Why this is more principled than a three-way softmax

A visual rotation residual, a temporal rotation residual, and a SignHPoser latent energy have different dimensions, scales, and semantics. A softmax over three heuristic scores does not make their resulting losses commensurate.

UC-VPSF instead separates two problems:

1. **within-source reliability** — handled by \(\kappa_V,\kappa_T\);
2. **between-energy scale** — handled by fixed/dev-calibrated \(\lambda_V,\lambda_T,\lambda_S\).

## 17.2 Optional normalized precision

To prevent one sequence from changing the overall source scale simply because all its variances are numerically small, optionally normalize precision over valid joints/frames:

\[
\tilde\kappa
=
\frac{\kappa}
{\operatorname{mean}_{valid}(\kappa)+\epsilon}.
\]

Then \(\lambda_V\) and \(\lambda_T\) retain interpretable experiment-level scale.

## 17.3 Probabilistic interpretation

Ignoring robustification constants, the local visual/temporal part can be interpreted as a MAP-like product of uncertain rotation observations:

\[
p(R|V,T,S)
\propto
p_V(R)\,p_T(R)\,p_S(R),
\]

where higher variance lowers observation precision.

This interpretation is useful for the paper, but the implementation must remain faithful to the actual calibrated energy being optimized; do not claim a fully probabilistic posterior if calibration is heuristic.

---

# 18. Visual uncertainty model

Visual uncertainty should be computed from **pre-optimization or stage-frozen evidence**, not from a quantity the current optimizer can freely manipulate.

For frame/side/joint:

\[
f^V_{t,s,j}
=
[
 c^{joint}_{2D},
 c_{bbox},
 a_{bbox},
 r^{joint}_{reproj},
 m_{visual},
 q_{crop},
 d_{V,T}
].
\]

Possible features:

### 18.1 Joint-level 2D confidence

Prefer joint-level confidence rather than only a hand average when available.

Finger \(j\) can receive low confidence even when the hand box is good.

### 18.2 Bounding-box confidence and normalized size

\[
a_{bbox}
=
\frac{w_{box}h_{box}}{WH}.
\]

Small/truncated crops generally receive larger uncertainty.

### 18.3 Frozen reprojection residual

Compute from the H4W++ initialization or from the **previous completed fitting stage**:

\[
r^{joint}_{reproj}
=
\|\Pi(J_j)-u_j\|_2.
\]

Then detach it before optimization:

```python
reproj_feature = reproj_feature.detach()
```

Do not recompute the feature after every optimizer step by default.

### 18.4 Visual–PAD disagreement

\[
d_{V,T,j}
=
d_{SO(3)}(R^V_j,R^T_j).
\]

This is a diagnostic feature, not a direct label of which source is wrong.

### 18.5 Deterministic visual variance baseline

A minimal monotonic model:

\[
\log \sigma_V^2
=
a_0
+a_1(1-c_{2D})
+a_2\hat r_{reproj}
+a_3(1-c_{bbox})
+a_4(1-m_{visual}).
\]

Then:

\[
\sigma_V^2=\operatorname{softplus}(\cdot)+\epsilon.
\]

Parameters are fixed from development data or deliberately hand-calibrated before final SGNify test.

---

# 19. Temporal/physics uncertainty model

Use the hierarchy defined in the PAD audit.

## 19.1 Tier A — native PAD variance

Only enable:

```yaml
uncertainty.temporal.mode: native_pad
```

when a dedicated validation script proves that the released model/checkpoint exposes the intended per-joint/per-time physics variance.

Required validation:

- shape = expected \([T,15]\) after removing root;
- nonnegative finite values;
- left/right parity;
- stable indexing across windows;
- values respond sensibly to controlled motion corruption.

## 19.2 Tier B — empirical diffusion/window variance

Recommended robust production fallback.

For \(K\) PAD samples/predictions:

\[
\bar R
=
\arg\min_R
\sum_k w_k d^2(R,R_k),
\]

\[
\sigma_T^2
=
\frac{\sum_k w_k d^2(\bar R,R_k)}{\sum_k w_k+\epsilon}.
\]

Sources of \(K\):

- overlapping windows;
- repeated stochastic diffusion samples;
- both.

## 19.3 Tier C — proxy temporal variance

If only one PAD prediction exists:

\[
f^T
=
[
 m_{pad},
 n_{support},
 edge_{window},
 d_{V,T},
 v_{motion},
 a_{motion},
 c_{2D}
].
\]

A safe rule is to make disagreement matter more when visual evidence is itself strong:

\[
\log \sigma_T^2
=
b_0
+b_1(1-m_{pad})
+b_2 q_{edge}
+b_3 c_{2D}\hat d_{V,T}
+b_4 q_{motion-outlier}.
\]

This avoids the incorrect rule "large disagreement → PAD unreliable" when the visual source is actually the failed source.

---

# 20. Precision calibration and clipping

## 20.1 Why clipping is mandatory

If \(\sigma^2\to0\), unconstrained precision can explode and effectively freeze the hand to one source.

Always use:

\[
\kappa
=
\operatorname{clip}
\left(
\frac{1}{\sigma^2+\epsilon},
\kappa_{min},
\kappa_{max}
\right).
\]

## 20.2 Calibration options

Recommended order:

1. deterministic monotonic calibration;
2. isotonic/temperature/linear calibration on held-out development data;
3. small learned uncertainty head only if the simpler calibration is insufficient.

The learned model should predict **variance/log-variance**, not arbitrary three-way source weights.

Example:

```python
class LogVarianceHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 15),
        )

    def forward(self, x):
        log_var = self.net(x)
        return F.softplus(log_var) + 1e-6
```

## 20.3 No test-GT calibration

Never calibrate variance/precision using final SGNify test 3D ground truth.

Allowed:

- train/dev partitions;
- separate sign videos with valid supervision;
- synthetic visual corruptions;
- self-consistency targets defined before final test.

---

# 21. Reliability freezing and training protocol

## 21.1 Stage-frozen default

For each fitting stage:

1. build pose initialization;
2. compute visual/temporal uncertainty;
3. convert to precision;
4. `detach()` all uncertainty/precision tensors;
5. optimize with fixed precisions for that stage.

This prevents a circular feedback loop.

## 21.2 Optional alternating update

After Stage 2 completes:

```text
Stage-2 result
   ↓
recompute visual reprojection features once
   ↓
recalibrate σ²_V
   ↓
detach
   ↓
Stage 3 temporal/sign optimization
```

This is allowed as an ablation.

Do not update reliability inside every LBFGS closure unless the method is explicitly reformulated and tested for stability.

## 21.3 Synthetic corruption calibration

Useful corruptions:

- hand occlusion;
- motion blur;
- crop truncation;
- keypoint dropout;
- one/two finger masking;
- dropped frames;
- short missing-hand spans.

The target behavior is not simply "PAD weight increases". The target is:

```text
visual corruption  → σ²_V increases
PAD instability    → σ²_T increases
both uncertain     → sign prior/other constraints become relatively more influential
```

---

# 22. Wrist policy

## 22.1 Finger joints

For \(j=1,\ldots,15\):

\[
L^j_{finger}
=
\lambda_V \tilde\kappa_V^j\rho(d^2(R_j,R_j^V))
+
\lambda_T \tilde\kappa_T^j\rho(d^2(R_j,R_j^T))
+
\lambda_S E^j_{sign}\;\text{(if exposed per-joint)}.
\]

If SignHPoser is only exposed as a hand-level latent energy, keep it hand-level rather than inventing a fake per-joint decomposition.

## 22.2 Wrist/root

Default:

```text
PAD wrist hard overwrite: OFF
PAD wrist soft prior:     OFF
H4W++ body wrist:         ON
DexAvatar body/2D fitting: ON
```

Optional ablation only:

```text
PAD root weak diagnostic prior
```

Do not reuse the finger precision \(\kappa_T\) as a wrist precision; they refer to different evidence.

---

# 23. DexAvatar temporal policy after review

Do **not** assume that DexAvatar already applies a generic temporal smoother to the fingers.

The paper formulates its temporal loss on body pose \(\theta_b\). Therefore the v3 default is:

```text
DexAvatar body temporal term: preserve
PAD temporal finger term: add
legacy hand/finger temporal term: change only if source-code audit confirms it exists
```

Before changing any temporal code, create an audit note:

```text
DEX_TEMP_AUDIT.md
- exact source file
- exact variable names
- body joints affected
- hand joints affected
- previous-frame state
- coefficient
```

Only if an explicit overlapping finger temporal term exists should it be reduced/disabled where PAD is valid.

---

# 24. Proposed full objective

Let the original DexAvatar objective be decomposed into:

\[
L_{Dex}
=
L_{body-obs}
+
L_{hand-obs}^{legacy}
+
L_{SignB}
+
L_{SignH-latent}
+
L_{shape}
+
L_{bio}
+
L_{angle}
+
L_{temp-body}
+
L_{collision}
+\cdots
\]

Our implementation replaces/deactivates only the **duplicated legacy hand visual observation part** when UC-VPSF visual evidence is active.

Define:

\[
L_{Dex}^{disentangled}
=
L_{Dex}
-
L_{hand-obs}^{legacy,duplicated}
\]

with source-code-verified bookkeeping rather than literal subtraction in code.

The core model is:

\[
\boxed{
L_{ours}
=
L_{Dex}^{disentangled}
+
\lambda_V L_V
+
\lambda_T L_T
+
\lambda_S L_S
+
\lambda_W L_{wrist}
+
\lambda_G L_{geom}^{optional}
+
\lambda_C L_{contact}^{optional}
}
\]

Default core-paper model:

```text
λ_C = 0
λ_G = 0 initially
PAD wrist loss = 0
```

## 24.1 Robust penalty

Use a robust penalty \(\rho\) for source observation residuals to prevent one bad target from dominating despite imperfect uncertainty calibration.

Recommended candidates:

- Geman–McClure, matching DexAvatar style where appropriate;
- Huber on geodesic angle;
- clipped squared geodesic loss.

Ablate only after the basic pipeline is stable.

## 24.2 Source-normalized reporting

Log both raw and weighted energies:

```text
raw_L_visual
weighted_L_visual
raw_L_temporal
weighted_L_temporal
L_sign_latent
L_sign_body
L_body_temporal
L_biomech
L_collision
```

This is mandatory for debugging whether improvement comes from calibration or simply from changing global loss scale.

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

Save SHA256 for all checkpoints.

## 28.2 Baseline lock

```bash
bash scripts/01_run_dexavatar_baseline.sh \
  --data /path/to/sgnify \
  --out results/b0
```

Then run the exact metric-lock procedure defined in the official-evaluation section of this document.

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

Mandatory before PAD:

```bash
pytest -q tests/test_wrist_preservation.py \
          tests/test_mano_smplx_semantic_bridge.py
```

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

Validate:

```bash
pytest -q tests/test_pad_windows.py \
          tests/test_pad_missing_frames.py \
          tests/test_pad_mask_leakage.py \
          tests/test_overlap_so3_merge.py
```

## 28.5 Build temporal uncertainty

Preferred empirical mode first:

```bash
conda run -n pad_hand \
python -m src.pad_adapter.uncertainty_ensemble \
  --pad-cache data/cache/pad/SEQ_ID \
  --mode window_or_diffusion_ensemble \
  --output data/cache/uncertainty/SEQ_ID/pad.npz
```

If native PAD variance has been source-verified:

```bash
conda run -n pad_hand \
python -m src.pad_adapter.uncertainty_native \
  --pad-cache data/cache/pad/SEQ_ID \
  --output data/cache/uncertainty/SEQ_ID/pad_native.npz
```

Do not silently switch uncertainty modes between experiments.

## 28.6 Sapiens observations

Reuse DexAvatar's original extraction path initially.

```bash
bash scripts/04_extract_sapiens.sh ...
```

## 28.7 Build visual uncertainty

```bash
conda run -n dexavatar \
python -m src.uncertainty.visual \
  --h4w-cache data/cache/h4wpp/SEQ_ID \
  --sapiens-cache data/cache/sapiens/SEQ_ID \
  --calibration configs/uncertainty.yaml \
  --output data/cache/uncertainty/SEQ_ID/visual.npz
```

The output precision tensors are frozen for the corresponding fitting stage.

## 28.8 DexAvatar hand-prior decomposition test

Before fitting ours:

```bash
conda run -n dexavatar \
python -m src.dex_adapter.decompose_dex_losses \
  --config configs/dexavatar.yaml \
  --sequence SEQ_ID
```

Required:

- legacy hand visual observation can be identified separately;
- SignHPoser latent energy can be logged separately;
- body temporal term's affected variables are documented.

## 28.9 Core fitting

```bash
conda run -n dexavatar \
python -m src.dex_adapter.fit_sequence \
  --config configs/experiments/b4_ucvpsf_empirical.yaml \
  --h4w-cache data/cache/h4wpp/SEQ_ID \
  --pad-cache data/cache/pad/SEQ_ID \
  --uncertainty-cache data/cache/uncertainty/SEQ_ID \
  --sapiens-cache data/cache/sapiens/SEQ_ID \
  --sign-meta /path/to/sign_metadata \
  --out results/b4_ucvpsf_empirical/SEQ_ID
```

## 28.10 Evaluation

```bash
python eval/official_sgnify_trv2v.py \
  --pred results/b4_ucvpsf_empirical \
  --gt /path/to/sgnify_gt \
  --sign-file /path/to/sign_file.txt \
  --sign-seg /path/to/sign_segments.json
```

Do not place final-test evaluation inside the optimization loop.

---

# 29. `run_all.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SEQ=$1
EXP=${2:-b4_ucvpsf_empirical}

python scripts/00_check_assets.py --config configs/sgnify.yaml

conda run -n h4wpp \
python -m src.h4wpp_adapter.infer_sequence \
  --input "$SGNIFY/$SEQ" \
  --output "data/cache/h4wpp/$SEQ"

conda run -n pad_hand \
python -m src.pad_adapter.two_hand_refiner \
  --h4w-cache "data/cache/h4wpp/$SEQ" \
  --output "data/cache/pad/$SEQ"

conda run -n pad_hand \
python -m src.pad_adapter.uncertainty_ensemble \
  --pad-cache "data/cache/pad/$SEQ" \
  --output "data/cache/uncertainty/$SEQ/pad.npz"

bash scripts/04_extract_sapiens.sh "$SEQ"

conda run -n dexavatar \
python -m src.uncertainty.visual \
  --h4w-cache "data/cache/h4wpp/$SEQ" \
  --sapiens-cache "data/cache/sapiens/$SEQ" \
  --output "data/cache/uncertainty/$SEQ/visual.npz"

conda run -n dexavatar \
python -m src.dex_adapter.fit_sequence \
  --config "configs/experiments/$EXP.yaml" \
  --h4w-cache "data/cache/h4wpp/$SEQ" \
  --pad-cache "data/cache/pad/$SEQ" \
  --uncertainty-cache "data/cache/uncertainty/$SEQ" \
  --out "results/$EXP/$SEQ"
```

Evaluation is a separate explicit command after all predictions are frozen.

---

# 30. Suggested config

```yaml
experiment:
  name: h4wpp_pad_dex_ucvpsf
  seed: 42

h4wpp:
  export_raw_wilor: true
  preserve_body_wrist: true

pad:
  seq_len: 16
  stride: 8
  diffusion_steps: 4
  finger_only: true
  use_root_as_hard_target: false
  missing_frame_mask: true
  overlap_merge: so3_karcher

  uncertainty:
    mode: ensemble     # native_pad | ensemble | proxy
    ensemble_source: overlap_windows
    num_diffusion_samples: 1

uncertainty:
  eps: 1.0e-6
  precision_min: 0.1
  precision_max: 10.0
  normalize_precision_mean: true
  freeze_within_stage: true

  visual:
    use_joint_kpt_conf: true
    use_bbox_score: true
    use_bbox_area: true
    use_frozen_reproj: true
    use_visual_pad_disagreement: true

  temporal:
    prefer_native_pad_variance: false
    use_ensemble_variance: true
    proxy_use_disagreement_conditioned_on_visual_conf: true
    use_motion_features: true

fusion:
  name: uc_vpsf
  lambda_visual: 1.0
  lambda_temporal: 1.0
  lambda_sign: 1.0
  robust_penalty: geman_mcclure
  shared_softmax_weights: false

  wrist:
    pad_hard_overwrite: false
    pad_soft_weight: 0.0
    h4w_consistency_weight: 0.0   # optional later ablation

dexavatar:
  keep_signbposer: true
  keep_signhposer_latent: true
  disable_duplicated_legacy_hand_visual_when_ucvpsf: true
  keep_biomechanics: true
  keep_collision: true

  temporal:
    keep_body_temporal: true
    hand_temporal_policy: audit_before_change

geometry_consistency:
  enabled: false

contact:
  enabled: false
```

All numeric loss coefficients above are placeholders until calibrated on a valid development protocol. They must not be selected from final SGNify test GT.

---

# 31. Ablation matrix

The experiment order must isolate both integration gain and methodological gain.

| ID | H4W++ | PAD fingers | Uncertainty | Visual/temporal fusion | Pure SignHPoser | Contact | Purpose |
|---|---:|---:|---|---|---:|---:|---|
| **B0** | ✗ | ✗ | ✗ | original DexAvatar | ✓ legacy | original | reproduce published baseline |
| **B1** | ✓ | ✗ | ✗ | H4W++ visual only | ✓ | original | effect of stronger body-hand initializer |
| **B2** | ✓ | ✓ | ✗ | fixed \(\lambda_V,\lambda_T\) | ✓ | original | effect of PAD temporal fingers |
| **B3** | ✓ | ✓ | proxy | UC-VPSF | ✓ | original | minimum viable proposed method |
| **B4** | ✓ | ✓ | empirical ensemble | UC-VPSF | ✓ | original | recommended core paper model |
| **B5** | ✓ | ✓ | native PAD variance | UC-VPSF | ✓ | original | preferred model if Tier-A reproduction is verified |
| **B6** | ✓ | ✓ | best | UC-VPSF | ✓ | ✓ | optional semantic-contact extension |

Required diagnostic ablations:

| Test | Variants |
|---|---|
| prior decomposition | legacy composite hand prior vs disentangled SignHPoser latent |
| weighting formulation | three-way heuristic softmax vs UC-VPSF precision fusion |
| uncertainty source | none vs proxy vs ensemble vs native PAD variance |
| precision clipping | unclipped diagnostic vs clipped production |
| precision normalization | off vs mean-normalized |
| reliability update | frozen vs stage-wise alternating |
| PAD usage | initialization only vs soft target vs init+target |
| wrist | PAD root 0 vs weak prior |
| PAD stride | 4 vs 8 vs 16 |
| missing frames | skip-window vs mask+interpolation |
| mask leakage | two placeholder values under identical mask |
| overlap merge | center select vs SO(3) Karcher mean |
| sign prior | with vs without SignHPoser latent |
| semantics bridge | raw convention vs validated adapter |

---

# 32. Evaluation protocol

## 32.1 Primary

Report the exact official SGNify metrics:

```text
TR above pelvis upper body ↓
TR left hand ↓
TR right hand ↓
```

using the locked evaluator described later in this document.

## 32.2 Secondary diagnostic metrics

Recommended:

- hand MPJPE/MPVPE if available under a valid common convention;
- wrist orientation error;
- fingertip/joint-chain diagnostic error;
- temporal acceleration error;
- frame-to-frame angular velocity/acceleration;
- uncertainty calibration plots;
- precision histograms;
- error vs predicted variance;
- penetration/contact metrics for optional modules.

## 32.3 Per-condition slices

Without changing the official aggregate metric, analyze:

- high visual confidence;
- low visual confidence;
- low/high PAD temporal variance;
- high H4W++–PAD disagreement;
- strong motion blur;
- fast articulation;
- one-hand visible;
- both hands visible;
- hand-hand overlap;
- hand-face/body proximity.

The purpose is to test the hypothesis:

```text
UC-VPSF should preserve already-good visual frames
while improving frames where one source becomes unreliable.
```

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

## 33.2 SO(3) validity

For every exported rotation:

\[
\|R^\top R-I\|_F<\epsilon,
\qquad
|\det(R)-1|<\epsilon.
\]

## 33.3 Left/right identity

Verify:

- left track always remains left;
- right track always remains right;
- missing detections never swap identities.

## 33.4 PAD missing-window test

A masked missing frame must not destroy the window or produce NaNs.

## 33.5 PAD masked-placeholder leakage test — mandatory

Construct two inputs that are identical on all valid frames and use the **same invalid-frame mask**, but assign very different numeric placeholders at the invalid frame:

```text
A: ... valid | identity-pose placeholder | valid ...
B: ... valid | extreme-pose placeholder  | valid ...
mask:        | invalid                  |
```

Run PAD on both.

Expected on unmasked output frames:

\[
d_{SO(3)}(R_A,R_B)<\epsilon_{leak}.
\]

If outputs change materially, the invalid placeholder leaks through a conditioning branch and masking must be fixed or the input representation redesigned.

## 33.6 Overlap merge test

- merged output remains on \(SO(3)\);
- identical predictions are unchanged;
- result lies geodesically between close predictions.

## 33.7 Wrist preservation test

Before DexAvatar optimization:

\[
d_{SO(3)}
(R^{wrist}_{H4W},
R^{wrist}_{H4W+PAD-fingers})
\approx0.
\]

## 33.8 MANO↔SMPL-X semantic hand-bridge test — mandatory

SO(3) validity is insufficient. We must validate semantics after handedness restoration and representation conversion.

Test procedure for each side:

1. take a real H4W++ frame with valid hand prediction;
2. export the H4W++-processed WiLoR local finger rotations;
3. run our adapter into the SMPL-X hand-pose representation;
4. reconstruct the SMPL-X hand with the same body/wrist context;
5. wrist-center the reference and adapter hand joints;
6. compare MCP→PIP→DIP→tip chain directions and fingertip positions;
7. render a mirrored diagnostic for left hand.

Fail on:

- mirrored fingers;
- wrong joint ordering;
- systematic 90°/180° local-axis errors;
- left/right pose-mean mismatch;
- finger chains bending in the wrong anatomical direction.

The tolerance should be established from H4W++'s own reference outputs, not guessed from SGNify test GT.

## 33.9 Uncertainty invariants

```python
assert np.isfinite(var).all()
assert (var >= 0).all()
assert np.isfinite(precision).all()
assert (precision >= kappa_min).all()
assert (precision <= kappa_max).all()
```

## 33.10 Dex hand-prior decomposition test

Verify that enabling our H4W++ visual energy while disabling the duplicated legacy visual hand term leaves SignHPoser latent energy active.

## 33.11 Camera projection test

Detect focal-length, principal-point, crop/full-image, and axis-convention errors.

## 33.12 Metric lock

Original DexAvatar predictions through our wrapper must reproduce the author evaluator within the required tolerance.

---

# 34. Debug visualizations

Every sequence should optionally render a 4-panel video:

```text
┌──────────────────┬──────────────────┐
│ RGB + 2D evidence│ H4W++ SMPL-X     │
├──────────────────┼──────────────────┤
│ PAD temporal hand│ Final UC-VPSF    │
│ mean + variance  │ SMPL-X           │
└──────────────────┴──────────────────┘
```

Overlay:

- hand boxes;
- 2D joint confidence;
- frozen reprojection residual;
- \(\sigma_V^2\) / \(\kappa_V\);
- \(\sigma_T^2\) / \(\kappa_T\);
- uncertainty mode (`native`, `ensemble`, `proxy`);
- H4W++–PAD angular disagreement;
- SignHPoser energy;
- contact state if enabled.

Do not visualize a fake `w_visual + w_pad + w_sign = 1` bar in the main method because the v3 model does not use such a simplex.

---

# 35. Uncertainty and failure diagnostics

Log per frame/side:

```json
{
  "frame": 37,
  "left": {
    "visual_valid": true,
    "pad_valid": true,
    "kpt_conf_mean": 0.32,
    "reproj_frozen_px": 14.2,
    "pad_disagreement_deg": 18.6,
    "visual_var_mean": 0.44,
    "temporal_var_mean": 0.18,
    "visual_precision_mean": 0.72,
    "temporal_precision_mean": 1.64,
    "uncertainty_mode": "ensemble",
    "signh_energy": 2.13
  }
}
```

Save diagnostics before and after fitting, but keep the stage-frozen precision values explicitly recorded so it is clear what actually governed optimization.

Required plots:

```text
predicted variance vs final diagnostic error
visual precision histogram
PAD precision histogram
TR-V2V change vs visual confidence
TR-V2V change vs PAD variance
TR-V2V change vs disagreement
```

---

# 36. Optimization strategy

Do not enable all losses from iteration zero.

### Stage 0 — preflight and decomposition

Before optimizing ours:

- metric lock passes;
- H4W++ projection/bridge tests pass;
- PAD mask/overlap tests pass;
- DexAvatar hand-prior decomposition is verified.

### Stage 1 — body/camera stabilization

Optimize:

- camera;
- global orientation;
- body pose;
- body shape.

Keep hands near H4W++ initialization.

Preserve original DexAvatar body temporal regularization.

### Stage 2 — visual + sign hand fitting

Compute and freeze \(\sigma_V^2,\kappa_V\).

Enable:

- 2D hand evidence;
- H4W++ visual finger energy \(L_V\);
- SignHPoser latent energy \(L_S\);
- hand biomechanics as in the audited DexAvatar schedule.

Do not enable PAD yet. This isolates the spatial/sign solution.

### Stage 3 — UC-VPSF temporal/physics fitting

Compute and freeze \(\sigma_T^2,\kappa_T\), then enable:

- PAD finger energy \(L_T\);
- \(L_V\) with frozen visual precision;
- \(L_S\);
- body temporal term;
- no PAD wrist overwrite.

Optional stage-boundary ablation: recompute visual uncertainty once from the completed Stage-2 solution and freeze again.

### Stage 4 — physical refinement

Enable/strengthen:

- biomechanics;
- interpenetration/collision;
- optional wrist consistency;
- optional hand geometry consistency;
- optional semantic contact.

This staging prevents a temporal prior from dragging a poorly initialized spatial solution into a local minimum.

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
params_stage4 = params_stage3
```

Uncertainty/precision tensors are **not** optimizer parameters in the default model:

```python
visual_precision = visual_precision.detach()
temporal_precision = temporal_precision.detach()
```

A learned variance head, if used, is pretrained/calibrated separately and frozen at test-time.

---

# 38. SignHPoser interaction with PAD after disentanglement

PAD and SignHPoser remain complementary:

```text
PAD-Hand   : generic temporal/physics evidence
SignHPoser : sign-domain handshape/manifold prior
H4W++      : frame-specific visual evidence
```

Typical cases:

### Clear fast articulation

```text
visual variance low
PAD may disagree because motion is unusual
→ visual observation remains strong
→ SignHPoser prevents implausible handshape
→ PAD precision may be lower if its own variance is high
```

### Blurred intermediate frame

```text
visual variance high
PAD variance low
→ temporal evidence dominates local finger correction
→ SignHPoser remains active
```

### Both visual and PAD uncertain

```text
visual variance high
PAD variance high
→ both observation energies weaken
→ sign manifold + biomechanics + neighboring/body evidence become relatively more influential
```

This behavior emerges from calibrated precisions and fixed source coefficients rather than a three-way softmax.

---

# 39. Why not simply use PAD output as initialization only?

This remains an important ablation.

### Option A — PAD initialization only

```text
PAD fingers → initialize hand → no PAD energy
```

### Option B — H4W++ initialization + PAD soft temporal energy

```text
H4W++ spatial init + UC-VPSF temporal energy
```

### Option C — PAD initialization + PAD temporal energy

```text
PAD init + PAD soft target
```

Default remains **Option B** because it preserves H4W++ body-hand spatial consistency and treats PAD as uncertain temporal evidence rather than absolute truth.

---

# 40. Why Hand4Whole++ is the main backbone

The sign-language task depends on the full chain:

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

Hand4Whole++ explicitly injects specialist hand information into a whole-body estimator and maintains body-aware wrist reasoning. This is more suitable than late-fusing an unrelated body estimator and hand estimator.

Our method therefore extends H4W++ in the missing dimensions:

```text
spatial body-hand coupling     ← H4W++
temporal/physics hand evidence ← PAD-Hand
sign-domain pose manifold      ← DexAvatar
uncertainty calibration        ← Ours
```

---

# 41. Source-code modifications summary

## 41.1 Hand4Whole++

Patch only the output interface needed for our cache.

Export:

- final SMPL-X initialization;
- processed L/R WiLoR local rotations;
- hand existence/bboxes/confidence;
- debug wrist/body-chain orientation;
- any 2D hand keypoints needed for visual uncertainty.

Do not alter:

- CHAM/HandControlNet behavior;
- H4W++ handedness restoration;
- body wrist ownership;
- internal rigid hand-body alignment.

## 41.2 PAD-Hand

Keep the official model/checkpoint intact.

Our wrapper changes only inference assumptions:

```text
one hand                    → explicit L/R tracks
right-priority selection    → fixed handedness identity
non-overlap windows         → overlap windows
missing-frame skip          → masked numeric placeholder
single demo output          → rotations + support + uncertainty provenance
```

Additional source audit:

- locate the paper's variance-related path if released;
- never label a tensor `physics variance` until verified;
- otherwise use the ensemble/proxy path and label it accordingly.

## 41.3 DexAvatar

Prefer a thin adapter plus minimal patch.

Required modifications:

```text
src/dex_adapter/dataset.py
src/dex_adapter/decompose_dex_losses.py
src/dex_adapter/ucvpsf_loss.py
src/dex_adapter/fit_sequence.py
```

The crucial patch is **loss disentanglement**:

```text
legacy hand visual target  -> separately addressable
SignHPoser latent energy   -> separately addressable
```

When UC-VPSF visual energy is active, disable only the duplicated legacy visual hand target, not the SignHPoser latent prior.

If direct changes to `third_party/DexAvatar` are required, maintain a small patch file and record the exact upstream SHA.

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
git SHA for all three third-party repos
local patch SHA/hash
checkpoint hashes
config YAML
random seed
CUDA version
PyTorch version
PyTorch3D version
GPU model
sequence IDs
evaluation protocol ID
uncertainty mode: native_pad | ensemble | proxy | none
visual calibration version/hash
temporal calibration version/hash
precision clipping bounds
precision normalization setting
loss-decomposition version
reliability freeze policy
```

Create:

```text
results/<exp>/run_manifest.json
```

A final table entry is invalid if its uncertainty/calibration provenance cannot be reconstructed from the manifest.

---

# 44. Caching policy

Caches are immutable by default.

Recommended namespaces:

```text
h4wpp_v1_<sha>/
pad_v1_stride8_<sha>/
pad_unc_ensemble_v1/
pad_unc_native_v1/        # only if verified
visual_unc_calib_v1/
sapiens_original/
fitting_ucvpsf_v1/
```

If code, checkpoint, calibration, uncertainty mode, or conversion logic changes, create a new cache version.

Never silently overwrite:

- H4W++ extraction;
- PAD rotations;
- temporal variance;
- visual variance;
- precision tensors.

Each cache manifest should contain its parent cache IDs so that the full lineage is recoverable.

---

# 45. Acceptance gates

## Gate A — baseline and evaluator

- original DexAvatar runs;
- official SGNify metric lock passes;
- no frame/topology/NaN silent skipping.

## Gate B — H4W++ spatial bridge

- all SGNify frames processed;
- camera projection passes;
- left/right identity stable;
- wrist preservation passes;
- MANO↔SMPL-X semantic bridge passes.

## Gate C — PAD temporal bridge

- both hands supported;
- missing frames supported;
- masked-placeholder leakage test passes;
- no NaNs;
- overlap merge valid on \(SO(3)\);
- left parity test passes or canonicalized fallback is implemented.

## Gate D — DexAvatar loss decomposition

- legacy visual hand term is separately identifiable;
- SignHPoser latent term remains active when visual legacy term is disabled;
- body temporal variables are documented;
- no unverified hand temporal term is removed.

## Gate E — uncertainty calibration

- variance finite/nonnegative;
- precision clipped and finite;
- uncertainty mode recorded;
- precision frozen during fitting stage;
- proxy/ensemble/native variants produce reproducible caches.

## Gate F — core method

- B1 H4W++→DexAvatar does not catastrophically regress baseline;
- B2 PAD fixed-weight behavior is measurable;
- UC-VPSF improves over the corresponding fixed-weight baseline or provides clear condition-specific gains without unacceptable high-confidence degradation.

## Gate G — final SOTA experiment

Only after A–F pass:

- freeze all hyperparameters;
- freeze uncertainty mode;
- run full SGNify test once for the final configuration;
- generate official aggregate table and diagnostics.

---

# 46. Expected failure modes and mitigation

## 46.1 PAD over-smooths sign-critical motion

**Symptom**

- fast finger transitions become averaged;
- handshape timing lags the video.

**Mitigation**

- uncertainty-calibrated temporal precision;
- visual evidence remains strong when clear;
- SignHPoser remains active;
- PAD remains finger-only;
- no unverified extra hand smoother.

## 46.2 Wrist/body discontinuity

**Mitigation**

- H4W++ owns body wrist;
- PAD root hard overwrite off;
- wrist preservation test;
- optional weak H4W wrist consistency only after core stability.

## 46.3 Left/right convention error

**Mitigation**

- use H4W++ post-handedness outputs;
- explicit side namespaces;
- parity test;
- semantic MANO↔SMPL-X bridge test.

## 46.4 PAD temporal seams

**Mitigation**

- overlapping windows;
- center weighting;
- Karcher mean;
- use overlap dispersion as uncertainty evidence.

## 46.5 Missing-frame leakage

**Symptom**

Changing only the numeric placeholder at a masked frame changes valid outputs.

**Mitigation**

- trace every conditioning branch;
- mask geometry/feature paths consistently;
- redesign placeholder injection if the model cannot guarantee masking.

## 46.6 Camera mismatch

**Mitigation**

- full-image camera conversion;
- projection tests before fitting;
- never confuse crop camera with full-image camera.

## 46.7 Visual double counting

**Symptom**

Adding H4W++ visual energy makes hands overly rigid to visual initialization and SignHPoser appears ineffective.

**Cause**

Legacy DexAvatar visual hand supervision remains active inside the composite hand prior.

**Mitigation**

- explicit loss decomposition;
- disable duplicated visual term only;
- retain SignHPoser latent energy.

## 46.8 Precision explosion

**Symptom**

A near-zero predicted variance freezes a joint to one source.

**Mitigation**

- \(\epsilon\) floor;
- \(\kappa_{max}\) clipping;
- optional mean precision normalization;
- diagnostics on precision histograms.

## 46.9 Reliability gaming

**Symptom**

Optimization reduces its own observation weight instead of improving the pose.

**Mitigation**

- stage-frozen detached reliability;
- only stage-wise alternating updates in controlled ablations.

---

# 47. Research-story boundary

Do **not** write the paper contribution as:

> "We combine Hand4Whole++, PAD-Hand, and DexAvatar."

The method should be described as:

> **A sign-aware 3D reconstruction framework that treats spatial visual evidence, physics-temporal hand evidence, and sign-domain pose priors as heterogeneous uncertain experts, and fuses them through calibrated rotation-space energies while preserving body–wrist kinematic ownership.**

A concise contribution statement:

1. **Hierarchical body–wrist–finger ownership** for integrating a body-aware whole-body estimator with a temporal hand model without breaking the SMPL-X kinematic chain.
2. **UC-VPSF**, an uncertainty-calibrated visual–physics–sign optimization that disentangles visual observations from sign-manifold priors and avoids heuristic shared-softmax weighting.
3. **A robust two-hand temporal bridge** for PAD-Hand with missing-frame masking, \(SO(3)\) overlap aggregation, uncertainty estimation, and left/right semantic validation.
4. **Strict official SGNify evaluation and ablation protocol** demonstrating where temporal/uncertainty reasoning helps.

The pretrained models provide evidence. The novelty is the **representation-safe, uncertainty-calibrated integration and optimization behavior for monocular sign-language reconstruction**.

---

# 48. Recommended experiment progression

## Experiment 0 — B0

Original DexAvatar + official metric lock.

## Experiment 1 — B1

H4W++ initialization + disentangled DexAvatar sign priors, no PAD.

Question:

> Does a stronger body-aware spatial initializer improve body/hand TR-V2V?

## Experiment 2 — B2

Add PAD fingers with a constant global coefficient, no uncertainty.

Question:

> Does temporal hand evidence help before calibration?

## Experiment 3 — B3

Use deterministic proxy variance.

Question:

> Does uncertainty calibration beat fixed weighting even without native PAD variance?

## Experiment 4 — B4

Use empirical diffusion/window geodesic variance.

Question:

> Does model-derived temporal dispersion improve source selection over heuristic proxies?

## Experiment 5 — B5

Use native PAD physics variance only if source-faithfully reproduced.

Question:

> Does the paper's physics uncertainty provide additional calibration gain?

## Experiment 6 — B6

Optional semantic contact refinement.

Question:

> Does contact improve contact-heavy signs without harming official TR-V2V elsewhere?

---

# 49. Core pseudocode

```python
def reconstruct_sign_sequence(frames, sign_meta, cfg):
    # --------------------------------------------------
    # 1. Spatial body-hand reconstruction
    # --------------------------------------------------
    h4w = h4wpp_infer_sequence(frames)
    validate_h4w_cache(h4w)
    validate_mano_smplx_bridge(h4w)

    # --------------------------------------------------
    # 2. Explicit two-hand temporal tracks
    # --------------------------------------------------
    tracks = {
        side: build_pad_track(h4w, side=side)
        for side in ["left", "right"]
    }

    # --------------------------------------------------
    # 3. PAD temporal refinement + uncertainty
    # --------------------------------------------------
    pad = {}
    for side in ["left", "right"]:
        pad[side] = pad_refine_track(
            tracks[side],
            side=side,
            seq_len=16,
            stride=cfg.pad.stride,
        )
        validate_pad_mask_leakage(pad[side])
        pad[side].temporal_var = build_temporal_variance(
            pad[side],
            mode=cfg.pad.uncertainty.mode,
        )

    # --------------------------------------------------
    # 4. Sapiens / 2D observations
    # --------------------------------------------------
    sapiens = load_or_extract_sapiens(frames)

    # --------------------------------------------------
    # 5. Visual uncertainty from frozen initial evidence
    # --------------------------------------------------
    visual_unc = {
        side: build_visual_uncertainty(
            h4w=h4w,
            sapiens=sapiens,
            side=side,
            calibration=cfg.uncertainty.visual,
        )
        for side in ["left", "right"]
    }

    # --------------------------------------------------
    # 6. Decompose DexAvatar hand prior once
    # --------------------------------------------------
    dex_model = load_dexavatar_disentangled(cfg.dexavatar)
    assert dex_model.signh_latent_is_separate

    # --------------------------------------------------
    # 7. Stage-wise fitting
    # --------------------------------------------------
    outputs = []
    temporal_state = None

    for t in range(len(frames)):
        init = build_smplx_init(h4w[t])

        evidence = {}
        for side in ["left", "right"]:
            evidence[side] = {
                "visual_R": h4w[t].finger_rotmat(side),
                "temporal_R": pad[side][t].finger_rotmat,
                "visual_precision": detach(
                    variance_to_precision(
                        visual_unc[side][t].visual_var,
                        cfg.uncertainty,
                    )
                ),
                "temporal_precision": detach(
                    variance_to_precision(
                        pad[side][t].temporal_var,
                        cfg.uncertainty,
                    )
                ),
                "visual_valid": h4w[t].finger_valid(side),
                "temporal_valid": pad[side][t].finger_valid,
            }

        result, temporal_state = dex_fit_frame_ucvpsf(
            frame=frames[t],
            init=init,
            evidence=evidence,
            sapiens=sapiens[t],
            sign_meta=sign_meta[t],
            temporal_state=temporal_state,
            preserve_body_wrist=True,
            keep_signh_latent=True,
            disable_duplicated_legacy_hand_visual=True,
            keep_body_temporal=True,
        )

        outputs.append(result)

    return outputs
```

---

# 50. `build_visual_uncertainty` pseudocode

```python
def build_visual_uncertainty(h4w, sapiens, side, calibration):
    records = []

    for t in range(len(h4w)):
        vis = side_view(h4w[t], side)

        joint_conf = get_joint_confidence(vis, sapiens[t])  # [15]
        bbox_conf = np.full(15, safe_bbox_score(vis))
        bbox_area = np.full(
            15,
            normalized_bbox_area(vis.bbox, h4w[t].image_hw),
        )

        # IMPORTANT: use initialization/stage-boundary pose and detach later
        reproj = frozen_joint_reprojection_error(
            h4w[t], sapiens[t], side
        )  # [15]

        valid = finger_validity(vis)  # [15]

        features = np.stack([
            joint_conf,
            bbox_conf,
            bbox_area,
            normalize(reproj),
            valid.astype(np.float32),
        ], axis=-1)

        visual_var = calibration.predict_variance(features)
        visual_var = np.maximum(visual_var, calibration.var_floor)

        records.append({
            "visual_var": visual_var,
            "features": features,
            "valid": valid,
        })

    return records
```

---

# 51. `ucvpsf_hand_loss` pseudocode

```python
def ucvpsf_hand_loss(
    current_R,          # [15,3,3]
    visual_R,           # [15,3,3]
    temporal_R,         # [15,3,3]
    visual_precision,   # [15]
    temporal_precision, # [15]
    visual_valid,       # [15]
    temporal_valid,     # [15]
    signh_latent_energy,
    cfg,
):
    d_vis = so3_geodesic(current_R, visual_R) ** 2
    d_tmp = so3_geodesic(current_R, temporal_R) ** 2

    d_vis = robustify(d_vis, cfg.fusion.robust_penalty)
    d_tmp = robustify(d_tmp, cfg.fusion.robust_penalty)

    L_visual = (
        visual_valid
        * visual_precision
        * d_vis
    ).sum()

    L_temporal = (
        temporal_valid
        * temporal_precision
        * d_tmp
    ).sum()

    return (
        cfg.fusion.lambda_visual * L_visual
        + cfg.fusion.lambda_temporal * L_temporal
        + cfg.fusion.lambda_sign * signh_latent_energy
    )
```

The **legacy DexAvatar visual hand supervision must not be active simultaneously** if it duplicates the H4W++ visual target.

---

# 52. Data leakage rules

Never use final SGNify test GT to:

- choose PAD stride;
- choose uncertainty mode;
- calibrate visual variance;
- set precision clipping;
- choose \(\lambda_V,\lambda_T,\lambda_S\);
- choose contact thresholds;
- decide which frames receive special handling;
- select native-vs-ensemble uncertainty after seeing final test scores.

Use:

- train/dev partitions;
- separate development sign videos;
- synthetic corruptions;
- fixed rules chosen before final test.

---

# 53. Implementation order checklist

- [ ] Pin official repository SHAs.
- [ ] Save checkpoint hashes and licenses.
- [ ] Reproduce original DexAvatar.
- [ ] Lock official SGNify TR-V2V.
- [ ] Audit DexAvatar Eq./code hand-prior decomposition.
- [ ] Audit DexAvatar temporal variables; document body vs hand scope.
- [ ] Run H4W++ official demo unchanged.
- [ ] Patch H4W++ output interface only.
- [ ] Build immutable H4W++ cache.
- [ ] Validate camera projection.
- [ ] Validate wrist preservation.
- [ ] Implement MANO↔SMPL-X semantic bridge test.
- [ ] Run PAD official demo unchanged.
- [ ] Build explicit L/R PAD tracks.
- [ ] Implement missing-frame numeric placeholders + masks.
- [ ] Implement PAD masked-placeholder leakage test.
- [ ] Implement overlapping windows.
- [ ] Implement SO(3) Karcher merge.
- [ ] Validate left-hand parity.
- [ ] Export decoded PAD rotations.
- [ ] Audit native PAD variance path.
- [ ] Implement empirical geodesic variance fallback.
- [ ] Implement proxy variance fallback.
- [ ] Implement variance→precision conversion and clipping.
- [ ] Build visual uncertainty from frozen evidence.
- [ ] Create DexAvatar-native parser for H4W++/PAD/uncertainty caches.
- [ ] Run B1 H4W++→DexAvatar without PAD.
- [ ] Run B2 constant PAD finger energy.
- [ ] Implement UC-VPSF with proxy uncertainty (B3).
- [ ] Implement UC-VPSF with empirical variance (B4).
- [ ] Enable native PAD variance only if verified (B5).
- [ ] Run required ablations.
- [ ] Add geometry/contact only after core stability.
- [ ] Freeze hyperparameters and uncertainty mode.
- [ ] Run final SGNify evaluation.
- [ ] Export per-frame diagnostics, calibration plots, and failure cases.

---

# 54. Source files to keep open while implementing

## Hand4Whole++

```text
main/model.py
common/nets/wilor.py
common/nets/module.py
```

Audit:

- exact output tensor names;
- handedness restoration;
- local/global rotations;
- hand existence masks;
- `combine_smplx_mano`;
- rigid alignment anchors;
- body-aware wrist output.

## PAD-Hand

```text
wilor_inference.py
demo.py
models/pad_hand.py
models/MANO.py
<variance / Laplace-related files if present in pinned release>
```

Audit:

- 16-pose layout;
- rotmat↔6D conversion;
- mask propagation through all branches;
- decoded rotation output;
- stochastic sampling interface;
- left/right MANO path;
- variance tensor provenance if available.

## DexAvatar

```text
Full_running_command.sh
run_dexavatar.py
dexavatar_fitting/smplifyx/data_parser.py
dexavatar_fitting/smplifyx/fit_single_frame.py
dexavatar_fitting/smplifyx/fitting.py
```

Audit:

- exact data dictionary;
- hand visual target term;
- SignHPoser latent term;
- SignBPoser term;
- temporal term variable scope;
- 2D/3D hand losses;
- biomechanics;
- collision;
- final SMPL-X export.

---

# 55. Final recommended core paper model

\[
\boxed{
\begin{aligned}
\mathcal V
&\xrightarrow{\text{Hand4Whole++}}
(\Theta^{H4W}_{body},H^V_L,H^V_R)
\\
(H^V_L,H^V_R)
&\xrightarrow{\text{two-hand PAD}}
(H^T_L,H^T_R,\sigma^2_{T,L},\sigma^2_{T,R})
\\
(H^V,\text{2D evidence})
&\xrightarrow{\text{visual calibration}}
\sigma^2_V
\\
(\sigma^2_V,\sigma^2_T)
&\xrightarrow{\text{precision calibration}}
(\kappa_V,\kappa_T)
\\
(\Theta^{H4W},H^V,H^T,\kappa_V,\kappa_T,E_{Sign})
&\xrightarrow{\text{DexAvatar + UC-VPSF}}
\Theta^{final}_{1:T}.
\end{aligned}
}
\]

Where:

- \(H^V\) = spatial visual finger evidence;
- \(H^T\) = temporal/physics finger evidence;
- \(\sigma_V^2,\sigma_T^2\) = source uncertainties;
- \(\kappa_V,\kappa_T\) = clipped/calibrated precisions;
- \(E_{Sign}\) = disentangled SignHPoser latent/manifold energy;
- final topology/parameterization = SMPL-X;
- wrist = H4W++/DexAvatar body chain;
- PAD = fingers only in the default model.

Recommended publishable core variant:

```text
H4W++ spatial init
+ two-hand PAD fingers
+ empirical temporal variance
+ visual uncertainty calibration
+ UC-VPSF
+ pure SignHPoser latent prior
+ DexAvatar SignBPoser/biomechanics/collision/body temporal
```

Native PAD physics variance replaces empirical variance only if faithfully reproduced from the released implementation.

---

# 56. Definition of success

The implementation is technically successful only if:

1. original DexAvatar baseline is reproduced;
2. evaluator parity is numerically locked;
3. H4W++ camera and wrist semantics are correct;
4. MANO↔SMPL-X semantic bridge passes for both hands;
5. PAD left/right tracks never swap;
6. missing frames do not destroy windows;
7. masked placeholders do not leak materially into valid predictions;
8. overlap merging stays on \(SO(3)\);
9. PAD never hard-overwrites body wrist;
10. DexAvatar visual hand supervision and SignHPoser latent energy are disentangled;
11. body temporal regularization is preserved unless source audit justifies a change;
12. uncertainty is finite, calibrated, clipped, and frozen inside fitting stages;
13. final output remains valid 10,475-vertex SMPL-X topology;
14. official SGNify TR-V2V runs on identical frame/sign lists;
15. ablations isolate H4W++, PAD, uncertainty source, and UC-VPSF;
16. no final-test GT is used for calibration/model selection.

Only then should the system be described as a SOTA-oriented method rather than an experimental integration.

---

# 57. Immediate coding target

Do **not** begin with the full uncertainty model.

Implement this verified chain first:

```text
SGNify frames
   ↓
Hand4Whole++
   ↓
export:
  SMPL-X body/wrist
  L/R processed WiLoR finger rotations
  hand validity/confidence
   ↓
MANO↔SMPL-X semantic bridge test
   ↓
two-hand PAD
  seq_len=16
  stride=8
  masks + overlapping windows
   ↓
mask-leakage test
   ↓
PAD L/R finger targets
   ↓
SO(3) overlap merge
   ↓
DexAvatar loss decomposition
   ↓
B1: H4W++ visual + pure SignHPoser
   ↓
B2: + constant PAD finger loss
   ↓
SGNify official evaluation
```

Only when B1/B2 are correct and measurable:

```text
add visual variance
add empirical PAD variance
add precision clipping
add UC-VPSF
→ B3/B4
```

This ordering prevents research novelty from being built on an unnoticed coordinate, handedness, temporal-mask, or loss-double-counting bug.

---

## Official sources used for the reviewed design

### DexAvatar

- Repository: https://github.com/kaustesseract/DexAvatar
- WACV 2026 paper: *DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors*.
- Review-critical point: the paper's hand prior includes both a visual hand supervisory term and SignHPoser latent regularization; its temporal term is formulated on body pose.

### Hand4Whole++

- Repository: https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE
- CVPR 2026 paper: *Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator*.
- Review-critical point: hand-specific features improve body-aware wrist orientation; detailed hand articulation is aligned into the whole-body mesh.

### PAD-Hand

- Repository: https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026
- CVPR 2026 paper: *PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery*.
- Review-critical point: the method estimates per-joint/per-time physics variance, but our use of any released variance tensor requires explicit checkpoint/source verification.

---

## Final implementation principle

> **Validate semantics first, calibrate uncertainty second, optimize fusion third.**

For this project, catastrophic failures are more likely to come from:

- left/right convention mismatch;
- crop/full-image camera mismatch;
- root/wrist ownership conflict;
- MANO/SMPL-X semantic mismatch;
- temporal mask leakage;
- double-counted visual supervision;
- unbounded precision;
- evaluator mismatch;

than from insufficient neural-network capacity.

The system is therefore deliberately designed as a sequence of **validated, immutable, versioned interfaces** before adding the final paper contribution.

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

Because the official metric removes **only the centroid translation of each evaluated anatomical subset**, it remains sensitive to:

- finger articulation;
- hand orientation;
- hand shape deformation;
- relative geometry inside the hand;
- torso/arm configuration;
- rotation errors;
- scale errors.

It is insensitive to a pure global translation of the selected subset.

Therefore UC-VPSF is aligned with the benchmark target because its core hand energies act on **local finger rotations and hand orientation/geometry**, not on an arbitrary global translation trick.

In particular:

- H4W++ improves the body-to-wrist kinematic frame;
- PAD can correct temporally ambiguous finger articulation;
- uncertainty prevents PAD from dominating clear but fast sign motion;
- SignHPoser constrains the sign-domain hand manifold when observations are weak.

However:

> **translation must still be reconstructed correctly even though subset-centroid translation is removed for TR-V2V.**

Correct translation remains necessary for:

- 2D reprojection;
- hand/body spatial consistency;
- collision/contact;
- downstream avatar use.

The metric simply means our benchmark gains should be explained primarily through improved **geometry and orientation**, not absolute subset position.

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

Use one consistent official table:

| Method | TR Upper Body ↓ | TR Left Hand ↓ | TR Right Hand ↓ |
|---|---:|---:|---:|
| B0 DexAvatar official reproduction | — | — | — |
| B1 + Hand4Whole++ spatial initialization | — | — | — |
| B2 + PAD-Hand fingers, fixed weight | — | — | — |
| B3 + UC-VPSF proxy uncertainty | — | — | — |
| B4 + UC-VPSF empirical uncertainty | — | — | — |
| B5 + UC-VPSF native PAD variance* | — | — | — |
| B6 + optional sign-aware contact | — | — | — |

`*` Report B5 only if the released native PAD variance path has been source-faithfully reproduced and validated.

Unit:

```text
mm
```

Footnote:

> TR-V2V follows the official SGNify evaluation script: each anatomical vertex subset is independently centered by its own centroid before vertex-to-vertex Euclidean error; no rotational, scale, or Procrustes alignment is performed.

For the paper, add a second ablation table for uncertainty/fusion design rather than overloading the primary benchmark table.

---

# 79. Per-frame diagnostic TR-V2V

Preserve per-frame means for:

- right hand;
- left hand;
- above-pelvis upper body.

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

Recommended columns:

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
visual_var_left
visual_var_right
visual_precision_left
visual_precision_right
temporal_var_left
temporal_var_right
temporal_precision_left
temporal_precision_right
uncertainty_mode_left
uncertainty_mode_right
h4w_pad_disagreement_left_deg
h4w_pad_disagreement_right_deg
signh_energy_left
signh_energy_right
```

For class-0 signs:

```text
tr_left_hand_mm = NaN
```

not zero.

This CSV is diagnostic only; the final aggregate metric must still use the official concatenate-then-mean behavior.

---

# 80. Error slicing for the proposed method

Because the core method is uncertainty-calibrated temporal hand refinement, report diagnostic slices that directly test its intended behavior.

### Low visual variance / high visual precision

Does UC-VPSF preserve frames that H4W++ already reconstructs well?

### High visual variance / low visual precision

Does temporal/sign evidence reduce hand TR-V2V?

### Low PAD temporal variance

Does trusting the temporal expert help when PAD is internally consistent?

### High PAD temporal variance

Does the method correctly avoid forcing an unstable temporal prediction?

### Large H4W++–PAD disagreement

Which source has lower predicted uncertainty, and does that correlate with the eventual diagnostic error?

### Fast motion

Does UC-VPSF avoid over-smoothing sign-critical articulation?

### Hand overlap / occlusion

Does temporal evidence improve recovery when visual evidence degrades?

The slicing labels may use model-side uncertainty/confidence signals, but the official primary TR-V2V computation remains unchanged.

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
