# DexFactor-4D — End-to-End Implementation Specification

**Mục tiêu:** tài liệu đủ chi tiết để một AI coding agent có thể triển khai DexFactor-4D trên codebase DexAvatar, kiểm thử từng module, xuất SMPL-X nhất quán và đánh giá bằng TR-V2V có audit trail.  
**Phiên bản đặc tả:** 1.0 — 25-08-2026  
**Companion research report:** `DexFactor4D_Method_Proposal_VI.md`  
**Evaluator đầu vào đã audit:** `evaluate_new_fitting(4).py`, SHA-256 `2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300`.

---

## 0. Kết quả cần tạo

Implementation hoàn chỉnh phải nhận một clip RGB và tạo:

1. một SMPL-X result cho **mọi frame đầu vào**;
2. PKL và OBJ được decode từ cùng một state;
3. cache observations có version/hash;
4. hand tracks trái/phải có missing mask;
5. log từng factor, stage, window và lỗi tối ưu;
6. report ablation;
7. kết quả TR-V2V gồm overall, per-frame, per-sign, denominator và hash protocol.

Full method là:

> **Sign-latent sequence fitting + identity-consistent multi-cue hands + reliability gating + SO(3) temporal factors + gated interaction/biomechanics + switchable PAD-Hand proposals.**

Không có neural network mới được train trong phiên bản 1.0. “Training” trong tài liệu này chỉ có nghĩa là **calibrate hyperparameter trên dev**, không update trọng số model.

---

## 1. Các invariant không được phá

AI triển khai phải coi các điều sau là assertions, không phải khuyến nghị:

- Không được drop frame vì HaMeR/WiLoR/SMPLer-X thất bại.
- Không dùng detection array index làm left/right identity.
- Không suy GT frame bằng vị trí list; mỗi prediction phải ghép với GT bằng manifest explicit.
- Không dùng SGNify GT để tune loss, threshold hoặc chọn ablation.
- Không gọi metric là official TR-V2V nếu chưa có exact official manifest và masks.
- Không Procrustes rotation hoặc scale alignment trong TR-V2V.
- Không silently skip NaN/Inf/missing output; official run phải fail.
- Không dùng chỉ depth `z` cho hand-3D factor.
- Không gọi disagreement score là calibrated uncertainty.
- Không cho PAD-Hand hoặc biomechanics ghi đè quan sát high-confidence.
- Không xuất PKL/OBJ từ hai pose state khác nhau.
- Không thay checkpoint/config sau khi đã nhìn SGNify test.

---

## 2. Dependency và repository plan

### 2.1 Runtime repositories

| Repo | Bắt buộc | Cách dùng |
|---|---:|---|
| [DexAvatar](https://github.com/kaustesseract/DexAvatar) | Có | Base pipeline, SignBPoser, SignHPoser, SMPLer-X integration |
| [Sapiens](https://github.com/facebookresearch/sapiens) | Có | Pose, part segmentation, relative depth observations |
| [HaMeR](https://github.com/geopavlakos/hamer) | Có | Hand detector/MANO/2D/3D observation thứ nhất |
| [WiLoR](https://github.com/rolpotamias/WiLoR) | Có | Hand detector/MANO/2D/3D observation thứ hai |
| [SMPL-X](https://github.com/vchoutas/smplx) | Có | Body model, topology, MANO↔SMPL-X assets |
| [PAD-Hand](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026) | Có cho full profile J | Frozen temporal hand proposal |
| PyTorch3D hoặc renderer tương đương | Có cho silhouette | Differentiable silhouette/depth rendering |

### 2.2 Reference-only repositories

Không import toàn bộ các repo sau vào runtime:

- Dyn-HaMR: chỉ port phần BMC/collision đã xác minh và giữ attribution/license;
- KNOWN-Hand: chỉ dùng formulation/limits sau khi kiểm tra convention;
- ACR, 4DHands, A2P: evidence và design reference, không là dependency;
- SGNify: classifier/evaluation assets nếu tác giả cung cấp, không dùng để tự suy official masks.

### 2.3 Asset có license riêng

Required assets:

- `SMPLX_NEUTRAL.npz` hoặc model SMPL-X tương ứng;
- MANO left/right models;
- `MANO_SMPLX_vertex_ids.pkl`;
- SignBPoser và SignHPoser checkpoints;
- official checkpoints của SMPLer-X, Sapiens, HaMeR, WiLoR;
- PAD-Hand checkpoint cho profile J;
- BMC convex-hull/ratio assets nếu bật BMC;
- exact TR-V2V masks và benchmark manifest.

Preflight phải kiểm license acknowledgement, file existence, SHA-256 và tensor/model version. Không commit model files vào Git nếu license cấm.

### 2.4 Không cần raw training data để inference

Không cần tải lại raw data đã train HaMeR/WiLoR/PAD-Hand. Để calibrate dev một cách hợp lệ, dùng:

- InterHand2.6M validation: mapping, geometry, bimanual interaction;
- Re:InterHand validation: temporal/occlusion;
- signer-disjoint SignAvatars/How2Sign subset: sign motion stress test;
- synthetic corruptions: missing detection, handedness swap, blur, occlusion, frame gaps.

SGNify chỉ là locked test.

---

## 3. Target repository layout

Giữ upstream DexAvatar càng nguyên vẹn càng tốt; thêm package mới:

```text
DexAvatar/
├── dexfactor4d/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── contracts.py
│   ├── preflight.py
│   ├── observations/
│   │   ├── schema.py
│   │   ├── smplerx_adapter.py
│   │   ├── sapiens_adapter.py
│   │   ├── hamer_adapter.py
│   │   ├── wilor_adapter.py
│   │   └── cache.py
│   ├── tracking/
│   │   ├── hand_tracks.py
│   │   └── activity.py
│   ├── geometry/
│   │   ├── joint_mapping.py
│   │   ├── rotations.py
│   │   ├── projection.py
│   │   └── hand_edges.py
│   ├── models/
│   │   ├── sign_priors.py
│   │   ├── smplx_state.py
│   │   └── pad_adapter.py
│   ├── factors/
│   │   ├── robust.py
│   │   ├── reprojection.py
│   │   ├── hand3d.py
│   │   ├── silhouette.py
│   │   ├── depth_order.py
│   │   ├── priors.py
│   │   ├── temporal_so3.py
│   │   ├── biomechanics.py
│   │   ├── collision.py
│   │   ├── contact.py
│   │   └── pad_proposal.py
│   ├── optimization/
│   │   ├── loss_graph.py
│   │   ├── stages.py
│   │   ├── windows.py
│   │   └── sequence_fitter.py
│   ├── export.py
│   └── diagnostics.py
├── evaluation/
│   ├── trv2v.py
│   ├── build_manifest.py
│   └── validate_protocol.py
├── configs/
│   ├── dexfactor4d_dev.yaml
│   └── dexfactor4d_frozen.yaml
└── tests/
    ├── test_joint_mapping.py
    ├── test_hand_tracks.py
    ├── test_rotations.py
    ├── test_factors.py
    ├── test_pad_adapter.py
    ├── test_export_consistency.py
    └── test_trv2v.py
```

Không tiếp tục nhồi logic vào `dexavatar_fitting/smplifyx/fitting.py` và `data_parser.py`; giữ một adapter mỏng để so sánh baseline.

---

## 4. Data contracts

Mọi array cần dtype/shape/frame/coordinate convention explicit. Dùng dataclass hoặc Pydantic; không truyền dictionaries không schema giữa modules.

### 4.1 Core identifiers

```python
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class FrameKey:
    clip_id: str
    frame_id: int
    timestamp_s: float

@dataclass(frozen=True)
class ModelVersion:
    name: str
    repo_commit: str
    checkpoint_sha256: str
    config_sha256: str
```

`frame_id` là ID từ input manifest, không phải vị trí sau filter.

### 4.2 Hand observation schema

```python
@dataclass
class HandDetection:
    source: str                    # "hamer" or "wilor"
    detection_id: int
    box_xyxy: torch.Tensor         # [4], pixels
    box_conf: float
    handedness_right_prob: float
    keypoints_2d: torch.Tensor     # [21, 2], full-image pixels
    keypoints_2d_conf: torch.Tensor# [21]
    keypoints_3d: torch.Tensor     # [21, 3], camera coordinates
    mano_rotmat: torch.Tensor      # [16, 3, 3], root + 15 joints
    mano_betas: torch.Tensor       # [10]
    camera_translation: torch.Tensor # [3]
    coordinate_system: str
    valid: bool
```

Requirements:

- Không mutate keypoint tensor in-place khi đổi crop→image.
- Tách raw output và canonical output.
- Với left hand, mirror/canonical transform phải có inverse và round-trip test.
- `valid=False` vẫn tạo record; không xóa frame.

### 4.3 Frame observations

```python
@dataclass
class FrameObservations:
    key: FrameKey
    image_size_wh: tuple[int, int]
    body_2d: torch.Tensor          # [Jb, 3] = x,y,confidence
    sapiens_hand_2d: torch.Tensor  # [2, 21, 3]
    segmentation_logits: torch.Tensor | None
    relative_depth: torch.Tensor | None
    hamer: list[HandDetection]
    wilor: list[HandDetection]
    smplerx_params: dict[str, torch.Tensor] | None
```

### 4.4 Semantic tracks

```python
@dataclass
class TrackFrame:
    key: FrameKey
    side: str                 # "left" | "right"
    hamer_detection_id: int | None
    wilor_detection_id: int | None
    observed: bool
    gap_length: int
    assignment_cost: float
    track_confidence: float
```

Hai track luôn dài đúng `T`; missing frame có `None`, không bị bỏ.

### 4.5 Optimization state

```python
@dataclass
class ClipState:
    betas: torch.nn.Parameter          # [1, 10], shared clip shape
    root_rotation_6d: torch.nn.Parameter # [T, 6]
    translation: torch.nn.Parameter    # [T, 3]
    body_latent: torch.nn.Parameter    # [T, 33]
    left_hand_latent: torch.nn.Parameter  # [T, 23]
    right_hand_latent: torch.nn.Parameter # [T, 23]
```

SignBPoser/SignHPoser decode latent thành axis-angle; chuyển sang rotation matrices trước temporal/BMC loss. Intrinsics `K` được freeze trong official profile.

### 4.6 Decoded state

```python
@dataclass
class DecodedState:
    vertices: torch.Tensor         # [T, 10475, 3]
    joints: torch.Tensor           # [T, J, 3]
    body_rotmat: torch.Tensor      # [T, 21, 3, 3]
    left_hand_rotmat: torch.Tensor # [T, 15, 3, 3]
    right_hand_rotmat: torch.Tensor# [T, 15, 3, 3]
    global_rotmat: torch.Tensor    # [T, 3, 3]
    translation: torch.Tensor      # [T, 3]
```

Mọi factor đọc cùng `DecodedState` trong một optimizer closure; không decode độc lập với convention khác nhau.

---

## 5. Configuration contract

Một frozen run phải dùng một YAML duy nhất và lưu bản copy cùng output:

```yaml
experiment:
  name: dexfactor4d_full_J
  seed: 20260825
  dtype: float32
  device: cuda
  strict: true

sequence:
  fps_from_timestamps: true
  window_size: 64
  overlap: 16
  missing_frames_are_observation_masks: true

shape:
  num_betas: 10
  pooling: huber_location
  optimize: true
  shared_per_clip: true

camera:
  optimize_intrinsics: false
  optimize_root: true
  optimize_translation: true

tracking:
  max_detections_per_source: 4
  states: [observed, missing, ambiguous]
  max_gap_for_track_prior: 12
  cost_terms: [handedness, bbox_iou, wrist_distance, pose_distance]

reliability:
  combine: geometric_mean
  alpha_confidence: 0.25
  alpha_cross_detector: 0.25
  alpha_track: 0.25
  alpha_silhouette: 0.25
  scales_from_dev_mad: true
  epsilon: 1.0e-6

residual_normalization:
  method: dev_median_absolute_deviation
  epsilon: 1.0e-6
  base_weight_after_normalization: 1.0

robustifier:
  type: geman_mcclure
  normalized_scale: 1.0

temporal:
  use_timestamps: true
  rotation_metric: so3_geodesic
  velocity: true
  acceleration: true
  gamma_min_ratio: 0.05
  gamma_max_ratio: 1.0

pad_hand:
  enabled: true
  sequence_length: 16
  overlap: 8
  diffusion_steps: 4
  use_released_uncertainty: false
  proposal_only: true
  switchable_constraint: true
  require_all_valid_inputs_v1: true

interaction:
  biomechanics: true
  collision: true
  contact_persistence: true
  require_evidence_for_contact: true

optimizer:
  type: lbfgs_line_search
  max_iterations_per_stage: 30
  learning_rate: 0.5
  ftol: 1.0e-9
  gtol: 1.0e-9

stage_anchor_ratio:
  stage_1: 1.0
  stage_2: 0.25
  stage_3: 0.05

evaluation:
  expected_unique_frames: 2872
  input_unit: m
  output_unit: mm
  strict_missing: true
  strict_finite: true
```

Các số reliability alpha bằng nhau và anchor ratios là **initial development profile**, không phải claim tối ưu. Dev search chỉ được chọn từ grid preregistered; frozen config phải thay bằng giá trị đã chọn và xóa mọi marker `TUNE`.

---

## 6. Phase 0 — tạo DexAvatar-CF

Không implement full method trước khi contract-fixed baseline pass.

### 6.1 Frame order

Trong data loading:

- parse numeric suffix đúng một lần;
- sort bằng `(numeric_id, filename)`;
- không overwrite bằng lexicographic sort;
- assert IDs unique;
- giữ record cho frame thiếu observation.

### 6.2 Left/right mapping

Xóa mọi magic index như `53:63`, `12:42` và detection `[0]/[1]`. Tạo named mapping:

```python
MANO_21_NAMES = (
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
```

Mỗi adapter trả joint theo thứ tự này. Mapping SMPL-X được tạo từ official joint mapper và test bằng joint names, không slice số bí ẩn.

### 6.3 No-drop policy

Nếu observation thiếu:

- `valid_mask=False` cho factor đó;
- state vẫn được khởi tạo từ nearest valid interpolation/Sign prior;
- output vẫn tồn tại;
- temporal factor dùng timestamp thực.

### 6.4 Export consistency

Chỉ có một hàm:

```python
def export_frame(decoded: DecodedState, t: int, pkl_path: Path, obj_path: Path) -> None:
    """Write parameters and vertices derived from the same decoded state."""
```

Test reload PKL → SMPL-X → vertices phải khớp OBJ trong tolerance `1e-6 m`.

### 6.5 Resume

Resume checkpoint phải chứa:

- full `ClipState`;
- optimizer stage/window index;
- overlap consensus buffer;
- previous track states;
- RNG states;
- config/checkpoint hashes.

Không resume dựa trên việc PNG visualization đã tồn tại.

---

## 7. Observation extraction

### 7.1 Cache format

Mỗi model ghi một file `.npz`/`.pt` hoặc Zarr có:

- ordered `FrameKey` list;
- raw outputs;
- canonical outputs;
- valid masks;
- coordinate convention;
- model/checkpoint/config hashes.

Cache loader từ chối dùng cache khi input frame hash hoặc model version đổi.

### 7.2 SMPLer-X

Giữ:

- shape betas per frame;
- global orientation;
- translation/camera;
- body/hand initialization;
- full SMPL-X params.

Clip shape initialization dùng robust Huber location/coordinate median trên frame có body confidence cao, không arithmetic mean vô điều kiện.

### 7.3 Sapiens

Chạy ba frozen heads:

- pose;
- part segmentation;
- human-relative depth.

Segmentation/depth giữ ở resolution gốc hoặc lưu transform chính xác về full image. Không dùng Sapiens depth như metric millimet; chỉ dùng relative ordering khi confidence đủ.

### 7.4 HaMeR và WiLoR

Giữ **mọi detection trước tracking**. Với mỗi detection:

1. clone output tensor;
2. chuyển crop keypoint về full-image pixel;
3. chuẩn hóa coordinate convention;
4. tính reprojection self-check;
5. gắn `valid=False` nếu transform fail.

Không chọn “right hand nếu có, nếu không lấy detection đầu” như PAD-Hand demo.

---

## 8. Global hand tracking

### 8.1 Frame assignment hypotheses

Sau NMS, tối đa bốn detections/source. Enumerate hypotheses cho hai semantic tracks:

- `(L=d_i, R=d_j)` với `i != j`;
- `(L=d_i, R=missing)`;
- `(L=missing, R=d_j)`;
- `(L=missing, R=missing)`.

Emission cost:

\[
C_{emit}(t,d,h)=
\lambda_{lr}C_{lr}
{}+\lambda_{box}C_{IoU}
{}+\lambda_w C_{wrist}
{}+\lambda_p C_{pose}.
\]

Trong đó distance được chia bbox/body scale. Transition cost penalize identity swap, implausible velocity và missing-gap start/end. Giải minimum path toàn clip bằng Viterbi; không greedy per-frame.

### 8.2 Cross-source pairing

Sau khi có track riêng từ mỗi detector, pair HaMeR–WiLoR cho cùng semantic side bằng bbox IoU + wrist distance. Không average pose; giữ hai observation residual độc lập.

### 8.3 Missing behavior

Track confidence ở missing frame:

\[
c_{track}(t,h)=c_{last}\exp(-g/\tau_g),
\]

với `g` là gap length. Giá trị chỉ dùng làm temporal gate, không tạo observation giả.

### 8.4 Tracking acceptance tests

- hoán vị detection order không đổi track;
- mirror clip hoán đổi L/R đúng;
- một detection mất 5 frame không làm đổi identity;
- first frame missing không truy cập state `None`;
- output track length luôn bằng input length;
- ambiguity phải xuất log, không silently resolve bằng index.

---

## 9. Reliability computation

Với side `h`, frame `t`:

\[
q_t^h = m_{coord}
\left(c_{det}+\epsilon\right)^{\alpha_c}
\left(c_{sil}+\epsilon\right)^{\alpha_s}
\exp\left[-\alpha_d d_{HW}/\sigma_d-\alpha_v e_{track}/\sigma_v\right].
\]

- `m_coord`: 0/1 từ coordinate/reprojection unit check;
- `c_det`: detector/keypoint confidence;
- `c_sil`: soft silhouette consistency;
- `d_HW`: normalized HaMeR–WiLoR disagreement;
- `e_track`: normalized innovation;
- scales `sigma` lấy từ MAD trên dev;
- clamp về `[0,1]`.

Nếu chỉ một detector tồn tại, bỏ `d_HW` khỏi geometric mean và renormalize các alpha còn lại. Không gán disagreement bằng 0 vì điều đó giả vờ hai model đồng thuận.

Log `q` cùng từng thành phần để debug.

---

## 10. State initialization

### 10.1 Shape

Tính robust clip shape từ SMPLer-X, sau đó optimize một `betas[1,10]` dùng chung. Add weak anchor tới robust initialization.

### 10.2 Body latent

Encode SMPLer-X body pose bằng SignBPoser encoder nếu encoder khả dụng; nếu không, giải latent inverse bằng 20–50 bước optimization chỉ trên reconstruction loss. Không mặc định zero latent nếu có initialization tốt.

### 10.3 Hand latent

Tương tự, initialize SignHPoser latent từ track-selected HaMeR/WiLoR fused initialization. Chọn pose source có reprojection residual nhỏ hơn, không lấy array index.

### 10.4 Missing initialization

- gap giữa hai valid frames: SLERP rotations, linear translation/latent interpolation;
- prefix/suffix missing: nearest valid state + strong prior;
- cả clip không thấy một hand: initialize prior mean, mark observation absent và không claim accuracy cho hand đó.

### 10.5 Camera/root

Freeze intrinsics. Root và translation warm-start từ SMPLer-X, optimize với weak anchor.

---

## 11. Factor implementation

Mỗi factor có interface thống nhất:

```python
class Factor(torch.nn.Module):
    def forward(
        self,
        decoded: DecodedState,
        observations: ObservationBatch,
        masks: FactorMasks,
        context: FactorContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return scalar normalized loss and detached diagnostics."""
```

Mỗi factor phải:

- trả zero scalar đúng device/dtype khi mask rỗng;
- không NaN ở zero variance;
- có gradient tới đúng state;
- log raw residual, normalized residual, active count và weighted loss;
- không normalize theo batch size hai lần.

### 11.1 Robust residual và scale normalization

Với residual vector `r`, dev scale `s_k`:

\[
\tilde r = r/(s_k+\epsilon),\qquad
\rho(\tilde r)=\frac{\tilde r^2}{\tilde r^2+1}.
\]

`s_k` là median absolute deviation của baseline residual trên dev và được lưu trong frozen config. Không tính lại từ SGNify.

Reference utilities:

```python
def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = mask.to(dtype=x.dtype)
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    denom = w.expand_as(x).sum().clamp_min(eps)
    return (x * w).sum() / denom

def geman_mcclure(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    z2 = (x / scale).square()
    return z2 / (z2 + 1.0)
```

### 11.2 Body/hand 2D reprojection

Input:

- Sapiens body joints;
- Sapiens/HaMeR/WiLoR hand joints;
- full-image pixel coordinates;
- keypoint/detector/reliability weights.

Loss:

\[
L_{2D}=\frac{1}{N_{valid}}\sum_{t,j,s}
m_{tjs}w_{tjs}\rho\left(
\frac{\pi_K(J_{tj})-u_{tjs}}{d_{box}+\epsilon}
\right).
\]

`s` là observation source. HaMeR và WiLoR tạo hai residual riêng; không average keypoint trước loss. Normalize pixel error bằng bbox diagonal hoặc body scale.

Tests:

- exact projection → zero;
- camera translation gradient khác zero;
- masked joint không có gradient;
- resize/crop round-trip dưới 0,5 pixel.

### 11.3 Full-XYZ hand geometry

Đây là replacement cho z-only branch.

MANO 21-joint graph:

```python
HAND_EDGES = (
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
)
```

Canonical geometry:

```python
def normalize_hand(joints: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    root_relative = joints - joints[..., :1, :]
    metacarpal = torch.stack([
        torch.linalg.norm(root_relative[..., i, :], dim=-1)
        for i in (5, 9, 13, 17)
    ], dim=-1)
    scale = metacarpal.median(dim=-1).values.clamp_min(eps)
    return root_relative / scale[..., None, None]

def unit_bones(joints: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bones = torch.stack([joints[..., j, :] - joints[..., i, :] for i, j in HAND_EDGES], dim=-2)
    return bones / torch.linalg.norm(bones, dim=-1, keepdim=True).clamp_min(eps)
```

Loss:

\[
L_{3D} = \sum_{s\in\{H,W\}} q_s
\left[\rho(\bar J-\bar Y_s)+\eta\rho(\hat B-\hat B_s)\right].
\]

Chỉ bật nếu 3D→2D self-projection của source pass tolerance. Toàn bộ `x,y,z` được dùng.

### 11.4 Silhouette

Implementation v1:

1. lấy SMPL-X hand vertex IDs;
2. tạo hand-only faces bằng các triangle có đủ ba vertex thuộc hand;
3. render left/right hand riêng với camera hiện tại;
4. crop ROI theo union track boxes + margin;
5. tạo target foreground từ Sapiens part mask trong ROI;
6. tính soft IoU + BCE/Dice;
7. tắt factor khi ROI/mask confidence thấp hoặc hand nhỏ hơn min pixels dev-fixed.

Không yêu cầu Sapiens phân biệt left/right ở pixel level; track bbox tách ROI. Khi hai ROI overlap mạnh, silhouette factor tổng hai tay và depth-order factor xử lý front/back.

### 11.5 Relative depth order

Không fit absolute Sapiens depth. Với cặp region overlap `(A,B)`:

1. lấy robust median Sapiens depth ở confident pixels của A/B;
2. chỉ tạo order label khi chênh lệch vượt dev-fixed uncertainty threshold;
3. render depth `z_A,z_B` từ current state;
4. dùng logistic ranking:

\[
L_{ord}=\operatorname{softplus}\left[-s_{AB}(z_B-z_A)/\tau_z\right].
\]

Nếu Sapiens depth và two-detector 3D order bất đồng mạnh, giảm reliability hoặc tắt factor; không chọn một cue tùy ý.

### 11.6 Sign latent priors và initialization anchors

```python
L_prior = body_latent.square().mean() \
        + left_latent.square().mean() \
        + right_latent.square().mean()
```

Initialization anchor so decoded rotations bằng SO(3) geodesic, không L2 axis-angle. Anchor stage ratios `1.0 → 0.25 → 0.05`; absolute normalized weight được chọn dev.

### 11.7 SO(3) temporal velocity/acceleration

Utilities:

```python
def so3_angle(relative_rotation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)

def relative_rotation(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
    return prev.transpose(-1, -2) @ curr
```

Với timestamps thật:

\[
\omega_{t,j}=\log(R_{t-1,j}^{T}R_{t,j})^\vee/\Delta t_t,
\]

\[
\alpha_{t,j}=(\omega_{t,j}-\omega_{t-1,j})/ar{\Delta t}_t.
\]

Gate:

\[
\gamma_{t,j}=\gamma_{min}+(\gamma_{max}-\gamma_{min})(1-q_{t,j}).
\]

Fast motion có multi-cue agreement cao nhận temporal weight gần `gamma_min`; missing/ambiguous observation nhận weight gần `gamma_max`.

Không sử dụng Gaussian smoothing làm final output.

### 11.8 Biomechanics

Implementation order:

1. axis-angle/rotmat → exact Euler convention mà BMC asset yêu cầu;
2. validate left/right mirroring;
3. bone-length consistency;
4. palm/finger ratio constraints;
5. convex-hull joint-angle feasibility;
6. reliability gate `(1-q)`;
7. capped normalized weight.

Nguồn tham khảo runtime có thể port từ:

- `Dyn-HaMR/dyn-hamr/optim/bio_loss.py`;
- BMC assets theo hướng dẫn `Dyn-HaMR/README.md`/Hand-BMC;
- KNOWN-Hand `losses.py` cho functional coupling.

Không copy code `.cuda()` hard-coded. Mọi tensor theo `decoded.device`. BMC là zero trong feasible set và không kéo pose về mean.

Nếu asset/convention unit tests chưa pass, profile full phải báo `BMC_DISABLED_UNVERIFIED`; không dùng một angle limit tự nghĩ ra.

### 11.9 Collision

Reuse một differentiable BVH/signed-distance implementation đã kiểm tra. Phạm vi:

- left-hand ↔ right-hand;
- hand ↔ forearm/torso;
- không tính self-collision toàn body nếu quá tốn.

Loss phải gần zero cho separated meshes, positive cho penetration, và có gradient đẩy ra ngoài. Capped weight để không hy sinh image alignment khi SGNify GT có interpenetration.

### 11.10 Contact persistence

Contact candidate chỉ tồn tại nếu trong ít nhất `k` frame liên tiếp:

- projected fingertip/palm distance nhỏ;
- silhouette/depth cho overlap/proximity;
- 3D detector hoặc current mesh distance cùng ủng hộ.

Loss giữ relative distance/transform ổn định trong candidate interval. Không có absolute attraction nếu chưa có candidate. `k` và thresholds chọn dev.

### 11.11 PAD-Hand proposal adapter

Không dùng `PAD-Hand/wilor_inference.py` vì release chọn một right hand nếu có. Adapter nhận semantic track đã giải quyết.

Pipeline per side:

1. lấy MANO root + 15 joint rotmats từ track source;
2. với left hand, mirror về right canonical frame bằng reflection `M=diag(-1,1,1)` và `R'=MRM`;
3. validate mirror round-trip;
4. chạy PAD-Hand sequence length 16, diffusion steps 4;
5. sliding window overlap 8;
6. v1 chỉ chạy window đủ valid input; core fitting vẫn xử lý mọi frame;
7. merge overlapping PAD proposals bằng SO(3) mean;
8. inverse mirror left output;
9. convert PAD pose thành soft proposal, không replace state.

Switchable factor:

\[
L_{PAD}=\sum_{t,h}(1-q_t^h)s_t^h\rho(d_{SO(3)}(R_t^h,P_t^h))
{}+\lambda_s\Phi(s_t^h).
\]

`s∈[0,1]` được parameterize bằng sigmoid. Prior `Phi` ưu tiên `s=1` nhẹ; observation disagreement có thể đẩy `s→0`. Không sử dụng “variance” vì public demo không expose đầy đủ uncertainty path của paper.

---

## 12. Loss graph và stage schedule

### 12.1 Objective

Sau residual normalization:

\[
L = L_{2D}^{body}+L_{2D}^{hand}+L_{3D}^{hand}
{}+L_{sil}+L_{ord}+L_{prior}+L_{init}+L_{root}
{}+L_{temp}+L_{bmc}+L_{coll}+L_{contact}+L_{PAD}.
\]

### 12.2 Stage 1 — root/body

Trainable:

- root rotation;
- translation;
- body latent;
- shared shape nếu dev cho phép.

Active factors:

- body 2D;
- body prior;
- root/translation/shape anchors;
- weak body temporal.

Hands frozen.

### 12.3 Stage 2 — hand observations

Mở two-hand latents. Active thêm:

- hand 2D;
- full-XYZ hand geometry;
- silhouette;
- depth order;
- hand priors.

Initialization anchors giảm còn 0,25 relative.

### 12.4 Stage 3 — sequence/interaction

Active thêm:

- SO(3) velocity/acceleration;
- BMC;
- collision;
- evidence-gated contact;
- switchable PAD proposals.

Anchor ratio 0,05.

### 12.5 Optimizer closure

LBFGS closure phải:

1. zero gradients;
2. decode state một lần;
3. tính factors và finite-check từng scalar;
4. backward tổng loss;
5. log stage/iteration/raw/normalized factors;
6. fail với diagnostic snapshot nếu NaN/Inf.

Không `continue` khi loss NaN.

---

## 13. Windowing và consensus

### 13.1 Windows

- size 64;
- overlap 16;
- last window được right-align để bao phủ frame cuối;
- every frame coverage ≥1;
- timestamp retained.

### 13.2 Warm start

Window sau lấy overlap state từ window trước. Non-overlap initialization từ original state/interpolation, không copy frame cuối cho cả đoạn.

### 13.3 Consensus

Trong overlap:

- Euclidean states (`latent`, `translation`, `betas`) dùng triangular confidence-weighted mean;
- root rotations dùng sign-aligned quaternion hoặc SO(3) Karcher mean;
- decoded hand rotations không average trực tiếp rồi lưu; average latent proposals hoặc re-optimize overlap consistency;
- final overlap polish bật observation + temporal factors.

### 13.4 Coverage assertion

Trước export:

```python
assert len(outputs) == len(input_manifest)
assert set(outputs) == set(input_manifest.frame_keys)
assert all(torch.isfinite(v).all() for v in decoded.vertices)
```

---

## 14. End-to-end CLI contract

Các command dưới đây là API mục tiêu để AI implement:

```bash
# 1. Assets, versions, frame order, coordinate conventions
python -m dexfactor4d.preflight \
  --config configs/dexfactor4d_frozen.yaml \
  --input-manifest data/clip_manifest.json

# 2. Frozen observations
python -m dexfactor4d.cli extract-observations \
  --config configs/dexfactor4d_frozen.yaml \
  --manifest data/clip_manifest.json \
  --output runs/exp/observations

# 3. Global semantic hand tracks
python -m dexfactor4d.cli build-tracks \
  --config configs/dexfactor4d_frozen.yaml \
  --observations runs/exp/observations \
  --output runs/exp/tracks.json

# 4. PAD proposals; optional for ablations A-I
python -m dexfactor4d.cli build-pad-proposals \
  --config configs/dexfactor4d_frozen.yaml \
  --observations runs/exp/observations \
  --tracks runs/exp/tracks.json \
  --output runs/exp/pad_proposals.pt

# 5. Sequence fitting
python -m dexfactor4d.cli fit \
  --config configs/dexfactor4d_frozen.yaml \
  --manifest data/clip_manifest.json \
  --observations runs/exp/observations \
  --tracks runs/exp/tracks.json \
  --pad-proposals runs/exp/pad_proposals.pt \
  --output runs/exp/fitting

# 6. Export from a single decoded state
python -m dexfactor4d.cli export \
  --run runs/exp/fitting \
  --output runs/exp/export

# 7. Validate every output before evaluation
python -m dexfactor4d.cli validate-export \
  --manifest data/clip_manifest.json \
  --export runs/exp/export
```

Mỗi command ghi `run_metadata.json`: Git commit, dirty state, command, config hash, checkpoint hashes, start/end time, host/GPU/software versions và status.

---

## 15. Audit `evaluate_new_fitting(4).py`

### 15.1 Hành vi evaluator đính kèm

File thực hiện:

- đọc OBJ prediction và GT;
- yêu cầu faces arrays giống nhau;
- với mỗi region, trừ centroid của **chính region đó** ở prediction và GT;
- tính Euclidean per-vertex error;
- concatenate errors và lấy mean ×1000;
- dùng masks `left_hand`, `right_hand`, `upper_body`, `upper_body_minus_head`, `upper_body_minus_face`;
- với sign class string `"0"`, bỏ left-hand vertices khỏi mọi region khác `left hand`, rồi bỏ luôn metric `TR left hand`;
- dùng segment `[start*2, end*2]` để chọn GT;
- ghép prediction và GT bằng cùng `inter_idx` trong hai lists.

Primary behavior quan trọng cần tái hiện trong compatibility profile:

\[
e_{t,r,i}=\left\|(P_{t,r,i}-\bar P_{t,r})-(G_{t,r,i}-\bar G_{t,r})\right\|_2.
\]

Đây là **per-region centroid translation alignment**.

### 15.2 Findings và cách sửa

| Finding | Loại | Cách xử lý trong reference evaluator |
|---|---|---|
| Paths/masks directory hard-coded | Reproducibility | CLI + manifest + hashes |
| `central` argument không ảnh hưởng control flow | Bug/API | Xóa flag; manifest quyết định frame |
| `start/end * 2` là implicit temporal mapping | Protocol risk | Manifest ghi explicit GT/pred path từng frame |
| Pairing bằng list position | Critical | Pair bằng one-record-per-frame manifest |
| Prediction NaN bị `continue` | Critical bias | Strict fail; không drop denominator |
| Missing/extra prediction không kiểm | Critical | Exact frame coverage assertion |
| Class-0 left-hand exclusion ẩn trong loop | Protocol ambiguity | Encode `regions` và `upper_exclude` trong manifest |
| `point_error_common_center` thêm cùng GT wrist cho cả hai | Redundant | Bỏ wrist metric khỏi primary; nếu cần wrist-align phải dùng source/target wrist riêng |
| Empty left-hand mean được in thành `0` | Misleading | Region không active → `null`/omitted, không zero |
| `args.method` là list nhưng được gán vào mapping values | Code bug | Một evaluation run = một explicit prediction root |
| `class_sign`, `left_hand_ids` là globals | Maintainability | Tất cả protocol data truyền explicit |
| Face equality phụ thuộc order/winding | Over-strict | So canonical triangle sets; vẫn yêu cầu vertex correspondence |
| Assume units meters rồi ×1000 | Protocol risk | Manifest declares input unit; output always mm |
| Không lưu per-frame/per-sign artifacts | Audit gap | JSON + CSV outputs |
| Không hash manifest/masks | Audit gap | SHA-256 in result |
| Không kiểm expected 2.872 unique frames | Benchmark gap | `expected_frame_count` assertion |

Lưu ý: class-0 exclusion và factor `×2` có thể là chủ ý của data layout. Vì file không tự chứng minh provenance/officialness, reference implementation gọi cấu hình này là `source_evaluator_compat_v1` cho đến khi tác giả xác nhận official contract.

---

## 16. TR-V2V manifest contract

Không để evaluator tự suy frame rate hoặc class policy.

```json
{
  "protocol": {
    "name": "source_evaluator_compat_v1",
    "alignment": "per_region_centroid_translation_only",
    "input_unit": "m",
    "output_unit": "mm",
    "expected_frame_count": 2872,
    "mask_file_sha256": "REPLACE_WITH_REAL_SHA256"
  },
  "frames": [
    {
      "sign_id": "SIGN_NAME",
      "frame_id": "000123",
      "gt_mesh": "SIGN_NAME/gt/000246.obj",
      "pred_mesh": "SIGN_NAME/pred/000123.obj",
      "regions": ["upper_body_minus_face", "right_hand"],
      "upper_exclude": ["left_hand"]
    },
    {
      "sign_id": "TWO_HAND_SIGN",
      "frame_id": "000010",
      "gt_mesh": "TWO_HAND_SIGN/gt/000020.obj",
      "pred_mesh": "TWO_HAND_SIGN/pred/000010.obj",
      "regions": ["upper_body_minus_face", "left_hand", "right_hand"],
      "upper_exclude": []
    }
  ]
}
```

Rules:

- `frames` có đúng 2.872 unique `(sign_id, frame_id)` cho official-compatible run;
- paths là relative, không `..`, được resolve dưới `--gt-root`/`--pred-root`;
- `regions` explicit nên denominator có thể audit;
- `upper_exclude` chỉ áp dụng cho upper-body mask;
- factor 30↔60/120 fps đã được giải quyết khi build manifest, không còn trong evaluator;
- mask `.npz` có keys `upper_body_minus_face`, `left_hand`, `right_hand` và int64 vertex IDs.

Manifest builder phải ưu tiên timestamp/author mapping. Chỉ dùng `gt_id=2*rgb_id` nếu dataset documentation hoặc author-provided mapping xác nhận; ghi rule vào metadata.

---

## 17. TR-V2V reference implementation

File mục tiêu: `evaluation/trv2v.py`. Code dưới đây là complete reference implementation cho manifest contract ở trên.

```python
# BEGIN TRV2V_REFERENCE
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REGIONS = (
    "upper_body_minus_face",
    "left_hand",
    "right_hand",
)


@dataclass(frozen=True)
class ManifestEntry:
    sign_id: str
    frame_id: str
    gt_mesh: str
    pred_mesh: str
    regions: tuple[str, ...]
    upper_exclude: tuple[str, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_join(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while preventing escape from its declared root."""
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes root: {relative_path}") from exc
    return candidate


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and triangular faces from an SMPL-X OBJ."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line.startswith("v "):
                tokens = line.split()
                if len(tokens) < 4:
                    raise ValueError(f"Malformed vertex at {path}:{line_number}")
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif line.startswith("f "):
                face_tokens = line.split()[1:]
                if len(face_tokens) != 3:
                    raise ValueError(
                        f"Expected triangular SMPL-X face at {path}:{line_number}"
                    )
                face: list[int] = []
                for token in face_tokens:
                    vertex_id = int(token.split("/")[0])
                    if vertex_id <= 0:
                        raise ValueError(
                            f"Negative/zero OBJ indices are not supported: {path}:{line_number}"
                        )
                    face.append(vertex_id - 1)
                faces.append(face)

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)

    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError(f"Invalid vertex array in {path}: {vertex_array.shape}")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError(f"Invalid face array in {path}: {face_array.shape}")
    if vertex_array.shape[0] == 0 or face_array.shape[0] == 0:
        raise ValueError(f"Empty mesh: {path}")
    if face_array.min() < 0 or face_array.max() >= vertex_array.shape[0]:
        raise ValueError(f"Face index outside vertex range: {path}")
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"NaN/Inf vertices: {path}")

    return vertex_array, face_array


def canonical_faces(faces: np.ndarray) -> np.ndarray:
    """Ignore face row order and triangle winding while retaining vertex labels."""
    canonical = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return canonical[order]


def load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REGIONS) - set(archive.files)
        if missing:
            raise KeyError(f"Mask archive lacks keys: {sorted(missing)}")
        masks = {name: np.asarray(archive[name], dtype=np.int64) for name in archive.files}

    for name, indices in masks.items():
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"Mask {name} must be a non-empty 1D array")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Mask {name} contains duplicate vertex IDs")
    return masks


def parse_manifest(path: Path) -> tuple[dict[str, Any], list[ManifestEntry]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "protocol" not in raw or "frames" not in raw:
        raise ValueError("Manifest requires top-level protocol and frames")

    protocol = raw["protocol"]
    required_protocol = {
        "name",
        "alignment",
        "input_unit",
        "output_unit",
        "expected_frame_count",
        "mask_file_sha256",
    }
    missing_protocol = required_protocol - set(protocol)
    if missing_protocol:
        raise KeyError(f"Protocol lacks: {sorted(missing_protocol)}")
    if protocol["alignment"] != "per_region_centroid_translation_only":
        raise ValueError("This implementation only supports per-region translation alignment")
    if protocol["output_unit"] != "mm":
        raise ValueError("TR-V2V output_unit must be mm")

    entries: list[ManifestEntry] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw["frames"]):
        try:
            entry = ManifestEntry(
                sign_id=str(item["sign_id"]),
                frame_id=str(item["frame_id"]),
                gt_mesh=str(item["gt_mesh"]),
                pred_mesh=str(item["pred_mesh"]),
                regions=tuple(str(value) for value in item["regions"]),
                upper_exclude=tuple(str(value) for value in item.get("upper_exclude", [])),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed frame record at index {index}") from exc

        key = (entry.sign_id, entry.frame_id)
        if key in seen:
            raise ValueError(f"Duplicate manifest frame: {key}")
        seen.add(key)
        if not entry.regions:
            raise ValueError(f"Frame has no active region: {key}")
        unknown_regions = set(entry.regions) - set(REGIONS)
        if unknown_regions:
            raise ValueError(f"Unknown regions at {key}: {sorted(unknown_regions)}")
        entries.append(entry)

    expected = int(protocol["expected_frame_count"])
    if len(entries) != expected:
        raise ValueError(f"Manifest has {len(entries)} frames; expected {expected}")
    return protocol, entries


def unit_to_mm_scale(unit: str) -> float:
    scales = {"m": 1000.0, "mm": 1.0}
    if unit not in scales:
        raise ValueError(f"Unsupported input unit: {unit}")
    return scales[unit]


def validate_masks_for_mesh(masks: dict[str, np.ndarray], vertex_count: int) -> None:
    for name, indices in masks.items():
        if indices.min() < 0 or indices.max() >= vertex_count:
            raise ValueError(
                f"Mask {name} is outside vertex range [0, {vertex_count})"
            )


def region_indices(
    region: str,
    upper_exclude: tuple[str, ...],
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    indices = masks[region]
    if region == "upper_body_minus_face" and upper_exclude:
        excluded_arrays: list[np.ndarray] = []
        for mask_name in upper_exclude:
            if mask_name not in masks:
                raise KeyError(f"Unknown upper_exclude mask: {mask_name}")
            excluded_arrays.append(masks[mask_name])
        excluded = np.unique(np.concatenate(excluded_arrays))
        indices = np.setdiff1d(indices, excluded, assume_unique=False)
    if indices.size == 0:
        raise ValueError(f"Region {region} became empty after exclusions")
    return indices


def translation_aligned_errors_mm(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    indices: np.ndarray,
    input_to_mm: float,
) -> np.ndarray:
    """Per-vertex error after independently centering the selected region."""
    pred_region = prediction[indices]
    gt_region = ground_truth[indices]
    pred_centered = pred_region - pred_region.mean(axis=0, keepdims=True)
    gt_centered = gt_region - gt_region.mean(axis=0, keepdims=True)
    errors = np.linalg.norm(pred_centered - gt_centered, axis=1) * input_to_mm
    if not np.isfinite(errors).all():
        raise ValueError("TR-V2V produced NaN/Inf")
    return errors


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    manifest_path: Path,
    mask_path: Path,
    gt_root: Path,
    pred_root: Path,
    output_dir: Path,
    check_topology: bool = True,
) -> dict[str, Any]:
    protocol, entries = parse_manifest(manifest_path)
    masks = load_masks(mask_path)

    actual_mask_hash = sha256_file(mask_path)
    expected_mask_hash = str(protocol["mask_file_sha256"])
    if actual_mask_hash != expected_mask_hash:
        raise ValueError(
            f"Mask SHA-256 mismatch: expected {expected_mask_hash}, got {actual_mask_hash}"
        )

    input_to_mm = unit_to_mm_scale(str(protocol["input_unit"]))
    expected_vertex_count = protocol.get("expected_vertex_count")

    totals = {
        region: {"sum_mm": 0.0, "vertex_count": 0, "frame_count": 0}
        for region in REGIONS
    }
    frame_means: dict[str, list[float]] = {region: [] for region in REGIONS}
    sign_totals: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {"sum_mm": 0.0, "vertex_count": 0, "frame_count": 0}
    )
    per_frame_rows: list[dict[str, Any]] = []
    masks_validated_for: set[int] = set()

    for entry in entries:
        gt_path = safe_join(gt_root, entry.gt_mesh)
        pred_path = safe_join(pred_root, entry.pred_mesh)
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT mesh: {gt_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction mesh: {pred_path}")

        gt_vertices, gt_faces = load_obj(gt_path)
        pred_vertices, pred_faces = load_obj(pred_path)
        if pred_vertices.shape != gt_vertices.shape:
            raise ValueError(
                f"Vertex shape mismatch at {entry.sign_id}/{entry.frame_id}: "
                f"pred={pred_vertices.shape}, gt={gt_vertices.shape}"
            )
        vertex_count = gt_vertices.shape[0]
        if expected_vertex_count is not None and vertex_count != int(expected_vertex_count):
            raise ValueError(
                f"Expected {expected_vertex_count} vertices, got {vertex_count} "
                f"at {entry.sign_id}/{entry.frame_id}"
            )
        if vertex_count not in masks_validated_for:
            validate_masks_for_mesh(masks, vertex_count)
            masks_validated_for.add(vertex_count)
        if check_topology and not np.array_equal(
            canonical_faces(pred_faces), canonical_faces(gt_faces)
        ):
            raise ValueError(f"Topology mismatch at {entry.sign_id}/{entry.frame_id}")

        row: dict[str, Any] = {
            "sign_id": entry.sign_id,
            "frame_id": entry.frame_id,
            "gt_mesh": entry.gt_mesh,
            "pred_mesh": entry.pred_mesh,
        }
        active_regions = set(entry.regions)
        for region in REGIONS:
            column = f"{region}_mean_mm"
            if region not in active_regions:
                row[column] = ""
                continue

            indices = region_indices(region, entry.upper_exclude, masks)
            errors = translation_aligned_errors_mm(
                pred_vertices,
                gt_vertices,
                indices,
                input_to_mm,
            )
            mean_mm = float(errors.mean())
            row[column] = mean_mm

            totals[region]["sum_mm"] += float(errors.sum())
            totals[region]["vertex_count"] += int(errors.size)
            totals[region]["frame_count"] += 1
            frame_means[region].append(mean_mm)

            sign_accumulator = sign_totals[(entry.sign_id, region)]
            sign_accumulator["sum_mm"] += float(errors.sum())
            sign_accumulator["vertex_count"] += int(errors.size)
            sign_accumulator["frame_count"] += 1

        per_frame_rows.append(row)

    per_sign_rows: list[dict[str, Any]] = []
    sign_means: dict[str, list[float]] = {region: [] for region in REGIONS}
    for (sign_id, region), accumulator in sorted(sign_totals.items()):
        vertex_count = int(accumulator["vertex_count"])
        if vertex_count <= 0:
            raise AssertionError(f"Zero denominator for {sign_id}/{region}")
        mean_mm = float(accumulator["sum_mm"]) / vertex_count
        sign_means[region].append(mean_mm)
        per_sign_rows.append(
            {
                "sign_id": sign_id,
                "region": region,
                "primary_micro_mean_mm": mean_mm,
                "frame_count": int(accumulator["frame_count"]),
                "vertex_count": vertex_count,
            }
        )

    overall: dict[str, Any] = {}
    for region in REGIONS:
        vertex_count = int(totals[region]["vertex_count"])
        frame_count = int(totals[region]["frame_count"])
        if vertex_count == 0:
            overall[region] = {
                "primary_micro_mean_mm": None,
                "mean_of_frame_means_mm": None,
                "macro_sign_mean_mm": None,
                "frame_count": 0,
                "vertex_count": 0,
            }
            continue
        overall[region] = {
            "primary_micro_mean_mm": float(totals[region]["sum_mm"]) / vertex_count,
            "mean_of_frame_means_mm": float(np.mean(frame_means[region])),
            "macro_sign_mean_mm": float(np.mean(sign_means[region])),
            "frame_count": frame_count,
            "vertex_count": vertex_count,
        }

    summary = {
        "protocol": protocol,
        "manifest_sha256": sha256_file(manifest_path),
        "mask_sha256": actual_mask_hash,
        "unique_frame_count": len(entries),
        "topology_check": check_topology,
        "overall": overall,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "per_frame.csv",
        [
            "sign_id",
            "frame_id",
            "gt_mesh",
            "pred_mesh",
            "upper_body_minus_face_mean_mm",
            "left_hand_mean_mm",
            "right_hand_mean_mm",
        ],
        per_frame_rows,
    )
    write_csv(
        output_dir / "per_sign.csv",
        ["sign_id", "region", "primary_micro_mean_mm", "frame_count", "vertex_count"],
        per_sign_rows,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict manifest-driven TR-V2V evaluator")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-topology-check",
        action="store_true",
        help="Only for debugging; never use for an official-compatible run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(
        manifest_path=args.manifest,
        mask_path=args.masks,
        gt_root=args.gt_root,
        pred_root=args.pred_root,
        output_dir=args.output,
        check_topology=not args.skip_topology_check,
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
# END TRV2V_REFERENCE
```

Run:

```bash
python -m evaluation.trv2v \
  --manifest evaluation/sgnify_trv2v_manifest.json \
  --masks evaluation/sgnify_trv2v_masks.npz \
  --gt-root /path/to/sgnify_gt \
  --pred-root runs/dexfactor4d/export \
  --output runs/dexfactor4d/trv2v
```

`summary.json → overall → region → primary_micro_mean_mm` tái hiện cách attachment concatenate mọi vertex error rồi lấy mean. Hai aggregate phụ giúp thấy hiệu ứng sign/frame weighting.

### 17.1 TR-V2V tests

File mục tiêu: `tests/test_trv2v.py`.

```python
# BEGIN TRV2V_TESTS
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.trv2v import evaluate, translation_aligned_errors_mm


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            one_based = face + 1
            handle.write(f"f {one_based[0]} {one_based[1]} {one_based[2]}\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_translation_is_removed() -> None:
    gt = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    pred = gt + np.array([10.0, -3.0, 5.0])
    errors = translation_aligned_errors_mm(pred, gt, np.arange(3), 1.0)
    np.testing.assert_allclose(errors, 0.0, atol=1e-12)


def test_rotation_is_not_removed() -> None:
    gt = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    rotation_90_z = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pred = gt @ rotation_90_z.T
    errors = translation_aligned_errors_mm(pred, gt, np.arange(3), 1.0)
    assert errors.mean() > 0.1


def test_alignment_is_region_specific() -> None:
    gt = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    pred = gt.copy()
    pred[:2] += np.array([10.0, 0.0, 0.0])
    pred[2:] += np.array([-10.0, 0.0, 0.0])
    np.testing.assert_allclose(
        translation_aligned_errors_mm(pred, gt, np.array([0, 1]), 1.0),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        translation_aligned_errors_mm(pred, gt, np.array([2, 3]), 1.0),
        0.0,
        atol=1e-12,
    )


def test_strict_end_to_end_and_denominators(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    pred_root = tmp_path / "pred"
    gt_root.mkdir()
    pred_root.mkdir()

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        dtype=np.int64,
    )
    _write_obj(gt_root / "frame.obj", vertices, faces)
    _write_obj(pred_root / "frame.obj", vertices + np.array([2.0, 3.0, 4.0]), faces)

    mask_path = tmp_path / "masks.npz"
    np.savez(
        mask_path,
        upper_body_minus_face=np.array([0, 1, 2, 3], dtype=np.int64),
        left_hand=np.array([0, 1], dtype=np.int64),
        right_hand=np.array([2, 3], dtype=np.int64),
    )
    manifest = {
        "protocol": {
            "name": "unit_test",
            "alignment": "per_region_centroid_translation_only",
            "input_unit": "m",
            "output_unit": "mm",
            "expected_frame_count": 1,
            "expected_vertex_count": 4,
            "mask_file_sha256": _sha256(mask_path),
        },
        "frames": [
            {
                "sign_id": "S1",
                "frame_id": "0",
                "gt_mesh": "frame.obj",
                "pred_mesh": "frame.obj",
                "regions": [
                    "upper_body_minus_face",
                    "left_hand",
                    "right_hand",
                ],
                "upper_exclude": [],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = evaluate(
        manifest_path,
        mask_path,
        gt_root,
        pred_root,
        tmp_path / "out",
    )
    assert summary["unique_frame_count"] == 1
    for region in ("upper_body_minus_face", "left_hand", "right_hand"):
        assert summary["overall"][region]["primary_micro_mean_mm"] == pytest.approx(0.0)
        assert summary["overall"][region]["frame_count"] == 1


def test_missing_prediction_is_failure(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    pred_root = tmp_path / "pred"
    gt_root.mkdir()
    pred_root.mkdir()

    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    _write_obj(gt_root / "frame.obj", vertices, faces)

    mask_path = tmp_path / "masks.npz"
    np.savez(
        mask_path,
        upper_body_minus_face=np.array([0, 1, 2], dtype=np.int64),
        left_hand=np.array([0, 1], dtype=np.int64),
        right_hand=np.array([1, 2], dtype=np.int64),
    )
    manifest = {
        "protocol": {
            "name": "missing_prediction_test",
            "alignment": "per_region_centroid_translation_only",
            "input_unit": "m",
            "output_unit": "mm",
            "expected_frame_count": 1,
            "expected_vertex_count": 3,
            "mask_file_sha256": _sha256(mask_path),
        },
        "frames": [
            {
                "sign_id": "S1",
                "frame_id": "0",
                "gt_mesh": "frame.obj",
                "pred_mesh": "missing.obj",
                "regions": ["upper_body_minus_face"],
                "upper_exclude": [],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        evaluate(
            manifest_path,
            mask_path,
            gt_root,
            pred_root,
            tmp_path / "out",
        )
# END TRV2V_TESTS
```

Thêm integration tests:

- class-0 manifest omits `left_hand` and removes its IDs khỏi upper mask; verify exact counts;
- NaN prediction raises;
- mask hash mismatch raises;
- duplicate `(sign_id, frame_id)` raises;
- prediction extra/missing frame fails ở manifest builder/export validator;
- face winding/order khác nhưng canonical topology giống thì pass;
- vertex-label topology khác thì fail;
- `m` input chuyển đúng sang `mm` ×1000;
- attachment evaluator và reference evaluator cho cùng số trên một compatibility fixture không NaN/missing.

---

## 18. Building the SGNify/TR-V2V manifest

### 18.1 Inputs

- exact sign list/class file;
- exact central-segment annotations;
- RGB evaluation frame IDs/timestamps;
- GT mesh frame IDs/timestamps;
- prediction frame IDs/timestamps;
- official or author-provided synchronization rule;
- region masks.

### 18.2 Builder algorithm

```text
read sign list in declared order
for each sign:
    read central RGB frame IDs exactly
    for each RGB frame:
        resolve one GT mesh using author timestamp/index mapping
        resolve exactly one prediction mesh with same RGB FrameKey
        determine explicit active regions
        if source-compatible class == "0":
            regions = [upper_body_minus_face, right_hand]
            upper_exclude = [left_hand]
        else:
            regions = [upper_body_minus_face, left_hand, right_hand]
            upper_exclude = []
        append manifest record
assert unique record count == 2872
assert every path exists before freezing manifest
hash manifest and masks
```

Nếu official policy thực ra chấm left hand cho class 0, tạo protocol khác; không sửa cùng một manifest sau khi chạy test.

### 18.3 Compatibility validation

Trước khi dùng cho paper:

1. chạy original attachment trên một subset không missing/NaN;
2. build manifest tái hiện đúng pairs;
3. chạy reference evaluator;
4. yêu cầu khác biệt absolute dưới `1e-8 mm` hoặc giải thích rounding/parser;
5. so denominator từng region/sign;
6. xin xác nhận tác giả về region-wise centering, class-0 policy và frame synchronization.

---

## 19. Evaluation outputs và statistics

Primary table dùng `primary_micro_mean_mm` cho:

- `upper_body_minus_face`;
- `left_hand`;
- `right_hand`.

Báo kèm:

- region frame count;
- vertex denominator;
- mean of frame means;
- macro sign mean;
- per-sign distribution;
- missing/invalid count, bắt buộc bằng 0;
- manifest/mask/config/checkpoint hashes.

Paired bootstrap:

- resample 57 signs, không resample frame độc lập;
- 10.000 resamples, fixed seed;
- tính paired delta method−baseline;
- báo 95% CI;
- guardrail từng region: upper CI không vượt +0,5 mm;
- Holm correction nếu chạy nhiều hypothesis tests.

Không được bootstrap trên vertex độc lập vì vertices/frame/sign tương quan mạnh.

---

## 20. Ablation implementation profiles

| Profile | Modules bật |
|---|---|
| B | Public DexAvatar commit |
| C | DexAvatar-CF only |
| D | C + global tracks + WiLoR observations |
| E | D + full-XYZ normalized hand geometry |
| F | E + silhouette + depth order |
| G | F + SO(3) sequence fitting |
| H | G + reliability gating |
| I | H + BMC/collision/contact |
| J | I + switchable PAD-Hand proposal = full method |

Mọi profile dùng cùng frozen observations và frame manifest. Không rerun detector với checkpoint/config khác giữa ablations.

`B→C` là correctness gain; `C→J` mới là algorithmic gain.

---

## 21. Dev calibration, không neural training

### 21.1 Residual scales

Trên dev baseline:

1. thu raw residual từng factor;
2. tính median và MAD;
3. lưu scale với dataset split hash;
4. normalize factor;
5. không cập nhật scale trên SGNify.

### 21.2 Hyperparameter grid

Grid nhỏ, preregistered:

- optional factor multiplier: `{0, 0.25, 1, 4}`;
- temporal `gamma_min/gamma_max` ratios: một danh sách hữu hạn khóa trước;
- reliability alpha: equal hoặc một số simplex candidates định trước;
- contact thresholds từ dev percentiles;
- PAD switch prior từ dev low-confidence subset.

Selection objective phải gồm positional accuracy, acceleration và failure rate. Không chọn chỉ bằng smoothness.

### 21.3 Frozen transition

Khi dev xong:

- tạo `dexfactor4d_frozen.yaml`;
- xóa grid/search code khỏi test command;
- commit code/config;
- hash container/checkpoints;
- khóa SGNify manifest/masks;
- chạy test một lần.

---

## 22. Diagnostics bắt buộc

Mỗi run lưu:

```text
runs/exp/
├── run_metadata.json
├── frozen_config.yaml
├── versions.json
├── observations/
├── tracks.json
├── reliability.csv
├── pad_proposals.pt
├── fitting/
│   ├── checkpoints/
│   ├── losses.csv
│   ├── window_status.json
│   └── final_state.pt
├── export/
│   ├── pkl/
│   ├── obj/
│   └── export_validation.json
└── trv2v/
    ├── summary.json
    ├── per_frame.csv
    └── per_sign.csv
```

Diagnostics plots:

- q reliability over time for L/R;
- HaMeR–WiLoR disagreement;
- per-factor loss curves;
- acceleration before/after;
- switch values PAD;
- collision/BMC violations;
- per-sign TR-V2V delta;
- missing/ambiguous track timeline.

---

## 23. AI coding-agent work packages

AI nên implement theo thứ tự sau và không nhảy tới full method:

### WP0 — Reproduction lock

- pin public DexAvatar commit;
- record environment/assets;
- run static/smoke tests;
- preserve upstream baseline command.

**Done:** baseline output reproducible trên một clip toy.

### WP1 — Contracts and correctness

- FrameKey/manifests;
- numeric order;
- no-drop records;
- named joint mapping;
- semantic export;
- resume state.

**Done:** all contract tests pass; PKL/OBJ agree.

### WP2 — Strict TR-V2V

- manifest builder;
- mask validation;
- reference evaluator;
- attachment compatibility test;
- CSV/JSON outputs.

**Done:** synthetic tests pass và subset compatibility parity đạt tolerance.

### WP3 — Observations and tracking

- adapters/cache;
- full-image coordinate transforms;
- global L/R Viterbi;
- missing/ambiguous state;
- cross-source pairing.

**Done:** permutation/mirror/gap tests pass.

### WP4 — Sequence state and core factors

- ClipState/DecodedState;
- Sign prior adapters;
- body/hand 2D;
- full-XYZ hand geometry;
- robust scale normalization;
- staged LBFGS.

**Done:** toy sequence converges và gradients finite.

### WP5 — Multi-cue and temporal

- Sapiens silhouette renderer;
- depth-order ranking;
- SO(3) velocity/acceleration;
- reliability gating;
- window/overlap consensus.

**Done:** synthetic occlusion/fast-motion tests pass without oversmoothing guardrail failure.

### WP6 — Interaction and PAD

- BMC adapter/assets;
- collision;
- contact candidates;
- semantic PAD adapter;
- switchable proposal factor.

**Done:** separated/contact/penetration/mirror/PAD switch tests pass.

### WP7 — Dev calibration and frozen evaluation

- dev scales/grid;
- ablations C–J;
- freeze config;
- official-compatible run;
- bootstrap/report.

**Done:** complete immutable evaluation bundle.

---

## 24. Definition of Done

Implementation chưa được coi là hoàn chỉnh nếu thiếu bất kỳ điều nào:

- [ ] all input frames produce finite SMPL-X outputs;
- [ ] named 21-joint mapping verified;
- [ ] detection permutation does not change semantic tracks;
- [ ] no z-only hand factor remains;
- [ ] SO(3) temporal uses real timestamps;
- [ ] reliability components are logged;
- [ ] PAD is proposal-only and switchable;
- [ ] BMC assets/conventions validated or module explicitly disabled;
- [ ] PKL→SMPL-X vertices equal OBJ;
- [ ] manifest has expected 2.872 unique frames;
- [ ] no missing/NaN frame is skipped;
- [ ] masks and manifest hashes recorded;
- [ ] attachment compatibility test passes;
- [ ] per-region denominators reported;
- [ ] correctness baseline C separated from method gains;
- [ ] dev config frozen before SGNify GT evaluation;
- [ ] every claimed result traceable to run metadata.

---

## 25. Những điểm cần xác nhận với tác giả/dataset owner

Không để AI tự quyết các điểm này:

1. `evaluate_new_fitting(4).py` có phải evaluator tạo bảng DexAvatar cuối cùng không?
2. Official alignment có thật sự center từng region độc lập không?
3. Với class `"0"`, official table có bỏ left-hand metric và bỏ left-hand vertices khỏi upper body không?
4. `frame_segment * 2` ánh xạ RGB 30 fps sang GT mesh nào; GT files chỉ có even IDs hay evaluator đang lấy mọi ID?
5. Exact 2.872-frame manifest và exact region masks có thể release không?
6. Kết quả paper lấy mesh OBJ hay PKL-rendered SMPL-X?
7. Khi prediction missing/NaN, official policy là fail hay loại frame?

Cho đến khi xác nhận, dùng tên `source_evaluator_compat_v1`, không ghi “official SGNify TR-V2V reproduction”.

---

## 26. Final implementation rule

AI coding agent phải ưu tiên correctness theo thứ tự:

> **Frame identity → hand identity → coordinate identity → state/export identity → strict metric → algorithmic improvements.**

Nếu một module mới làm metric tốt hơn nhưng vi phạm manifest coverage, finite checks, frozen protocol hoặc export consistency, kết quả đó không hợp lệ.
