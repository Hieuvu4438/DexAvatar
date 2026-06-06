# Hướng dẫn đọc hiểu Codebase DexAvatar

> Tài liệu này dành cho người mới bắt đầu muốn hiểu toàn bộ cấu trúc và luồng chạy của dự án DexAvatar.

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc Pipeline](#2-kiến-trúc-pipeline)
3. [Cấu trúc thư mục gốc](#3-cấu-trúc-thư-mục-gốc)
4. [Chi tiết từng thư mục và file](#4-chi-tiết-từng-thư-mục-và-file)
   - 4.1 [Top-level files](#41-top-level-files-các-file-ở-thư-mục-gốc)
   - 4.2 [data/](#42-datathư-mục-dữ-liệu)
   - 4.3 [scripts/](#43-scriptsthư-mục-script-pipeline)
   - 4.4 [methods/](#44-methodsthư-mục-các-phương-pháp-variant)
   - 4.5 [dexavatar_fitting/](#45-dexavatar_fittingthư-mục-core-fitting-engine)
   - 4.6 [SMPLer-X/](#46-smpler-xthư-mục-3d-body-estimation)
   - 4.7 [WiLoR/](#47-wilorthư-mục-hand-mesh-recovery)
   - 4.8 [hamer/](#48-hamerthư-mục-hand-mesh-recovery-ban-đầu)
   - 4.9 [sapiens/](#49-sapiensthư-mục-pose-estimation)
   - 4.10 [Các thư mục khác](#410-các-thư-mục-khác)
5. [Thứ tự đọc hiểu codebase](#5-thứ-tự-đọc-hiểu-codebase)
6. [Bảng tóm tắt Method Variants](#6-bảng-tóm-tắt-method-variants)
7. [Môi trường Conda](#7-môi-trường-conda)

---

## 1. Tổng quan dự án

**DexAvatar** là một hệ thống tái tạo ngôn ngữ ký hiệu 3D (3D Sign Language Reconstruction) từ video đơn thị (monocular video). Dự án đạt **Best Paper Award Finalist tại WACV 2026**.

**Bài toán:** Cho một video quay người đang ký hiệu (sign language), hệ thống cần tái tạo lại mesh 3D chi tiết của toàn bộ cơ thể, bao gồm cả các khớp ngón tay chính xác về mặt sinh học cơ học (biomechanics).

**Điểm mới (Contribution):**
- **SignBPoser**: VAE học pose prior cho cơ thể khi ký hiệu
- **SignHPoser**: VAE học pose prior cho tay khi ký hiệu
- Pipeline tối ưu hóa (optimization) kết hợp nhiều nguồn supervision: 2D keypoints, 3D hand mesh, pose priors, và ràng buộc sinh học cơ học

**Dữ liệu:** 57 ký hiệu từ Ngôn ngữ ký hiệu Đức (DGS), trích từ dataset SGNify.

---

## 2. Kiến trúc Pipeline

Pipeline gồm **4 giai đoạn** chạy tuần tự cho mỗi ký hiệu:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Video frames (PNG)                 │
│                    data/frames/<sign_name>/                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1A: Sapiens Extraction (sapiens_lite env)            │
│  - Person detection + 133-keypoint whole-body pose          │
│  - Output: sapiens_1b/*.json per frame                      │
│  - Script: scripts/S1_sapiens_extract.sh                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1B: Aggregate Sapiens (dexavatar env)                │
│  - Gộp per-frame JSON → 1 file sapiens.pkl                 │
│  - Script: scripts/aggregate_sapiens.py                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: SMPLer-X Extraction (smpler_x env)               │
│  - 3D body shape/pose từ image                              │
│  - Output: smplerx/smplx/*.pkl (body_pose, betas, etc.)     │
│  - Script: scripts/S1_smplerx_extract.sh                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Hand Extraction (dexavatar env)                   │
│  - Tính mean shape từ betas (M3_mean_shape_smplerx.py)      │
│  - WiLoR/HaMeR 3D hand pose estimation                     │
│  - Output: hamer/hamer.pkl, mean_shape_smplx.npy            │
│  - Script: scripts/M3.5_wilor_extract.sh                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: SMPLify-X Fitting (dexavatar env)  ← CORE        │
│  - Optimization-based fitting với SignBPoser + SignHPoser   │
│  - Multi-stage L-BFGS-LS optimization                       │
│  - Output: smplifyx/meshes/*.obj, results/*.pkl             │
│  - Script: scripts/M4_smplifyx_pose.sh                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Evaluation: TR-V2V metric (mm)                             │
│  - So sánh mesh output vs ground truth                      │
│  - Metrics: UBody, LHand, RHand                             │
│  - Script: evaluation_trv2v_wilor.py                        │
└─────────────────────────────────────────────────────────────┘
```

**3 môi trường Conda riêng biệt:**
- `sapiens_fix` — chạy Sapiens (Python 3.9)
- `smpler_x` — chạy SMPLer-X (Python 3.8)
- `dexavatar` — chạy fitting chính (Python 3.10)

---

## 3. Cấu trúc thư mục gốc

```
DexAvatar/
├── README.md                    # Hướng dẫn cài đặt và sử dụng
├── requirements.txt             # Dependencies cho dexavatar env
├── LICENSE                      # MIT License
├── .gitignore
│
├── run_dexavatar_wilor.py       # ★ ENTRY POINT chính
├── Full_running_command_wilor.sh # ★ Pipeline orchestrator
├── evaluation_trv2v_wilor.py    # ★ Evaluation script
├── eval_wilor_full.sh           # Shell chạy eval 57 signs
├── M3_mean_shape_smplerx.py     # Tính mean shape
│
├── data/                        # Dữ liệu input
├── scripts/                     # Các script pipeline stage
├── methods/                     # Các variant method
├── dexavatar_fitting/           # ★ CORE: Fitting engine
├── SMPLer-X/                    # 3D body estimation
├── WiLoR/                       # Hand mesh recovery
├── hamer/                       # Hand mesh recovery (cũ)
├── sapiens/                     # Pose estimation
│
├── checkpoints/                 # Model weights
├── _DATA/                       # Checkpoints shared (HaMeR, ViTPose)
├── outputs/                     # Kết quả runtime
├── assets/                      # Ảnh cho README
├── neural_renderer/             # Third-party: neural renderer
├── torch-mesh-isect/            # Third-party: mesh intersection
├── LHM-plusplus/                # External project (không dùng)
└── docs/                        # Tài liệu
```

---

## 4. Chi tiết từng thư mục và file

### 4.1 Top-level files (các file ở thư mục gốc)

#### `run_dexavatar_wilor.py` — Entry Point chính

```python
# Chức năng: Duyệt qua tất cả subfolder trong data/frames/
# → Với mỗi ký hiệu, gọi Full_running_command_wilor.sh
#
# Tham số:
#   --input_img_folder: thư mục chứa frames (data/frames/)
#   --output_path: thư mục output
#   --fitting_experiment: tên experiment
#
# Ví dụ chạy:
#   python run_dexavatar_wilor.py --input_img_folder data/frames --output_path outputs/output_wilor
```

**Cách đọc:** File này rất ngắn (~30 dòng). Chỉ là vòng lặp `for` qua các subfolder, gọi shell script cho mỗi ký hiệu.

#### `Full_running_command_wilor.sh` — Pipeline Orchestrator

```bash
# Chức năng: Chạy toàn bộ pipeline cho 1 ký hiệu
# Thứ tự: Sapiens → Aggregate → SMPLer-X → WiLoR → SMPLify-X
#
# Biến môi trường:
#   ROOT_PATH: thư mục frames của ký hiệu
#   OUTPUT_PATH: thư mục output
#   FITTING_EXPERIMENT: tên experiment
```

**Cách đọc:** Đây là file quan trọng để hiểu thứ tự chạy. Đọc từ trên xuống, mỗi block là một stage.

#### `evaluation_trv2v_wilor.py` — Evaluation

```python
# Chức năng: Tính metric TR-V2V (Translation-Removed Vertex-to-Vertex)
# - Load mesh output (.obj) và ground truth (.obj)
# - Align bằng cách trừ translation
# - Tính L2 distance per vertex
# - Report theo 3 vùng: UBody (upper body - face), LHand, RHand
#
# Output: DataFrame với cột sign, UBody_mm, LHand_mm, RHand_mm
```

#### `M3_mean_shape_smplerx.py` — Mean Shape Calculator

```python
# Chức năng: Trung bình hóa shape parameters (betas) across frames
# - Load tất cả .pkl từ SMPLer-X output
# - Lấy betas từ mỗi frame
# - Tính mean → lưu thành mean_shape_smplx.npy
# Mục đích: Giữ shape nhất quán across frames trong fitting
```

---

### 4.2 `data/`— Thư mục dữ liệu

```
data/
├── signs.txt              # Danh sách 57 ký hiệu + nhãn (0=1 tay, ~0=2 tay)
├── segment.json           # Frame ranges [start, end] cho mỗi ký hiệu
├── frames/                # 57 subfolder, mỗi cái chứa PNG frames
│   ├── Ablehnen/          #   VD: low_121.png, low_123.png, ...
│   ├── Abnehmen/
│   └── ...
├── smplx_gt/              # 57 subfolder, mỗi cái chứa GT meshes
│   ├── Ablehnen/          #   VD: 00242.obj, 00246.obj, ...
│   └── ...
└── dummy_test/            # Subset nhỏ để test (chỉ Ablehnen)
```

**Quy ước đặt tên:**
- Frame: `low_NNN.png` với NNN là số lẻ (121, 123, 125, ...)
- GT mesh: `XXXXX.obj` với XXXXX = NNN × 2 (VD: frame 121 → GT 00242)
- Mapping: frame index `i` → GT index `2*i`, zero-padded 5 chữ số

**Cách đọc:**
- `signs.txt`: mỗi dòng = `tên_ký_hiệu\tclass_label`
- `segment.json`: dict `{sign_name: [start_frame, end_frame]}`
- `frames/`: ảnh PNG input cho pipeline
- `smplx_gt/`: ground truth để đánh giá (không dùng trong training)

---

### 4.3 `scripts/`— Thư mục script pipeline

#### Scripts cấu hình môi trường

| File | Chức năng |
|------|-----------|
| `config.sh` | Activate conda env `dexavatar` |
| `config_sapiens.sh` | Activate conda env `sapiens_fix` |
| `config_smplerx.sh` | Activate conda env `smpler_x` |
| `env_install.sh` | Cài đặt đầy đủ dependencies |
| `bug_fix.sh` | Patch torchgeometry trong smpler_x env |
| `bug_fix_dexavatar.sh` | Đã rỗng (fix đã apply) |

#### Scripts từng stage

| File | Stage | Mô tả |
|------|-------|-------|
| `S1_sapiens_extract.sh` | 1A | Chạy Sapiens lite pose estimation (133 keypoints) |
| `aggregate_sapiens.py` | 1B | Gộp per-frame JSON → `sapiens.pkl` |
| `S1_smplerx_extract.sh` | 2 | Chạy SMPLer-X 3D shape/pose estimation |
| `M3.5_wilor_extract.sh` | 3 | Mean shape + WiLoR hand extraction |
| `M3.5_hamer_extract.sh` | 3 | Mean shape + HaMeR hand extraction |
| `M3_ensemble_init.sh` | 3 | Mean shape + HaMeR cho ensemble init |
| `M4_smplifyx_pose.sh` | 4 | SMPLify-X fitting (default config) |
| `M4_smplifyx_pose_biomech.sh` | 4 | SMPLify-X với biomechanics config |
| `M4_smplifyx_pose_ensemble.sh` | 4 | SMPLify-X với ensemble init config |
| `M4_smplifyx_pose_hand2d.sh` | 4 | SMPLify-X với 2D hand keypoint supervision |
| `M4_smplifyx_pose_temporal.sh` | 4 | SMPLify-X với temporal window fitting |

#### Scripts chạy toàn bộ pipeline + eval

| File | Mô tả |
|------|-------|
| `run_m2_hand2d.sh` | Pipeline Hand2D + auto-evaluation |
| `run_m4_ensemble.sh` | Pipeline Ensemble + auto-evaluation |
| `run_m5_biomech.sh` | Pipeline Biomechanics + auto-evaluation |
| `eval_baseline_when_ready.sh` | Đợi baseline xong rồi evaluate |

**Cách đọc:** Bắt đầu từ `S1_sapiens_extract.sh` để hiểu cách gọi Sapiens. Sau đó đọc `M4_smplifyx_pose.sh` để hiểu cách gọi fitting.

---

### 4.4 `methods/`— Thư mục các phương pháp variant

```
methods/
├── run_dexavatar.py                        # Baseline (HaMeR)
├── run_dexavatar_wilor_biomech.py          # WiLoR + Biomechanics
├── run_dexavatar_wilor_ensemble.py         # WiLoR + Ensemble Init
├── run_dexavatar_wilor_hand2d.py           # WiLoR + 2D Hand Supervision
├── run_dexavatar_wilor_temporal.py         # WiLoR + Temporal Window
│
├── Full_running_command.sh                 # Shell orchestrator cho Baseline
├── Full_running_command_wilor_biomech.sh   # Shell orchestrator cho Biomech
├── Full_running_command_wilor_ensemble.sh  # Shell orchestrator cho Ensemble
├── Full_running_command_wilor_hand2d.sh    # Shell orchestrator cho Hand2D
├── Full_running_command_wilor_temporal.sh  # Shell orchestrator cho Temporal
│
├── eval_hand2d_full.sh                     # Eval cho Hand2D
└── eval_temporal_full.sh                   # Eval cho Temporal
```

**Cách đọc:** Mỗi cặp `(run_*.py, Full_running_command_*.sh)` tạo thành một method variant. Python file duyệt qua signs, shell file chạy pipeline. Cấu trúc giống hệt `run_dexavatar_wilor.py` ở root nhưng với config khác nhau.

---

### 4.5 `dexavatar_fitting/`— Thư mục Core Fitting Engine

**Đây là phần quan trọng nhất của dự án.** Nơi implement thuật toán tối ưu hóa SMPLify-X với các sign language priors.

```
dexavatar_fitting/
├── script.py                    # Entry point: gọi smplifyx/main.py
├── script_hand2d.py             # Entry point cho Hand2D variant
├── script_temporal.py           # Entry point cho Temporal variant
├── rewrite_body_model.py        # ★ Quan trọng: Modified SMPL/SMPLX body models
│
├── cfg_files/                   # YAML configs cho fitting
│   ├── fit_smplx_vposer_x.yaml           # Default (baseline)
│   ├── fit_smplx_vposer_x_biomech.yaml   # + Biomechanics
│   ├── fit_smplx_vposer_x_hand2d.yaml    # + 2D hand keypoints
│   ├── fit_smplx_vposer_x_temporal.yaml  # + Temporal window
│   ├── fit_smplx_vposer_x_ensemble.yaml  # + Ensemble init
│   ├── fit_smpl.yaml                     # Original SMPLify-X configs
│   ├── fit_smplx.yaml
│   └── fit_smplh.yaml
│
├── smplifyx/                    # ★ Core optimization code
│   ├── main.py                  # Main loop: load models → iterate frames → fit
│   ├── fit_single_frame.py      # Single-frame optimization
│   ├── fit_temporal_window.py   # Temporal sliding window optimization
│   ├── fitting.py               # ★ Loss functions (SMPLifyLoss)
│   ├── data_parser.py           # Dataset class: load keypoints + params
│   ├── camera.py                # Camera model
│   ├── prior.py                 # Pose priors (L2, angle, GMM)
│   ├── utils.py                 # Utilities (JointMapper, GMoF)
│   ├── body_constants.py        # Biomechanical joint angle bounds
│   ├── cmd_parser.py            # Argument parsing từ YAML
│   ├── ensemble_init.py         # Multi-model ensemble initialization
│   ├── mesh_viewer.py           # PyRender visualization
│   ├── render_pkl.py            # Render từ .pkl results
│   ├── render_results.py        # Render results
│   ├── test_bposer.py           # Load SignBPoser model
│   ├── test_hposer.py           # Load SignHPoser model
│   │
│   ├── optimizers/              # L-BFGS with Strong Wolfe Line Search
│   │   └── optim_factory.py
│   │
│   ├── signbposer/              # ★ Learned body pose prior
│   │   ├── signbposer.py        # VAE: 21 body joints → 33-dim latent
│   │   ├── snapshots/
│   │   │   └── TR00_E078.pt     # Pretrained weights
│   │   └── TR00_signbposer.ini  # Training config
│   │
│   └── signhposer/              # ★ Learned hand pose prior
│       ├── signhposer/
│       │   ├── signhposer.py    # VAE: 15 hand joints → 23-dim latent
│       │   └── snapshots/
│       │       └── TR00_E100.pt # Pretrained weights
│       └── signhposer/
│           └── TR00_signhposer.ini
│
└── assets/                      # Region indices cho evaluation
    ├── smplx_upper_body_minus_face_vidx.npy  # 5401 vertices: UBody(-F)
    ├── smplx_left_hand_vidx.npy              # 809 vertices: left hand
    ├── smplx_right_hand_vidx.npy             # 808 vertices: right hand
    ├── smplx_parts_segm.pkl                  # Per-triangle part segmentation
    ├── smplx_region_manifest.json            # Metadata
    ├── build_region_indices.py               # Script build vertex indices
    ├── joint_mapping.py                      # COCO-WholeBody → SMPL-X mapping
    └── mapping_func.py                       # Mapping utilities
```

#### Chi tiết các file quan trọng:

##### `rewrite_body_model.py` — Modified Body Models

```python
# Chức năng: Override các class SMPL, SMPLH, SMPLX, MANO, FLAME từ thư viện smplx
#
# Modification chính:
#   - body_pose được CHIA THÀNH 2 phần:
#     + body_pose_fore: 15 joints đầu (fore body) — được optimize riêng
#     + body_pose_op: các joints còn lại — có thể freeze hoặc optimize khác
#   - Mục đích: Cho phép optimize các joints quan trọng (tay, vai) với
#     learning rate/prior khác nhau so với các joints ít quan trọng hơn
#
# Classes: SMPL, SMPLH, SMPLX, MANO, FLAME (ghi đè từ smplx library)
```

##### `smplifyx/main.py` — Main Fitting Loop

```python
# Chức năng: Orchestrator chính cho toàn bộ fitting process
#
# Flow:
#   1. Parse config từ YAML (cmd_parser.parse_config)
#   2. Load body models cho male/female/neutral
#   3. Tạo priors: body_pose_prior, jaw_prior, left_hand_prior, right_hand_prior
#   4. Tạo camera model
#   5. Load dataset (data_parser.create_dataset)
#   6. Với mỗi frame:
#      a. Load keypoints, init SMPL-X params
#      b. Gọi fit_single_frame() hoặc fit_temporal_window()
#      c. Lưu mesh (.obj), params (.pkl), hình ảnh
#
# Output: smplifyx/meshes/*.obj, smplifyx/results/*.pkl, smplifyx/images/*.png
```

##### `smplifyx/fit_single_frame.py` — Single Frame Optimization

```python
# Chức năng: Tối ưu hóa SMPL-X params cho 1 frame
#
# Flow:
#   1. guess_init(): Ước tính camera translation ban đầu
#   2. Stage 1: Optimize camera translation + global orient
#   3. Stage 2: Optimize body pose (via SignBPoser latent) + shape
#   4. Stage 3: Optimize hand pose (via SignHPoser latent) + thêm 2D hand loss
#
# Hàm chính:
#   - fit_single_frame(): entry point
#   - guess_init(): ước tính camera z từ 2D/3D joint ratio
#   - FittingMonitor: context manager chạy L-BFGS optimization
```

##### `smplifyx/fitting.py` — Loss Functions ★

```python
# Chức năng: Định nghĩa tất cả loss functions cho optimization
#
# Class SMPLifyLoss(nn.Module):
#   - forward(): tính tổng loss
#
# Các loss components:
#   1. reprojection_loss: 2D reprojection error (GT 2D joints vs projected 3D)
#   2. hand_loss: 3D hand joint error (từ WiLoR/HaMeR)
#   3. body_pose_prior_loss: regularization từ SignBPoser VAE latent
#   4. jaw_prior_loss: regularization cho jaw pose
#   5. left/right_hand_prior_loss: regularization từ SignHPoser VAE latent
#   6. shape_prior_loss: L2 regularization cho betas
#   7. angle_prior_loss: biomechanical joint angle constraints
#   8. pen_loss: interpenetration penalty (dùng torch-mesh-isect)
#   9. hand_2d_loss: 2D hand keypoint reprojection (optional, Hand2D variant)
#   10. temporal_loss: velocity/acceleration/jerk smoothness (optional, Temporal variant)
#   11. hand_contact_loss: hand-hand contact penalty (optional, Biomech variant)
#   12. hand_body_contact_loss: hand-body contact penalty (optional, Biomech variant)
#   13. finger_prior_loss: finger articulation prior (optional, Biomech variant)
#
# Hàm helper:
#   - guess_init(): ước tính camera translation
#   - FittingMonitor: chạy optimization loop với convergence checks
```

##### `smplifyx/data_parser.py` — Data Loading

```python
# Chức năng: Load tất cả dữ liệu cần thiết cho fitting
#
# Class OpenPose(Dataset):
#   - __init__(): load keypoints, SMPLer-X params, hand params
#   - __getitem__(): trả về data cho 1 frame
#
# Dữ liệu load:
#   1. Sapiens 2D keypoints (từ sapiens.pkl) → 133 keypoints
#   2. SMPLer-X init params (từ smplerx/smplx/*.pkl) → body_pose, betas, cam_param
#   3. WiLoR/HaMeR hand params (từ hamer/hamer.pkl) → hand_pose, hand_3d_joints
#   4. Sign segmentation (từ segment.json) → frame ranges
#   5. Mean shape (từ mean_shape_smplx.npy) → shape consistency
```

##### `smplifyx/signbposer/signbposer.py` — Body Pose Prior

```python
# Chức năng: VAE học pose distribution của cơ thể khi ký hiệu
#
# Kiến trúc:
#   Encoder: BatchNorm → FC(63→1024) → FC(1024→1024) → mu, logvar
#   Decoder: FC(33→1024) → FC(1024→1024) → FC(1024→126) → 6D rotation
#
# Input: 21 body joints × 3 axis-angle = 63 dims
# Latent: 33 dims
# Output: 21 body joints × 6D rotation = 126 dims
#
# Hàm:
#   - encode(pose) → mu, logvar
#   - decode(z, output_type='aa') → body_pose (axis-angle)
#   - sample() → random pose from prior
```

##### `smplifyx/signhposer/signhposer.py` — Hand Pose Prior

```python
# Chức năng: VAE học pose distribution của tay khi ký hiệu
#
# Kiến trúc: Tương tự SignBPoser nhưng cho hand joints
# Input: 15 hand joints × 3 axis-angle = 45 dims
# Latent: 23 dims
# Output: 15 hand joints × 6D rotation = 90 dims
```

##### `smplifyx/ensemble_init.py` — Ensemble Initialization

```python
# Chức năng: Chọn init SMPL-X params tốt nhất từ nhiều estimators
#
# Flow:
#   1. Load params từ nhiều nguồn (SMPLer-X, HaMeR, etc.)
#   2. Với mỗi source, project 3D joints → 2D
#   3. So sánh với GT 2D keypoints (reprojection error)
#   4. Chọn source có error thấp nhất cho mỗi frame
#
# Mục đích: Khởi tạo optimization tốt hơn → kết quả tốt hơn
```

---

### 4.6 `SMPLer-X/`— Thư mục 3D Body Estimation

```
SMPLer-X/
├── main/
│   ├── script_smplerx.py        # ★ Entry point: chạy inference
│   ├── SMPLer_X.py              # Model definition (ViT + heads)
│   ├── inference.py             # Inference loop
│   └── config.py                # Configuration
│
├── common/
│   ├── utils/
│   │   └── human_model_files/   # SMPL-X model files, joint regressors
│   └── nets/                    # Network architectures
│       ├── PositionNet.py       # 3D position estimation
│       ├── HandRotationNet.py   # Hand rotation estimation
│       ├── BodyRotationNet.py   # Body rotation estimation
│       ├── FaceRegressor.py     # Face parameter estimation
│       ├── BoxNet.py            # Person detection
│       └── HandRoI.py           # Hand region of interest
│
├── data/                        # Dataset definitions (30+ datasets)
└── main/config/                 # Training configs
```

**Cách đọc:** Bắt đầu từ `main/script_smplerx.py` → `main/inference.py` → `main/SMPLer_X.py`.

**Output:** Mỗi frame → 1 file `.pkl` chứa:
- `body_pose`: axis-angle cho body joints
- `left_hand_pose`, `right_hand_pose`: axis-angle cho hand joints
- `betas`: shape parameters (10 dims)
- `cam_param`: camera parameters
- `expression`: face expression params

---

### 4.7 `WiLoR/`— Thư mục Hand Mesh Recovery

```
WiLoR/
├── export_hamer_pkl.py          # ★ Critical bridge: WiLoR → HaMeR format
├── demo.py                      # Demo script
├── gradio_demo.py               # Gradio web demo
│
├── wilor/                       # Core model code
│   ├── models/                  # Model architectures
│   ├── datasets/                # Dataset loaders
│   ├── utils/                   # Utilities
│   └── configs/                 # Model configs
│
├── pretrained_models/
│   ├── wilor_final.ckpt         # WiLoR weights
│   ├── detector.pt              # YOLO hand detector
│   ├── model_config.yaml        # Model config
│   └── dataset_config.yaml      # Dataset config
│
└── mano_data/
    └── mano_mean_params.npz     # MANO mean hand parameters
```

**Cách đọc:** Bắt đầu từ `export_hamer_pkl.py` — đây là bridge script quan trọng.

**`export_hamer_pkl.py` làm gì:**
1. Load WiLoR model
2. Load YOLO hand detector
3. Với mỗi ảnh: detect hands → chạy WiLoR inference
4. Export kết quả ra format `hamer.pkl` (tương thích với pipeline hiện tại)

**Tại sao cần bridge:** Pipeline ban đầu dùng HaMeR. Khi thay bằng WiLoR, cần export cùng format để các stage sau không cần thay đổi.

---

### 4.8 `hamer/`— Thư mục Hand Mesh Recovery (ban đầu)

```
hamer/
├── demo.py                      # Inference script
├── hamer/                       # Model code
│   ├── models/
│   ├── datasets/
│   └── configs/
├── third-party/
│   └── ViTPose/                 # ViTPose dependency
└── _DATA/hamer_ckpts/           # Checkpoints
```

**Cách đọc:** Tương tự WiLoR nhưng là version gốc. Nếu chỉ muốn hiểu pipeline WiLoR, có thể bỏ qua thư mục này.

---

### 4.9 `sapiens/`— Thư mục Pose Estimation

```
sapiens/
├── lite/                        # ★ Phần được sử dụng
│   ├── scripts/demo/torchscript/
│   │   └── pose_keypoints133.sh # Script được gọi bởi S1_sapiens_extract.sh
│   └── torchscript/             # Model checkpoints
│
├── pose/                        # Full pose estimation (không dùng)
├── seg/                         # Segmentation (không dùng)
├── det/                         # Detection (không dùng)
├── pretrain/                    # Pretrained models
└── sapiens_lite_host/           # Host environment setup
```

**Cách đọc:** Chỉ cần đọc `lite/scripts/demo/torchscript/pose_keypoints133.sh` để hiểu cách Sapiens được gọi.

**Sapiens output:** Mỗi frame → 1 file JSON chứa 133 keypoints (body + hands + face) với confidence scores.

---

### 4.10 Các thư mục khác

#### `checkpoints/` — Model Weights
```
checkpoints/
├── smpler_x_h32.pth.tar         # SMPLer-X weights
└── mmdet/                       # Faster R-CNN detector
    ├── faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    └── faster_rcnn_r50_fpn_1x_coco.py
```

#### `_DATA/` — Shared Checkpoints
```
_DATA/
├── hamer_ckpts/checkpoints/hamer.ckpt    # HaMeR weights
├── vitpose_ckpts/vitpose+_huge/wholebody.pth  # ViTPose weights
└── data/mano_mean_params.npz             # MANO mean params
```

#### `outputs/` — Runtime Outputs
```
outputs/
├── output/                      # Original HaMeR baseline
├── output_baseline/             # WiLoR baseline
├── output_wilor/                # WiLoR pipeline
├── output_wilor_hand2d/         # WiLoR + 2D hand supervision
├── output_wilor_temporal/       # WiLoR + temporal window
├── method_biomech/              # WiLoR + biomechanics
├── method_ensemble/             # WiLoR + ensemble init
└── method_hand2d/               # Hand2D variant
```

**Cấu trúc output cho mỗi ký hiệu:**
```
outputs/<method>/<sign_name>/
├── sapiens_1b/                  # Sapiens JSON + visualization PNG
├── smplerx/smplx/              # SMPLer-X .pkl files
├── hamer/hamer.pkl             # WiLoR/HaMeR hand params
├── mean_shape_smplx.npy        # Mean shape
├── gender.txt                  # Gender prediction
└── smplifyx/                   # ★ Final fitting results
    ├── meshes/*.obj            # 3D meshes
    ├── results/*.pkl           # SMPL-X params
    └── images/*.png            # Visualization
```

#### `neural_renderer/` — Third-party
- Neural mesh renderer cho visualization
- Không cần đọc除非 bạn muốn hiểu rendering

#### `torch-mesh-isect/` — Third-party
- Mesh self-intersection detection (BVH-based)
- Dùng cho interpenetration penalty trong loss
- Không cần đọc除非 bạn muốn hiểu collision detection

---

## 5. Thứ tự đọc hiểu codebase

### Mức 1: Hiểu tổng quan (1-2 giờ)

```
Bước 1: Đọc README.md
  → Hiểu cài đặt, cấu trúc, cách chạy

Bước 2: Đọc run_dexavatar_wilor.py
  → Hiểu entry point, cách duyệt qua signs

Bước 3: Đọc Full_running_command_wilor.sh
  → Hiểu thứ tự 4 stages trong pipeline

Bước 4: Đọc docs/CODEBASE_GUIDE.md (file này)
  → Hiểu chức năng từng phần
```

### Mức 2: Hiểu data flow (2-3 giờ)

```
Bước 5: Đọc scripts/aggregate_sapiens.py
  → Hiểu cách gộp Sapiens outputs

Bước 6: Đọc M3_mean_shape_smplerx.py
  → Hiểu cách tính mean shape

Bước 7: Đọc dexavatar_fitting/smplifyx/data_parser.py
  → Hiểu cách load tất cả dữ liệu cho fitting
  → Đây là file QUAN TRỌNG để hiểu data flow

Bước 8: Đọc WiLoR/export_hamer_pkl.py
  → Hiểu bridge giữa WiLoR và pipeline
```

### Mức 3: Hiểu fitting engine (4-6 giờ)

```
Bước 9: Đọc dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml
  → Hiểu cấu hình fitting, các loss weights

Bước 10: Đọc dexavatar_fitting/smplifyx/main.py
  → Hiểu main loop: load models → iterate → fit

Bước 11: Đọc dexavatar_fitting/smplifyx/fit_single_frame.py
  → Hiểu single-frame optimization flow
  → Bắt đầu từ hàm fit_single_frame()

Bước 12: Đọc dexavatar_fitting/smplifyx/fitting.py ★ QUAN TRỌNG NHẤT
  → Hiểu tất cả loss functions
  → Bắt đầu từ class SMPLifyLoss
  → Hiểu từng loss component và tại sao cần nó

Bước 13: Đọc dexavatar_fitting/rewrite_body_model.py
  → Hiểu modification: body_pose_fore vs body_pose_op
```

### Mức 4: Hiểu priors và advanced features (3-4 giờ)

```
Bước 14: Đọc dexavatar_fitting/smplifyx/signbposer/signbposer.py
  → Hiểu VAE architecture cho body pose prior

Bước 15: Đọc dexavatar_fitting/smplifyx/signhposer/signhposer/signhposer.py
  → Hiểu VAE architecture cho hand pose prior

Bước 16: Đọc dexavatar_fitting/smplifyx/prior.py
  → Hiểu cách tạo các priors (L2, angle, GMM)

Bước 17: Đọc dexavatar_fitting/smplifyx/ensemble_init.py
  → Hiểu ensemble initialization strategy

Bước 18: Đọc dexavatar_fitting/smplifyx/fit_temporal_window.py
  → Hiểu temporal sliding window fitting
```

### Mức 5: Hiểu external models (tùy nhu cầu)

```
Bước 19: SMPLer-X/main/script_smplerx.py → inference.py → SMPLer_X.py
  → Hiểu 3D body estimation

Bước 20: WiLoR/wilor/models/ (nếu muốn hiểu WiLoR architecture)

Bước 21: hamer/hamer/models/ (nếu muốn hiểu HaMeR architecture)

Bước 22: sapiens/lite/ (nếu muốn hiểu Sapiens)
```

### Tóm tắt thứ tự ưu tiên:

```
Ưu tiên cao (PHẢI đọc):
  1. run_dexavatar_wilor.py + Full_running_command_wilor.sh  [5 phút]
  2. dexavatar_fitting/smplifyx/data_parser.py                [30 phút]
  3. dexavatar_fitting/smplifyx/main.py                       [30 phút]
  4. dexavatar_fitting/smplifyx/fit_single_frame.py           [1 giờ]
  5. dexavatar_fitting/smplifyx/fitting.py                    [2 giờ]

Ưu tiên trung bình (NÊN đọc):
  6. dexavatar_fitting/rewrite_body_model.py                  [30 phút]
  7. dexavatar_fitting/smplifyx/signbposer/signbposer.py      [30 phút]
  8. dexavatar_fitting/smplifyx/signhposer/signhposer/signhposer.py [30 phút]
  9. dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml      [15 phút]

Ưu tiên thấp (TÙY chọn):
  10. SMPLer-X/main/script_smplerx.py                         [1 giờ]
  11. WiLoR/export_hamer_pkl.py                               [30 phút]
  12. evaluation_trv2v_wilor.py                               [30 phút]
```

---

## 6. Bảng tóm tắt Method Variants

| Method | Config File | Entry Point | Feature đặc biệt |
|--------|-------------|-------------|------------------|
| **Baseline** | `fit_smplx_vposer_x.yaml` | `run_dexavatar_wilor.py` | SignBPoser + SignHPoser, 3-stage optimization |
| **M1 Temporal** | `fit_smplx_vposer_x_temporal.yaml` | `methods/run_dexavatar_wilor_temporal.py` | Sliding window (size=15), velocity/acceleration/jerk penalties |
| **M2 Hand2D** | `fit_smplx_vposer_x_hand2d.yaml` | `methods/run_dexavatar_wilor_hand2d.py` | 2D hand keypoint supervision từ WiLoR |
| **M4 Ensemble** | `fit_smplx_vposer_x_ensemble.yaml` | `methods/run_dexavatar_wilor_ensemble.py` | Multi-model ensemble SMPL-X initialization |
| **M5 Biomech** | `fit_smplx_vposer_x_biomech.yaml` | `methods/run_dexavatar_wilor_biomech.py` | Hand contact + body contact + finger prior |

---

## 7. Môi trường Conda

| Tên env | Python | Dùng cho | Packages chính |
|---------|--------|----------|----------------|
| `dexavatar` | 3.10 | Fitting engine, evaluation | smplx, mmcv, mmhuman3d, torch, pyrender |
| `sapiens_fix` | 3.9 | Sapiens pose estimation | torch, torchvision |
| `smpler_x` | 3.8 | SMPLer-X body estimation | mmcv, mmdet, torch |

**Lưu ý:** Mỗi stage trong pipeline cần activate đúng env. Các script `config*.sh` xử lý việc này.

---

## Appendix: Glossary

| Thuật ngữ | Ý nghĩa |
|-----------|----------|
| **SMPL-X** | Skinned Multi-Person Linear Model - eXpressive. Body model parametric đại diện cơ thể 3D |
| **SMPLify-X** | Algorithm tối ưu hóa SMPL-X params để khớp với observations |
| **SignBPoser** | Sign Body Pose Prior. VAE học pose distribution cơ thể khi ký hiệu |
| **SignHPoser** | Sign Hand Pose Prior. VAE học pose distribution tay khi ký hiệu |
| **MANO** | Hand body model. Đại diện mesh 3D của bàn tay |
| **TR-V2V** | Translation-Removed Vertex-to-Vertex. Metric đánh giá (mm) |
| **L-BFGS-LS** | Limited-memory BFGS with Line Search. Optimization algorithm |
| **VAE** | Variational AutoEncoder. Kiến trúc generative model |
| **2D Reprojection** | Project 3D joints lên 2D image plane, so sánh với detected keypoints |
| **Interpenetration** | Mesh self-intersection (tay xuyên qua cơ thể) |
| **Biomechanics** | Ràng buộc sinh học cơ học (joint angle limits, contact constraints) |
| **Sapiens** | Facebook's person detection + whole-body pose estimation |
| **SMPLer-X** | 3D human shape/pose estimation từ single image |
| **WiLoR** | Wide-range Lightweight Renderer. Hand mesh recovery model |
| **HaMeR** | Hand Mesh Recovery. Hand mesh recovery model (version cũ) |
