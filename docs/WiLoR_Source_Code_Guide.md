# WiLoR Source Code - Hướng Dẫn Đọc Toàn Bộ Codebase

> **WiLoR**: "End-to-end 3D Hand Localization and Reconstruction in-the-wild" — CVPR 2025
> Tác giả: Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, Stefanos Zafeiriou
> Viện: Imperial College London, Shanghai Jiao Tong University

---

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Cấu Trúc Thư Mục](#2-cấu-trúc-thư-mục)
3. [Thứ Tự Đọc Hiểu Source Code (Recommended)](#3-thứ-tự-đọc-hiểu-source-code)
4. [Chi Từng File - Vai Trò, Input/Output](#4-chi-tiết-từng-file)
5. [Kiến Trúc Mạng (Model Architecture)](#5-kiến-trúc-mạng)
6. [Data Flow End-to-End](#6-data-flow-end-to-end)
7. [Cấu Hình Hệ Thống](#7-cấu-hình-hệ-thống)
8. [Dependencies Và External Resources](#8-dependencies)
9. [Bảng Kích Thước Tensor Tổng Hợp](#9-bảng-kích-thước-tensor)
10. [Integration với DexAvatar](#10-integration-với-dexavatar)

---

## 1. Tổng Quan Dự Án

WiLoR là hệ thống **phát hiện và tái tạo 3D bàn tay** từ ảnh RGB đơn. Pipeline gồm 3 giai đoạn:

```
Ảnh RGB → [YOLO Detector] → Bounding Box + Left/Right
         → [ViT Backbone]  → MANO params ban đầu + features
         → [RefineNet]     → MANO params tinh chỉnh
         → [MANO Model]    → 3D mesh vertices + joints
         → [Projection]    → 2D keypoints + render
```

### Vai trò trong DexAvatar
WiLoR đóng vai trò **hand pose estimator** — cung cấp pose/shape bàn tay cho hệ thống avatar animation. Script `export_hamer_pkl.py` chuyển đổi output WiLoR sang format HaMeR-compatible để tích hợp vào pipeline DexAvatar.

---

## 2. Cấu Trúc Thư Mục

```
/home/haipd/DexAvatar/WiLoR/
│
├── README.md                              # Documentation dự án
├── requirements.txt                       # Python dependencies
├── license.txt                            # CC-BY-NC-ND 4.0
│
├── demo.py                                # ★ Entry point: CLI demo
├── gradio_demo.py                         # ★ Entry point: Web demo (Gradio)
├── export_hamer_pkl.py                    # ★ Custom: Export sang HaMeR format
├── verify_is_right.py                     # ★ Custom: Verify YOLO left/right
├── download_videos.py                     # Utility: Download WHIM videos
│
├── mano_data/
│   └── mano_mean_params.npz               # MANO mean pose/shape/camera params
│
├── pretrained_models/
│   ├── model_config.yaml                  # ★ Model architecture config
│   ├── dataset_config.yaml                # ★ Training dataset config
│   ├── detector.pt                        # YOLO hand detector weights
│   └── wilor_final.ckpt                   # WiLoR model checkpoint
│
├── demo_img/                              # Sample input images (8 ảnh)
├── demo_out/                              # Demo output (rendered + OBJ meshes)
├── whim/                                  # WHIM dataset metadata
│
└── wilor/                                 # ★ Main Python package
    ├── configs/
    │   └── __init__.py                    # Config loading (YACS CfgNode)
    │
    ├── datasets/
    │   ├── vitdet_dataset.py              # Inference dataset (hand crops)
    │   └── utils.py                       # Augmentation, cropping, transforms
    │
    ├── models/
    │   ├── __init__.py                    # Exports + load_wilor()
    │   ├── wilor.py                       # ★ Main model (PyTorch Lightning)
    │   ├── mano_wrapper.py                # MANO hand model wrapper
    │   ├── losses.py                      # Loss functions
    │   ├── discriminator.py               # Adversarial discriminator
    │   ├── backbones/
    │   │   ├── __init__.py                # Backbone factory
    │   │   └── vit.py                     # ★ Vision Transformer backbone
    │   └── heads/
    │       ├── __init__.py
    │       └── refinement_net.py          # ★ RefineNet head
    │
    └── utils/
        ├── __init__.py                    # Exports renderers + recursive_to()
        ├── geometry.py                    # Rotation math + projection
        ├── renderer.py                    # Full mesh renderer (pyrender)
        ├── mesh_renderer.py               # Training mesh visualizer
        ├── skeleton_renderer.py           # Keypoint visualizer
        ├── render_openpose.py             # OpenPose-style rendering
        ├── pose_utils.py                  # Evaluation metrics (MPJPE, RE, PCK)
        ├── misc.py                        # Hydra/Lightning utilities
        ├── pylogger.py                    # Rank-zero logger
        └── rich_utils.py                  # Rich terminal output
```

---

## 3. Thứ Tự Đọc Hiểu Source Code

### Mức 1: Hiểu tổng thể (đọc theo thứ tự này)

| Thứ tự | File | Thời gian | Lý do |
|--------|------|-----------|-------|
| 1 | `README.md` | 10 min | Hiểu dự án làm gì, cách cài đặt |
| 2 | `demo.py` | 15 min | Entry point — thấy toàn bộ pipeline inference |
| 3 | `pretrained_models/model_config.yaml` | 10 min | Hiểu cấu hình model đầy đủ |
| 4 | `wilor/models/__init__.py` | 5 min | `load_wilor()` — cách load model |

### Mức 2: Hiểu kiến trúc model (core)

| Thứ tự | File | Thời gian | Lý do |
|--------|------|-----------|-------|
| 5 | `wilor/models/wilor.py` | 30 min | ★★★ MAIN MODEL — toàn bộ forward pass |
| 6 | `wilor/models/backbones/vit.py` | 40 min | ★★★ ViT backbone — kiến trúc transformer |
| 7 | `wilor/models/heads/refinement_net.py` | 25 min | ★★ RefineNet — refinement head |
| 8 | `wilor/models/mano_wrapper.py` | 15 min | MANO hand model — output format |

### Mức 3: Hiểu data pipeline

| Thứ tự | File | Thời gian | Lý do |
|--------|------|-----------|-------|
| 9 | `wilor/datasets/vitdet_dataset.py` | 20 min | Dataset inference — preprocessing |
| 10 | `wilor/datasets/utils.py` | 45 min | Augmentation, cropping, transforms |
| 11 | `wilor/configs/__init__.py` | 15 min | Config system (YACS) |

### Mức 4: Hiểu loss và training

| Thứ tự | File | Thời gian | Lý do |
|--------|------|-----------|-------|
| 12 | `wilor/models/losses.py` | 15 min | Loss functions (2D, 3D, param) |
| 13 | `wilor/models/discriminator.py` | 15 min | Adversarial training |

### Mức 5: Utilities và rendering

| Thứ tự | File | Thời gian | Lý do |
|--------|------|-----------|-------|
| 14 | `wilor/utils/geometry.py` | 15 min | Rotation conversions + projection |
| 15 | `wilor/utils/renderer.py` | 30 min | Mesh rendering (pyrender) |
| 16 | `wilor/utils/pose_utils.py` | 20 min | Evaluation metrics |
| 17 | `wilor/utils/mesh_renderer.py` | 10 min | Training visualization |
| 18 | `wilor/utils/skeleton_renderer.py` | 10 min | Keypoint visualization |
| 19 | `wilor/utils/render_openpose.py` | 10 min | OpenPose drawing |
| 20 | `wilor/utils/misc.py` | 10 min | Hydra utilities |
| 21 | `wilor/utils/pylogger.py` | 2 min | Logger |
| 22 | `wilor/utils/rich_utils.py` | 5 min | Rich output |

### Mức 6: Custom scripts (DexAvatar-specific)

| Thứ tự | File | Thời gian | Lý do |
|--------|------|-----------|-------|
| 23 | `export_hamer_pkl.py` | 20 min | ★ Bridge WiLoR → DexAvatar |
| 24 | `verify_is_right.py` | 5 min | Debug utility |
| 25 | `gradio_demo.py` | 10 min | Web UI |
| 26 | `download_videos.py` | 5 min | Data utility |

---

## 4. Chi Tiết Từng File

### 4.1 Entry Points

---

#### `demo.py` — CLI Demo Entry Point

**Vai trò**: Điểm vào chính để chạy inference trên folder ảnh.

**Tham số dòng lệnh**:
| Argument | Default | Mô tả |
|----------|---------|-------|
| `--img_folder` | `demo_img/` | Thư mục ảnh đầu vào |
| `--out_folder` | `demo_out/` | Thư mục output |
| `--save_mesh` | `False` | Lưu OBJ mesh files |
| `--rescale_factor` | `2.5` | Hệ số phóng to bounding box |
| `--file_type` | `jpg,png,jpeg` | Loại file ảnh |
| `--fast` | `False` | FP16 + skip blocks → 1.6x faster |

**Pipeline bên trong**:
```python
# 1. Load models
model, model_cfg = load_wilor('pretrained_models/wilor_final.ckpt',
                               'pretrained_models/model_config.yaml')
detector = YOLO('pretrained_models/detector.pt')

# 2. Với mỗi ảnh:
for img_path in image_paths:
    # 2a. Detect hands
    det_results = detector(img, conf=0.3)
    boxes = det_results[0].boxes.xyxy.cpu().numpy()       # (N, 4)
    is_right = det_results[0].boxes.cls.cpu().numpy()      # (N,) 0=left, 1=right

    # 2b. Create dataset (crop + preprocess)
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, is_right, ...)

    # 3. Forward pass
    batch = {k: v.to(device) for k, v in batch.items()}
    output = model.forward_step(batch)

    # 4. Render results
    # - Overlay mesh lên ảnh gốc
    # - Lưu OBJ mesh (nếu --save_mesh)
```

**`--fast` mode**:
- `torch.float16` precision
- `torch.compile(model.backbone, mode='max-autotune')`
- `skip_blocks=True` → bỏ qua 11/32 ViT blocks: [25,27,26,23,24,29,22,13,14,15,20]

---

#### `gradio_demo.py` — Web Demo

**Vai trò**: Gradio web interface cho demo tương tác.

**Input**: Ảnh upload + confidence threshold slider (0.0–1.0)
**Output**: Ảnh với mesh overlay

---

#### `export_hamer_pkl.py` — Custom: HaMeR Format Exporter

**Vai trò**: ★ Quan trọng cho DexAvatar — chuyển đổi output WiLoR sang format HaMeR-compatible.

**Tham số**:
| Argument | Default | Mô tả |
|----------|---------|-------|
| `--img_folder` | required | Thư mục ảnh |
| `--out_folder` | required | Thư mục output |
| `--rescale_factor` | `2.5` | Box rescale |

**Output**:
- `{image_name}/hamer.pkl` — Format HaMeR-compatible
- `{image_name}/wilor.pkl` — Raw WiLoR format

**`hamer.pkl` format** (per image):
```python
{
    'pred_keypoints_2d': np.array,     # (N, 21, 2) — 2D keypoints
    'pred_keypoints_3d': np.array,     # (N, 21, 3) — 3D keypoints
    'hand_pose': np.array,             # (N, 15, 3, 3) — rotation matrices
    'box_center': np.array,            # (N, 2)
    'box_size': np.array,              # (N,)
    'is_right': np.array,              # (N,) — boolean
    'cam_t': np.array,                 # (N, 3) — camera translation
}
```

**`wilor.pkl` format** (per image):
```python
{
    'right' or 'left': {               # key là 'right' hoặc 'left'
        'hand_pose': np.array,         # (1, 15, 3) — axis-angle
        'betas': np.array,             # (1, 10)
        'global_orient': np.array,     # (1, 1, 3) — axis-angle
        'pred_keypoints_2d': np.array, # (1, 21, 2)
        'pred_keypoints_3d': np.array, # (1, 21, 3)
        'pred_vertices': np.array,     # (1, 778, 3)
        'box_center': np.array,        # (1, 2)
        'box_size': np.array,          # (1,)
        'is_right': np.array,          # (1,)
        'cam_t': np.array,             # (1, 3)
    }
}
```

**Hàm quan trọng**: `rotmat_to_axis_angle_batch(rot_matrices)`
- Input: `(N, 3, 3)` rotation matrices
- Output: `(N, 3)` axis-angle vectors
- Sử dụng Rodrigues formula: θ = arccos((trace(R) - 1) / 2), axis = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] / (2sin(θ))

---

#### `verify_is_right.py` — Debug Utility

**Vai trò**: Kiểm tra YOLO detector class mapping (cls=0 → left, cls=1 → right).

**Cách dùng**: `python verify_is_right.py --img_path path/to/hand_image.jpg`

---

### 4.2 Config System

---

#### `wilor/configs/__init__.py` — Configuration System

**Vai trò**: Quản lý toàn bộ cấu hình dự án sử dụng YACS CfgNode.

**Hàm chính**:
- `default_config()` → trả về config mặc định
- `get_config(config_file, merge=True)` → load YAML, merge với defaults
- `dataset_config(name)` → load dataset YAML

**Cấu trúc config đầy đủ**:

```yaml
GENERAL:
  RESUME: True
  TIME_TO_RUN: 3300          # seconds
  VAL_STEPS: 100
  LOG_STEPS: 100
  CHECKPOINT_STEPS: 20000
  NUM_GPUS: 1
  NUM_WORKERS: 4
  MIXED_PRECISION: True       # FP16 training
  DISTRIBUTED: False

TRAIN:
  NUM_EPOCHS: 100
  BATCH_SIZE: 32
  SHUFFLE: True
  WARMUP: False
  CLIP_GRAD: False

LOSS_WEIGHTS:                  # Loaded from YAML
  KEYPOINTS_3D: 0.05
  KEYPOINTS_2D: 0.01
  GLOBAL_ORIENT: 0.001
  HAND_POSE: 0.001
  BETAS: 0.0005
  ADVERSARIAL: 0.0005

DATASETS:
  CONFIG:
    SCALE_FACTOR: 0.3         # Box scale jitter
    ROT_FACTOR: 30            # Max rotation (degrees)
    TRANS_FACTOR: 0.02        # Translation jitter
    COLOR_SCALE: 0.2          # Color jitter
    ROT_AUG_RATE: 0.6
    DO_FLIP: False
    FLIP_AUG_RATE: 0.5
    EXTREME_CROP_AUG_RATE: 0.10

MODEL:
  IMAGE_SIZE: 256
  BACKBONE:
    TYPE: 'vit'
  BBOX_SHAPE: [192, 256]

EXTRA:
  FOCAL_LENGTH: 5000
```

---

### 4.3 Models (Core)

---

#### `wilor/models/__init__.py` — Model Exports

**Vai trò**: Package entry point + convenience loading function.

**Hàm `load_wilor(checkpoint_path, cfg_path)`**:
```python
def load_wilor(checkpoint_path, cfg_path):
    model_cfg = get_config(cfg_path)
    # Adjust BBOX_SHAPE for ViT
    if model_cfg.MODEL.BACKBONE.TYPE == 'vit':
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    # Set MANO paths
    model_cfg.MANO.DATA_DIR = './mano_data/'
    model_cfg.MANO.MODEL_PATH = './mano_data/MANO_RIGHT.pkl'
    model_cfg.MANO.MEAN_PARAMS = './mano_data/mano_mean_params.npz'
    # Load from checkpoint
    model = WiLoR.load_from_checkpoint(checkpoint_path, cfg=model_cfg, init_renderer=False)
    return model, model_cfg
```

---

#### `wilor/models/wilor.py` — ★★★ MAIN MODEL

**Vai trò**: Model chính — PyTorch Lightning module kết hợp backbone + head + MANO + losses.

**Class**: `WiLoR(pl.LightningModule)`

**Constructor `__init__(cfg, init_renderer=True)`**:
```python
# Submodules
self.backbone = create_backbone(cfg)           # ViT or FastViT
self.backbone_mean = create_backbone(cfg)      # (same architecture, for mean params)
self.refine_head = RefineNet(cfg, feat_dim=1280, upscale=3)

# MANO model
self.mano = MANO(cfg)                          # smplx MANOLayer wrapper

# Losses
self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
self.parameter_loss = ParameterLoss()

# Optional: Discriminator (adversarial training)
if cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
    self.discriminator = Discriminator()

# Optional: Renderers (visualization)
if init_renderer:
    self.skeleton_renderer = SkeletonRenderer(cfg)
    self.mesh_renderer = MeshRenderer(cfg, self.mano.faces)
```

**Forward pass `forward_step(batch, train=False) → Dict`**:
```python
# Input
x = batch['img']           # (B, 3, 256, 256)

# 1. Center-crop width cho ViT: 256→192
x = x[:, :, :, 32:-32]     # (B, 3, 256, 192)

# 2. ViT backbone → initial params + features
temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(x)
# temp_mano_params: {global_orient: (B,1,3,3), hand_pose: (B,15,3,3), betas: (B,10)}
# pred_cam: (B, 3)
# pred_mano_feats: {hand_pose: (B,96), betas: (B,10), cam: (B,3)}
# vit_out: (B, 1280, H, W) spatial features

# 3. MANO forward → initial 3D vertices
temp_output = self.mano(**temp_mano_params)
temp_vertices = temp_output.vertices  # (B, 778, 3)

# 4. RefineNet → refined params
pred_mano_params, pred_cam = self.refine_head(
    vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length
)

# 5. MANO forward (refined) → final output
output = self.mano(**pred_mano_params)
# output.vertices: (B, 778, 3)
# output.joints: (B, 21+, 3)

# 6. Camera projection → 2D keypoints
focal_length = batch['focal_length']  # (B, 2)
camera_center = torch.zeros(B, 2)
pred_cam_t = torch.stack([
    pred_cam[:, 1], pred_cam[:, 2], 2 * focal_length[:, 0] / (pred_cam[:, 0] * 256 + 1e-9)
], dim=-1)  # (B, 3)

pred_keypoints_2d = perspective_projection(
    output.joints, translation=pred_cam_t, focal_length=focal_length
)  # (B, N, 2)

pred_keypoints_3d = output.joints  # (B, N, 3)
pred_vertices = output.vertices    # (B, 778, 3)

return {
    'pred_cam': pred_cam,              # (B, 3) — weak-perspective [scale, tx, ty]
    'pred_cam_t': pred_cam_t,          # (B, 3) — 3D camera translation
    'pred_mano_params': pred_mano_params,  # dict of refined params
    'pred_keypoints_3d': pred_keypoints_3d,
    'pred_vertices': pred_vertices,
    'pred_keypoints_2d': pred_keypoints_2d,
    'focal_length': focal_length,
}
```

**Loss computation `compute_loss(batch, output, train=True)`**:
```python
loss = (
    cfg.LOSS_WEIGHTS.KEYPOINTS_3D * keypoint_3d_loss(pred_3d, gt_3d)
  + cfg.LOSS_WEIGHTS.KEYPOINTS_2D * keypoint_2d_loss(pred_2d, gt_2d)
  + cfg.LOSS_WEIGHTS.GLOBAL_ORIENT * parameter_loss(pred_go, gt_go, has_go)
  + cfg.LOSS_WEIGHTS.HAND_POSE * parameter_loss(pred_hp, gt_hp, has_hp)
  + cfg.LOSS_WEIGHTS.BETAS * parameter_loss(pred_betas, gt_betas, has_betas)
)
```

**Training step** (manual optimization for adversarial training):
```python
def training_step(self, joint_batch, batch_idx):
    # joint_batch có 2 keys: 'img' (real images) và 'mocap' (motion capture data)
    output = self.forward_step(joint_batch['img'], train=True)
    loss = self.compute_loss(joint_batch['img'], output, train=True)

    # Adversarial loss (nếu có)
    if self.discriminator:
        disc_out = self.discriminator(output['pred_mano_params']['hand_pose'],
                                       output['pred_mano_params']['betas'])
        adv_loss = F.binary_cross_entropy_with_logits(disc_out, torch.ones_like(disc_out))
        loss += cfg.LOSS_WEIGHTS.ADVERSARIAL * adv_loss

    # Manual backward
    self.manual_backward(loss)
    if cfg.TRAIN.CLIP_GRAD:
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), cfg.TRAIN.CLIP_GRAD_VALUE)
    opt.step()

    # Train discriminator
    if self.discriminator:
        # Real samples from mocap
        real_disc_out = self.discriminator(real_poses, real_betas)
        # Fake samples from model output
        fake_disc_out = self.discriminator(pred_poses.detach(), pred_betas.detach())
        disc_loss = (
            F.binary_cross_entropy_with_logits(real_disc_out, torch.ones_like(real_disc_out))
          + F.binary_cross_entropy_with_logits(fake_disc_out, torch.zeros_like(fake_disc_out))
        ) / 2
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        disc_opt.step()
```

---

#### `wilor/models/backbones/vit.py` — ★★★ Vision Transformer Backbone

**Vai trò**: Backbone trích xuất features từ ảnh hand crop, đồng thời dự đoán MANO params ban đầu.

**Factory function**: `vit(cfg)`
```python
ViT(img_size=(256, 192), patch_size=16, embed_dim=1280,
    depth=32, num_heads=16, mlp_ratio=4, drop_path_rate=0.55)
```

**Các class con**:

| Class | Vai trò | Input → Output |
|-------|---------|----------------|
| `PatchEmbed` | Chia ảnh thành patch tokens | `(B, 3, 256, 192)` → `(B, 192, 1280)` |
| `Attention` | Multi-head self-attention | `(B, N, 1280)` → `(B, N, 1280)` |
| `Mlp` | Feed-forward network | `(B, N, 1280)` → `(B, N, 1280)` |
| `Block` | Transformer block (LN→Attn→LN→MLP) | `(B, N, 1280)` → `(B, N, 1280)` |
| `DropPath` | Stochastic depth regularization | identity or drop |
| `ViT` | ★ Main backbone class | `(B, 3, 256, 192)` → 4 outputs |

**Class `ViT` — chi tiết**:

**Learned tokens** (prepended vào patch sequence):
```python
self.pose_emb = Linear(joint_rep_dim, 1280)   # 16 pose tokens (1 joint/token)
self.shape_emb = Linear(10, 1280)              # 1 shape token
self.cam_emb = Linear(3, 1280)                 # 1 camera token
# Total: 18 extra tokens + 192 patch tokens = 210 tokens

# Decoders
self.decpose = Linear(1280, 6)     # 6D rotation per joint
self.decshape = Linear(1280, 10)   # 10 shape params
self.deccam = Linear(1280, 3)      # 3 camera params

# Registered buffers (from MANO mean params)
self.init_hand_pose    # (96,) — mean hand pose in 6D
self.init_betas        # (10,) — mean shape
self.init_cam          # (3,) — mean camera
```

**Forward `forward_features(x)`**:
```python
# Input: x (B, 3, 256, 192)

# 1. Patch embedding
x = self.patch_embed(x)              # (B, 192, 1280)

# 2. Prepend learned tokens
pose_tokens = self.pose_emb(init_pose)    # (B, 16, 1280)
shape_token = self.shape_emb(init_betas)  # (B, 1, 1280)
cam_token = self.cam_emb(init_cam)        # (B, 1, 1280)
x = torch.cat([pose_tokens, shape_token, cam_token, x], dim=1)  # (B, 210, 1280)

# 3. Add positional embedding
x = x + self.pos_embed[:, :210, :]

# 4. Pass through 32 Transformer blocks
for blk in self.blocks:
    x = blk(x)                       # (B, 210, 1280)

# 5. Split and decode
pose_feat = x[:, :16, :]             # (B, 16, 1280)
shape_feat = x[:, 16, :]             # (B, 1280)
cam_feat = x[:, 17, :]               # (B, 1280)
img_feat = x[:, 18:, :]              # (B, 192, 1280)

pred_hand_pose = self.decpose(pose_feat) + self.init_hand_pose  # (B, 16, 6)
pred_betas = self.decshape(shape_feat) + self.init_betas        # (B, 10)
pred_cam = self.deccam(cam_feat) + self.init_cam                # (B, 3)

# 6. Convert 6D rotation → 3x3 rotation matrix
pred_hand_pose = rot6d_to_rotmat(pred_hand_pose)  # (B, 16, 3, 3)

# 7. Reshape image features to spatial
img_feat = img_feat.transpose(1, 2).reshape(B, 1280, Hp, Wp)  # (B, 1280, 12, 16)

return pred_mano_params, pred_cam, pred_mano_feats, img_feat
```

**Skip blocks (fast mode)**:
```python
self.blocks_to_skip = [25, 27, 26, 23, 24, 29, 22, 13, 14, 15, 20]
# Khi skip_blocks=True: bỏ qua 11/32 blocks → 34% faster, slight accuracy drop
```

**Freezing support**:
- `_freeze_stages()` — freeze patch_embed, individual blocks, attention, hoặc FFN
- Dùng cho fine-tuning hoặc transfer learning

---

#### `wilor/models/heads/refinement_net.py` — ★★ RefineNet

**Vai trò**: Tinh chỉnh MANO params ban đầu bằng cách sample features tại vị trí 3D vertices.

**Class `DeConvNet`**: Multi-scale upsampling
```
Input features (B, 1280, H, W)
  → 1x1 conv → (B, 640, H, W)
  → DeConv 2x → (B, 320, 2H, 2W)
  → DeConv 4x → (B, 160, 4H, 4W)
  → DeConv 8x → (B, 80, 8H, 8W)   [unused, only 3 scales used]
```

**Class `RefineNet`**:
```python
# Multi-scale feature dimensions (after concat)
total_dim = 160 + 320 + 640 = 1120

# Decoders
dec_pose = Linear(1120, 96)    # 16 joints × 6D rotation
dec_shape = Linear(1120, 10)
dec_cam = Linear(1120, 3)
```

**Forward `forward(img_feat, verts_3d, pred_cam, pred_mano_feats, focal_length)`**:
```python
# Input:
#   img_feat: (B, 1280, H, W) — backbone spatial features
#   verts_3d: (B, 778, 3) — initial MANO vertices
#   pred_cam: (B, 3) — initial camera params
#   pred_mano_feats: {hand_pose (B,96), betas (B,10), cam (B,3)}
#   focal_length: (B, 2)

# 1. Upsample features multi-scale
multi_scale_feats = self.deconv(img_feat)
# Returns list: [(B, 640, 2H, 2W), (B, 320, 4H, 4W), (B, 160, 8H, 8W)]

# 2. Với mỗi scale:
vertex_features = []
for scale_idx, feats in enumerate(multi_scale_feats):
    # 2a. Compute camera translation at this scale
    cam_t = compute_cam_t(pred_cam, focal_length, scale_factor)

    # 2b. Project 3D vertices → 2D
    verts_2d = perspective_projection(verts_3d + cam_t, focal_length)
    # Normalize to [-1, 1] for grid_sample
    verts_2d_normalized = verts_2d / (img_size / 2) - 1  # (B, 778, 2)

    # 2c. Sample features at vertex locations
    sampled = F.grid_sample(feats, verts_2d_normalized)  # (B, C, 1, 778)
    sampled = sampled.squeeze(2).permute(0, 2, 1)        # (B, 778, C)

    # 2d. Max-pool across vertices
    pooled = sampled.max(dim=1).values                    # (B, C)
    vertex_features.append(pooled)

# 3. Concatenate multi-scale features
fused = torch.cat(vertex_features, dim=1)  # (B, 1120)

# 4. Decode residual deltas
delta_pose = self.dec_pose(fused)    # (B, 96)
delta_shape = self.dec_shape(fused)  # (B, 10)
delta_cam = self.dec_cam(fused)      # (B, 3)

# 5. Add deltas to initial predictions
refined_hand_pose = pred_mano_feats['hand_pose'] + delta_pose  # (B, 96)
refined_betas = pred_mano_feats['betas'] + delta_shape         # (B, 10)
refined_cam = pred_mano_feats['cam'] + delta_cam                # (B, 3)

# 6. Convert 6D → rotation matrices
refined_hand_pose = rot6d_to_rotmat(refined_hand_pose.reshape(-1, 6)).reshape(B, 16, 3, 3)

return refined_mano_params, refined_cam
```

---

#### `wilor/models/mano_wrapper.py` — MANO Hand Model

**Vai trò**: Wrapper around `smplx.MANOLayer` với OpenPose keypoint mapping.

**Class `MANO(smplx.MANOLayer)`**:

**MANO → OpenPose joint mapping** (21 joints):
```python
mano_to_openpose = [
    0,   # Wrist
    5, 6, 7, 8,    # Index finger (MCP, PIP, DIP, TIP)
    9, 10, 11, 12, # Middle finger
    13, 14, 15, 16,# Ring finger
    17, 18, 19, 20,# Pinky finger
    1, 2, 3, 4,    # Thumb
]
```

**Forward `forward(**kwargs) → MANOOutput`**:
```python
# Input kwargs:
#   global_orient: (B, 1, 3, 3) — wrist orientation
#   hand_pose: (B, 15, 3, 3) — 15 finger joint rotations
#   betas: (B, 10) — shape parameters

output = super().forward(**kwargs)
# output.vertices: (B, 778, 3) — mesh vertices
# output.joints: (B, 21, 3) — joint positions

# Reorder to OpenPose convention
output.joints = output.joints[:, self.mano_to_openpose, :]

return output
```

---

#### `wilor/models/losses.py` — Loss Functions

| Loss Class | Input Shape | Mô tả |
|------------|-------------|-------|
| `Keypoint2DLoss` | pred: `(B,S,N,2)`, gt: `(B,S,N,3)` | Confidence-weighted L1/L2 loss |
| `Keypoint3DLoss` | pred: `(B,S,N,3)`, gt: `(B,S,N,4)` | Centered at pelvis, L1/L2 loss |
| `ParameterLoss` | pred: `(*)`, gt: `(*)`, has: `(*)` | MSE with validity mask |

**Chi tiết `Keypoint2DLoss`**:
```python
def forward(self, pred_keypoints_2d, gt_keypoints_2d):
    # gt_keypoints_2d[:, :, :, 2] là confidence
    conf = gt_keypoints_2d[:, :, :, 2:3]  # (B, S, N, 1)
    loss = F.l1_loss(pred_keypoints_2d, gt_keypoints_2d[:, :, :, :2], reduction='none')
    loss = (loss * conf).sum()  # Weighted by confidence
    return loss
```

**Chi tiết `Keypoint3DLoss`**:
```python
def forward(self, pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0):
    # Center at pelvis
    pred = pred_keypoints_3d - pred_keypoints_3d[:, :, pelvis_id:pelvis_id+1, :]
    gt = gt_keypoints_3d[:, :, :, :3] - gt_keypoints_3d[:, :, pelvis_id:pelvis_id+1, :3]
    conf = gt_keypoints_3d[:, :, :, 3:4]
    loss = F.l1_loss(pred, gt, reduction='none')
    loss = (loss * conf).sum()
    return loss
```

---

#### `wilor/models/discriminator.py` — Adversarial Discriminator

**Vai trò**: Discriminator cho adversarial training (từ HMR).

**Architecture**:
```
Pose branch (per-joint):
  (B, 15, 3, 3) → reshape (B, 15, 1, 9) → permute (B, 9, 15, 1)
  → Conv2d(9→32, 1x1) → ReLU → Conv2d(32→32, 1x1) → ReLU
  → 15 separate Linear(32→1) heads → (B, 15)

Betas branch:
  (B, 10) → Linear(10→10) → ReLU → Linear(10→5) → ReLU → Linear(5→1) → (B, 1)

Joint-pose branch:
  flatten conv features (B, 32*15) → Linear(480→1024) → ReLU → Linear(1024→1024) → ReLU → Linear(1024→1) → (B, 1)

Output: concat → (B, 17) [15 pose + 1 betas + 1 joint-pose]
```

---

### 4.4 Datasets

---

#### `wilor/datasets/vitdet_dataset.py` — Inference Dataset

**Vai trò**: Dataset cho inference — crop và preprocess hand regions từ ảnh gốc.

**Class `ViTDetDataset`**: Chỉ dùng cho inference (`train=False`).

**Constructor**:
```python
ViTDetDataset(cfg, img_cv2, boxes, right, rescale_factor=2.5, train=False, fp16=False)
# img_cv2: (H, W, 3) — BGR image
# boxes: (N, 4) — [x1, y1, x2, y2]
# right: (N,) — 1=right, 0=left
```

**`__getitem__(idx)` output**:
```python
{
    'img': torch.Tensor,       # (3, 256, 256) — normalized
    'personid': int,           # detection index
    'box_center': np.array,    # (2,) — original box center
    'box_size': float,         # crop size after aspect ratio expansion
    'img_size': np.array,      # (2,) — [W, H] original image
    'right': float,            # 1.0 or 0.0
}
```

**Processing pipeline**:
1. Compute center + scale from box
2. Expand to target aspect ratio (`cfg.MODEL.BBOX_SHAPE = [192, 256]`)
3. Anti-aliasing Gaussian blur (nếu downscale > 1.1×)
4. `generate_image_patch_cv2()` → crop + affine warp
5. **Flip left hands** (`flip = (right == 0)`)
6. BGR → RGB, HWC → CHW
7. Normalize: `(pixel - IMAGE_MEAN) / IMAGE_STD`
8. Optional FP16 cast

---

#### `wilor/datasets/utils.py` — Augmentation & Transforms (995 lines)

**Vai trò**: Toàn bộ augmentation, cropping, keypoint processing cho training.

**Hàm chính**:

| Hàm | Input | Output | Mô tả |
|-----|-------|--------|-------|
| `do_augmentation(aug_config)` | config dict | 8 augmentation params | Random scale, rot, flip, color, trans |
| `get_example(...)` | image + annotations | processed data | ★ Main training data pipeline |
| `generate_image_patch_cv2(...)` | image + box params | crop + affine matrix | Core cropping function |
| `gen_trans_from_patch_cv(...)` | box params | 2×3 affine matrix | Build affine transform |
| `fliplr_keypoints(...)` | joints + permutation | flipped joints | Left-right flip |
| `fliplr_params(...)` | mano params | flipped params | Flip MANO for L/R swap |
| `rot_aa(aa, rot)` | axis-angle + degrees | rotated axis-angle | Rotate around Z-axis |
| `expand_to_aspect_ratio(...)` | shape + target | expanded shape | Aspect ratio normalization |

**`get_example()` pipeline chi tiết**:
```python
def get_example(img_path, center_x, center_y, width, height,
                keypoints_2d, keypoints_3d, mano_params,
                flip_permutation, patch_height, patch_width,
                mean, std, is_right, augment=True, ...):

    # 1. Load image
    img = cv2.imread(img_path)  # BGR

    # 2. Augmentation params
    if augment:
        scale, rot, do_flip, extreme_crop, color_scale, tx, ty = do_augmentation(aug_config)
    else:
        scale, rot, do_flip = 1.0, 0.0, False
        color_scale, tx, ty = [1,1,1], 0.0, 0.0

    # 3. Left hands always flip
    if not is_right:
        do_flip = True

    # 4. Optional extreme cropping
    if extreme_crop:
        center_x, center_y, width, height = extreme_cropping(keypoints_2d, ...)

    # 5. Translation jitter
    center_x += tx * width
    center_y += ty * height

    # 6. Process 3D keypoints
    keypoints_3d = keypoint_3d_processing(keypoints_3d, flip_perm, rot, do_flip)

    # 7. Generate image patch (crop + affine warp)
    img_patch, trans = generate_image_patch_cv2(
        img, center_x, center_y, width, height,
        patch_width, patch_height, do_flip, scale, rot
    )

    # 8. BGR → RGB, HWC → CHW
    img_patch = img_patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)

    # 9. Process MANO params
    mano_params = mano_param_processing(mano_params, rot, do_flip)

    # 10. Color scale + normalize
    for i in range(3):
        img_patch[i] = np.clip(img_patch[i] * color_scale[i], 0, 255)
        img_patch[i] = (img_patch[i] - mean[i]) / std[i]

    # 11. Transform 2D keypoints
    if do_flip:
        keypoints_2d[:, 0] = width - keypoints_2d[:, 0]  # Flip x
    for i in range(len(keypoints_2d)):
        keypoints_2d[i, :2] = trans_point2d(keypoints_2d[i, :2], trans)
    keypoints_2d[:, 0:2] = keypoints_2d[:, 0:2] / patch_width - 0.5  # Normalize to [-0.5, 0.5]

    return img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size
```

---

### 4.5 Utilities

---

#### `wilor/utils/geometry.py` — Rotation & Projection Math

**Vai trò**: Các phép toán hình học cơ bản, tất cả đều differentiable (PyTorch).

| Hàm | Input Shape | Output Shape | Mô tả |
|-----|-------------|--------------|-------|
| `aa_to_rotmat(theta)` | `(B, 3)` | `(B, 3, 3)` | Axis-angle → rotation matrix (via quaternion) |
| `quat_to_rotmat(quat)` | `(B, 4)` | `(B, 3, 3)` | Quaternion (w,x,y,z) → rotation matrix |
| `rot6d_to_rotmat(x)` | `(B, 6)` | `(B, 3, 3)` | 6D representation → rotation matrix (Zhou et al.) |
| `perspective_projection(points, translation, focal_length, camera_center, rotation)` | `(B, N, 3)` | `(B, N, 2)` | 3D → 2D pinhole projection |

**`perspective_projection` chi tiết**:
```python
def perspective_projection(points, translation, focal_length, camera_center, rotation=None):
    # points: (B, N, 3)
    # translation: (B, 3)
    # focal_length: (B, 2)

    if rotation is not None:
        points = torch.einsum('bij,bnj->bni', rotation, points)

    points = points + translation.unsqueeze(1)

    # Perspective division
    points_2d = points[:, :, :2] / points[:, :, 2:3]

    # Apply intrinsics
    K = torch.zeros(B, 3, 3)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 0, 2] = camera_center[:, 0]
    K[:, 1, 2] = camera_center[:, 1]
    K[:, 2, 2] = 1.0

    points_2d = torch.einsum('bij,bnj->bni', K, torch.cat([points_2d, torch.ones_like(points_2d[:,:,:1])], dim=-1))

    return points_2d[:, :, :2]
```

---

#### `wilor/utils/renderer.py` — Full Mesh Renderer (424 lines)

**Vai trò**: Render MANO mesh lên ảnh sử dụng pyrender (EGL headless).

**Class `Renderer`**:

**Constructor**: `Renderer(cfg, faces)`
- Thêm 14 extra faces để mesh watertight (hở ở cổ tay)
- Lưu cả left-hand faces (đảo winding order: `faces[:, [0,2,1]]`)

**Hàm chính**:

| Hàm | Mô tả |
|-----|-------|
| `__call__(vertices, cam_t, image, ...)` | Main entry — render mesh overlay |
| `render_rgba(vertices, cam_t, ...)` | Render RGBA image |
| `render_rgba_multiple(vertices, cam_t, ...)` | Multi-hand render |
| `vertices_to_trimesh(vertices, ...)` | Convert to trimesh object |

**Hỗ trợ functions**:
```python
cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length)
# Convert weak-perspective cam → full-frame 3D translation

rotx(theta), roty(theta), rotz(theta)
# Basic 3×3 rotation matrices

make_rotation(rx, ry, rz, order='xyz')
# Composed Euler rotation (all 6 orderings)

make_4x4_pose(R, t)
# Rigid body transform → 4×4 homogeneous matrix
```

---

#### `wilor/utils/mesh_renderer.py` — Training Mesh Visualizer

**Vai trò**: Render mesh cho TensorBoard visualization trong training.

**Class `MeshRenderer`**:

**`visualize_tensorboard()`**: Tạo grid 5 columns:
1. Original image
2. Front-view rendered mesh
3. Side-view rendered mesh (90° Y rotation)
4. Predicted 2D keypoints (OpenPose style)
5. Ground truth 2D keypoints

---

#### `wilor/utils/skeleton_renderer.py` — Keypoint Visualizer

**Vai trò**: Lightweight keypoint-only renderer (nhanh hơn mesh renderer).

**Class `SkeletonRenderer`**:

**`__call__()` output**: Grid 5 panels per sample:
1. Input image + GT 2D keypoints
2. Image + projected GT 3D keypoints
3. Image + projected predicted 3D keypoints
4. Side view GT 3D keypoints
5. Side view predicted 3D keypoints

---

#### `wilor/utils/render_openpose.py` — OpenPose Drawing

**Vai trò**: Vẽ keypoints theo style OpenPose.

**`render_hand_keypoints(img, keypoints, ...)`**:
- 21 hand keypoints với 20 bone connections
- Màu theo ngón tay: gray (wrist/thumb), red (index), yellow (middle), green (ring), blue (pinky), magenta (palm)
- Thickness tỷ lệ với bounding box size

---

#### `wilor/utils/pose_utils.py` — Evaluation Metrics (352 lines)

**Vai trò**: Metrics đánh giá accuracy của 3D hand pose.

| Hàm/Class | Input | Output | Mô tả |
|-----------|-------|--------|-------|
| `eval_pose(pred, gt)` | `(B,N,3)` | `(mpjpe_mm, re_mm)` | MPJPE + Reconstruction Error |
| `compute_similarity_transform(S1, S2)` | `(B,N,3)` | `(B,N,3)` | Procrustes alignment (SVD) |
| `Evaluator` | batches | accumulated metrics | Full evaluation pipeline |
| `EvaluatorPCK` | batches | PCK@thresholds | Percentage of Correct Keypoints |

**Metrics**:
- **MPJPE** (Mean Per-Joint Position Error): Euclidean distance sau khi align về origin
- **RE** (Reconstruction Error): Euclidean distance sau Procrustes alignment (similarity transform)
- **PCK** (Percentage of Correct Keypoints): % keypoints within threshold

---

#### `wilor/utils/misc.py` — Hydra/Lightning Utilities (204 lines)

**Vai trò**: Training infrastructure — logging, callbacks, config management.

| Hàm | Mô tả |
|-----|-------|
| `task_wrapper(func)` | Decorator cho Hydra task — logging, error handling |
| `extras(cfg)` | Pre-task setup — disable warnings, print config tree |
| `instantiate_callbacks(cfg)` | Tạo Lightning Callbacks từ config |
| `instantiate_loggers(cfg)` | Tạo Lightning Loggers (TensorBoard, WandB) |
| `log_hyperparameters(obj)` | Log config + param counts |
| `close_loggers()` | Đóng tất cả loggers |

---

#### `wilor/utils/pylogger.py` — Rank-Zero Logger

**Vai trò**: Logger chỉ log trên main process (tránh duplicate trong multi-GPU).

```python
def get_pylogger(name):
    logger = logging.getLogger(name)
    for level in ['debug', 'info', 'warning', 'error', 'exception', 'fatal', 'critical']:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger
```

---

## 5. Kiến Trúc Mạng

### Tổng quan 3-stage pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: YOLO Detection                                         │
│   Input:  Raw image (H, W, 3)                                   │
│   Output: Bounding boxes (N, 4) + is_right (N,)                 │
│   Model:  YOLOv8 (detector.pt)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: ViT Backbone                                           │
│   Input:  Hand crop (B, 3, 256, 192)                            │
│   Output: Initial MANO params + spatial features                │
│   Architecture:                                                 │
│     - PatchEmbed: Conv2d(3→1280, k=16, s=16) → 192 patches     │
│     - 18 learned tokens (16 pose + 1 shape + 1 cam)            │
│     - 32 Transformer Blocks (embed_dim=1280, heads=16)          │
│     - Decoders: Linear(1280→6) per joint, Linear(1280→10/3)    │
│   Params: ~100M                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: RefineNet                                               │
│   Input:  ViT features (B, 1280, H, W) + initial vertices       │
│   Output: Refined MANO params (residual deltas)                 │
│   Architecture:                                                 │
│     - DeConvNet: Multi-scale upsampling (2x, 4x)               │
│     - Vertex feature sampling via grid_sample                   │
│     - Max-pool across vertices → (B, 1120)                      │
│     - Decoders: pose (1120→96), shape (1120→10), cam (1120→3)  │
│   Params: ~5M                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ MANO Model                                                       │
│   Input:  global_orient (B,1,3,3), hand_pose (B,15,3,3),        │
│           betas (B,10)                                           │
│   Output: vertices (B, 778, 3), joints (B, 21, 3)               │
│   Source: smplx.MANOLayer + OpenPose mapping                    │
└─────────────────────────────────────────────────────────────────┘
```

### ViT Architecture Detail

```
Input: (B, 3, 256, 192)
    │
    ▼ PatchEmbed (Conv2d k=16 s=16)
Tokens: (B, 192, 1280)     ← 16×12 patches
    │
    ├─ Prepend: pose_tokens (B, 16, 1280)
    ├─ Prepend: shape_token (B, 1, 1280)
    └─ Prepend: cam_token (B, 1, 1280)
    │
    ▼ Concat
Tokens: (B, 210, 1280)
    │
    ├─ + positional_embed (1, 210, 1280) [learned]
    │
    ▼ 32× Transformer Block
       ┌──────────────────────────────┐
       │  LayerNorm (1280)            │
       │  Multi-Head Attention (16h)  │  ← scaled_dot_product_attention
       │  + DropPath (p=0.55)         │
       │  + Residual                  │
       │  LayerNorm (1280)            │
       │  MLP (1280→5120→1280, GELU)  │
       │  + DropPath (p=0.55)         │
       │  + Residual                  │
       └──────────────────────────────┘
    │
    ▼ Split output
pose_feat  = x[:, :16, :]     → decpose(pose_feat) + init_pose   → (B, 16, 6) → rot6d_to_rotmat → (B, 16, 3, 3)
shape_feat = x[:, 16, :]      → decshape(shape_feat) + init_betas → (B, 10)
cam_feat   = x[:, 17, :]      → deccam(cam_feat) + init_cam       → (B, 3)
img_feat   = x[:, 18:, :]     → reshape → (B, 1280, 12, 16)
```

### RefineNet Architecture Detail

```
ViT features: (B, 1280, 12, 16)
    │
    ▼ DeConvNet
    ├─ 1×1 Conv: (B, 640, 12, 16)
    ├─ DeConv 2×: (B, 320, 24, 32)
    └─ DeConv 4×: (B, 160, 48, 64)
    │
    ▼ For each scale (3 scales):
    │   ├─ Project vertices → 2D (perspective projection)
    │   ├─ Normalize to [-1, 1]
    │   ├─ grid_sample features at vertex locations
    │   └─ Max-pool across 778 vertices → (B, C)
    │
    ▼ Concatenate all scales
fused: (B, 1120)    ← 640 + 320 + 160
    │
    ├─ dec_pose: Linear(1120→96)   → delta_pose
    ├─ dec_shape: Linear(1120→10)  → delta_shape
    └─ dec_cam: Linear(1120→3)     → delta_cam
    │
    ▼ Add residuals
refined_pose  = initial_pose + delta_pose     → (B, 96) → rot6d_to_rotmat → (B, 16, 3, 3)
refined_shape = initial_shape + delta_shape   → (B, 10)
refined_cam   = initial_cam + delta_cam       → (B, 3)
```

---

## 6. Data Flow End-to-End

### Inference Flow (demo.py)

```
┌──────────────┐
│  Raw Image   │  (H, W, 3) BGR uint8
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ YOLO Detect  │  conf=0.3
└──────┬───────┘
       │ boxes: (N, 4) [x1,y1,x2,y2]
       │ is_right: (N,) 0=left, 1=right
       ▼
┌──────────────────┐
│ ViTDetDataset    │  rescale_factor=2.5
│ - Crop hand      │  boxes expanded to [192, 256] aspect ratio
│ - Flip left hand │  left hands are horizontally flipped
│ - Normalize      │  ImageNet mean/std
└──────┬───────────┘
       │ batch['img']: (B, 3, 256, 256) float32
       │ batch['right']: (B,)
       │ batch['box_center']: (B, 2)
       │ batch['box_size']: (B,)
       │ batch['img_size']: (B, 2)
       ▼
┌──────────────────┐
│ WiLoR.forward    │
│ - Center crop    │  256→192 width: (B, 3, 256, 192)
│ - ViT backbone   │  → initial MANO params + features
│ - MANO forward   │  → temp vertices (B, 778, 3)
│ - RefineNet      │  → refined MANO params
│ - MANO forward   │  → final vertices + joints
│ - Projection     │  → 2D keypoints
└──────┬───────────┘
       │ output dict:
       │   pred_vertices: (B, 778, 3)
       │   pred_keypoints_3d: (B, 21, 3)
       │   pred_keypoints_2d: (B, 21, 2)
       │   pred_cam_t: (B, 3)
       │   pred_mano_params: dict
       ▼
┌──────────────────┐
│ Render / Export  │
│ - Mesh overlay   │  pyrender + compositing
│ - OBJ export     │  trimesh export
│ - PKL export     │  HaMeR-compatible format
└──────────────────┘
```

### Training Flow (wilor.py)

```
┌────────────────────────────┐
│ WebDataset (tar files)     │  14 datasets
│ - Image + 2D/3D keypoints  │
│ - MANO params (optional)   │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ get_example()              │  (datasets/utils.py)
│ - Random augmentation      │  scale, rot, flip, color, trans
│ - Left hand flip           │  always flip for left hands
│ - Crop + affine warp       │
│ - Normalize                │
└──────────┬─────────────────┘
           │ joint_batch = {
           │   'img': {...},    # real images
           │   'mocap': {...}   # motion capture data
           │ }
           ▼
┌────────────────────────────┐
│ WiLoR.training_step()      │
│ 1. Forward pass            │
│ 2. Compute losses:         │
│    - Keypoint3D (L1)       │  weight=0.05
│    - Keypoint2D (L1)       │  weight=0.01
│    - Global Orient (MSE)   │  weight=0.001
│    - Hand Pose (MSE)       │  weight=0.001
│    - Betas (MSE)           │  weight=0.0005
│    - Adversarial (BCE)     │  weight=0.0005
│ 3. Manual backward         │
│ 4. Gradient clipping       │  (optional)
│ 5. Step optimizer          │
│ 6. Train discriminator     │  (if adversarial)
└────────────────────────────┘
```

---

## 7. Cấu Hình Hệ Thống

### `pretrained_models/model_config.yaml`

```yaml
MODEL:
  IMAGE_SIZE: 256
  BACKBONE:
    TYPE: 'vit'
    PRETRAINED_WEIGHTS: 'training_data/vitpose_backbone.pth'
  BBOX_SHAPE: [192, 256]
  NUM_HAND_JOINTS: 15

MANO:
  MODEL_PATH: 'mano_data/MANO_RIGHT.pkl'
  MEAN_PARAMS: 'mano_data/mano_mean_params.npz'
  DATA_DIR: 'mano_data/'

EXTRA:
  FOCAL_LENGTH: 5000

LOSS_WEIGHTS:
  KEYPOINTS_3D: 0.05
  KEYPOINTS_2D: 0.01
  GLOBAL_ORIENT: 0.001
  HAND_POSE: 0.001
  BETAS: 0.0005
  ADVERSARIAL: 0.0005

TRAIN:
  LR: 1e-5
  BATCH_SIZE: 32
  TOTAL_STEPS: 1000000
```

### `pretrained_models/dataset_config.yaml`

14 training datasets (webdataset tar format):
1. FREIHAND
2. INTERHAND26M
3. MTC
4. RHD
5. COCOW (COCO Wholebody)
6. HALPE
7. MPIINZSL
8. HO3D
9. H2O3D
10. DEX (Dexter+Object)
11. BEDLAM
12. REINTER
13. HOT3D
14. ARCTIC

---

## 8. Dependencies

### Core Dependencies
```
torch>=2.0.0
torchvision
pytorch-lightning
smplx==0.1.28          # MANO hand model
ultralytics==8.1.34    # YOLO detector
opencv-python
numpy
pyrender               # 3D rendering
trimesh                # Mesh processing
```

### Config & Infrastructure
```
yacs                   # Config system
hydra-core             # Experiment management
rich                   # Terminal output
```

### Optional
```
gradio                 # Web demo
timm                   # FastViT backbone
webdataset             # Training data loading
chumpy                 # MANO dependency
```

### External Resources
| Resource | Path | Nguồn |
|----------|------|-------|
| MANO model | `mano_data/MANO_RIGHT.pkl` | mano.is.tue.mpg.de |
| MANO mean params | `mano_data/mano_mean_params.npz` | Included |
| YOLO detector | `pretrained_models/detector.pt` | HuggingFace |
| WiLoR checkpoint | `pretrained_models/wilor_final.ckpt` | HuggingFace |
| ViTPose backbone | `training_data/vitpose_backbone.pth` | Pre-trained |

---

## 9. Bảng Kích Thước Tensor

### Input Tensors

| Tensor | Shape | Dtype | Mô tả |
|--------|-------|-------|-------|
| Raw image | `(H, W, 3)` | uint8 BGR | Ảnh gốc từ camera |
| Bounding boxes | `(N, 4)` | float32 | `[x1, y1, x2, y2]` |
| Hand laterality | `(N,)` | int | 0=left, 1=right |

### Intermediate Tensors

| Tensor | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `batch['img']` | `(B, 3, 256, 256)` | float32 | ViTDetDataset output |
| `x` (center-cropped) | `(B, 3, 256, 192)` | float32 | WiLoR.forward_step |
| Patch tokens | `(B, 192, 1280)` | float32 | PatchEmbed output |
| Full sequence | `(B, 210, 1280)` | float32 | After prepending tokens |
| ViT output features | `(B, 1280, 12, 16)` | float32 | Spatial feature map |
| Temp vertices | `(B, 778, 3)` | float32 | Initial MANO output |
| Multi-scale features | `[(B,640,24,32), (B,320,48,64), (B,160,96,128)]` | float32 | DeConvNet output |
| Vertex features | `(B, 1120)` | float32 | After max-pool |

### Output Tensors

| Tensor | Shape | Dtype | Mô tả |
|--------|-------|-------|-------|
| `pred_vertices` | `(B, 778, 3)` | float32 | MANO mesh vertices |
| `pred_keypoints_3d` | `(B, 21, 3)` | float32 | 3D joint positions |
| `pred_keypoints_2d` | `(B, 21, 2)` | float32 | Projected 2D joints |
| `pred_cam` | `(B, 3)` | float32 | Weak-perspective camera [s, tx, ty] |
| `pred_cam_t` | `(B, 3)` | float32 | 3D camera translation |
| `global_orient` | `(B, 1, 3, 3)` | float32 | Wrist rotation matrix |
| `hand_pose` | `(B, 15, 3, 3)` | float32 | Joint rotation matrices |
| `betas` | `(B, 10)` | float32 | Shape parameters |

### MANO Model Constants

| Constant | Value | Mô tả |
|----------|-------|-------|
| Num vertices | 778 | MANO mesh vertices |
| Num joints | 21 | After OpenPose remapping |
| Num pose params | 15 joints × 3×3 rotation | Excluding global_orient |
| Shape params | 10 | PCA shape components |
| Face count | 1538 + 14 (watertight) | Triangle faces |

---

## 10. Integration với DexAvatar

### Pipeline tích hợp

```
DexAvatar Pipeline:
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Video/Images │ ──→ │ WiLoR (here) │ ──→ │ hamer.pkl   │ ──→ DexAvatar
│              │     │ export_hamer │     │ wilor.pkl   │     data_parser.py
└─────────────┘     └──────────────┘     └─────────────┘
```

### Export formats

**`hamer.pkl`** — Format chính cho DexAvatar:
```python
{
    'pred_keypoints_2d': np.ndarray,   # (N, 21, 2)
    'pred_keypoints_3d': np.ndarray,   # (N, 21, 3)
    'hand_pose': np.ndarray,           # (N, 15, 3, 3) — rotation matrices
    'box_center': np.ndarray,          # (N, 2)
    'box_size': np.ndarray,            # (N,)
    'is_right': np.ndarray,            # (N,) — boolean
    'cam_t': np.ndarray,               # (N, 3)
}
```

**`wilor.pkl`** — Raw format (per-hand dict):
```python
{
    'right': {
        'hand_pose': np.ndarray,       # (1, 15, 3) — axis-angle
        'betas': np.ndarray,           # (1, 10)
        'global_orient': np.ndarray,   # (1, 1, 3) — axis-angle
        'pred_keypoints_2d': np.ndarray,
        'pred_keypoints_3d': np.ndarray,
        'pred_vertices': np.ndarray,   # (1, 778, 3)
        'box_center': np.ndarray,
        'box_size': np.ndarray,
        'is_right': np.ndarray,
        'cam_t': np.ndarray,
    }
}
```

### Chuyển đổi Rotation

**Rotation Matrix → Axis-Angle** (trong `export_hamer_pkl.py`):
```python
def rotmat_to_axis_angle_batch(rot_matrices):
    # Input: (N, 3, 3)
    # Output: (N, 3)
    # Rodrigues: θ = arccos((trace(R) - 1) / 2)
    #            axis = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] / (2sin(θ))
```

---

## Appendix A: Class Hierarchy

```
pl.LightningModule
  └── WiLoR                          # Main training/inference module

nn.Module
  ├── ViT                            # Vision Transformer backbone
  │     ├── PatchEmbed               # Image → patch tokens
  │     ├── Block                    # Transformer block × 32
  │     │     ├── Attention          # Multi-head self-attention
  │     │     ├── Mlp                # Feed-forward network
  │     │     └── DropPath           # Stochastic depth
  │     └── (pose/shape/cam decoders)
  │
  ├── RefineNet                      # Refinement head
  │     └── DeConvNet                # Multi-scale upsampling
  │
  ├── Discriminator                  # Adversarial training
  │
  ├── Keypoint2DLoss
  ├── Keypoint3DLoss
  └── ParameterLoss

smplx.MANOLayer
  └── MANO                           # Hand model with OpenPose mapping
```

## Appendix B: File Size Summary

| File | Lines | Vai trò |
|------|-------|---------|
| `wilor/datasets/utils.py` | 995 | ★ Largest — augmentation pipeline |
| `wilor/utils/renderer.py` | 424 | Full mesh renderer |
| `wilor/utils/pose_utils.py` | 352 | Evaluation metrics |
| `wilor/utils/render_openpose.py` | 192 | OpenPose drawing |
| `wilor/utils/misc.py` | 204 | Hydra utilities |
| `wilor/utils/mesh_renderer.py` | 149 | Training visualizer |
| `wilor/utils/skeleton_renderer.py` | 125 | Keypoint visualizer |
| `wilor/utils/geometry.py` | 101 | Rotation math |
| `wilor/utils/rich_utils.py` | 106 | Rich output |
| `wilor/models/heads/refinement_net.py` | ~300 | RefineNet |
| `wilor/models/backbones/vit.py` | ~500 | ViT backbone |
| `wilor/models/wilor.py` | ~400 | Main model |
| `wilor/models/mano_wrapper.py` | ~80 | MANO wrapper |
| `wilor/models/discriminator.py` | ~100 | Discriminator |
| `wilor/models/losses.py` | ~80 | Loss functions |
| `wilor/datasets/vitdet_dataset.py` | ~150 | Inference dataset |
| `wilor/configs/__init__.py` | ~200 | Config system |
| `export_hamer_pkl.py` | ~200 | HaMeR export |
| `demo.py` | ~200 | CLI demo |

---

*Tài liệu này được tạo tự động bằng cách đọc toàn bộ source code dự án WiLoR/DexAvatar.*
*Cập nhật: 2026-06-01*
