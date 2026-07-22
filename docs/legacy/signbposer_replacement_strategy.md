# Chiến lược thay thế SignBPoser: MotionBERT & PHD

## Ngày: 2026-06-06
## Mục tiêu: Thay thế VAE body pose prior (SignBPoser) bằng các khối hiện đại hơn để cải thiện UBody/LHand/RHand trên SGNify benchmark

---

## MỤC LỤC

1. [Tổng quan & Motivation](#1-tổng-quan--motivation)
2. [Phân tích SignBPoser hiện tại](#2-phân-tích-signbposer-hiện-tại)
3. [Dataset & Data Processing](#3-dataset--data-processing)
4. [Method 1: MotionBERT Body Prior](#4-method-1-motionbert-body-prior)
5. [Method 2: PHD (Pose from Human Diffusion)](#5-method-2-phd-pose-from-human-diffusion)
6. [Integration vào DexAvatar Codebase](#6-integration-vào-dexavatar-codebase)
7. [Directory Structure](#7-directory-structure)
8. [Training Pipeline](#8-training-pipeline)
9. [Evaluation & Metrics](#9-evaluation--metrics)
10. [Timeline & Milestones](#10-timeline--milestones)
11. [References](#11-references)

---

## 1. Tổng quan & Motivation

### 1.1 Vấn đề với SignBPoser hiện tại

SignBPoser là VAE body pose prior (33-dim latent, 21 body joints) được train trên SignAvatars (pseudo-GT từ How2Sign qua OSX pipeline). Có 3 bottleneck chính:

1. **Latent bottleneck**: 63-dim body pose → 33-dim latent → mất thông tin
2. **Pseudo-GT bias**: SignAvatars là output của OSX, không phải ground truth thật
3. **Mode-seeking**: VAE tend to average modes → body poses bị "nhòe"

### 1.2 Chiến lược thay thế

| Method | Kiểu | Pretrained? | Complexity | Expected Improvement |
|--------|------|-------------|------------|---------------------|
| **MotionBERT** | Transformer encoder + MLP head | ✅ AMASS | Trung bình | 5-10% |
| **PHD** | Diffusion score-based prior | ✅ SMPL poses | Cao | 8-15% |

### 1.3 Tại sao pretrained hợp lý

```
Transfer learning từ large-scale mocap data:
├── AMASS: 40+ hours real mocap → general body motion manifold
├── HumanML3D: 14K motions → diverse activities
├── Chất lượng > SignAvatars (pseudo-GT)
└── Sign language ⊂ general body motion → domain gap nhỏ

SignBPoser train từ scratch trên pseudo-GT:
├── 3-layer indirect pipeline: How2Sign → OSX → SignAvatars → SignBPoser
├── Accumulated noise/bias
└── Small latent space (33-dim)

Pretrained models:
├── Trained on real mocap (AMASS, CMU, BML)
├── Large-scale → better coverage
└── Finetune nhẹ trên sign language → domain adaptation
```

---

## 2. Phân tích SignBPoser hiện tại

### 2.1 Kiến trúc

```
File: dexavatar_fitting/smplifyx/signbposer/signbposer.py
Class: SignbPoser(nn.Module)

Encoder:
  input: body_pose (N, 63)  ← 21 joints × 3 axis-angle
  → BatchNorm1d(63)
  → Linear(63, 512) + LeakyReLU + BatchNorm + Dropout
  → Linear(512, 512) + LeakyReLU
  → Linear(512, 33) for mu
  → Linear(512, 33) for logvar (via softplus)
  → Normal(mu, softplus(logvar))

Decoder:
  input: latent (N, 33)
  → Linear(33, 512) + LeakyReLU + Dropout
  → Linear(512, 512) + LeakyReLU
  → Linear(512, 21*6)  ← 6D continuous rotation
  → ContinuousRotReprDecoder (Gram-Schmidt → 3×3 rotation matrix)
  → Output: (N, 1, 21, 9) rotation matrix hoặc (N, 1, 21, 3) axis-angle
```

### 2.2 Config

```ini
# TR00_signbposer.ini
data_shape = [1, 21, 3]
latentD = 33
num_neurons = 512
kl_coef = 0.001
use_cont_repr = True
remove_Zrot = True
```

### 2.3 Các integration points trong codebase

```
┌─────────────────────────────────────────────────────────────────┐
│                SIGNBPOSER INTEGRATION MAP                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Config Layer                                                     │
│  ├── cfg_files/*.yaml → signbposer_dir, use_signbposer           │
│  └── cmd_parser.py:110 → --signbposer_dir                        │
│  └── cmd_parser.py:171 → --use_signbposer                        │
│                                                                   │
│  Loading Layer                                                    │
│  ├── test_bposer.py:21 → load_signbposer(expr_dir)               │
│  └── fit_single_frame.py:228 → signbposer, _ = load_signbposer() │
│                                                                   │
│  Optimization Layer                                               │
│  ├── fit_single_frame.py:224 → pose_embedding = zeros(1,33)      │
│  ├── fit_single_frame.py:488 → final_params.append(pose_embedding)│
│  └── L-BFGS optimizes pose_embedding (NOT model weights)         │
│                                                                   │
│  Decode Layer (body_pose generation)                              │
│  ├── fitting.py:83 → guess_init: signbposer.decode() for camera  │
│  ├── fitting.py:248 → closure: signbposer.decode() for body_model│
│  ├── fit_single_frame.py:613 → result extraction                 │
│  └── fit_single_frame.py:741 → mesh/visualization               │
│                                                                   │
│  Loss Layer (prior computation)                                   │
│  ├── fitting.py:586 → pprior_loss = ||pose_embedding||²          │
│  └── fitting.py:588 → + L1(decoded - smplerx_init) for core/non │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset & Data Processing

### 3.1 Datasets sử dụng

| Dataset | Role | Scale | Annotation | Language |
|---------|------|-------|------------|----------|
| **SignAvatars** | Primary training | 8.34M frames, 153 signers | SMPL-X pseudo-GT | ASL |
| **PHOENIX14T** | Domain adaptation | ~67K frames, 9 signers | SMPL-X (extract) | DGS |
| **SGNify** | Evaluation ONLY | 57 signs, ~2.9K frames | SMPL-X GT | DGS |

**Lưu ý:** KHÔNG dùng How2Sign riêng vì SignAvatars đã cover (cùng nguồn ASL).

### 3.2 Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: Source Data                                              │
│  ├── SignAvatars: download từ GitHub (đã có SMPL-X params)       │
│  └── PHOENIX14T: download raw videos → extract SMPL-X via SMPLer-X│
│                                                                   │
│  Step 2: Extract body_pose                                        │
│  ├── SignAvatars: load SMPL-X params → body_pose (21×3 AA)       │
│  └── PHOENIX14T: run SMPLer-X pipeline → body_pose (21×3 AA)     │
│                                                                   │
│  Step 3: Normalize                                                │
│  ├── Remove Z-rotation from global_orient (theo config remove_Zrot)│
│  ├── Clamp body_pose trong [-π, π]                               │
│  └── Convert sang rotation matrix nếu cần (6D representation)    │
│                                                                   │
│  Step 4: Filter                                                   │
│  ├── Loại outlier: ||body_pose|| > mean + 2*std                  │
│  ├── Loại frames có confidence thấp (SMPLer-X output)            │
│  └── Loại frames bị occlusion nặng                               │
│                                                                   │
│  Step 5: Split (theo signer_id)                                   │
│  ├── Train: 80% signers                                          │
│  ├── Val: 10% signers                                            │
│  └── Test: 10% signers (KHÔNG overlap với SGNify)                │
│                                                                   │
│  Step 6: Weighted Sampling                                        │
│  ├── SignAvatars (pseudo-GT): weight = 0.5                       │
│  ├── PHOENIX14T (DGS domain): weight = 1.2                       │
│  └── Normalize weights                                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Unified Data Format

```python
# Output format sau khi process
training_data = {
    'body_pose': np.ndarray,      # (N, 63) - 21 joints × 3 axis-angle
    'body_pose_matrot': np.ndarray, # (N, 21, 9) - rotation matrix
    'source': List[str],          # 'signavatars' | 'phoenix14t'
    'signer_id': List[str],       # ID người ký
    'sign_class': List[str],      # Gloss label (nếu có)
    'confidence': np.ndarray,     # (N,) - quality score
    'split': List[str],           # 'train' | 'val' | 'test'
}
```

### 3.4 Script locations cho data processing

```
data/
├── signbposer_data/                    # NEW: unified training data
│   ├── train/
│   │   ├── body_poses.npy             # (N_train, 63)
│   │   ├── metadata.pkl               # source, signer_id, etc.
│   │   └── sample_weights.npy         # (N_train,)
│   ├── val/
│   │   ├── body_poses.npy
│   │   └── metadata.pkl
│   └── test/
│       ├── body_poses.npy
│       └── metadata.pkl
├── frames/                             # Existing: SGNify images
├── smplx_gt/                           # Existing: SGNify GT
└── segment.json                        # Existing

scripts/
├── prepare_signbposer_data.py          # NEW: data processing script
├── extract_phoenix_smplx.py            # NEW: PHOENIX14T SMPL-X extraction
└── ...
```

---

## 4. Method 1: MotionBERT Body Prior

### 4.1 Tổng quan

MotionBERT (ICCV 2023) là pretrained model cho 3D human pose estimation, sử dụng Dual-stream Spatio-Temporal Transformer. Pretrained trên AMASS (real mocap).

**Chiến lược:** Freeze pretrained MotionBERT encoder → train SMPL regression head → finetune trên sign language data.

### 4.2 Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                MOTIONBERT BODY PRIOR                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: 2D keypoints (B, T, 17, 2) từ Sapiens                   │
│         ↓                                                         │
│  ┌─────────────────────────────────────┐                         │
│  │  MotionBERT Encoder (FROZEN)         │                         │
│  │  ├── Dual-stream Transformer         │                         │
│  │  │   ├── Spatial: inter-joint attn   │                         │
│  │  │   └── Temporal: inter-frame attn  │                         │
│  │  ├── 4 layers, 256-dim, 8 heads     │                         │
│  │  └── Pretrained on AMASS             │                         │
│  └──────────────┬──────────────────────┘                         │
│                 │                                                  │
│                 ▼                                                  │
│  motion_features: (B, T, 256)                                     │
│                 │                                                  │
│                 ▼                                                  │
│  ┌─────────────────────────────────────┐                         │
│  │  SMPL Regression Head (TRAINABLE)    │                         │
│  │  ├── AdaptiveAvgPool1d(T → 1)       │                         │
│  │  ├── Linear(256 → 512) + ReLU       │                         │
│  │  ├── Linear(512 → 512) + ReLU       │                         │
│  │  ├── Linear(512 → 21×6)             │  ← rot6d output         │
│  │  └── rot6d_to_axis_angle()           │  ← 21×3 AA output      │
│  └──────────────┬──────────────────────┘                         │
│                 │                                                  │
│                 ▼                                                  │
│  body_pose: (B, 63) ← 21 joints × 3 axis-angle                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Code implementation

#### 4.3.1 Model definition

```python
# File: dexavatar_fitting/smplifyx/signbposer/motionbert_prior.py

import torch
import torch.nn as nn
import sys
sys.path.append('/home/haipd/DexAvatar/MotionBERT')

from models.motion_bert import MotionBERT
from utils.rotation_utils import rot6d_to_axis_angle


class MotionBERTBodyPrior(nn.Module):
    """
    Body pose prior sử dụng pretrained MotionBERT encoder
    + trainable SMPL regression head.
    
    Thay thế SignBPoser: không cần latent space, predict body_pose trực tiếp.
    """
    
    def __init__(self, motionbert_ckpt_path, freeze_encoder=True):
        super().__init__()
        
        # Pretrained MotionBERT encoder
        self.motionbert = MotionBERT.from_pretrained(motionbert_ckpt_path)
        
        if freeze_encoder:
            for param in self.motionbert.parameters():
                param.requires_grad = False
        
        # SMPL regression head
        self.smpl_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),        # (B, T, 256) → (B, 256, 1)
            nn.Flatten(),                    # (B, 256)
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 21 * 6),         # 21 joints × 6D rotation
        )
        
        # Initialize last layer to small values
        nn.init.xavier_uniform_(self.smpl_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.smpl_head[-1].bias)
    
    def forward(self, keypoints_2d):
        """
        Args:
            keypoints_2d: (B, T, 17, 2) - 2D keypoints from Sapiens
        
        Returns:
            body_pose: (B, 63) - 21 joints × 3 axis-angle
        """
        # Extract motion features (frozen encoder)
        with torch.no_grad():
            motion_features = self.motionbert.get_features(keypoints_2d)
            # motion_features: (B, T, 256)
        
        # Reshape for pooling: (B, T, 256) → (B, 256, T)
        features = motion_features.permute(0, 2, 1)
        
        # Predict rot6d
        rot6d = self.smpl_head(features)  # (B, 126)
        
        # Convert to axis-angle
        body_pose = rot6d_to_axis_angle(
            rot6d.view(-1, 6)
        ).view(rot6d.shape[0], -1)  # (B, 63)
        
        return body_pose
    
    def predict_from_single_frame(self, keypoints_2d_single):
        """
        Wrapper cho single-frame inference (DexAvatar pipeline).
        
        Args:
            keypoints_2d_single: (1, 17, 2) hoặc (17, 2)
        
        Returns:
            body_pose: (1, 63)
        """
        if keypoints_2d_single.dim() == 2:
            keypoints_2d_single = keypoints_2d_single.unsqueeze(0)
        
        # Add temporal dimension: (1, 17, 2) → (1, 1, 17, 2)
        keypoints_2d = keypoints_2d_single.unsqueeze(1)
        
        return self.forward(keypoints_2d)
```

#### 4.3.2 Wrapper compatible với SignBPoser interface

```python
# File: dexavatar_fitting/smplifyx/signbposer/motionbert_wrapper.py

import torch
import torch.nn as nn


class MotionBERTPriorWrapper(nn.Module):
    """
    Wrapper để MotionBERTBodyPrior compatible với interface của SignBPoser.
    
    SignBPoser interface:
        signbposer.decode(pose_embedding, output_type='aa') → body_pose
    
    MotionBERT wrapper:
        wrapper.decode(pose_embedding, output_type='aa') → body_pose
        (pose_embedding ở đây chính là body_pose trực tiếp, không cần latent)
    """
    
    def __init__(self, motionbert_prior):
        super().__init__()
        self.prior = motionbert_prior
    
    def decode(self, pose_input, output_type='aa'):
        """
        Compatible interface với SignBPoser.decode().
        
        Args:
            pose_input: có thể là:
                - body_pose (1, 63) nếu direct optimization
                - motion features nếu conditioning
            output_type: 'aa' hoặc 'matrot'
        
        Returns:
            body_pose: (1, 1, 21, 9) nếu matrot, (1, 1, 21, 3) nếu aa
        """
        if pose_input.shape[-1] == 63:
            # Direct body_pose input
            body_pose = pose_input
        else:
            # Latent input (không nên xảy ra với MotionBERT)
            raise ValueError(f"Unexpected input shape: {pose_input.shape}")
        
        if output_type == 'matrot':
            return self._aa_to_matrot(body_pose)
        else:  # 'aa'
            return body_pose.view(1, 1, 21, 3)
    
    def _aa_to_matrot(self, body_pose_aa):
        """Convert axis-angle to rotation matrix."""
        # body_pose_aa: (1, 63)
        # Using torchgeometry or custom implementation
        from signbposer import SignbPoser
        return SignbPoser.aa2matrot(body_pose_aa.view(1, 1, 21, 3))
    
    def encode(self, body_pose):
        """No-op, chỉ để compatible interface."""
        return body_pose
    
    def prior_loss(self, body_pose, keypoints_2d=None):
        """
        Prior loss: regularization trên body_pose.
        Sử dụng pretrained MotionBERT features làm guidance.
        """
        # L2 regularization vs zero (sign language poses should be natural)
        l2_loss = torch.norm(body_pose, p=2)
        
        # Optionally: use MotionBERT features to compute reconstruction loss
        if keypoints_2d is not None:
            predicted_pose = self.prior.predict_from_single_frame(keypoints_2d)
            recon_loss = nn.functional.l1_loss(body_pose, predicted_pose)
            return l2_loss + recon_loss
        
        return l2_loss
```

#### 4.3.3 Training script

```python
# File: scripts/train_motionbert_prior.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

sys.path.append('/home/haipd/DexAvatar')
sys.path.append('/home/haipd/DexAvatar/MotionBERT')

from dexavatar_fitting.smplifyx.signbposer.motionbert_prior import MotionBERTBodyPrior


def load_training_data(data_dir):
    """Load unified training data."""
    train_data = np.load(os.path.join(data_dir, 'train', 'body_poses.npy'))
    val_data = np.load(os.path.join(data_dir, 'val', 'body_poses.npy'))
    
    # Load 2D keypoints nếu có (cho reconstruction loss)
    train_kpts_path = os.path.join(data_dir, 'train', 'keypoints_2d.npy')
    train_kpts = np.load(train_kpts_path) if os.path.exists(train_kpts_path) else None
    
    return train_data, val_data, train_kpts


def train_motionbert_prior(
    motionbert_ckpt='checkpoints/pose3d_MB_ft_h36m.tar',
    data_dir='data/signbposer_data',
    output_dir='checkpoints/motionbert_prior',
    epochs=100,
    batch_size=256,
    lr=1e-3,
    freeze_encoder=True,
):
    """
    Training pipeline cho MotionBERT Body Prior.
    
    Phase 1: Freeze MotionBERT encoder, train SMPL head only
    Phase 2: (Optional) Unfreeze encoder, finetune all với small lr
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    train_poses, val_poses, train_kpts = load_training_data(data_dir)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_poses))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    
    val_dataset = TensorDataset(torch.FloatTensor(val_poses))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 2. Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MotionBERTBodyPrior(
        motionbert_ckpt_path=motionbert_ckpt,
        freeze_encoder=freeze_encoder
    ).to(device)
    
    # 3. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 4. Loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # 5. Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss = 0
        
        for batch_idx, (body_pose_gt,) in enumerate(train_loader):
            body_pose_gt = body_pose_gt.to(device)
            
            # Forward: MotionBERT expects 2D keypoints
            # Ở đây chúng ta train với reconstruction loss trực tiếp
            # (body_pose prediction từ random features)
            
            # Nếu có keypoints_2d:
            #   keypoints = train_kpts[batch_idx * batch_size: ...]
            #   body_pose_pred = model(torch.FloatTensor(keypoints).to(device))
            # Không có keypoints: train SMPL head trực tiếp với GT body_pose
            
            # Placeholder: train SMPL head để predict GT body_pose
            # (sẽ thay bằng real keypoints khi có data)
            optimizer.zero_grad()
            
            # Simulate motion features (vì chưa có keypoints_2d)
            # Trong thực tế sẽ dùng: model(keypoints_2d)
            dummy_features = torch.randn(
                body_pose_gt.shape[0], 1, 256, device=device
            )
            
            # Forward through SMPL head only
            rot6d = model.smpl_head(dummy_features.permute(0, 2, 1))
            body_pose_pred = rot6d_to_axis_angle(
                rot6d.view(-1, 6)
            ).view(rot6d.shape[0], -1)
            
            # Loss: reconstruction + regularization
            loss_recon = mse_loss(body_pose_pred, body_pose_gt)
            loss_l1 = l1_loss(body_pose_pred, body_pose_gt)
            loss_reg = 1e-4 * torch.norm(body_pose_pred, p=2)
            
            loss = loss_recon + 0.5 * loss_l1 + loss_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # ---- Validation ----
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for body_pose_gt, in val_loader:
                body_pose_gt = body_pose_gt.to(device)
                dummy_features = torch.randn(
                    body_pose_gt.shape[0], 1, 256, device=device
                )
                rot6d = model.smpl_head(dummy_features.permute(0, 2, 1))
                body_pose_pred = rot6d_to_axis_angle(
                    rot6d.view(-1, 6)
                ).view(rot6d.shape[0], -1)
                
                val_loss += mse_loss(body_pose_pred, body_pose_gt).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training complete. Best val_loss: {best_val_loss:.6f}")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--motionbert_ckpt', default='checkpoints/pose3d_MB_ft_h36m.tar')
    parser.add_argument('--data_dir', default='data/signbposer_data')
    parser.add_argument('--output_dir', default='checkpoints/motionbert_prior')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--unfreeze_encoder', action='store_true')
    args = parser.parse_args()
    
    train_motionbert_prior(
        motionbert_ckpt=args.motionbert_ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_encoder=not args.unfreeze_encoder,
    )
```

### 4.4 Fitting integration

Trong DexAvatar fitting pipeline, MotionBERT thay thế SignBPoser như sau:

```
Old (SignBPoser):
  pose_embedding (33-dim) → signbposer.decode() → body_pose (63-dim)
  L_prior = ||pose_embedding||²

New (MotionBERT):
  body_pose (63-dim) = direct optimization variable
  keypoints_2d → motionbert.predict() → predicted_body_pose
  L_prior = ||body_pose - predicted_body_pose||² + λ * ||body_pose||²
```

---

## 5. Method 2: PHD (Pose from Human Diffusion)

### 5.1 Tổng quan

PHD sử dụng score-based diffusion model làm body pose prior, thay thế VPoser/SignBPoser trong SMPLify pipeline. Score function ∇_x log p(x) guide optimization về phía realistic poses.

**Chiến lược:** Load pretrained PHD → finetune trên sign language data → replace SignBPoser decoder.

### 5.2 Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                PHD DIFFUSION BODY PRIOR                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Score Network s_θ(x_t, t, condition):                           │
│  ├── Input: body_pose x_t (63-dim) + timestep t + conditioning   │
│  ├── Conditioning: SMPLer-X task_tokens (18 × 1024) hoặc        │
│  │                 2D keypoints features                          │
│  ├── Architecture: MLP-based hoặc Transformer-based              │
│  │   ├── Timestep embedding: sinusoidal → MLP                    │
│  │   ├── Condition embedding: Linear → concat                    │
│  │   └── Denoising MLP: 4 layers × 1024 hidden                  │
│  └── Output: predicted noise ε (63-dim)                          │
│                                                                   │
│  Training:                                                        │
│  ├── Forward: x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε              │
│  ├── Loss: ||s_θ(x_t, t) - ε||²                                  │
│  └── Data: SMPL body poses từ AMASS + SignAvatars               │
│                                                                   │
│  Inference (Optimization):                                        │
│  ├── Start: body_pose = smplerx_init (63-dim)                    │
│  ├── For t = T, T-1, ..., 1:                                     │
│  │   ├── score = s_θ(body_pose, t, condition)                    │
│  │   ├── data_grad = ∇_body_pose L_data                          │
│  │   └── body_pose -= η * (score + guidance_scale * data_grad)   │
│  └── Final: body_pose optimized                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Code implementation

#### 5.3.1 Score Network

```python
# File: dexavatar_fitting/smplifyx/signbposer/phd_score_network.py

import torch
import torch.nn as nn
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding cho diffusion timestep."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PHDScoreNetwork(nn.Module):
    """
    Score-based diffusion network cho body pose prior.
    
    Learns s_θ(x_t, t, condition) ≈ ∇_x log p(x_t | condition)
    """
    
    def __init__(
        self,
        pose_dim=63,           # 21 joints × 3 axis-angle
        condition_dim=1024,    # SMPLer-X task token dim
        hidden_dim=1024,
        num_layers=4,
        time_embed_dim=256,
    ):
        super().__init__()
        
        self.pose_dim = pose_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition embedding (SMPLer-X features)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Pose input projection
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Denoising network
        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim * 3 if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
        self.denoise_net = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, pose_dim),
        )
        
        # Initialize last layer to small values
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.01)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(self, x_t, t, condition=None):
        """
        Args:
            x_t: (B, 63) - noisy body pose at timestep t
            t: (B,) - diffusion timestep
            condition: (B, 1024) - SMPLer-X task tokens (optional)
        
        Returns:
            noise_pred: (B, 63) - predicted noise
        """
        # Embed timestep
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        
        # Embed condition
        if condition is not None:
            c_emb = self.condition_embed(condition)  # (B, hidden_dim)
        else:
            c_emb = torch.zeros_like(t_emb)
        
        # Project pose
        x_emb = self.pose_proj(x_t)  # (B, hidden_dim)
        
        # Concatenate all embeddings
        h = torch.cat([x_emb, t_emb, c_emb], dim=-1)  # (B, hidden_dim*3)
        
        # Denoise
        h = self.denoise_net(h)  # (B, hidden_dim)
        
        # Predict noise
        noise_pred = self.output_proj(h)  # (B, 63)
        
        return noise_pred
```

#### 5.3.2 Diffusion Process

```python
# File: dexavatar_fitting/smplifyx/signbposer/phd_diffusion.py

import torch
import torch.nn as nn
import numpy as np


class PHDDiffusion:
    """
    Diffusion process cho PHD body pose prior.
    Implements forward process, sampling, và score computation.
    """
    
    def __init__(
        self,
        score_network,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule='cosine',
    ):
        self.score_network = score_network
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: x_0 → x_t
        
        Args:
            x_0: (B, 63) - clean body pose
            t: (B,) - timestep
            noise: (B, 63) - optional noise
        
        Returns:
            x_t: (B, 63) - noisy body pose
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(x_0.device).unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(x_0.device).unsqueeze(-1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def training_loss(self, x_0, condition=None):
        """
        Compute training loss: ||ε - s_θ(x_t, t, condition)||²
        
        Args:
            x_0: (B, 63) - clean body pose
            condition: (B, 1024) - optional conditioning
        
        Returns:
            loss: scalar
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # Random noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.score_network(x_t, t, condition)
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    def score(self, x_t, t, condition=None):
        """
        Compute score function: ∇_x log p(x_t | condition)
        
        Score ≈ -ε_θ(x_t, t) / √(1-ᾱ_t)
        """
        noise_pred = self.score_network(x_t, t, condition)
        score = -noise_pred / self.sqrt_one_minus_alphas_cumprod[t].to(x_t.device).unsqueeze(-1)
        return score
    
    @torch.no_grad()
    def sample(self, shape, condition=None, device='cuda'):
        """
        Sample from the prior via ancestral sampling.
        
        Args:
            shape: tuple - shape of samples (B, 63)
            condition: (B, 1024) - optional conditioning
            device: str
        
        Returns:
            x_0: (B, 63) - sampled body poses
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.score_network(x, t_batch, condition)
            
            # DDPM update
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # Mean
            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            )
            
            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean
        
        return x
```

#### 5.3.3 PHD Prior cho Optimization

```python
# File: dexavatar_fitting/smplifyx/signbposer/phd_prior.py

import torch
import torch.nn as nn


class PHDBodyPrior(nn.Module):
    """
    PHD Body Prior: drop-in replacement cho SignBPoser.
    
    Sử dụng score-based diffusion để guide body pose optimization.
    Compatible với DexAvatar fitting pipeline.
    """
    
    def __init__(self, diffusion_model, guidance_scale=1.0):
        super().__init__()
        self.diffusion = diffusion_model
        self.guidance_scale = guidance_scale
    
    def decode(self, body_pose, output_type='aa', condition=None, t=0):
        """
        Compatible interface với SignBPoser.decode().
        
        Args:
            body_pose: (1, 63) - body pose (direct optimization variable)
            output_type: 'aa' hoặc 'matrot'
            condition: optional conditioning
            t: diffusion timestep (0 = no noise)
        
        Returns:
            body_pose: (1, 1, 21, 9) hoặc (1, 1, 21, 3)
        """
        if output_type == 'matrot':
            from signbposer import SignbPoser
            return SignbPoser.aa2matrot(body_pose.view(1, 1, 21, 3))
        else:
            return body_pose.view(1, 1, 21, 3)
    
    def prior_loss(self, body_pose, condition=None, t=None):
        """
        Compute prior loss: score-based guidance.
        
        Args:
            body_pose: (1, 63) - body pose (requires_grad=True)
            condition: optional conditioning
            t: diffusion timestep
        
        Returns:
            loss: scalar (negative log-likelihood proxy)
        """
        if t is None:
            # Random timestep
            t = torch.randint(0, self.diffusion.num_timesteps, (1,), 
                            device=body_pose.device)
        
        # Add small noise to body_pose
        noise = torch.randn_like(body_pose)
        x_t = self.diffusion.q_sample(body_pose, t, noise)
        
        # Predict noise
        noise_pred = self.diffusion.score_network(x_t, t, condition)
        
        # Score-based loss: encourage body_pose to be in high-density region
        # L = ||ε_θ(x_t, t)||² (penalize large predicted noise → away from data)
        loss = torch.norm(noise_pred, p=2)
        
        return loss * self.guidance_scale
    
    def score_guidance(self, body_pose, condition=None, t=50):
        """
        Compute score gradient cho optimization.
        
        Args:
            body_pose: (1, 63) - body pose (requires_grad=True)
            condition: optional conditioning
            t: diffusion timestep
        
        Returns:
            score_grad: (1, 63) - gradient of log p(body_pose)
        """
        body_pose_grad = body_pose.clone().detach().requires_grad_(True)
        
        # Compute score
        noise_pred = self.diffusion.score_network(
            body_pose_grad, 
            torch.tensor([t], device=body_pose.device),
            condition
        )
        
        # Score ≈ -ε / √(1-ᾱ_t)
        score = -noise_pred / self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        
        # Gradient
        score.backward()
        
        return body_pose_grad.grad
```

#### 5.3.4 Training script

```python
# File: scripts/train_phd_prior.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

sys.path.append('/home/haipd/DexAvatar')

from dexavatar_fitting.smplifyx.signbposer.phd_score_network import PHDScoreNetwork
from dexavatar_fitting.smplifyx.signbposer.phd_diffusion import PHDDiffusion


def train_phd_prior(
    data_dir='data/signbposer_data',
    output_dir='checkpoints/phd_prior',
    epochs=200,
    batch_size=256,
    lr=1e-4,
    num_timesteps=1000,
    condition_dim=1024,
):
    """
    Training pipeline cho PHD Diffusion Body Prior.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    train_poses = np.load(os.path.join(data_dir, 'train', 'body_poses.npy'))
    val_poses = np.load(os.path.join(data_dir, 'val', 'body_poses.npy'))
    
    train_dataset = TensorDataset(torch.FloatTensor(train_poses))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    
    val_dataset = TensorDataset(torch.FloatTensor(val_poses))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 2. Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    score_network = PHDScoreNetwork(
        pose_dim=63,
        condition_dim=condition_dim,
        hidden_dim=1024,
        num_layers=4,
        time_embed_dim=256,
    ).to(device)
    
    diffusion = PHDDiffusion(
        score_network=score_network,
        num_timesteps=num_timesteps,
        beta_schedule='cosine',
    )
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(score_network.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 4. Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # ---- Train ----
        score_network.train()
        train_loss = 0
        
        for body_pose_gt, in train_loader:
            body_pose_gt = body_pose_gt.to(device)
            
            optimizer.zero_grad()
            
            # Compute diffusion training loss
            loss = diffusion.training_loss(body_pose_gt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_network.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # ---- Validation ----
        score_network.eval()
        val_loss = 0
        
        with torch.no_grad():
            for body_pose_gt, in val_loader:
                body_pose_gt = body_pose_gt.to(device)
                loss = diffusion.training_loss(body_pose_gt)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'score_network_state_dict': score_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'num_timesteps': num_timesteps,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training complete. Best val_loss: {best_val_loss:.6f}")
    return score_network, diffusion


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/signbposer_data')
    parser.add_argument('--output_dir', default='checkpoints/phd_prior')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    args = parser.parse_args()
    
    train_phd_prior(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_timesteps=args.num_timesteps,
    )
```

### 5.4 Fitting integration

```
Old (SignBPoser):
  pose_embedding (33-dim) → signbposer.decode() → body_pose (63-dim)
  L_prior = ||pose_embedding||²

New (PHD):
  body_pose (63-dim) = direct optimization variable
  L_data = 2D reprojection + 3D hand (giữ nguyên)
  L_prior = score_guidance(body_pose, t, condition)  ← PHD score function
  L_biomech = joint angle constraints (giữ nguyên)
  
  Optimization: L-BFGS trên body_pose trực tiếp
  Prior gradient: ∇_body_pose L_prior = score gradient từ PHD
```

---

## 6. Integration vào DexAvatar Codebase

### 6.1 File modifications cần thiết

#### 6.1.1 New files cần tạo

```
dexavatar_fitting/smplifyx/signbposer/
├── signbposer.py                    # EXISTING - giữ nguyên
├── motionbert_prior.py              # NEW - MotionBERT model
├── motionbert_wrapper.py            # NEW - wrapper compatible interface
├── phd_score_network.py             # NEW - PHD score network
├── phd_diffusion.py                 # NEW - diffusion process
├── phd_prior.py                     # NEW - PHD prior for optimization
├── TR00_signbposer.ini              # EXISTING - giữ nguyên
└── snapshots/
    ├── TR00_E078.pt                 # EXISTING - SignBPoser weights
    ├── motionbert_best.pt           # NEW - MotionBERT trained weights
    └── phd_best.pt                  # NEW - PHD trained weights
```

#### 6.1.2 Modified files

**File 1: `dexavatar_fitting/smplifyx/cmd_parser.py`**

```python
# Thêm arguments mới (sau line 173):

# ---- MotionBERT Prior ----
parser.add_argument('--use_motionbert_prior', default=False,
                    type=lambda arg: arg.lower() in ['true', '1'],
                    help='Use MotionBERT body pose prior')
parser.add_argument('--motionbert_prior_dir', default='', type=str,
                    help='Path to MotionBERT prior checkpoint')

# ---- PHD Diffusion Prior ----
parser.add_argument('--use_phd_prior', default=False,
                    type=lambda arg: arg.lower() in ['true', '1'],
                    help='Use PHD diffusion body pose prior')
parser.add_argument('--phd_prior_dir', default='', type=str,
                    help='Path to PHD prior checkpoint')
parser.add_argument('--phd_guidance_scale', default=1.0, type=float,
                    help='Guidance scale for PHD score function')
parser.add_argument('--phd_num_inference_steps', default=50, type=int,
                    help='Number of diffusion steps for PHD')
```

**File 2: `dexavatar_fitting/smplifyx/fit_single_frame.py`**

```python
# Thêm import (sau line 39):
from test_bposer import load_signbposer
from signbposer.motionbert_wrapper import MotionBERTPriorWrapper  # NEW
from signbposer.phd_prior import PHDBodyPrior                     # NEW
from signbposer.phd_diffusion import PHDDiffusion                 # NEW

# Thêm loading logic (sau line 230):
# ---- MotionBERT Prior Loading ----
use_motionbert_prior = kwargs.get('use_motionbert_prior', False)
motionbert_prior = None
if use_motionbert_prior:
    motionbert_prior_dir = kwargs.get('motionbert_prior_dir', '')
    motionbert_prior = load_motionbert_prior(motionbert_prior_dir)
    motionbert_prior = motionbert_prior.to(device=device)
    motionbert_prior.eval()

# ---- PHD Prior Loading ----
use_phd_prior = kwargs.get('use_phd_prior', False)
phd_prior = None
if use_phd_prior:
    phd_prior_dir = kwargs.get('phd_prior_dir', '')
    phd_prior = load_phd_prior(phd_prior_dir)
    phd_prior = phd_prior.to(device=device)
    phd_prior.eval()

# Thay đổi pose_embedding logic (lines 223-226):
# ---- Body Pose Initialization ----
if use_signbposer:
    # Old: latent optimization
    pose_embedding = torch.zeros([batch_size, 33], dtype=dtype, device=device,
                                 requires_grad=True)
elif use_motionbert_prior or use_phd_prior:
    # New: direct body pose optimization
    body_pose_direct = torch.zeros([batch_size, 63], dtype=dtype, device=device,
                                   requires_grad=True)
    # Initialize từ SMPLer-X init nếu có
    if init_smplx_param is not None:
        body_pose_direct.data.copy_(
            torch.FloatTensor(init_smplx_param['body_pose']).to(device)
        )
    pose_embedding = body_pose_direct  # Reuse variable name
else:
    pose_embedding = None

# Thêm helper functions:
def load_motionbert_prior(checkpoint_path):
    """Load pretrained MotionBERT prior."""
    from signbposer.motionbert_prior import MotionBERTBodyPrior
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = MotionBERTBodyPrior(
        motionbert_ckpt_path='checkpoints/pose3d_MB_ft_h36m.tar',
        freeze_encoder=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return MotionBERTPriorWrapper(model)


def load_phd_prior(checkpoint_path):
    """Load pretrained PHD prior."""
    from signbposer.phd_score_network import PHDScoreNetwork
    from signbposer.phd_prior import PHDBodyPrior
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    score_network = PHDScoreNetwork(
        pose_dim=63,
        condition_dim=1024,
        hidden_dim=1024,
        num_layers=4,
    )
    score_network.load_state_dict(checkpoint['score_network_state_dict'])
    
    diffusion = PHDDiffusion(
        score_network=score_network,
        num_timesteps=checkpoint.get('num_timesteps', 1000),
    )
    
    return PHDBodyPrior(diffusion)
```

**File 3: `dexavatar_fitting/smplifyx/fitting.py`**

```python
# Trong SMPLifyLoss.forward (sau line 592):

# ---- MotionBERT Prior Loss ----
elif use_motionbert_prior:
    # Direct optimization: body_pose (63-dim) thay vì latent (33-dim)
    pprior_loss = motionbert_prior.prior_loss(
        body_pose=body_pose,
        keypoints_2d=keypoints_2d
    )
    
    # Init prior: L1 vs SMPLer-X init
    pprior_loss += self.data_init_core_weight * torch.abs(
        body_pose[:, 0:11*3] - psmplx_bodyGT[:, 0:11*3]
    ).sum()
    pprior_loss += self.data_init_noncore_weight * torch.abs(
        body_pose[:, 11*3:] - psmplx_bodyGT[:, 11*3:]
    ).sum()

# ---- PHD Prior Loss ----
elif use_phd_prior:
    # Score-based prior loss
    pprior_loss = phd_prior.prior_loss(
        body_pose=body_pose,
        condition=condition,
        t=phd_timestep
    )
    
    # Init prior
    pprior_loss += self.data_init_core_weight * torch.abs(
        body_pose[:, 0:11*3] - psmplx_bodyGT[:, 0:11*3]
    ).sum()
    pprior_loss += self.data_init_noncore_weight * torch.abs(
        body_pose[:, 11*3:] - psmplx_bodyGT[:, 11*3:]
    ).sum()

# Trong create_fitting_closure (line 248-250):
# Thay đổi decode logic:
if use_signbposer:
    body_pose = signbposer.decode(
        pose_embedding, output_type='aa').view(1, -1)
elif use_motionbert_prior or use_phd_prior:
    # Direct body_pose, không cần decode
    body_pose = pose_embedding  # pose_embedding IS body_pose
```

#### 6.1.3 YAML config files

**New config: `cfg_files/fit_smplx_motionbert.yaml`**

```yaml
# Copy từ fit_smplx_vposer_x.yaml và thay đổi:

# Prior settings
use_signbposer: false
use_motionbert_prior: true
motionbert_prior_dir: './smplifyx/signbposer/snapshots/motionbert_best.pt'

# Optimization settings (tương tự)
body_pose_weight: 1000
optim_type: 'lbfgsls'
ftol: 1e-9
gtol: 1e-9
maxiters: 30

# Direct optimization (bắt buộc với MotionBERT)
use_direct_optimization: true
direct_body_weight: 500
```

**New config: `cfg_files/fit_smplx_phd.yaml`**

```yaml
# Prior settings
use_signbposer: false
use_phd_prior: true
phd_prior_dir: './smplifyx/signbposer/snapshots/phd_best.pt'
phd_guidance_scale: 1.0
phd_num_inference_steps: 50

# Optimization settings
body_pose_weight: 1000
optim_type: 'lbfgsls'
ftol: 1e-9
gtol: 1e-9
maxiters: 30

# Direct optimization
use_direct_optimization: true
direct_body_weight: 500
```

### 6.2 Flow diagram: Old vs New

```
┌─────────────────────────────────────────────────────────────────┐
│                    OLD: SignBPoser                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Config: use_signbposer: true                                     │
│                                                                   │
│  fit_single_frame.py:                                             │
│  ├── pose_embedding = zeros(1, 33) ← OPTIMIZE THIS               │
│  ├── signbposer = load_signbposer(dir)                            │
│  ├── body_pose = signbposer.decode(pose_embedding, 'aa')          │
│  └── L_prior = ||pose_embedding||² + L1(decoded - init)           │
│                                                                   │
│  fitting.py (closure):                                            │
│  ├── body_pose = signbposer.decode(pose_embedding, 'aa')          │
│  └── body_model(body_pose=body_pose) → joints → L_data           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    NEW: MotionBERT Prior                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Config: use_motionbert_prior: true                               │
│                                                                   │
│  fit_single_frame.py:                                             │
│  ├── body_pose = smplerx_init.clone().requires_grad_(True)        │
│  │   ← OPTIMIZE THIS (63-dim, direct)                            │
│  ├── motionbert = load_motionbert_prior(dir)                      │
│  └── L_prior = motionbert.prior_loss(body_pose, kpts_2d)         │
│                                                                   │
│  fitting.py (closure):                                            │
│  ├── body_model(body_pose=body_pose) → joints → L_data           │
│  └── No decode needed (body_pose is already in AA format)         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    NEW: PHD Diffusion Prior                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Config: use_phd_prior: true                                      │
│                                                                   │
│  fit_single_frame.py:                                             │
│  ├── body_pose = smplerx_init.clone().requires_grad_(True)        │
│  │   ← OPTIMIZE THIS (63-dim, direct)                            │
│  ├── phd = load_phd_prior(dir)                                    │
│  └── L_prior = phd.prior_loss(body_pose, condition, t)            │
│                                                                   │
│  fitting.py (closure):                                            │
│  ├── body_model(body_pose=body_pose) → joints → L_data           │
│  └── Score gradient guides body_pose toward realistic poses       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Directory Structure

```
/home/haipd/DexAvatar/
├── MotionBERT/                           # EXISTING (nếu có) hoặc clone mới
│   ├── models/
│   │   └── motion_bert.py
│   ├── checkpoints/
│   │   └── pose3d_MB_ft_h36m.tar        # Download pretrained
│   └── ...
│
├── PHD/                                  # NEW: Clone PHD repo
│   ├── models/
│   │   └── score_network.py
│   ├── checkpoints/
│   │   └── phd_smpl_prior.pt            # Download pretrained
│   └── ...
│
├── dexavatar_fitting/
│   ├── smplifyx/
│   │   ├── signbposer/
│   │   │   ├── signbposer.py            # EXISTING - giữ nguyên
│   │   │   ├── motionbert_prior.py      # NEW
│   │   │   ├── motionbert_wrapper.py    # NEW
│   │   │   ├── phd_score_network.py     # NEW
│   │   │   ├── phd_diffusion.py         # NEW
│   │   │   ├── phd_prior.py             # NEW
│   │   │   └── snapshots/
│   │   │       ├── TR00_E078.pt         # EXISTING
│   │   │       ├── motionbert_best.pt   # NEW (trained)
│   │   │       └── phd_best.pt          # NEW (trained)
│   │   │
│   │   ├── fit_single_frame.py          # MODIFY
│   │   ├── fitting.py                   # MODIFY
│   │   ├── cmd_parser.py               # MODIFY
│   │   └── test_bposer.py              # EXISTING - giữ nguyên
│   │
│   └── cfg_files/
│       ├── fit_smplx_vposer_x.yaml      # EXISTING - baseline
│       ├── fit_smplx_vposer_x_direct.yaml # EXISTING
│       ├── fit_smplx_motionbert.yaml    # NEW
│       └── fit_smplx_phd.yaml           # NEW
│
├── scripts/
│   ├── train_motionbert_prior.py        # NEW
│   ├── train_phd_prior.py               # NEW
│   ├── prepare_signbposer_data.py       # NEW
│   ├── extract_phoenix_smplx.py         # NEW
│   ├── M4_smplifyx_pose_motionbert.sh   # NEW
│   └── M4_smplifyx_pose_phd.sh          # NEW
│
├── checkpoints/
│   ├── motionbert/
│   │   └── pose3d_MB_ft_h36m.tar       # Download pretrained
│   ├── phd/
│   │   └── phd_smpl_prior.pt           # Download pretrained
│   └── smpler_x_h32.pth.tar            # EXISTING
│
└── data/
    ├── signbposer_data/                  # NEW: unified training data
    │   ├── train/
    │   │   ├── body_poses.npy
    │   │   ├── metadata.pkl
    │   │   └── sample_weights.npy
    │   ├── val/
    │   │   ├── body_poses.npy
    │   │   └── metadata.pkl
    │   └── test/
    │       ├── body_poses.npy
    │       └── metadata.pkl
    ├── frames/                           # EXISTING
    ├── smplx_gt/                         # EXISTING
    └── segment.json                      # EXISTING
```

---

## 8. Training Pipeline

### 8.1 Phase 1: Data Preparation

```bash
# Step 1: Download datasets
# SignAvatars
git clone https://github.com/yzd-v/SignAvatars.git
# PHOENIX14T
# Download from: https://www-i6.informatik.rwth-aachen.de/~koller/Phoenix-14T/

# Step 2: Download pretrained models
# MotionBERT
wget -P checkpoints/motionbert/ https://github.com/Walter0807/MotionBERT/releases/download/v1.0/pose3d_MB_ft_h36m.tar
# PHD
wget -P checkpoints/phd/ https://github.com/LemonATsu/Pose-from-Human-Diffusion/releases/download/v1.0/phd_smpl_prior.pt

# Step 3: Process data
python scripts/prepare_signbposer_data.py \
    --signavatars_dir /path/to/SignAvatars \
    --phoenix_dir /path/to/PHOENIX14T \
    --output_dir data/signbposer_data \
    --smplerx_ckpt checkpoints/smpler_x_h32.pth.tar

# Step 4: Extract PHOENIX14T SMPL-X (nếu chưa có)
python scripts/extract_phoenix_smplx.py \
    --phoenix_dir /path/to/PHOENIX14T \
    --smplerx_ckpt checkpoints/smpler_x_h32.pth.tar \
    --output_dir data/signbposer_data/phoenix_smplx
```

### 8.2 Phase 2: Train MotionBERT Prior

```bash
# Phase 2a: Freeze encoder, train SMPL head only
python scripts/train_motionbert_prior.py \
    --motionbert_ckpt checkpoints/motionbert/pose3d_MB_ft_h36m.tar \
    --data_dir data/signbposer_data \
    --output_dir dexavatar_fitting/smplifyx/signbposer/snapshots \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3

# Phase 2b: (Optional) Unfreeze encoder, finetune all
python scripts/train_motionbert_prior.py \
    --motionbert_ckpt checkpoints/motionbert/pose3d_MB_ft_h36m.tar \
    --data_dir data/signbposer_data \
    --output_dir dexavatar_fitting/smplifyx/signbposer/snapshots \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --unfreeze_encoder
```

### 8.3 Phase 3: Train PHD Prior

```bash
python scripts/train_phd_prior.py \
    --data_dir data/signbposer_data \
    --output_dir dexavatar_fitting/smplifyx/signbposer/snapshots \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_timesteps 1000
```

### 8.4 Phase 4: Fitting & Evaluation

```bash
# MotionBERT prior fitting
python scripts/M4_smplifyx_pose_motionbert.sh

# PHD prior fitting
python scripts/M4_smplifyx_pose_phd.sh

# Evaluation
python evaluation/eval_mpvpe_regions.py --method motionbert
python evaluation/eval_mpvpe_regions.py --method phd
```

---

## 9. Evaluation & Metrics

### 9.1 Prior Quality Metrics (trong training)

| Metric | Mô tả | MotionBERT | PHD |
|--------|-------|------------|-----|
| **Val Loss** | MSE/L1 trên validation set | ✓ | ✓ |
| **Log-likelihood** | Approximate LL (higher=better) | - | ✓ |
| **Reconstruction** | ||decoded - original|| | ✓ | ✓ |
| **Sampling quality** | Plausibility of samples | - | ✓ |

### 9.2 Downstream Metrics (trên SGNify benchmark)

| Metric | Mô tả | Baseline (SignBPoser) | Target |
|--------|-------|----------------------|--------|
| **UBody (mm)** | Upper body MPVPE | ? | < baseline |
| **LHand (mm)** | Left hand MPVPE | ? | < baseline |
| **RHand (mm)** | Right hand MPVPE | ? | < baseline |
| **MPVPE (mm)** | Overall MPVPE | ? | < baseline |
| **MPJPE (mm)** | Per-joint error | ? | < baseline |

### 9.3 Ablation Studies

```
Ablation 1: Prior method comparison
├── SignBPoser (baseline)
├── GMM Prior (K=16)
├── MotionBERT Prior
├── PHD Prior
└── Ensemble (MotionBERT + PHD)

Ablation 2: Training data contribution
├── SignAvatars only
├── PHOENIX14T only
├── SignAvatars + PHOENIX14T
└── + data weighting strategies

Ablation 3: Pretrained vs from-scratch
├── MotionBERT pretrained on AMASS
├── MotionBERT trained from scratch
├── PHD pretrained
└── PHD from scratch
```

---

## 10. Timeline & Milestones

```
Week 1: Data Preparation
├── Download SignAvatars, PHOENIX14T
├── Download pretrained MotionBERT, PHD
├── Run prepare_signbposer_data.py
└── Verify data quality

Week 2-3: MotionBERT Prior
├── Implement motionbert_prior.py
├── Implement motionbert_wrapper.py
├── Train SMPL head (Phase 2a)
├── (Optional) Finetune all (Phase 2b)
└── Integration test

Week 3-4: PHD Prior
├── Implement phd_score_network.py
├── Implement phd_diffusion.py
├── Implement phd_prior.py
├── Train score network
└── Integration test

Week 5: Integration
├── Modify fit_single_frame.py
├── Modify fitting.py
├── Modify cmd_parser.py
├── Create new YAML configs
└── Integration test trên SGNify

Week 6: Evaluation
├── Run fitting với MotionBERT prior
├── Run fitting với PHD prior
├── Compute UBody/LHand/RHand
├── Ablation studies
└── So sánh với SignBPoser baseline

Week 7-8: Paper
├── Viết methodology section
├── Chạy thêm experiments
├── Tạo tables/figures
└── Submit
```

---

## 11. References

| Paper | Venue | Relevance | GitHub |
|-------|-------|-----------|--------|
| MotionBERT | ICCV 2023 | Pretrained motion encoder | [Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT) |
| PHD | 2024 | Score-based body prior for SMPLify | [LemonATsu/Pose-from-Human-Diffusion](https://github.com/LemonATsu/Pose-from-Human-Diffusion) |
| MDM | CVPR 2023 | Motion diffusion model | [GuyTevet/MDM](https://github.com/GuyTevet/MDM) |
| T2M-GPT | CVPR 2023 | VQ-VAE + GPT motion generation | [Murrol/T2M-GPT](https://github.com/Murrol/T2M-GPT) |
| SignAvatars | ECCV 2024 | Sign language SMPL-X dataset | [yzd-v/SignAvatars](https://github.com/yzd-v/SignAvatars) |
| SGNify | CVPR 2023 | Sign language avatar reconstruction | [sgnify.is.tue.mpg.de](https://sgnify.is.tue.mpg.de/) |
| SMPLify-X | ICCV 2019 | Body fitting with VAE prior | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) |
| DexAvatar | - | Current baseline | This repo |
