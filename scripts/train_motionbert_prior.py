#!/usr/bin/env python3
"""
Train MotionBERT Body Pose Prior
=================================
Phase 1: Freeze MotionBERT encoder, train SMPL regression head only
Phase 2: (Optional) Unfreeze encoder, finetune all với small lr

Usage:
    # Phase 1: Train SMPL head only
    python scripts/train_motionbert_prior.py \
        --data_dir data/signbposer_data \
        --output_dir dexavatar_fitting/smplifyx/signbposer/snapshots \
        --epochs 100 --batch_size 256 --lr 1e-3

    # Phase 2: Finetune all
    python scripts/train_motionbert_prior.py \
        --data_dir data/signbposer_data \
        --output_dir dexavatar_fitting/smplifyx/signbposer/snapshots \
        --epochs 50 --batch_size 128 --lr 1e-4 \
        --unfreeze_encoder --resume_from checkpoints/motionbert_prior/best_model.pt
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Thêm project root vào path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dexavatar_fitting', 'smplifyx'))
sys.path.insert(0, os.path.join(project_root, 'MotionBERT'))

from signbposer.motionbert_prior import MotionBERTBodyPrior


def load_data(data_dir, split='train'):
    """Load training/validation data."""
    split_dir = os.path.join(data_dir, split)

    body_poses_path = os.path.join(split_dir, 'body_poses.npy')
    if not os.path.exists(body_poses_path):
        raise FileNotFoundError(f"Data not found: {body_poses_path}")

    body_poses = np.load(body_poses_path)
    print(f"[{split}] Loaded {len(body_poses)} body poses, shape: {body_poses.shape}")

    # Load weights (optional)
    weights_path = os.path.join(split_dir, 'sample_weights.npy')
    if os.path.exists(weights_path):
        weights = np.load(weights_path)
    else:
        weights = np.ones(len(body_poses), dtype=np.float32)

    return body_poses, weights


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_reg = 0
    n_batches = 0

    for batch_idx, (body_pose_gt,) in enumerate(dataloader):
        body_pose_gt = body_pose_gt.to(device)

        optimizer.zero_grad()

        # Forward: predict body_pose từ dummy features
        # (trong thực tế sẽ dùng 2D keypoints)
        # Ở đây train SMPL head để learn mapping từ features → body_pose
        batch_size = body_pose_gt.shape[0]

        # Simulate motion features (placeholder)
        # Khi có real keypoints, thay bằng: features = model.encode(keypoints_2d)
        dummy_features = torch.randn(batch_size, 1, model.encoder_dim, device=device)

        # Predict
        body_pose_pred = model.smpl_head(dummy_features)

        # Loss
        loss_recon = nn.functional.mse_loss(body_pose_pred, body_pose_gt)
        loss_l1 = nn.functional.l1_loss(body_pose_pred, body_pose_gt)
        loss_reg = 1e-4 * torch.norm(body_pose_pred, p=2)

        loss = loss_recon + 0.5 * loss_l1 + loss_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_reg += loss_reg.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={loss.item():.6f} recon={loss_recon.item():.6f}")

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'reg': total_reg / n_batches,
    }


def validate(model, dataloader, device):
    """Validate."""
    model.eval()
    total_loss = 0
    total_mpe = 0  # mean per-joint error
    n_batches = 0

    with torch.no_grad():
        for body_pose_gt, in dataloader:
            body_pose_gt = body_pose_gt.to(device)
            batch_size = body_pose_gt.shape[0]

            dummy_features = torch.randn(batch_size, 1, model.encoder_dim, device=device)
            body_pose_pred = model.smpl_head(dummy_features)

            loss = nn.functional.mse_loss(body_pose_pred, body_pose_gt)
            mpe = nn.functional.l1_loss(body_pose_pred, body_pose_gt)

            total_loss += loss.item()
            total_mpe += mpe.item()
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mpe': total_mpe / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train MotionBERT Body Prior')
    parser.add_argument('--data_dir', type=str, default='data/signbposer_data')
    parser.add_argument('--output_dir', type=str,
                        default='dexavatar_fitting/smplifyx/signbposer/snapshots')
    parser.add_argument('--motionbert_ckpt', type=str,
                        default='checkpoints/motionbert/pose3d_MB_ft_h36m.tar')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--unfreeze_encoder', action='store_true',
                        help='Unfreeze MotionBERT encoder for finetuning')
    parser.add_argument('--resume_from', type=str, default='',
                        help='Resume training from checkpoint')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_poses, train_weights = load_data(args.data_dir, 'train')
    val_poses, val_weights = load_data(args.data_dir, 'val')

    train_dataset = TensorDataset(torch.FloatTensor(train_poses))
    val_dataset = TensorDataset(torch.FloatTensor(val_poses))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # Model
    model = MotionBERTBodyPrior(
        motionbert_ckpt_path=args.motionbert_ckpt if not args.unfreeze_encoder else None,
        freeze_encoder=not args.unfreeze_encoder,
        input_dim=256,
        hidden_dim=args.hidden_dim,
        num_joints=21,
        output_type='aa',
        dropout=args.dropout,
    ).to(device)

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from {args.resume_from}, epoch {start_epoch}")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training MotionBERT Body Prior")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Train samples: {len(train_poses)}, Val samples: {len(val_poses)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) "
              f"lr={lr:.6f}")
        print(f"  Train: loss={train_metrics['loss']:.6f} "
              f"recon={train_metrics['recon']:.6f}")
        print(f"  Val:   loss={val_metrics['loss']:.6f} "
              f"mpe={val_metrics['mpe']:.6f}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            save_path = os.path.join(args.output_dir, 'motionbert_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': vars(args),
            }, save_path)
            print(f"  → Saved best model (val_loss={val_metrics['loss']:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'motionbert_best.pt')}")


if __name__ == '__main__':
    main()
