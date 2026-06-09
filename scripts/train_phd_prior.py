#!/usr/bin/env python3
"""
Train PHD (Pose from Human Diffusion) Body Pose Prior
======================================================
Score-based diffusion model cho body pose distribution learning.

Usage:
    python scripts/train_phd_prior.py \
        --data_dir data/signbposer_data \
        --output_dir dexavatar_fitting/smplifyx/signbposer/snapshots \
        --epochs 200 --batch_size 256 --lr 1e-4 \
        --num_timesteps 1000 --hidden_dim 1024 --num_layers 4
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Thêm project root vào path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dexavatar_fitting', 'smplifyx'))

from signbposer.phd_score_network import PHDScoreNetwork
from signbposer.phd_diffusion import PHDDiffusion


def load_data(data_dir, split='train'):
    """Load training/validation data."""
    split_dir = os.path.join(data_dir, split)
    body_poses_path = os.path.join(split_dir, 'body_poses.npy')

    if not os.path.exists(body_poses_path):
        raise FileNotFoundError(f"Data not found: {body_poses_path}")

    body_poses = np.load(body_poses_path)
    print(f"[{split}] Loaded {len(body_poses)} body poses, shape: {body_poses.shape}")

    return body_poses


def train_epoch(diffusion, dataloader, optimizer, device, epoch):
    """Train one epoch."""
    diffusion.score_network.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (body_pose_gt,) in enumerate(dataloader):
        body_pose_gt = body_pose_gt.to(device)

        optimizer.zero_grad()

        # Compute diffusion training loss
        loss, info = diffusion.training_loss(body_pose_gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.score_network.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.6f}")

    return {'loss': total_loss / n_batches}


def validate(diffusion, dataloader, device):
    """Validate."""
    diffusion.score_network.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for body_pose_gt, in dataloader:
            body_pose_gt = body_pose_gt.to(device)
            loss, _ = diffusion.training_loss(body_pose_gt)
            total_loss += loss.item()
            n_batches += 1

    return {'loss': total_loss / n_batches}


def sample_and_evaluate(diffusion, device, num_samples=100):
    """
    Sample từ trained model và evaluate quality.
    """
    diffusion.score_network.eval()
    with torch.no_grad():
        samples = diffusion.sample(
            shape=(num_samples, 63),
            device=device,
        )

    # Statistics
    mean_pose = samples.mean(dim=0)
    std_pose = samples.std(dim=0)
    mean_norm = torch.norm(samples, dim=1).mean()

    return {
        'samples': samples,
        'mean_pose': mean_pose,
        'std_pose': std_pose,
        'mean_norm': mean_norm.item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Train PHD Body Pose Prior')
    parser.add_argument('--data_dir', type=str, default='data/signbposer_data')
    parser.add_argument('--output_dir', type=str,
                        default='dexavatar_fitting/smplifyx/signbposer/snapshots')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'])
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--time_embed_dim', type=int, default=256)
    parser.add_argument('--condition_dim', type=int, default=1024)
    parser.add_argument('--use_condition', action='store_true',
                        help='Use SMPLer-X conditioning')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--resume_from', type=str, default='')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--sample_interval', type=int, default=20,
                        help='Sample every N epochs')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='DDIM steps for fast sampling')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_poses = load_data(args.data_dir, 'train')
    val_poses = load_data(args.data_dir, 'val')

    train_dataset = TensorDataset(torch.FloatTensor(train_poses))
    val_dataset = TensorDataset(torch.FloatTensor(val_poses))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # Model
    score_network = PHDScoreNetwork(
        pose_dim=63,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_embed_dim=args.time_embed_dim,
        dropout=args.dropout,
        use_condition=args.use_condition,
    ).to(device)

    diffusion = PHDDiffusion(
        score_network=score_network,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
    )

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        score_network.load_state_dict(checkpoint['score_network_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from {args.resume_from}, epoch {start_epoch}")

    # Optimizer
    trainable_params = list(score_network.parameters())
    num_params = sum(p.numel() for p in trainable_params)
    print(f"Score network parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training PHD Body Pose Prior")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Timesteps: {args.num_timesteps}, Schedule: {args.beta_schedule}")
    print(f"Hidden: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f"Train: {len(train_poses)}, Val: {len(val_poses)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(diffusion, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate(diffusion, val_loader, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) lr={lr:.6f}")
        print(f"  Train: loss={train_metrics['loss']:.6f}")
        print(f"  Val:   loss={val_metrics['loss']:.6f}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            save_path = os.path.join(args.output_dir, 'phd_best.pt')
            torch.save({
                'epoch': epoch,
                'score_network_state_dict': score_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': {
                    'pose_dim': 63,
                    'condition_dim': args.condition_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'time_embed_dim': args.time_embed_dim,
                    'dropout': args.dropout,
                    'use_condition': args.use_condition,
                    'num_timesteps': args.num_timesteps,
                    'beta_schedule': args.beta_schedule,
                    'guidance_scale': 1.0,
                    'num_inference_steps': args.ddim_steps,
                },
            }, save_path)
            print(f"  → Saved best model (val_loss={val_metrics['loss']:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Periodic sampling
        if (epoch + 1) % args.sample_interval == 0:
            print(f"\n  Sampling from model...")
            sample_stats = sample_and_evaluate(diffusion, device, num_samples=50)
            print(f"  Sample stats: mean_norm={sample_stats['mean_norm']:.4f}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'phd_best.pt')}")


if __name__ == '__main__':
    main()
