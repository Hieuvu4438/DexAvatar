#!/usr/bin/env python3
"""
Train a SOKE-style VQVAE on SMPL-X hand pose.

Reads hand poses from /home/haipd/DexAvatar/data/vqvae_hand_data/{train,val}/
and trains a `SignHVQVAE` (SOKE VQVAE with nfeats=45, code_num=192).

This is the **optional** fine-tuning path. The default is to use a
pretrained SOKE VQVAE checkpoint via `load_signhposer_vqvae`.

Output (created, never overwrites source):
    /home/haipd/DexAvatar/checkpoints/vqvae_hand/signhposer_vqvae/last.ckpt
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make our wrapper importable.
DEXAVATAR_ROOT = "/home/haipd/DexAvatar"
sys.path.insert(0, os.path.join(DEXAVATAR_ROOT, "dexavatar_fitting", "smplifyx"))

from signhposer_vqvae.vqvae_hand import SignHVQVAE


def load_split(data_dir, split):
    path_l = os.path.join(data_dir, split, "lhand_poses.npy")
    path_r = os.path.join(data_dir, split, "rhand_poses.npy")
    if not (os.path.exists(path_l) and os.path.exists(path_r)):
        return None
    l = np.load(path_l).astype(np.float32)
    r = np.load(path_r).astype(np.float32)
    # Concatenate lhand and rhand along the time dim (nfeats=90 -> 30 joints);
    # the VQVAE is per-nfeats, so we keep them separate and merge into a
    # single dataset of (N, 45) poses.
    return np.concatenate([l, r], axis=0)


def to_loader(poses, batch_size, shuffle):
    # Wrap in time axis T=min_T (8) for the SOKE temporal conv stack.
    t_min = 8
    N = poses.shape[0]
    poses_repeat = np.repeat(poses[:, None, :], t_min, axis=1).astype(np.float32)
    tensor = torch.from_numpy(poses_repeat)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/haipd/DexAvatar/data/vqvae_hand_data")
    parser.add_argument("--output_dir", default="/home/haipd/DexAvatar/checkpoints/vqvae_hand/signhposer_vqvae")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--code_num", type=int, default=192)
    parser.add_argument("--code_dim", type=int, default=512)
    parser.add_argument("--lambda_commit", type=float, default=0.02)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init_ckpt", default="",
                        help="Optional: SOKE pretrained .ckpt to fine-tune from.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Data.
    train = load_split(args.data_dir, "train")
    val = load_split(args.data_dir, "val")
    if train is None:
        raise SystemExit(f"No train data at {args.data_dir}/train. "
                         f"Run scripts/extract_sign_hand_pose.py first.")
    print(f"train: {train.shape}, val: {val.shape if val is not None else None}")
    train_loader = to_loader(train, args.batch_size, shuffle=True)
    val_loader = to_loader(val, args.batch_size, shuffle=False) if val is not None else None

    # Model.
    model = SignHVQVAE(nfeats=45, code_num=args.code_num, code_dim=args.code_dim,
                       latent_dim=23, down_t=2, stride_t=2, width=512, depth=3,
                       dilation_growth_rate=3, quantizer="ema_reset").to(args.device)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        from signhposer_vqvae.loaders import load_signhposer_vqvae
        pretrained, _ = load_signhposer_vqvae(ckpt_path=args.init_ckpt, map_location=args.device)
        model.vqvae.load_state_dict(pretrained.vqvae.state_dict(), strict=False)
        print(f"Loaded pretrained vqvae weights from {args.init_ckpt}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for (x,) in train_loader:
            x = x.to(args.device)
            opt.zero_grad()
            recon, commit, perplexity = model(x)
            rec_l1 = (recon - x).abs().mean()
            loss = args.lambda_recon * rec_l1 + args.lambda_commit * commit
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        sched.step()
        train_loss = float(np.mean(train_losses))

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for (x,) in val_loader:
                    x = x.to(args.device)
                    recon, commit, _ = model(x)
                    rec_l1 = (recon - x).abs().mean()
                    v_losses.append((args.lambda_recon * rec_l1 + args.lambda_commit * commit).item())
            val_loss = float(np.mean(v_losses))

        dt = time.time() - t0
        print(f"epoch {epoch+1:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)")

        # Save last + best.
        ckpt_path = os.path.join(args.output_dir, "last.ckpt")
        torch.save({"state_dict": model.state_dict(),
                    "config": dict(nfeats=45, code_num=args.code_num, code_dim=args.code_dim,
                                   latent_dim=23, down_t=2, stride_t=2, width=512,
                                   depth=3, dilation_growth_rate=3, quantizer="ema_reset")},
                   ckpt_path)
        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "best.ckpt")
            torch.save({"state_dict": model.state_dict(),
                        "config": dict(nfeats=45, code_num=args.code_num, code_dim=args.code_dim,
                                       latent_dim=23, down_t=2, stride_t=2, width=512,
                                       depth=3, dilation_growth_rate=3, quantizer="ema_reset")},
                       best_path)
            print(f"  best so far → {best_path}")


if __name__ == "__main__":
    main()
