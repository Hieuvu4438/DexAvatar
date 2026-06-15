#!/usr/bin/env python3
"""
End-to-end smoke test for the SOKE VQVAE hand + DPoser-X body priors.

Loads both priors, runs a 1-frame forward pass through:
  * VQVAE encode_aa (fitting-time differentiable decode)
  * VQVAE forward (recon + commit loss)
  * DPoser-X prior_loss (body pose)
  * A simulated L-BFGS step (manual gradient descent on body + hand latents)

Pass criteria: all losses finite, gradients non-NaN, no exceptions.
"""
import os
import sys
import argparse
import numpy as np
import torch

DEXAVATAR_ROOT = "/home/haipd/DexAvatar"
sys.path.insert(0, os.path.join(DEXAVATAR_ROOT, "dexavatar_fitting", "smplifyx"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dposerx_ckpt",
                        default="/home/haipd/DexAvatar/checkpoints/dposerx_body/body.ckpt")
    parser.add_argument("--dposerx_config",
                        default="/home/haipd/DexAvatar/DPoser-X/configs/body/subvp/timefc.py")
    parser.add_argument("--dposerx_normalizer_dir",
                        default="/home/haipd/DexAvatar/checkpoints/dposerx_body/body_normalizer")
    parser.add_argument("--vqvae_ckpt",
                        default="/home/haipd/DexAvatar/checkpoints/vqvae_hand/signhposer_vqvae/best.ckpt",
                        help="Set to '' to use random-init VQVAE (smoke test only).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[smoke_test] device: {args.device}")
    print()

    # ---- VQVAE hand ----
    print("[smoke_test] Loading SOKE VQVAE hand prior...")
    from signhposer_vqvae.loaders import load_signhposer_vqvae
    vqvae, _ = load_signhposer_vqvae(
        ckpt_path=args.vqvae_ckpt, map_location=args.device)
    vqvae = vqvae.to(args.device)
    vqvae.eval()
    n_params = sum(p.numel() for p in vqvae.parameters())
    print(f"  VQVAE: code_num={vqvae.code_num}, code_dim={vqvae.code_dim}, params={n_params/1e6:.2f}M")

    # 1) Fitting-time differentiable decode
    z_hand = torch.randn(1, 23, requires_grad=True, device=args.device)
    decoded = vqvae.decode_aa(z_hand)
    assert decoded.shape == (1, 1, 15, 3), decoded.shape
    print(f"  decode_aa({z_hand.shape}) -> {decoded.shape}  ok")
    decoded.sum().backward()
    assert torch.isfinite(z_hand.grad).all(), "z_hand grad is non-finite"
    print(f"  backward through decode_aa  ok  (grad norm: {z_hand.grad.norm().item():.4f})")

    # 2) Standard VQVAE forward
    pose = torch.randn(1, 8, 45, device=args.device)
    recon, commit, perp = vqvae(pose)
    assert torch.isfinite(recon).all() and torch.isfinite(commit).all()
    print(f"  forward(1, 8, 45) -> recon {recon.shape}  commit {commit.item():.4f}  "
          f"perp {perp.item():.2f}/{vqvae.code_num}  ok")

    # 3) Recon loss
    loss = vqvae.recon_loss(pose)
    loss.backward()
    print(f"  recon_loss={loss.item():.4f}  ok")
    print()

    # ---- DPoser-X body ----
    print("[smoke_test] Loading DPoser-X body prior...")
    if not os.path.exists(args.dposerx_ckpt):
        print(f"  SKIP: DPoser-X ckpt not found: {args.dposerx_ckpt}")
        return 0
    from signbposer_dposerx.loaders import load_signbposer_dposerx
    prior = load_signbposer_dposerx(
        config_path=args.dposerx_config,
        ckpt_path=args.dposerx_ckpt,
        body_normalizer_path=args.dposerx_normalizer_dir,
        device=args.device,
    )
    n_params = sum(p.numel() for p in prior.parameters())
    print(f"  DPoser-X: n_poses={prior._n_poses}, pose_dim={prior._pose_dim}, "
          f"params={n_params/1e6:.2f}M, sde={prior.sde.__class__.__name__}")

    # 4) Prior loss on a random body pose
    body_pose = torch.randn(1, 63, requires_grad=True, device=args.device)
    loss = prior.prior_loss(body_pose)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    print(f"  prior_loss(random body pose) = {loss.item():.4f}  ok")
    loss.backward()
    assert torch.isfinite(body_pose.grad).all(), "body_pose grad is non-finite"
    print(f"  backward  ok  (grad norm: {body_pose.grad.norm().item():.4f})")

    # 5) Prior loss on a real sign body pose (from existing data)
    body_npy = "/home/haipd/DexAvatar/data/signbposer_data/train/body_poses.npy"
    if os.path.exists(body_npy):
        bodies = np.load(body_npy)
        real = torch.from_numpy(bodies[:1]).float().to(args.device).requires_grad_(True)
        real_loss = prior.prior_loss(real)
        print(f"  prior_loss(real sign pose, mean abs={real.abs().mean().item():.3f}) = "
              f"{real_loss.item():.4f}  ok")

    # 6) Multi-step denoise (sanity check, no fitting-time relevance)
    body = torch.randn(1, 63, device=args.device)
    decoded = prior.decode_to_pose(body, num_steps=5)
    assert torch.isfinite(decoded).all()
    print(f"  decode_to_pose(random, 5 steps) -> {decoded.shape}  ok")
    print()

    # ---- Combined L-BFGS-style step ----
    print("[smoke_test] Combined gradient step (body + hand latents together)...")
    body = torch.randn(1, 63, requires_grad=True, device=args.device)
    lhand = torch.randn(1, 23, requires_grad=True, device=args.device)
    rhand = torch.randn(1, 23, requires_grad=True, device=args.device)
    body_loss = prior.prior_loss(body)
    lhand_pose = vqvae.decode_aa(lhand)
    rhand_pose = vqvae.decode_aa(rhand)
    hand_loss = (lhand_pose.pow(2).mean() + rhand_pose.pow(2).mean())
    total = body_loss + 0.5 * hand_loss
    total.backward()
    assert torch.isfinite(body.grad).all() and torch.isfinite(lhand.grad).all() and \
           torch.isfinite(rhand.grad).all(), "grads not finite"
    print(f"  body_loss={body_loss.item():.4f}  hand_loss={hand_loss.item():.4f}  "
          f"total={total.item():.4f}  ok")
    print(f"  body.grad.norm={body.grad.norm().item():.4f}  "
          f"lhand.grad.norm={lhand.grad.norm().item():.4f}  "
          f"rhand.grad.norm={rhand.grad.norm().item():.4f}")
    print()
    print("=" * 60)
    print("[smoke_test] ALL OK — SOKE VQVAE + DPoser-X priors wire up end-to-end")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
