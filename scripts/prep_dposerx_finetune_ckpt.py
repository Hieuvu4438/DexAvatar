#!/usr/bin/env python3
"""
Prepare a DPoser-X body checkpoint for FINE-TUNING (weights-only init).

Problem: the DPoser-X trainer's --resume-ckpt does a full PyTorch-Lightning
resume, which restores `global_step`. The released AMASS body.ckpt was saved at
step ~200000; with fine-tune max_steps=30000, a plain resume would run 0 steps.

Fix: load the AMASS ckpt, KEEP `state_dict` + `model_ema` (the weights we want to
start from), DROP optimizer/scheduler/callback/loop state, and RESET
global_step=0 / epoch=0. Saving the result lets `--resume-ckpt` load the AMASS
weights while training a fresh 30k-step schedule.

Output (default): DPoser-X/checkpoints/dposer/sign/sign_body_ft/sign_init.ckpt
(this is under the ckpt_dir the trainer resolves --resume-ckpt against:
   checkpoints/dposer/{config.dataset=sign}/{name=sign_body_ft}/ )

Usage:
    python scripts/prep_dposerx_finetune_ckpt.py
"""
import os
import argparse
import torch

SRC_DEFAULT = "/home/haipd/DexAvatar/checkpoints/dposerx_body/body.ckpt"
DST_DEFAULT = "/home/haipd/DexAvatar/DPoser-X/checkpoints/dposer/sign/sign_body_ft/sign_init.ckpt"

# Keys that carry optimizer/scheduler/training-loop state -> drop for a fresh schedule.
DROP_KEYS = ("optimizer_states", "lr_schedulers", "loops", "callbacks")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT, help="Released AMASS body.ckpt to init from.")
    ap.add_argument("--dst", default=DST_DEFAULT, help="Output weights-only init checkpoint.")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"Source ckpt not found: {args.src}")

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"Unexpected checkpoint format (not a dict): {type(ckpt)}")

    print(f"Loaded {args.src}")
    print(f"  keys: {sorted(ckpt.keys())}")
    for k in DROP_KEYS:
        ckpt.pop(k, None)
    if "state_dict" not in ckpt:
        raise SystemExit("Checkpoint has no 'state_dict' -- cannot init from it.")
    if "model_ema" not in ckpt:
        print("  [WARN] no 'model_ema' in ckpt (trainer's on_load_checkpoint expects it).")

    # Strip the frozen SMPL eval/rendering aids (body_model_vis/eval/train). They are
    # NOT learned: they are built fresh from the SMPLX npz at construction and were
    # serialized into the released ckpt with num_expression=10 / older smplx buffers,
    # which mismatch the current model (num_expression=100 + extra persistent buffers)
    # and break strict restore. Drop them so the init ckpt carries ONLY score-model
    # weights; the trainer rebuilds the body models from the npz and loads relaxed.
    sd = ckpt["state_dict"]
    n_before = len(sd)
    ckpt["state_dict"] = {k: v for k, v in sd.items() if not k.startswith("body_model")}
    print(f"  stripped body_model* keys: {n_before - len(ckpt['state_dict'])} removed "
          f"-> {len(ckpt['state_dict'])} score-model tensors remain")

    ckpt["global_step"] = 0
    ckpt["epoch"] = 0

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"  -> saved weights-only init ckpt: {args.dst}")
    print(f"     state_dict tensors: {len(ckpt['state_dict'])}, "
          f"global_step={ckpt['global_step']}, epoch={ckpt['epoch']}")


if __name__ == "__main__":
    main()
