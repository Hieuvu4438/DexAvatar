# SignEFT-X

Implementation of `docs/proposal12/SignEFT_X_NoMarker_Implementation_v3.md`.
The implementation is RGB-only, frame-independent, checksum-locks the author
evaluator, and always keeps byte-exact A3f rollback as candidate zero.

Current implementation order follows the blueprint: C0 parity, C1 real
heatmap likelihood, C2 NLF bone evidence, C3 kinematic wrist protection, then
the hand and dense-evidence ladders only after their prerequisites pass.

```bash
export PYTHONPATH="$PWD/src"
python -m pytest -q
python -m signeft.cli --config configs/ablations/c0_a3f.yaml protocol-lock
python -m signeft.cli --config configs/ablations/c0_a3f.yaml prepare-manifest
python -m signeft.cli --config configs/ablations/c0_a3f.yaml refine
python -m signeft.cli --config configs/ablations/c0_a3f.yaml materialize
python -m signeft.cli --config configs/ablations/c0_a3f.yaml preflight
python -m signeft.cli --config configs/ablations/c0_a3f.yaml evaluate-official \
  --gt-root /absolute/path/to/smplx_gt
```

The fitting/refinement process never receives a ground-truth path. Ground truth
is passed only to the locked evaluator subprocess.

Real observation export (no synthetic Gaussian heatmaps):

```bash
PYTHONPATH=src python scripts/export_sapiens_heatmaps.py \
  --manifest manifests/trv2v.jsonl \
  --out observations/sapiens_pose_v1 \
  --checkpoint /absolute/path/sapiens_1b_goliath_torchscript.pt2 \
  --sapiens-root /absolute/path/sapiens --batch-size 4

PYTHONPATH=src python scripts/export_nlf.py \
  --manifest manifests/trv2v.jsonl \
  --out observations/nlf_v1 \
  --checkpoint /absolute/path/nlf_l_multi_0.3.2.torchscript \
  --nlf-root /absolute/path/nlf --batch-size 4

PYTHONPATH=src python scripts/validate_observations.py \
  --kind pose --manifest manifests/trv2v.jsonl \
  --root observations/sapiens_pose_v1 --out observations/pose_validation.json
```

C0 is verified on the attached 57-sign/1,493-frame protocol with exact A3f
parity: All 42.0936, UBody 25.8311, UBody-F 29.1458, UBody-H 39.6963,
LHand 12.8466, RHand 12.1275 mm. This protocol has not yet been reconciled
with the reported 2,872-frame paper setup, so results must not be described as
paper-comparable SOTA.
