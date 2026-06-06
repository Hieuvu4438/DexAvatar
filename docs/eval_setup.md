# Evaluation setup (TR-V2V Table-1 style)

This repository includes `evaluation/evaluation_trv2v_wilor.py` for evaluating DexAvatar outputs on the 57-sign split.

## 1) Required assets

You need:

1. **Predictions** from DexAvatar pipeline (HAMER baseline or WiLor variant), organized per-sign.
2. **GT vertices** (`smplxgt`) with matching per-sign + per-frame relative paths.
3. **Region index files** for SMPL-X vertices:
   - `UBody(-F)` indices
   - `LHand` indices
   - `RHand` indices

> Note: The script intentionally requires region index files as inputs to avoid hardcoding ambiguous mappings.

## 2) Run WiLor pipeline

```bash
python runners/run_dexavatar_wilor.py \
  --input_img_folder <DATA_PATH> \
  --output_path <OUTPUT_WILOR> \
  --fitting_experiment ./dexavatar_fitting
```

## 3) Run evaluation

```bash
python evaluation/evaluation_trv2v_wilor.py \
  --pred_root <OUTPUT_WILOR> \
  --gt_root <SMPLXGT_ROOT> \
  --signs_txt data/signs.txt \
  --segment_json data/segment.json \
  --ubody_indices <ubody_minus_face_indices.npy> \
  --lhand_indices <left_hand_indices.npy> \
  --rhand_indices <right_hand_indices.npy> \
  --method_name DexAvatar-WiLor
```

Output includes:
- overall mean: `UBody(-F), LHand, RHand` (mm)
- frame count used
- sign coverage
- per-sign summary table

## 4) Baseline comparison (HAMER)

Run original pipeline:

```bash
python methods/run_dexavatar.py \
  --input_img_folder <DATA_PATH> \
  --output_path <OUTPUT_HAMER> \
  --fitting_experiment ./dexavatar_fitting
```

Then evaluate with same GT + index files:

```bash
python evaluation/evaluation_trv2v_wilor.py \
  --pred_root <OUTPUT_HAMER> \
  --gt_root <SMPLXGT_ROOT> \
  --ubody_indices <ubody_minus_face_indices.npy> \
  --lhand_indices <left_hand_indices.npy> \
  --rhand_indices <right_hand_indices.npy> \
  --method_name DexAvatar-HAMER
```
