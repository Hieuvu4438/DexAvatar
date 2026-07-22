# Reproduction Plan: Replicating TR-V2V Paper Evaluation Results

This document serves as a step-by-step reproduction plan and technical guide to replicate the exact TR-V2V evaluation metrics reported in Table 1 of the paper using the original evaluation script and configurations provided by the main author.

## 1. Reproduction Target (Table 1 Metrics)

Our goal is to reproduce the following quantitative results for the baseline and proposed methods:
* **UBody(-F)**: **30.13 mm** (Upper Body excluding face)
* **LHand**: **13.53 mm** (Left Hand - two-handed active average)
* **RHand**: **13.08 mm** (Right Hand - average over all signs)

---

## 2. Directory Structure and Files

The author's unzipped files and datasets are organized as follows in the workspace:

### Author's Evaluation Code and Metadata:
* [evaluate_new_fitting.py](file:///home/haipd/DexAvatar/evaluation/evaluate_new_fitting.py): Original evaluation script containing hardcoded desktop paths.
* [segment.json](file:///home/haipd/DexAvatar/data/segment.json): Metadata defining active frame segment ranges.
* [signs.txt](file:///home/haipd/DexAvatar/data/signs.txt): Mapping of signs to subclass `0` (one-handed) or `~0` (two-handed).

### Localized Model Configuration Data (Unzipped from `data.zip`):`
* [MANO_SMPLX_vertex_ids.pkl](file:///home/haipd/DexAvatar/data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl): Hand vertices lookup table for SMPL-X.
* [SMPLX_NEUTRAL.npz](file:///home/haipd/DexAvatar/data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz): Standard neutral SMPL-X body model.
* [sgnify_part_segm_above_pelvis_joint](file:///home/haipd/DexAvatar/data/evaluation_from_author/data/data/sgnify_part_segm_above_pelvis_joint): Folder containing upper body vertex masks.

### Evaluation Inputs (Mesh Folders):
* **Ground Truth**: [data/smplx_gt](file:///home/haipd/DexAvatar/data/smplx_gt)
* **Baseline Predictions**: [outputs/output_baseline](file:///home/haipd/DexAvatar/outputs/output_baseline)
* **WiLoR Predictions**: [outputs/output_wilor](file:///home/haipd/DexAvatar/outputs/output_wilor)

---

## 3. Step-by-Step Execution Guide

### Step 1: Create the Localized Script
Copy the original script to a local version so we can resolve the hardcoded desktop paths safely:
```bash
cp evaluate_new_fitting.py evaluate_new_fitting_local.py
```

### Step 2: Edit Hardcoded Paths in `evaluate_new_fitting_local.py`
Open [evaluate_new_fitting_local.py](file:///home/haipd/DexAvatar/evaluation/evaluate_new_fitting_local.py) and modify the following hardcoded paths:

1. **MANO SMPL-X Vertex Mapping (Line 522)**:
   ```python
   # Replace:
   with open('/home/kaustubh/Desktop/Fitting/experiments/data/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
   # With:
   with open('/home/haipd/DexAvatar/data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
   ```

2. **SMPL-X Neutral Model (Line 526)**:
   ```python
   # Replace:
   smplx_model_data = np.load('/home/kaustubh/Desktop/Fitting/experiments/data/SMPLX_NEUTRAL.npz', allow_pickle=True)
   # With:
   smplx_model_data = np.load('/home/haipd/DexAvatar/data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz', allow_pickle=True)
   ```

3. **Part Segmentation Directory (Line 554)**:
   ```python
   # Replace:
   'above pelvis': '/home/kaustubh/Desktop/Fitting/experiments/data/sgnify_part_segm_above_pelvis_joint',
   # With:
   'above pelvis': '/home/haipd/DexAvatar/data/evaluation_from_author/data/data/sgnify_part_segm_above_pelvis_joint',
   ```

### Step 3: Run the Localized Script
Activate the `dexavatar` conda environment and execute the evaluation command.

#### For Baseline:
```bash
conda activate dexavatar
python evaluation/evaluate_new_fitting_local.py \
    --method slrt \
    --central false \
    --evaluate_folder /home/haipd/DexAvatar/outputs/output_baseline \
    --gt_folder /home/haipd/DexAvatar/data/smplx_gt \
    --sign_file /home/haipd/DexAvatar/data/evaluation_from_author/data/data/signs.txt \
    --sign_seg /home/haipd/DexAvatar/data/evaluation_from_author/data/data/segment.json
```

#### For WiLoR:
```bash
python evaluation/evaluate_new_fitting_local.py \
    --method slrt \
    --central false \
    --evaluate_folder /home/haipd/DexAvatar/outputs/output_wilor \
    --gt_folder /home/haipd/DexAvatar/data/smplx_gt \
    --sign_file /home/haipd/DexAvatar/data/evaluation_from_author/data/data/signs.txt \
    --sign_seg /home/haipd/DexAvatar/data/evaluation_from_author/data/data/segment.json
```

*Note on Arguments:*
* `--method slrt`: Specifies the baseline method name. (Since we are evaluating SLRT/Baseline, and no rotation is needed, `slrt` is correct).
* `--central false`: The author's evaluation loop ignores the central frames filter and evaluates the full active segments. Setting this to `false` ensures identical frame counts.

---

## 4. Key Protocol Details for Claude to Observe

When running and verifying the results, pay close attention to these differences from the repository's native evaluation scripts:

### A. Subclass 0 Inactive Hand Exclusion (Body & Hand)
For one-handed signs (Class `0`), the left hand is inactive. The author's script applies two filters:
* **Metric Exclusion**: Left hand error calculation is completely skipped for subclass `0` signs. The `LHand` mean metric is calculated only over two-handed signs (`~0`), matching SGNify's protocol.
* **Vertex Exclusion**: Left-hand vertices are subtracted from `UBody` and `all` regions for subclass `0` signs using `np.setdiff1d(vertex_index_set, left_hand_ids)`. This prevents noise on the inactive hand from inflating the body fitting metrics.

### B. Index-based Frame Pairing
The author's script pairs files sequentially by zip index:
```python
method_vertices, faces = read_verts_and_faces(mocap_objs[idx][inter_idx], method)
soma_vertices, soma_faces = read_verts_and_faces(gt_objs[idx][inter_idx], 'soma')
```
* **Implication**: Ensure that the prediction folder contains exactly the same frames as the GT segment list. If any extra frames exist in the prediction output folder, the files will misalign, leading to incorrect calculations.

### C. The Inoperative `--central` Flag
Although the script defines a `--central` parameter, it is never passed to or utilized in the main evaluation logic. The script always defaults to evaluating the entire segment listed in `segment.json`. Do not attempt to use `central=True` inside the evaluation process.
