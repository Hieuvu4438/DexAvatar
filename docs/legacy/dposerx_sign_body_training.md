# Training a Sign-Language DPoser-X Body Prior on How2Sign + PHOENIX-2014-T

**Goal.** Fine-tune the DPoser-X diffusion body prior on **two sign-language datasets jointly** — How2Sign (ASL, high-res) and PHOENIX-2014-T (DGS, low-res) — producing a domain-specific checkpoint that replaces the generic AMASS-trained body prior in `dexavatar_fitting/smplifyx/signbposer_dposerx/dposerx_body.py` for sign-aware SMPLify-X regularization.

**Why.** Stock DPoser-X is trained on AMASS / ARCTIC / EgoBody / GRAB / BEAT2 — **zero sign-language data** (verified by repo-wide grep). A sign-specific prior is both a fairness win (domain-matched comparison against SignBPoser) and a defensible standalone contribution.

> **A complete, staged pipeline already exists in this repo** (built across prior sessions).
> This doc describes that canonical pipeline, its current state, and the remaining steps.
> Do **not** create parallel scripts — the ones below are canonical.

---

## 0. Current status (verified 2026-06-24)

The pipeline is **fully staged for fine-tuning on How2Sign-only data right now**. PHOENIX extraction and the training launch are the only remaining steps.

| # | Step | Canonical script | State |
|---|---|---|---|
| 1 | Extract PHOENIX body poses | `scripts/extract_phoenix_sign.py` | ❌ **not run** (`data/signbposer_data/raw/phoenix_sign/` empty) |
| 2 | Merge H2S + PHX (split-preserving) | `scripts/merge_h2s_phoenix_for_dposerx.py` | ✅ done, **H2S-only** (train 1449 / valid 181 / test 182) |
| 3 | Convert → DPoser-X layout | `scripts/convert_sign_to_dposerx_layout.py` | ✅ done → `data/body_data/sign_v1/` |
| 4 | Fit normalizer (min-max) | `scripts/fit_sign_normalizer.py` | ✅ done → `axis_normalize1.pt` (both consumers) |
| 5 | Sign fine-tune config | `DPoser-X/configs/body/subvp/sign_timefc.py` | ✅ exists |
| 6 | Weights-only fine-tune init ckpt | `scripts/prep_dposerx_finetune_ckpt.py` | ✅ done → `sign_init.ckpt` (195 MB) |
| 7 | Fine-tune | `scripts/train_dposerx_sign_body.sh` | ⏳ **not launched** (only init ckpt present) |
| ★ | End-to-end orchestration | `scripts/run_sign_dposerx_train_pipeline.sh` | ready (runs 1→7) |

**Validation performed:** DPoser-X `AMASSDataset` loads all three splits of `sign_v1` cleanly (shapes `body_pose=(63,)`, `global_orient=(3,)`), and `Posenormalizer` round-trips with reconstruction error `1e-7`. The staged data is correct and ready.

**Two ways forward:**
- **(A) H2S-only fine-tune now** — launch step 7 immediately on the already-staged How2Sign data; add PHOENIX later. Fastest path to a first checkpoint.
- **(B) Full H2S+PHX** — run the orchestrator (step ★), which extracts PHOENIX first, then re-merges/converts/normalizes, then trains.

---

## 1. Architecture & data contract (what DPoser-X expects)

### 1.1 Training pipeline (complete and runnable)
DPoser-X ships a full PyTorch-Lightning body trainer:

| Component | Path |
|---|---|
| Trainer (PL module + `main()`) | `DPoser-X/run/trainer/body/diffusion.py` (`DPoserTrainer`, DDP, configurable iters) |
| Base body config | `DPoser-X/configs/body/subvp/timefc.py` → `configs/body/default_amass_configs.py` |
| **Sign fine-tune config** | `DPoser-X/configs/body/subvp/sign_timefc.py` (extends `timefc.py`) |
| Dataset / DataModule | `DPoser-X/lib/dataset/body/__init__.py` (`AMASSDataModule`), `lib/dataset/body/AMASS.py` (`AMASSDataset`) |
| Normalizer | `DPoser-X/lib/dataset/utils.py` (`Posenormalizer`) |
| Model | sub-VP SDE score model, `TimeFC` backbone (HIDDEN_DIM 1024, EMBED_DIM 512) |

### 1.2 On-disk data layout
`AMASSDataset.read_data()` loads exactly two tensors per split:
```
{data_root}/{version}/{train|valid|test}/
    pose_body.pt     # (N, 63) = 21 SMPL body joints × 3 axis-angle, float32
    root_orient.pt   # (N, 3)  global rotation axis-angle  (zeros — body prior is root-invariant)
```
- `N_POSES = 21` (SMPL body, **no hands/face**). Split names are `train` / `valid` / `test` (note `valid`).

### 1.3 Normalizer contract — RESOLVED (Option B)
`Posenormalizer` is built with `data_path = {data_root}/{version}/train` and loads from **that dir**:

| `data.min_max` | File loaded | Keys |
|---|---|---|
| `True`  *(chosen)* | `axis_normalize1.pt` | `min_poses`, `max_poses` |
| `False` *(AMASS default)* | `axis_normalize2.pt` | `mean_poses`, `std_poses` |

**Decision: Option B (`min_max = True`).** The fitting integration `dposerx_body.py` hardcodes `min_max=True`, so training with the same mode keeps train/fit normalization identical. `sign_timefc.py` sets `data.min_max=True`, and `fit_sign_normalizer.py` writes `axis_normalize1.pt` to **both** the trainer dir (`data/body_data/sign_v1/train/`) and the fitting dir (`checkpoints/dposerx_body_sign/body_normalizer/`). Nothing left to reconcile.

> Historical note: the default AMASS body config uses z-score (`min_max=False`); an earlier one-off `scripts/fit_dposerx_normalizer.py` wrote min-max to a different path. Both are superseded by `fit_sign_normalizer.py` for this pipeline.

---

## 2. Prerequisites (all verified present)

| Requirement | Location | Status |
|---|---|---|
| DPoser-X repo | `./DPoser-X` | ✅ |
| SMPLer-X (pose estimator) | `./SMPLer-X/main/inference.py` | ✅ |
| Conda env (training) | `dexavatar` | ✅ |
| Conda env (SMPLer-X extraction) | `smpler_x` | ✅ |
| SMPL-X body model | config points at `SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz` | verify |
| Released AMASS body ckpt | `checkpoints/dposerx_body/body.ckpt` | ✅ |
| PHOENIX-2014-T raw (read-only) | `/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T` | ✅ |
| How2Sign videos | `/home/shared_data/sign_language/How2Sign/{train,test,eval}/raw_videos` | ✅ |

---

## 3. The canonical pipeline (7 steps + orchestrator)

All scripts live under `scripts/`. The orchestrator runs them in order.

```
raw videos/frames
   │  (1) extract_*_sign.py     → SMPLer-X  → body_poses.npy (63-dim) per split
   ▼
data/signbposer_data/raw/{how2sign, phoenix_sign}/
   │  (2) merge_h2s_phoenix…    → split-preserving merge + PHX cap + weights
   ▼
data/signbposer_data_sign/{train,valid,test}/body_poses.npy
   │  (3) convert_sign_to_dposerx_layout.py  → pose_body.pt + root_orient.pt(=0)
   ▼
data/body_data/sign_v1/{train,valid,test}/
   │  (4) fit_sign_normalizer.py  → axis_normalize1.pt (train dir + fitting dir)
   ▼
   │  (5) sign_timefc.py config + (6) prep_dposerx_finetune_ckpt.py → sign_init.ckpt
   ▼
(7) train_dposerx_sign_body.sh  → fine-tune  → checkpoints/dposer/sign/sign_body_ft/
```

### 3.1 Extract (step 1)
- **How2Sign:** already extracted → `data/signbposer_data/{train,val,test}/body_poses.npy` (1449/181/182). Regenerate via `scripts/extract_how2sign_body_pose.py` if needed.
- **PHOENIX-2014-T:** `scripts/extract_phoenix_sign.py` is the **batched** extractor (one SMPLer-X call per split, not one per clip — ~1000× fewer process spawns than the legacy `extract_phoenix14t_body_pose.py`). Read-only on `/home/dongvk`, writes to `data/signbposer_data/raw/phoenix_sign/{train,dev,test}/`.
  ```bash
  conda activate smpler_x   # or run via dexavatar env as the orchestrator does
  python scripts/extract_phoenix_sign.py --frames_per_clip 3 --gpu_id 0
  ```
  - `--frames_per_clip 3` → ≤3 frames/clip uniformly sampled. At 7096 train clips × 3 ≈ **21k images in one SMPLer-X call** — minutes-to-~1h on one GPU.
  - PHOENIX frames are 210×260 px → SMPLer-X poses are noisier than How2Sign (acceptable for a prior; the merge step can cap them).

### 3.2 Merge (step 2) — split-preserving, imbalance-aware
`merge_h2s_phoenix_for_dposerx.py` maps **both** datasets' official splits to DPoser-X splits (no random re-split, **no How2Sign test leakage**):

| DPoser-X split | How2Sign | PHOENIX |
|---|---|---|
| `train` | train | train |
| `valid` | val | dev |
| `test`  | test | test |

```bash
python scripts/merge_h2s_phoenix_for_dposerx.py --phx_cap_mult 10
```
- `--phx_cap_mult 10` caps PHOENIX per split to ≤10× the How2Sign count (random subsample), keeping the blend ≤~10:1 instead of ~48:1. This is the **effective** balance lever.
- Writes per split: `body_poses.npy (N,63)`, `metadata.pkl` (source tags), `sample_weights.npy` (inverse-source-frequency).
  > Note: DPoser-X's `AMASSDataModule` hardcodes `sample_weights=ones(...)` and does **not** use a `WeightedRandomSampler`, so `sample_weights.npy` is for the record / a future weighted sampler. Physical PHX capping is what actually balances training.
- Missing PHOENIX splits are skipped — How2Sign is still emitted (this is why H2S-only data already exists).

### 3.3 Convert (step 3)
`convert_sign_to_dposerx_layout.py` writes the DPoser-X `.pt` layout (`pose_body.pt` + zero `root_orient.pt`) under `data/body_data/sign_v1/{train,valid,test}/`. Idempotent; run after any re-merge.

### 3.4 Normalizer (step 4)
`fit_sign_normalizer.py` computes per-dim min/max on the **combined** train poses and writes `axis_normalize1.pt` to **both** `data/body_data/sign_v1/train/` (trainer consumer) and `checkpoints/dposerx_body_sign/body_normalizer/` (fitting consumer). Re-run after re-merge so stats reflect PHOENIX too.

### 3.5 Config + init ckpt (steps 5–6)
- **`sign_timefc.py`** *extends* `timefc.py` (keeps the TimeFC architecture identical to the AMASS ckpt) and overrides: `devices=[0]`, `dataset='sign'`, `data.min_max=True`, `batch_size=512`, `n_iters=30000`, `num_workers=2`, `eval.sample_interval=1`.
- **`prep_dposerx_finetune_ckpt.py`** defeats a PL trap: the released AMASS `body.ckpt` was saved at `global_step≈200000`; a plain `--resume-ckpt` would restore that step and, under `max_steps=30000`, run **0 steps**. This script keeps `state_dict` + `model_ema`, drops optimizer/scheduler/loops/callbacks, and resets `global_step=0`/`epoch=0` → a weights-only init (`sign_init.ckpt`) for a fresh 30k schedule.

### 3.6 Fine-tune (step 7)
```bash
bash scripts/train_dposerx_sign_body.sh
```
Launches `python -m run.trainer.body.diffusion -c configs.body.subvp.sign_timefc.get_config --data-root ../data/body_data --version sign_v1 --resume-ckpt sign_init.ckpt --name sign_body_ft` on GPU 0. Checkpoints/TensorBoard land under `DPoser-X/checkpoints/dposer/sign/sign_body_ft/` and `DPoser-X/logs/dposer/sign/`.

### 3.7 Full orchestrator (★)
```bash
tmux new -s sign_dposerx -d "bash scripts/run_sign_dposerx_train_pipeline.sh"
tail -f logs/sign_dposerx_pipeline.log     # progress; sentinels: .done / .failed
```
Runs steps 1→7 in order with failure trapping. **This is the path for the full H2S+PHX run.**

---

## 4. Two-dataset design considerations

| Dimension | How2Sign | PHOENIX-2014-T | Implication |
|---|---|---|---|
| Language | ASL (English) | DGS (German) | Good cross-lingual diversity |
| Domain | Instructional, studio | Broadcast weather | PHOENIX noisier / more cropped |
| Resolution | High, multi-view | 210×260, single-view | PHOENIX SMPLer-X quality lower |
| Scale (poses) | ~1.4k | up to ~21k (3/clip) | Imbalance → `--phx_cap_mult` |
| Continuous? | Yes | Yes | Compatible |

**Split integrity is the hard constraint:** How2Sign `test` is the DexAvatar downstream fitting eval set, so it must never enter prior training — the merge preserves it. PHOENIX is never a DexAvatar eval set, so its poses are pure training data (a small PHOENIX holdout still lands in `valid`/`test` for prior-level completion/denoise checks).

**Balance:** with `--phx_cap_mult 10`, train blend ≈ 1.4k H2S + ≤14k PHX. If How2Sign (the target domain) seems under-represented, lower the multiplier (e.g. 5) or use the inverse-freq `sample_weights.npy` after wiring a `WeightedRandomSampler` into `AMASSDataModule`.

---

## 5. Validation & sanity checks
Already passing on staged data (2026-06-24):
```bash
cd DPoser-X && conda activate dexavatar
python - <<'PY'
import torch
from lib.dataset.body.AMASS import AMASSDataset
from lib.dataset.utils import Posenormalizer
for split in ['train','valid','test']:
    ds = AMASSDataset('../data/body_data','sign_v1',split)
    print(split, len(ds), tuple(ds[0]['body_pose'].shape))
nz = Posenormalizer('../data/body_data/sign_v1/train', device='cpu',
                    normalize=True, min_max=True, rot_rep='axis')
bp = torch.load('../data/body_data/sign_v1/train/pose_body.pt')[:8]
print('round-trip err:', (nz.offline_denormalize(nz.offline_normalize(bp, True), True)-bp).abs().max().item())
PY
```
Before a full run, also do a 10-step debug launch (`--sample 64`, temporarily set `n_iters=10`) to confirm loss decreases.

---

## 6. Risks & gotchas
- **PHOENIX pose noise** from low-res crops — mitigated by `--phx_cap_mult` and the outlier filter in the extraction collect step.
- **Class imbalance** — addressed by physical capping at merge (DPoser-X's loader ignores `sample_weights`).
- **PL `global_step` resume trap** — handled by `prep_dposerx_finetune_ckpt.py` (weights-only init). Do **not** pass the raw AMASS `body.ckpt` to `--resume-ckpt`.
- **Train/fit normalizer consistency** — resolved by Option B (min-max in both places via `fit_sign_normalizer.py`).
- **Compute** — full PHOENIX extraction is one batched SMPLer-X call per split (~minutes–1h/GPU); validate on a clip subset first via `extract_phoenix_sign.py --max_clips`.

---

## 7. Recommended next actions
1. **First checkpoint (fast):** launch `bash scripts/train_dposerx_sign_body.sh` now on the staged H2S-only data.
2. **Full run:** `tmux new -s sign_dposerx -d "bash scripts/run_sign_dposerx_train_pipeline.sh"` (extracts PHOENIX, then trains H2S+PHX).
3. **Evaluate downstream:** point `dposerx_body.py` / `fit_smplx_vposer_x_dposerx_sign.yaml` at the new ckpt + `checkpoints/dposerx_body_sign/body_normalizer/`, then run `methods/run_dexavatar_nlf_dposerx_sign.py` vs. the AMASS-DPoser-X / SignBPoser baselines.

---

## 8. File / path reference
- Raw datasets (read-only): `/home/shared_data/sign_language/How2Sign`, `/home/dongvk/datasets/phoenix14T`
- Raw extractions: `data/signbposer_data/raw/{how2sign,phoenix_sign}/`
- How2Sign aggregated (ready): `data/signbposer_data/{train,val,test}/body_poses.npy`
- Merged sign data: `data/signbposer_data_sign/{train,valid,test}/`
- DPoser-X training data: `data/body_data/sign_v1/{train,valid,test}/{pose_body.pt,root_orient.pt}` + `train/axis_normalize1.pt`
- Fitting-side normalizer: `checkpoints/dposerx_body_sign/body_normalizer/axis_normalize1.pt`
- Trainer: `DPoser-X/run/trainer/body/diffusion.py`
- Sign config: `DPoser-X/configs/body/subvp/sign_timefc.py`
- Fine-tune init ckpt: `DPoser-X/checkpoints/dposer/sign/sign_body_ft/sign_init.ckpt`
- Pipeline scripts: `scripts/{extract_phoenix_sign,merge_h2s_phoenix_for_dposerx,convert_sign_to_dposerx_layout,fit_sign_normalizer,prep_dposerx_finetune_ckpt}.py`, `scripts/{train_dposerx_sign_body,run_sign_dposerx_train_pipeline}.sh`
- Inference integration: `dexavatar_fitting/smplifyx/signbposer_dposerx/dposerx_body.py`
