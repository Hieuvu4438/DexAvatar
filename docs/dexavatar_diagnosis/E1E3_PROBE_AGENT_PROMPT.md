# Coding Agent Brief — DexAvatar Diagnostic Probes E1–E3

> Paste this whole document as the opening instruction to the coding agent
> (Claude Code or equivalent) working inside a fresh clone of
> `https://github.com/kaustesseract/DexAvatar`.
> Fill every `<...>` placeholder before sending.

---

## 0. Role and objective

You are a research engineer. Your job is to implement and run three diagnostic
experiments (E1, E2, E3) plus a statistics pass (E4) against the DexAvatar
codebase, and to report measured numbers.

You are **not** improving DexAvatar. You are **not** proposing methods. You are
producing measurements that will later inform a research direction. Optimising
for "good-looking" numbers is a failure, not a success. If a result contradicts
the paper, report it plainly.

---

## 1. PRIME DIRECTIVE — the repository must remain pristine

I will continue developing my own method on top of this clone. Upstream code
must stay byte-identical so that I can rebase, diff against upstream, and reuse
the original fitting pipeline unchanged.

### 1.1 Path policy

**WRITE-ALLOWED — the only place you may create or modify files:**

```
probes/
```

**READ-ONLY — you may read, import, and execute, but NEVER modify, reformat,
rename, move, or delete:**

```
dexavatar_fitting/**
data/**
configs/**            (any upstream config directory)
*.py at repository root
requirements*.txt, setup.py, pyproject.toml, environment.yml
README.md, LICENSE, and every other pre-existing tracked file
```

This includes cosmetic changes. Do not run formatters, linters with `--fix`,
`isort`, `black`, or auto-import tools over upstream paths.

### 1.2 Version control policy

- Work on a branch: `git checkout -b probes/diagnostics`.
- Never commit to `main`/`master`.
- Never `git add` a path outside `probes/`.
- Record the upstream commit SHA at the start; include it in the final report.

### 1.3 Mandatory guard

Create `probes/tools/guard_upstream.sh` as the **first file** you write:

```bash
#!/usr/bin/env bash
# Fails if anything outside probes/ has been touched.
set -euo pipefail
DIRTY=$(git status --porcelain | grep -v '^.. probes/' || true)
if [ -n "$DIRTY" ]; then
  echo "UPSTREAM CONTAMINATION DETECTED:"
  echo "$DIRTY"
  exit 1
fi
echo "OK: upstream clean, all changes confined to probes/"
```

Run it:
- immediately after creating it (baseline),
- after every experiment phase,
- as the final step before writing the report.

If it ever fails, **stop**, revert the offending change with
`git checkout -- <path>`, and report what happened. Do not continue past a
failed guard.

### 1.4 How to change upstream behaviour without editing upstream

You will need to alter hyperparameters that the original code hardcodes (e.g.
initialisation weight `1200` in the fitting YAML, the temporal factor `2000` in
`fitting.py`). Use this escalation ladder and always prefer the earliest rung
that works:

1. **New config file** in `probes/configs/`, passed through the existing CLI.
   Copying an upstream YAML into `probes/configs/` and editing the copy is fine.
2. **Runtime monkeypatch** in `probes/runtime/overrides.py`, applied by the probe
   entry script *before* the upstream module executes. Every patch must carry a
   comment naming the exact upstream file and line it shadows, and the original
   value.
3. **Vendored copy** of a single module into `probes/vendor/`, with a header
   stating the source path and upstream commit SHA, plus a
   `probes/vendor/<module>.patch` file showing the diff versus upstream. Use
   this only if 1 and 2 are impossible; justify it in the report.

Never rung 4. There is no rung 4.


## 2. Target directory layout

```
probes/
├── README.md                      # how to reproduce every number
├── requirements-probe.txt         # can use existing environment dexavatar
├── configs/                       # copied + edited YAMLs (never upstream ones)
├── runtime/overrides.py           # monkeypatch layer
├── vendor/                        # last-resort module copies + .patch files
├── metrics/trv2v.py               # independent metric reimplementation
├── experiments/
│   ├── e0_audit.py
│   ├── e1_orientation_split.py
│   ├── e2_anchor_sweep.py
│   └── e3_gt_noise_floor.py
├── stats/bootstrap.py
├── tools/guard_upstream.sh
├── results/                       # all outputs (csv/json), gitignored
└── REPORT.md                      # final deliverable
```

Add `probes/results/` to a **new** `probes/.gitignore`. Do not touch the
repository-root `.gitignore`.

---

## 3. Phase 0 — Asset audit, then STOP

Before writing any experiment code, run `probes/experiments/e0_audit.py` and
report findings. **Do not proceed to Phase 1 until I reply.**

Determine and report, with concrete file paths and shapes:

1. **Predictions.** Are DexAvatar per-frame SMPL-X output parameters available
   (as `.pkl`/`.npz`), or only meshes? Which fields exist: `body_pose`,
   `left_hand_pose`, `right_hand_pose`, `betas`, `global_orient`, `transl`,
   `expression`? Are the hand poses in full axis-angle (45-D) or PCA
   coefficients? If PCA, what `num_pca_comps` and is `flat_hand_mean` set?
2. **Ground truth.** Does the SGNify evaluation set expose GT **SMPL-X
   parameters**, or only GT **meshes/vertices**? This determines whether E1 is
   fully or only partially feasible. State clearly which case applies.
3. **SMPL-X model files.** Are the body model `.npz`/`.pkl` assets present and
   loadable? Which gender/variant does the evaluator use?
4. **Frame manifest.** How does the author evaluator pair prediction frames to
   GT frames? Extract the exact pairing logic and the resulting frame count.
   Confirm whether it equals 2,872.
5. **Region masks.** Locate the vertex index sets for `UBody(-F)`, `LHand`,
   `RHand`. Report their sizes and their source (file or hardcoded).
6. **One-handed handling.** Confirm the class-0 behaviour at
   `data/evaluation_from_author/evaluate_new_fitting.py:380-395`: which signs
   are excluded from LHand, and the resulting LHand/RHand sign counts.
7. **Baseline outputs.** Are EVA*, SGNify, or other baseline outputs available
   for paired comparison, or only the published table numbers?
8. **Rectifier.** Does a reusable implementation of the bio-mechanical hand
   rectifier (paper Fig. 4 / Sec. 3.2.2) exist in the repo? Give its path, or
   state that it must be reimplemented for E3.
9. **Runtime cost.** Measure wall-clock time to fit one sign end-to-end so we
   can budget the E2 sweep.

Report blockers explicitly. **If an asset is missing, say so and stop. Never
substitute synthetic, randomly generated, or placeholder data to keep the
pipeline running.** A blocked experiment reported honestly is a good outcome; a
fabricated number is a catastrophic one.

---

## 4. Phase 1 — Independent metric, then parity check

Implement `probes/metrics/trv2v.py` from scratch, reading (not importing, not
editing) the author evaluator as the specification:

```
E_TR(R) = mean_{i in R} || (v̂_i − mean_R(v̂)) − (v_i − mean_R(v)) ||_2
```

with prediction and reference each centred **independently per evaluated
region** — matching `transl_point_error`
(`data/evaluation_from_author/evaluate_new_fitting.py:159-169`).

Then run a **parity check**: reproduce the published DexAvatar row
(UBody(-F) 30.13, LHand 13.53, RHand 13.08) using released outputs.

- Report your numbers to two decimals next to the published ones.
- If any column differs by more than 0.05 mm, **stop and report** the
  discrepancy with your best hypothesis. Do not tune your metric to match.
- Everything downstream depends on this parity. Treat failure here as a
  blocking finding, not an inconvenience.

Also emit, for every subsequent experiment, both:
- **micro** average (mean over all frames, as the paper does), and
- **macro** average (mean over per-sign means).

Report both everywhere. Never silently pick one.

---

## 5. Phase 2 — E1: orientation vs articulation decomposition

**Question.** The residual hand error is 13.53 / 13.08 mm. TR-V2V is
translation-removed per region, so hand *placement* contributes 0 mm. What
remains is articulation, wrist/palm orientation, within-hand relative depth, and
shape/scale. How is the residual split?

**Method.** Oracle substitution: take DexAvatar predictions, replace selected
parameter groups with GT values, re-run SMPL-X forward, re-evaluate.

**Critical implementation note.** In SMPL-X, finger articulation lives in
`left_hand_pose` / `right_hand_pose` (15 joints each), while the **wrist**
rotation lives inside `body_pose`. Do **not** assume joint indices. Resolve them
programmatically from the SMPL-X model's joint-name list and `assert` the
mapping before use. Log the resolved indices in the output.

**Variants** (all evaluated on the identical frame manifest):

| ID | Substitution | Isolates |
|----|--------------|----------|
| V0 | none (baseline) | must reproduce Phase 1 parity |
| V1 | GT `hand_pose` (15 finger joints), predicted wrist | finger articulation |
| V2 | GT wrist rotation, predicted `hand_pose` | palm/wrist orientation |
| V3 | GT wrist **and** GT elbow rotation, predicted `hand_pose` | orientation incl. forearm pronation/supination |
| V4 | GT `hand_pose` + GT wrist | articulation + orientation jointly |
| V5 | GT `betas` only, all pose predicted | shape / scale mismatch |
| V6 | GT everything for the hand region | sanity floor; should approach 0 |

**Reporting.** For each variant and region: mm, absolute reduction versus V0,
and percentage of the V0 residual. State explicitly in the report that these
contributions are **not additive** (V1 + V2 ≠ V4 in general) and give the
measured non-additivity gap.

**If GT SMPL-X parameters are unavailable** (Phase 0 item 2), E1 cannot be run
as specified. Do not improvise a substitute. Instead propose — and wait for my
approval before implementing — a fallback such as fitting SMPL-X parameters to
the GT meshes and quantifying that fit's own residual as an error bar on the
whole experiment.

---

## 6. Phase 3 — E2: initializer ceiling and anchor sweep

**Question.** Is DexAvatar an independent reconstruction, or a plausibility-
corrected version of its initialisers? `L_bprior` anchors to SMPLerX and
`L_hprior` anchors to HaMeR, both at initialisation weight 1200 across all three
stages, with 30 LBFGS iterations.

**E2-A — Raw initializer baseline.** Evaluate the *unoptimised* initialiser
output under the identical protocol: HaMeR for hands, SMPLerX for UBody(-F).
HaMeR outputs MANO; document the MANO→SMPL-X hand-pose conversion you use and
quantify any conversion loss (e.g. round-trip error). If the conversion is
lossy enough to confound the comparison, say so rather than burying it.

**E2-B — Anchor weight sweep.** Vary the initialisation weight over
`{1200, 600, 300, 100, 0}` via `probes/configs/`. Also run two asymmetric
conditions: body anchor zeroed with hand anchor at default, and the reverse.

**E2-C — Observation vs target separation.** HaMeR is simultaneously an
observation (keypoints in `L_joint`) and a prior target (`L_hprior`). Run:
(i) HaMeR keypoints kept, hand anchor zeroed; (ii) hand anchor kept, HaMeR
keypoints down-weighted. This tests whether the reported gain is circular.

**E2-D — Convergence.** Iterations `{30 (default), 100, 300}` at default
weights. If results keep improving past 30, the published configuration is
under-converged and the anchor is doing the work.

**Reporting.** One curve per region across the sweep. Answer directly: at what
anchor weight does performance collapse, and how close is the default setting to
the raw-initialiser baseline? If E2-A ≈ V0 within the E4 confidence interval,
state that conclusion in one unambiguous sentence.

---

## 7. Phase 4 — E3: ground-truth noise floor

**Question.** Supplement S6 admits the SGNify reference contains implausible
hand configurations. How large is that reference noise in millimetres, and is
the paper's 0.20 mm LHand improvement above or below it?

**Steps.**

1. Obtain the bio-mechanical hand rectifier. Use the repo implementation if
   Phase 0 found one; otherwise implement per Sec. 3.2.2 / Fig. 4 — per-joint
   bend/splay/twist limits over 15 hand joints, with the MANO axis alignment
   described in the paper. State clearly which path you took, and treat a
   reimplementation as a source of uncertainty in the result.
2. Apply it to the **GT** hand poses → `GT_rect`.
3. Compute `q_R = TR-V2V(GT, GT_rect)` per hand region. This is the empirical
   reference-noise estimate.
4. Compute `TR-V2V(pred, GT_rect)` and compare to `TR-V2V(pred, GT)`. Report
   whether evaluating against a rectified reference changes DexAvatar's score,
   and (if baseline outputs exist) whether it changes the **ranking**.
5. Quantify GT implausibility directly: percentage of frames with at least one
   joint outside limits, distribution of violation magnitudes in degrees, and
   per-sign breakdown. Identify the worst signs by name.

**Reporting.** Give `q_R` with a confidence interval. Then state plainly whether
0.20 mm (LHand) and 0.60 mm (RHand) lie above or below this floor.

---

## 8. Phase 5 — E4: statistics

The paper reports point estimates to two decimals with no uncertainty. Fix that.

- **Cluster bootstrap by sign**, B = 10,000, resampling *signs* rather than
  frames — frames within a sign are strongly correlated and frame-level
  bootstrap will understate the interval. Report 95% CI for every column of
  every variant above.
- **Minimum detectable difference** at the achieved sample size.
- **Paired comparison** DexAvatar vs EVA* if baseline outputs exist; if only
  published numbers are available, say so and skip rather than approximating.
- **Per-sign distribution**: median, IQR, 90th percentile, and the five worst
  signs per region. The mean may be hiding catastrophic frames.
- Fix and log all random seeds.

---

## 9. Deliverable

`probes/REPORT.md`, containing:

1. Upstream commit SHA; output of the final `guard_upstream.sh` run.
2. Phase 0 audit results and any blockers.
3. Parity check: your numbers vs published, with the delta.
4. E1 table with the orientation/articulation/shape split and the
   non-additivity gap.
5. E2 curves and the direct verdict on anchor domination.
6. E3 `q_R` with CI and the above/below-floor verdict.
7. E4 confidence intervals and minimum detectable difference.
8. **Threats to validity** — every assumption you made, every conversion whose
   loss you could not fully quantify, every place you reimplemented rather than
   reused.
9. Exact commands to reproduce each number.

Machine-readable results as CSV/JSON in `probes/results/`.

---

## 10. Prohibitions

- Do not modify anything outside `probes/`.
- Do not "fix" upstream bugs, typos, or code smells, even obvious ones. Note
  them in the report instead.
- Do not fabricate, synthesise, interpolate, or placeholder any data. Missing
  asset → stop and report.
- Do not tune the metric, the frame manifest, or any hyperparameter toward a
  desired outcome.
- Do not quietly drop frames that error out. Count them, report the count, and
  report results both with and without them.
- Do not report a number without stating which frames and which averaging
  (micro/macro) produced it.
- Do not proceed past Phase 0 or past a failed parity check without my reply.

---

## 11. Definition of done

- [ ] `guard_upstream.sh` passes; `git diff` against upstream is empty outside `probes/`
- [ ] Parity check reproduces the published row within 0.05 mm, or the failure is documented
- [ ] E1, E2, E3, E4 complete or explicitly blocked with the blocker named
- [ ] Every number is reproducible from a logged command
- [ ] `REPORT.md` includes the threats-to-validity section
- [ ] The upstream fitting pipeline still runs unchanged from a clean checkout

---

## 12. Working style

Work phase by phase. After each phase, report results and wait rather than
chaining straight into the next. Show me numbers before conclusions. If you are
uncertain whether something counts as an upstream modification, ask instead of
guessing.
