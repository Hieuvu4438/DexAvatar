# Baseline Paper, Code, and Evaluator Forensics

**Evidence cut:** 2026-08-21  
**Paper:** Kundu et al., *DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors*  
**Code:** `kaustesseract/DexAvatar@a0dfd427f60f5811aadb35c8657b3856d47f56b5`

## 1. Baseline task and reported method

- [VERIFIED] Input: monocular signing video; output: per-frame SMPL-X mesh sequence with articulated body and hands (attached paper, pp. 1–5).
- [REPORTED] DexAvatar combines SMPLer-X body initialisation, HaMeR hand evidence, a learned SignBPoser body VAE, a learned SignHPoser hand VAE, biomechanical constraints, interpenetration loss, and a previous-frame temporal term (paper Secs. 3.1–3.4, Eqs. 1–16).
- [REPORTED] Priors use three fully connected layers with hidden size 512; training uses Adam with learning rate `1e-3`; fitting uses LBFGS on an RTX 4090 24 GB system (paper p. 6, Sec. 4).
- [REPORTED] Table 1 gives `30.13 / 13.53 / 13.08 mm` on `UBody(-F) / LHand / RHand`.
- [VERIFIED] The supplement states that some prior hyperparameters were selected using both DEV **and TEST** (`paper.txt` lines 534–545; attached PDF p. 12, Sec. S4). This is a protocol risk and must not be repeated in the follow-up.

## 2. Actual repository execution path

`run_dexavatar.py` sequentially invokes Sapiens/RTMPose, SMPLer-X, HaMeR, then `dexavatar_fitting/script.py` and the SMPLify-X-derived fitting stack. The audited fitting config uses:

- LBFGS, learning rate `0.5`, three stages, `maxiters=30`;
- SignBPoser latent dimension 33 and SignHPoser latent dimension 23 per hand;
- interpenetration enabled;
- hand joint weights `0.5 / 1.5 / 2.5` and body biomechanical weight `100` in all stages;
- temporal loss multiplier `2000` hard-coded in Python.

At each stage, only the body latent and active hand latent(s) are placed in the optimiser parameter list (`fit_single_frame.py:476–503`). Camera, shape, expression, global orientation, and direct SMPL-X pose parameters are effectively fixed on this path.

## 3. Paper–code discrepancy and implementation-risk table

| Item | Paper says | Code does | Impact | Status |
|---|---|---|---|---|
| Dominant hand for one-handed signs | A sign-aware fitting treatment is described at a high level. | Class `0` comes from `signs.txt`, but active side is inferred from mean 2D wrist speed with a ratio threshold (`data_parser.py:201–203, 300–346`). | May flip which hand is optimised; must verify every sign. | [VERIFIED] code / [INFERENCE] impact |
| Frame coverage | Evaluate all 2,872 central frames. | Frames lacking HaMeR output or SMPLer-X parameters are silently dropped (`data_parser.py:182–199`). | Different methods may be evaluated on different populations. | [VERIFIED] |
| Segment filtering | Central portion of each sign. | Builds `selected` but ignores it; then requires exact endpoint IDs and slices the original list (`data_parser.py:155–168`). | Missing endpoints can crash or alter coverage. | [VERIFIED] |
| 3D hand evidence | Uses recovered 3D hands in fitting. | Only the **z coordinate** of selected joints is used, wrist-relative and standardised per frame without epsilon (`fitting.py:457–496`). | Loses x/y geometry; zero variance can create NaN. | [VERIFIED] |
| Biomechanical optimisation term | Paper Eq. 11 describes penalising invalid angle ranges and discusses body/hand constraints. | Runtime total loss contains a body Euler-angle hinge mean; no explicit hand-biomechanics term was found (`fitting.py:512–517, 654–662`). | Paper objective and released path are not fully aligned. | [VERIFIED] code; final intent needs author clarification |
| Temporal weight | Paper denotes a tunable `lambda_4`. | Multiplier `2000` is hard-coded and absent from YAML (`fitting.py:499`). | Cannot ablate/configure without code change. | [VERIFIED] |
| Temporal resume | Previous fitted pose regularises current frame. | Existing overlay output causes `continue`, so temporal state is not restored/advanced (`main.py:301–330`). | Resumed runs can differ from clean runs. | [VERIFIED] |
| Optimised variables | Method presentation can be read as fitting an SMPL-X parameter set under the full objective. | Only prior latents are passed to LBFGS (`fit_single_frame.py:476–503`). | Several loss terms are constant with respect to the optimiser on this path. | [VERIFIED] code; [INFERENCE] paper-code interpretation |
| CPU support | Config exposes CUDA options. | Multiple direct `.cuda()` calls occur in the loss (`fitting.py:435,443,509`); collision mode asserts CUDA. | CPU-only reproduction is unavailable without a patch. | [VERIFIED] |
| Rendering | Visualisation is ancillary in the method description. | Rendering executes unconditionally in `fit_single_frame.py`. | Headless/GPU render dependencies become part of baseline correctness. | [VERIFIED] |
| Dependency lock | Paper gives hardware/software overview. | `requirements.txt` is partially pinned, duplicates `tqdm`, and combines legacy `mmcv==1.3.9`, `mmhuman3d==0.11.0`, `numpy==1.23.5`; install docs use multiple PyTorch/CUDA stacks. | High API-drift risk. | [VERIFIED] |

These findings are not evidence that the reported results are wrong. They define checks required before calling the baseline reproduced.

## 4. Locked evaluator specification

For a region with predicted vertices `P` and ground-truth vertices `G`, the attached evaluator computes

\[
e_i = \left\| (P_i-\bar P) - (G_i-\bar G) \right\|_2,
\qquad
\mathrm{TR\text{-}V2V}=1000\,\mathrm{mean}_{f,i}(e_{f,i}).
\]

The centroids are computed **independently for each evaluated region and frame**. There is no rigid rotation or scale fit. Errors are concatenated across included frames/vertices, so signs with more included frames have more weight.

### Evaluator sanity tests

`audit/evaluator_sanity.py` extracts the actual metric functions from the hashed attachment via the Python AST and executes them with NumPy. Result: [VERIFIED] **8/8 PASS**:

1. identity gives zero;
2. pure translation is removed;
3. rotation is not removed;
4. Euclidean distance is correct on a known offset;
5. common-centre helper matches independent centroid alignment;
6. explicit centres remove translation;
7. pelvis alignment removes a global translation;
8. concatenate-then-mean aggregation gives the expected millimetre value.

Machine-readable result: `audit/evaluator_sanity_report.json`.

### Evaluator hazards to resolve before benchmark use

| Finding | Evidence | Consequence | Required control |
|---|---|---|---|
| `--central` is parsed but not used by `main` | `evaluate_new_fitting.py:479–482,511` | CLI suggests a control that has no effect. | Log the exact segment JSON and included IDs; remove misleading flag in wrapper. |
| Data assets use a developer-specific absolute path | line 535 | Script cannot run portably. | Parameterise path without changing metric math. |
| GT/method files are paired by list position | lines 350–369 | Missing frame can shift all later pairings or raise. | Pair by explicit frame ID and assert equality. |
| NaN predictions are skipped | lines 364–366 | A failing method can improve its mean by dropping hard frames. | Fail the run and report completeness. |
| One-handed class `0` excludes LHand and removes left-hand vertices from other regions | lines 381–392 | LHand and RHand are computed on different sign populations. | Preserve for exact reproduction, but report per-region sample counts and an all-sign diagnostic separately. |
| Empty arrays can produce NaN; no finite/count checks | aggregation/report loop | Undefined results may propagate quietly. | Assert non-empty finite arrays and expected counts. |
| Topology equality assertion exists | lines 373–377 | Positive safeguard. | Keep unchanged. |

The original evaluator will remain untouched. A minimal audit wrapper may enforce pairing/completeness and make paths configurable; original and wrapper must agree bit-for-bit on a complete valid fixture before use.

## 5. Reproduction status

| Check | Command / evidence | Outcome |
|---|---|---|
| Repository integrity | `git status --short --branch`; `git rev-parse HEAD` | [VERIFIED] clean `main`, commit `a0dfd427...` |
| Shell syntax | `bash -n scripts/*.sh` | [VERIFIED] PASS |
| Python syntax | AST parse over 1,274 `.py` files | [VERIFIED] 0 failures |
| Pipeline CLI | `python3 run_dexavatar.py --help` | [VERIFIED] exit 0 |
| Evaluator CLI | `python3 upload/evaluate_new_fitting.py --help` | [BLOCKED] `ModuleNotFoundError: torch` |
| Pure metric sanity | `python3 audit/evaluator_sanity.py` | [VERIFIED] 8/8 PASS |
| Official checkpoint evaluation | Requires models/data/GPU | [BLOCKED] not run |
| Baseline end-to-end inference | Requires models/data/GPU/legacy environments | [BLOCKED] not run |

### Exact blocker set

1. No CUDA-visible GPU and no `conda` on this host.
2. No PyTorch/loguru/tqdm in the current Python environment.
3. No SGNify evaluation images or SMPL-X ground-truth OBJ files.
4. No official DexAvatar/Sapiens/SMPLer-X/HaMeR/SignBPoser/SignHPoser checkpoints.
5. No licensed SMPL-X neutral model or MANO model.
6. Evaluator data-root and frame-pairing defects must be wrapped and dual-checked.

Accordingly, the current result is **research/design + static verification only**, not a reproduced baseline and not a SOTA result.

