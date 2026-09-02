# SignEFT-X beyond H1: research and optimization log

> Status: active research log, 2026-09-01  
> Frozen reference: `Final H1 = A3f + gated canonical WiLoR finger-only refinement`  
> Rule: all new candidates use separate configs/run roots; the frozen H1 artifacts are never overwritten.

## Material Passport

- Material type: research brief + methodology blueprint + experiment ledger
- Research mode: quantitative, hypothesis-driven ablation
- Ground truth policy: GT and official metrics are evaluation-only; no per-frame fitting or selection may access them
- Development policy: engineering12 is the only split allowed for design decisions
- Confirmation policy: untouched45 and full57 are run only after configuration freeze
- Post-confirmation policy: after H14 opened untouched45, hypotheses motivated by its
  failure analysis (H15 onward) are explicitly exploratory; untouched45/full57 can
  measure them but cannot be presented as an untouched confirmation set.
- Current reference artifacts:
  - `SignEFT-X/reports/final_result_card.md`
  - `SignEFT-X/reports/engineering12_hand_ablation.json`
  - `SignEFT-X/runs/signeft_final_h1_full57/`
- AI disclosure: this log and the experiment design were prepared with AI-assisted code and literature analysis; all numerical claims must remain traceable to stored evaluator artifacts.

## 1. Research question brief

### Primary question

Can a GT-free, compositional and risk-controlled hand refinement improve every official TR-V2V region over frozen H1 while adding a defensible methodological contribution beyond whole-hand expert transfer?

### FINER assessment

| Criterion | Score | Reason |
|---|---:|---|
| Feasible | 5/5 | Full H1 caches, canonical SMPL-X states, RGB heatmaps, WiLoR/HaMeR observations, SignHPoser assets and the official evaluator already exist locally. |
| Interesting | 5/5 | H1 improves all six metrics but still accepts a measurable subset of harmful hand candidates. |
| Novel | 4/5 | The promising gap is not another frozen expert; it is localized candidate construction plus independent, hierarchical risk control for sign reconstruction. Novelty must be checked against the literature below. |
| Ethical | 5/5 | No new human-subject collection is needed; the work remains within the existing benchmark and model licenses. |
| Relevant | 5/5 | A successful selector improves accuracy and turns a small empirical gain into a clearer method contribution. |
| **Average** | **4.8/5** | |

### Scope

In scope: framewise RGB-only refinement of A3f/H1, canonical SMPL-X output, finger articulation, uncertainty, sign-pose plausibility, candidate selection, exact rollback, official six-region evaluation.

Out of scope for the first research loop: retraining a large image encoder, using marker/GT cues, tuning on untouched45/full57, changing evaluator alignment, free wrist rotation, free beta/camera, or temporal pose smoothing.

Sub-questions:

1. Which H1 accepted candidates are harmful, and are their failures random or structured?
2. Does per-finger compositional selection retain beneficial joints while avoiding whole-hand contamination?
3. Can a pure SignHPoser manifold delta act as an independent veto without suppressing visually correct sign articulation?

## 2. Methodology blueprint

- Paradigm: positivist/quantitative benchmark study.
- Design: controlled ablation with frozen C0 and H1 references.
- Independent variables: candidate construction, evidence locality, selection rule, optional prior veto.
- Dependent variables: TR All, UBody, UBody-F, UBody-H, LHand and RHand; paired per-sign bootstrap; accepted/fallback counts.
- Controls: identical manifest, A3f state, evaluator hash, topology, WiLoR/Sapiens caches, beta, camera, wrist and non-hand pose.
- Promotion gate on engineering12: all six metrics must be non-worse than H1, at least one target hand metric must improve, and no safety/provenance test may fail.
- Confirmation gate: freeze code/config/hash first, then require directionally consistent gains on untouched45 and full57.
- Multiple-comparison discipline: hypotheses and rejection rules are logged before each run; negative rows remain in the ledger.

Known threats to validity:

- Engineering12 has only 12 signs; sign-specific clusters can overstate selector quality.
- WiLoR, HaMeR and Sapiens are not fully independent because their training corpora can overlap.
- SignHPoser was already part of DexAvatar fitting, so its energy is a plausibility signal, not an independent RGB observation.
- The attached protocol contains 1,493 frames rather than the larger count reported by prior papers; claims stay scoped to the attached protocol.

## 3. Frozen H1 audit

The stored engineering12 run contains 298 frames and 193 accepted hand instances with a defined side-specific official metric. Of these, 164 improve and 29 regress relative to C0; their total hand-error delta is `-151.9748 mm` across accepted instances. The right hand is reliable (`83/88` improve), while the left hand carries `24/29` harmful accepts.

The errors are structured rather than uniformly random:

| Sign/side | Accepted | Improved | Regressed | Mean side delta (mm) |
|---|---:|---:|---:|---:|
| Akzeptieren / left | 9 | 2 | 7 | +0.4590 |
| Akzeptieren / right | 5 | 0 | 5 | +0.6601 |
| BesuchenEinmischen / left | 19 | 3 | 16 | +0.1899 |
| Ablehnen / left | 14 | 14 | 0 | -1.0684 |
| Blume / left | 21 | 21 | 0 | -1.3250 |
| BroetchenAufschneiden / right | 36 | 36 | 0 | -1.3312 |

Interpretation: the existing two-family gate is strong at recognizing average improvement, but it can approve a coherent expert bias across a whole sign/side. Raising one global threshold is unlikely to solve this without losing many good candidates.

### SignHPoser diagnostic (not yet a promoted method)

For the accepted engineering instances, the change in original DexAvatar SignHPoser posterior KL separates some failures: mean KL change is `-94.73` for improved instances and `+117.25` for regressed instances. A preliminary threshold sweep shows that `delta_KL <= 100` would reject 8 harmful and 3 beneficial instances and improve the accepted-delta sum from `-151.97` to `-154.72 mm`. This sweep used GT only for diagnosis; the threshold is therefore **not eligible for production**. A production prior threshold must be calibrated without GT, for example from baseline perturbation noise or a fixed reference quantile.

## 4. Primary-source evidence map

Search date: 2026-09-01. Sources are included only when an official paper page, proceedings PDF, arXiv record or official repository is available.

| Source | Primary evidence relevant here | Consequence for our design |
|---|---|---|
| [DexAvatar](https://arxiv.org/abs/2512.21054) | Uses sign-domain hand/body priors for monocular sign reconstruction. | SignHPoser is inherited prior evidence; using its pure energy is not itself novel. |
| [WiLoR](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html) | In-the-wild hand localization and MANO reconstruction. | Keep as candidate generator, not as the paper contribution. |
| [HaMeR](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html) | Large transformer/data scaling improves hand mesh recovery. | Useful as cross-expert evidence, but the hard veto already failed H4. |
| [HUMR](https://openaccess.thecvf.com/content/WACV2025/html/Wehrbein_Utilizing_Uncertainty_in_2D_Pose_Detectors_for_Probabilistic_3D_Human_WACV_2025_paper.html) | Preserves 2D detector uncertainty instead of reducing it to a point. | Continue using heatmap likelihood/entropy; localize it per finger. |
| [MaskHand](https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html) | Uses discrete MANO tokens and confidence-guided iterative sampling under ambiguity. | Supports confidence-aware partial hypotheses; our gap is inference-only composition across frozen experts without retraining. |
| [Hierarchical Sampling Optimization](https://openaccess.thecvf.com/content_iccv_2015/html/Tang_Opening_the_Black_ICCV_2015_paper.html) | Organizes candidate generation by the hand kinematic hierarchy and selects partial hypotheses with surrogate energies. | Direct precedent for per-finger candidate decomposition; novelty cannot be claimed merely for “hierarchical fingers.” |
| [HMP](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_Video_WACV_2024_paper.html) | Optimizes hand pose under a learned motion prior and 2D observations. | Confirms prior-constrained refinement, but HMP is temporal; our first branch remains framewise. |
| [DPoser-X](https://arxiv.org/abs/2508.00599) | Provides diffusion pose priors for whole-body and hand fitting. | Optional independent plausibility comparator; do not backpropagate it before a veto ablation passes. |
| [Neural Localizer Fields](https://arxiv.org/abs/2407.07532) | Predicts continuous 3D human locations from images. | Relevant to the rejected upper-body branch, not the first hand branch. |
| [Sapiens2](https://arxiv.org/abs/2604.21681) | Releases human-centric pose/segmentation/pointmap tasks. | Whole-body segmentation C4 already regressed; any return to dense evidence must be hand-local and separately gated. |

Literature limitation: this is a targeted method search, not a complete systematic review. The current set is concentrated in computer-vision proceedings and recent preprints because the research question is implementation-specific and rapidly moving.

## 5. Novelty and hypothesis matrix

| ID | Hypothesis | Mechanism | Falsification on engineering12 | Novelty value | Priority |
|---|---|---|---|---|---:|
| H7 | Whole-hand candidates contain mixtures of good and harmful finger updates. | Generate baseline, full-hand, five single-finger and at most two trusted multi-finger candidates; score only the joints each candidate changes; compose non-overlapping winners. | Reject if any official metric regresses vs H1 or no hand metric improves. | High when combined with local multi-family risk control and exact rollback; hierarchical fingers alone are prior art. | 1 |
| H8 | H1 regressions include off-sign-manifold outliers. | Use the original pure SignHPoser posterior/reconstruction delta only as a separately calibrated veto. | Reject if it removes net-beneficial candidates or cannot define a GT-free threshold. | Medium; contribution is heterogeneous risk factorization, not the prior. | 2 |
| H9 | Expert reliability is coherent within a sign/side and cannot be captured framewise. | Estimate a sequence-level reliability variable from RGB-evidence consistency, without using neighboring pose as a target. | Reject if improvement disappears on untouched signs or violates frame-independent claims. | Potentially high, but leakage/generalization risk is high. | 4 |
| H10 | Whole-body segmentation failed because the signal was too broad, not because dense evidence is useless. | Crop-local hand silhouette/skin boundary candidate veto only. | Reject unless H7/H8 saturate and dense hand evidence improves hands without UBody drift. | Medium. | 5 |
| H11 | A3f and WiLoR errors differ by finger and pose family. | Fit a calibrated mixture-of-experts reliability model using GT-free heatmap entropy, expert disagreement and manifold delta. | Reject if calibration requires official per-frame labels or overfits engineering signs. | High in principle, but needs a valid calibration source. | 3 |
| H12 | The fixed 8° production radius clips useful canonical fits. | Sweep a frozen radius response curve; select only on engineering12 and audit bound saturation. | Reject if any untouched45 metric regresses versus H1. | Low alone; useful mechanistic calibration for later modules. | complete |
| H13 | HaMeR is more useful as a diverse proposal generator than as the H4 hard veto. | Preserve every H1 accept; let a canonical HaMeR proposal rescue only H1 rejects under Sapiens + HaMeR wins and a WiLoR conflict veto. | Reject/repair if hands improve but All/UBody regions regress. | High: monotonic multi-expert proposal diversity plus factorized evidence. | active |
| H14 | Canonical finger fitting can improve centered hand shape while shifting the hand centroid inside whole-body metrics. | Add an explicit centroid-neutral articulation constraint during proposal fitting. | Reject if it removes hand gain or cannot fix H13 All/UBody-H regression. | High when coupled to metric-aware regional protection. | rejected after confirmation |
| H15-v1 | A single fitting regularizer should not alter the already-promoted incumbent expert merely to protect a rescue expert. | Regenerate WiLoR with weight `0` and apply weight `0.5` only to HaMeR. | Reject if either official metrics fail or an artifact-level incumbent-identity audit finds any non-rescue side changed. | High in intent, but implementation must prove identity rather than infer it from counts. | rejected by invariant audit |
| H15-v2 | A true rescue stage must consume the frozen H1 state/decision as its incumbent rather than regenerate the incumbent path. | Load frozen H1 states, preserve/copy them exactly, fit HaMeR only as a rescue for H1-rejected sides, and materialize a new SMPL-X state only on rescued frames. | Reject on any incumbent/hash/fallback audit violation, any engineering metric regression, or no hand gain. Later 45/57-sign results remain exploratory. | High: exact incumbent preservation + asymmetric expert-specific proposal geometry + factorized rescue evidence. | exploratory pass; external confirmation required |

## 6. Experiment ledger

| Experiment | Status | Configuration | Result | Decision |
|---|---|---|---|---|
| H1 frozen | complete | full-hand canonical WiLoR + Sapiens/WiLoR 2-family gate | Full57 improves all six metrics; see final result card. | Frozen reference |
| H7a compositional fingers | rejected | local candidates may override a full H1 candidate | Engineering12: L/R regress because local ring z-score displaced good whole-hand candidates. | Reject |
| H7b monotonic compositional rescue | rejected | preserve H1; local fingers rescue rejects only | Engineering12 improves all six slightly; untouched45 improves 5/6 but LHand regresses `12.6482→12.6486`. | Reject under strict no-regression gate |
| H8 SignHPoser veto | rejected | H7b + pure original SignHPoser KL delta, threshold 4.5 per latent dim | Engineering12 improves all six; untouched45 regresses all six, including RHand `11.9869→12.0361`. | Strong evidence of dev-specific prior bias; do not promote or retune on holdout |
| H12 radius response | rejected, near-pass | H1 radius 4/6/8/10/12° | Engineering curve improves nearly monotonically; 8° clips 34/193 accepted hands while 12° clips 0. Untouched45 at 12° improves 5/6 but RHand regresses `11.9869→11.9871`. | Do not promote alone; retain saturation finding |
| H13 multi-expert bank | repair active | H1 accepts + HaMeR rescues | Engineering12: `LHand 12.0412→12.0235`, `RHand 11.6415→11.5035`; however All `41.1480→41.1485` and UBody-H `38.9090→38.9112` regress. | Preserve proposal bank; add metric-aware centroid protection before deciding |
| H14 symmetric centroid-neutral bank | rejected, near-pass | centroid weight `0.5` applied to both WiLoR and HaMeR fits | Engineering12 improves all six. Frozen untouched45 improves 5/6 but UBody-H regresses `39.8056→39.8073`; sign-weighted bootstrap CIs cross zero. | Reject under strict gate; do not retune the symmetric weight on holdout |
| H15-v1 asymmetric regenerated bank | rejected after invariant audit | regenerate weight-0 WiLoR path + centroid-neutral HaMeR rescues | Metrics improve all six on engineering12, 45 signs and full57, but full artifact audit finds 389 non-rescue hand sides that differ from frozen H1 even though the total WiLoR accept count is also 991. | Do not promote; counts are not proof of incumbent identity |
| H15-v2 exact-incumbent rescue | exploratory pass | frozen H1 states/decisions + centroid-neutral HaMeR rescue only | Zero invariant violations on engineering12, exploratory45 and full57. Full57 preserves all 991 H1 WiLoR sides, adds 429 HaMeR rescue sides, and improves all six official aggregates to `42.0640 / 25.7991 / 29.1057 / 39.6121 / 12.5060 / 11.8431`. | Best verified method on the attached protocol; retain H1 as frozen confirmatory reference and require an external/new sanctioned split before an unbiased promotion claim |

## 7. Strongest reviewer objections

1. “This is only an ensemble of existing models.” Response required: the claimed method must be the compositional risk-control formulation and its verified safety/selection behavior, while frozen experts are components.
2. “The gate is tuned on the benchmark.” Mitigation: engineering12 is explicitly development-only; untouched45/full57 remain sealed until freeze, and no GT enters per-frame inference.
3. “Per-finger hierarchy is old.” Mitigation: do not claim the decomposition alone; test the specific combination of canonical cross-model retargeting, changed-variable-local evidence, heterogeneous prior veto and exact canonical rollback for sign reconstruction.
4. “Small gains are not meaningful.” Mitigation: report effect sizes, paired bootstrap, coverage/acceptance and failure reductions; avoid SOTA claims outside the attached 1,493-frame protocol.

## 8. Iteration findings

### 8.1 Local evidence is not globally rank-equivalent

H7a proved that a high per-finger Sapiens/WiLoR z-score is insufficient to replace a full-hand candidate. Ring-only candidates were selected 23 times and made the aggregate hand result worse. H7b therefore introduced a monotonic hierarchy: full H1 consensus has priority; local hypotheses can only rescue a reject. The monotonic version generalized directionally to five of six untouched metrics but missed the strict left-hand gate. This is retained as a negative ablation rather than described as a success.

### 8.2 A sign prior is not an uncertainty estimator

The SignHPoser KL diagnostic separated the two visible engineering failure clusters, but the frozen threshold vetoed 113/822 proposed hands on untouched45 and regressed every metric. This falsifies the assumption that distance from the inherited sign manifold is calibrated risk for unseen signs. SignHPoser remains part of the A3f ancestry; it is not used by the post-H1 final candidate.

### 8.3 Trust-region saturation exists, but a global radius is insufficient

Accepted-hand bound counts on engineering12 were `185/196` at 4°, `98/194` at 6°, `34/193` at 8°, `11/193` at 10° and `0/193` at 12°. Metrics improve almost monotonically through 12°, so the original 8° radius underfits some candidates. The 12° candidate nevertheless missed untouched RHand by 0.0002 mm; it is not promoted in isolation.

### 8.4 Proposal diversity works; regional protection is the current bottleneck

H13 keeps all 193 H1 WiLoR accepts and adds 98 HaMeR rescues on engineering12. The resulting right-hand gain is substantial (`-0.1380 mm` versus H1), demonstrating that H4's failure was a misuse of HaMeR as a universal veto rather than evidence that HaMeR had no complementary signal. The same rescues slightly regress All and UBody-H. Per-frame audit shows a candidate can improve the translation-centered hand metric while shifting the finger/hand centroid relative to the whole upper-body region. H14 will therefore change proposal construction, not merely tighten an acceptance threshold.

### 8.5 Symmetric protection perturbs the incumbent path

H14 at frozen centroid weight `0.5` was deterministic after the provenance fix:
all 298 engineering OBJ files and all state arrays matched the earlier H14 run,
with 292 accepted hand-sides. It improved all six engineering metrics, but the
first untouched45 run gave `All 42.2990`, `UBody 26.1397`, `UBody-F 29.4660`,
`UBody-H 39.8073`, `LHand 12.6457`, and `RHand 11.9543`; UBody-H is `+0.0017 mm`
worse than H1, so H14 is rejected.

Post-hoc failure decomposition is diagnostic rather than confirmation evidence.
Frames containing one of the 300 HaMeR rescues improved mean per-frame deltas in
all six regions, including UBody-H `-0.0521 mm`. The 834 WiLoR primary accepts,
which H14 had also refit with the centroid penalty, regressed UBody-H by
`+0.0280 mm/frame`. H15 therefore freezes the incumbent H1 proposal construction
exactly and regularizes only the rescue expert. Because this hypothesis was formed
after opening untouched45, subsequent results must be labelled exploratory until
an external or newly sanctioned confirmation split is available.

### 8.6 Asymmetric proposal geometry passes the development gate

H15 changes no evidence threshold and no trust radius. Its only change is to set
the incumbent WiLoR canonical centroid weight back to the frozen H1 value `0.0`
while retaining `0.5` for HaMeR rescues. On engineering12 it selects exactly 193
WiLoR hands and 99 HaMeR rescues. Every selected WiLoR hand-pose array is identical
to the corresponding frozen H1 output. Official metrics versus H1 change from
`41.1480 / 24.4635 / 27.7001 / 38.9090 / 12.0412 / 11.6415` to
`41.1458 / 24.4597 / 27.6955 / 38.9032 / 12.0184 / 11.5082`, passing the all-six
development gate. The 12-sign paired bootstrap intervals remain wide and cross
zero; this is an effect-size/coverage limitation, not evidence of significance.

### 8.7 Aggregate counts do not prove monotonic incumbent preservation

H15-v1 subsequently improved all six metrics on the exploratory 45-sign and
full57 evaluations, reaching full57 `42.0640 / 25.7991 / 29.1057 / 39.6121 /
12.5060 / 11.8431`. It is nevertheless **not promotable**. A new artifact-level
auditor compared every candidate state to frozen H1 and found 389 non-rescue hand
sides that changed. The regenerated path selected 991 WiLoR sides in aggregate,
equal to H1's count, but not the same complete set of side identities. This is a
methodological failure independent of favorable metrics.

H15-v2 therefore makes the monotonic relation structural: frozen H1 states and
decisions are explicit inputs, non-rescue frames are copied from H1, incumbent
hand poses are retained on mixed incumbent/rescue frames, and only H1-rejected
sides are eligible for HaMeR rescue. The invariant auditor is now a mandatory
promotion gate, not an optional diagnostic.

The first H15-v2 implementation run was stopped before evaluation by that gate:
on mixed incumbent/rescue frames it decoded both rotation matrices back to
axis-angle and rewrote both hand arrays. Although the non-rescue rotation was
geometrically equivalent, 79 engineering non-rescue arrays were not byte/array
identical to H1. The corrected implementation overwrites only the explicitly
rescued side; this failed run is not eligible for metric comparison.

The corrected engineering run passes the mandatory audit with zero violations:
209/298 frames are exact incumbent copies, all 109 no-consensus fallbacks are
exact A3f/H1, and the side histogram is 193 WiLoR incumbents, 99 HaMeR rescues
and 304 baseline sides. Its official metrics remain
`41.1458 / 24.4597 / 27.6955 / 38.9032 / 12.0184 / 11.5082`, so the exactness
fix does not sacrifice the development gain. Code/config are frozen before the
exploratory 45-sign run.

### 8.8 Exact-incumbent rescue passes the exploratory full-protocol audit

The corrected H15-v2 implementation has SHA-256
`e627e54c460c400c87ca4c9d73fde59e087d5c6631c72790d61eb82c64f79ac0`.
The exploratory 45-sign run wrote all 1,195 frames and passed the mandatory
artifact audit with zero violations: 910 frames are exact H1 copies when no
rescue occurs, 414 fallbacks are exact A3f/H1, and the selected-side histogram
is 798 WiLoR incumbents, 330 HaMeR rescues and 1,262 baseline sides. Official
metrics improve all six H1 aggregates:

| Exploratory45 | All | UBody | UBody-F | UBody-H | LHand | RHand |
|---|---:|---:|---:|---:|---:|---:|
| Frozen H1 | 42.3001 | 26.1411 | 29.4671 | 39.8056 | 12.6482 | 11.9869 |
| H15-v2 | 42.2937 | 26.1344 | 29.4590 | 39.7905 | 12.6341 | 11.9266 |

The 12-sign and 45-sign artifacts were then hardlinked into a disjoint full57
run. The full config resumed 1,493/1,493 frames with `written=0`, confirming a
single implementation/configuration rather than a new refit. The full audit
again reports zero violations: 1,119 frames exact H1, 523 exact fallbacks, all
991 H1 WiLoR sides preserved, and 429 additional HaMeR rescue sides. The
official and audited evaluators agree after rounding:

| Full57 | All | UBody | UBody-F | UBody-H | LHand | RHand |
|---|---:|---:|---:|---:|---:|---:|
| Frozen H1 | 42.0696 | 25.8053 | 29.1131 | 39.6254 | 12.5219 | 11.9180 |
| H15-v2 | **42.0640** | **25.7991** | **29.1057** | **39.6121** | **12.5060** | **11.8431** |
| Delta | -0.0056 | -0.0062 | -0.0074 | -0.0133 | -0.0159 | -0.0749 |

Paired sign bootstrap versus H1 gives mean deltas and 95% percentile intervals:
All `-0.0060 [-0.0134, -0.0005]`, UBody
`-0.0065 [-0.0162, 0.0013]`, UBody-F
`-0.0078 [-0.0189, 0.0007]`, UBody-H
`-0.0144 [-0.0306, -0.0016]`, LHand
`-0.0112 [-0.0400, 0.0182]`, and RHand
`-0.0500 [-0.1138, 0.0076]` mm. Against C0/A3f, all six intervals are fully
negative. Therefore the aggregate result is directionally favorable in every
region, with the clearest incremental evidence over H1 in All and UBody-H; the
small incremental hand effects still have wide sign-level intervals.

This is an **exploratory pass**, not a clean holdout confirmation. H15 was
motivated by a post-hoc decomposition after H14 opened untouched45. The correct
paper claim is that EI-AMER is the best verified method on the attached
1,493-frame protocol and satisfies exact incumbent/fallback invariants. A new
external or prospectively sanctioned confirmation set is required before
claiming unbiased generalization or replacing frozen H1 as the confirmatory
result.

## 9. Current paper contribution

The contribution is not WiLoR, HaMeR, SignHPoser or SignBPoser themselves. The
post-H1 method does not call SignHPoser or SignBPoser. A defensible formulation
is **SignEFT-X with Exact-Incumbent Asymmetric Multi-Expert Rescue (EI-AMER)**:

1. canonical cross-model finger retargeting into a shared-beta SMPL-X state,
   with body, wrist, shape and camera held fixed;
2. an exact-incumbent cascade in which a frozen, previously validated expert is
   structurally immutable and alternate experts can act only on its rejects;
3. expert-specific proposal geometry: centroid-neutral regularization is
   applied only to the HaMeR rescue expert, avoiding perturbation of the proven
   WiLoR incumbent;
4. factorized GT-free rescue evidence: Sapiens and HaMeR must support the rescue,
   while conflicting WiLoR evidence vetoes it; and
5. an artifact-level safety contract proving that non-rescue outputs are exact
   H1 and rejected outputs are exact A3f, rather than inferring safety from
   aggregate counts or metrics.

The empirical claim must stay narrow: all six aggregates improve on the attached
full57 protocol and all six bootstrap intervals versus C0 are negative. The
incremental H15-v2-versus-H1 effects are small, only All and UBody-H have fully
negative 95% percentile intervals, and the experiment is post-holdout
exploratory.
