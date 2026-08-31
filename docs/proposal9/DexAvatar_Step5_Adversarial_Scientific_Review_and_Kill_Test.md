# DexAvatar Step 5 — Adversarial Scientific Review and Kill Test of CLRA-Dex

**Audit date:** 2026-08-26 (Asia/Bangkok)  
**Frozen candidate under review:** *CLRA-Dex — Counterfactual Local-Rotation Arbitration for DexAvatar*  
**Final verdict:** **`REJECT`**  
**Novelty level:** **`WEAK / ENGINEERING`**  
**Review mode:** static scientific audit only. No repository, checkpoint, SGNify data, evaluator, optimization, or score was executed. No result in this document is a reproduced result.

Evidence labels follow the Step 5 contract: `[VERIFIED]`, `[INFERENCE]`, `[UNRESOLVED]`; descriptions of the frozen candidate retain `[PROPOSED]` where appropriate.

---

## 1. Executive verdict 

**Verdict: `REJECT`.**

`[VERIFIED]` CLRA-Dex has a clean state boundary: it would preserve one unified SMPL-X model, freeze body/wrist/shape/camera/translation, discard HandFlow root/shape/translation/vertices, and change only a 23-D SignHPoser latent that decodes 15 local finger rotations. Those rotations can alter centered hand vertices, so the intervention is conceptually aligned with official LHand/RHand TR-V2V and the retained hand vertices in UBody(-F).

The central mechanism nevertheless fails the mandatory missing-observation kill test. HandFlow forms each target condition as $m_t c_t+(1-m_t)c_{mask}$. When the factual observation is already missing, $m_{t,s}=0$. Setting the target confidence to zero again leaves the factual and masked inputs identical. With the same checkpoint, state, and initial noise, the deterministic ODE therefore gives $H^+_{t,s}=H^-_{t,s}$ in exact arithmetic. The two constraints become duplicates and CLRA-Dex reduces to an ordinary frozen-HandFlow-to-SignHPoser projection. The proposed paired intervention contributes no information in precisely the missing regime named by the central hypothesis. HandFlow may still perform temporal infilling, but that is HandFlow's contribution, not CLRA-Dex's counterfactual arbitration.

`[VERIFIED]` For $0<m<1$, the pair isolates the frozen model's response to changing one scalar box-confidence gate; it does not isolate physical causality, local-finger uncertainty, or correctness. The two responses share the same model and correlated HaMeR/WiLoR-derived evidence. Common noise controls draw mismatch only. CLRA-Dex also never optimizes the response difference itself; it seeks a latent close to both outputs.

`[VERIFIED]` The proposed “safe” energy omits DexAvatar's heavily weighted direct HaMeR axis-angle anchor, creating a new post-fit objective. Its non-worsening guarantee concerns only robust 2D reprojection plus latent regularization, not 3D geometry or TR-V2V. Improvement therefore cannot be attributed to the paired mechanism without decisive controls.

Finally, $\mathcal C_s$, side identity, overlap-window semantics, numerical acceptance tolerance, and solver tie-breaking remain operationally unresolved. StableHand and ATDEdit also occupy the closest scientific ideas—state-selective quality-aware hand flow and common-noise condition-sensitivity/projection—leaving the surviving delta as an underspecified inference wrapper. Fixing the missing-regime collapse or restoring a meaningful central contribution would require changing the frozen mechanism, which meets the Step 5 definition of `REJECT`, not `MAJOR REVISION`.

---

## 2. Steelman summary of CLRA-Dex

`[PROPOSED]` CLRA-Dex targets unreliable monocular hand evidence in retained sign-language frames while preserving DexAvatar's final SMPL-X parameterization. It starts only after released DexAvatar fitting. For each physical hand side and eligible frame, it freezes the RGB-derived observations, complete DexAvatar state, body and wrist chain, shape, expression, camera, translation, HandFlow, and SignHPoser.

The intervention queries frozen HandFlow twice over its temporal sequence: a factual query with the released confidence sequence and a target-masked query in which only the target frame's confidence is set to zero. Both queries reuse one initial noise realization. Only their 45 finger-axis-angle values are retained; root orientation, shape, translation, camera/world trajectory, MANO mesh, and vertices are discarded. A side-dependent interface $\mathcal C_s$ is intended to express those rotations in the DexAvatar/SMPL-X local hand frames.

The sole optimized variable is a 23-D SignHPoser latent for one side–frame. Its decoded 15 local rotations must not worsen a truncated fixed-observation energy, must not move farther from either HandFlow response than the DexAvatar latent, and must improve at least one support distance. Otherwise the method returns the exact baseline state. The selected hand block is inserted into the unchanged parameter vector and one unified SMPL-X forward pass produces the final mesh.

The hypothesis is falsified if the paired masked query adds no behavior beyond factual HandFlow projection, if accepted changes do not improve centered local-hand geometry, if reliable frames regress, if missing/low-confidence coverage is negligible, or if the handed local-frame mapping cannot be validated. This is internally describable, although several operational details remain unspecified.

---

## 3. Top three strongest contributions

These are the strongest *proposed* contributions under a steelman reading; none is an empirically established contribution.

1. **Metric-relevant state isolation.** `[VERIFIED]` The candidate confines changes to 15 local finger rotations and regenerates one unified SMPL-X mesh. Pure translation, camera, wrist-root copying, MANO vertex stitching, and evaluator changes are excluded. This is unusually disciplined relative to the centered TR-V2V contract (Step 4 §§6–8; evaluator `transl_point_error`, lines 159–169).

2. **Controlled paired model query with exact fallback.** `[PROPOSED]` Reusing the same noise removes branch-wise random-draw mismatch, and returning the released DexAvatar state avoids a compulsory estimator overwrite. The strongest defensible interpretation is a conservative *condition-ablation consistency* query, not causal or calibrated uncertainty inference (HandFlow `condition_builder.py`, lines 29–66; `inference_utils.py`, lines 206–278).

3. **Conceptually separable causal controls.** `[INFERENCE]` The factual support, masked support, changed post-fit energy, dual-support rule, and refusal behavior can in principle be contrasted separately. This makes attribution testable if all controls in §16 are reported; it does not itself establish novelty or benefit.

---

## 4. Top three strongest reasons for rejection

1. **Core collapse in the missing regime — `FATAL`.** `[VERIFIED]` If $m_{t,s}=0$, factual and target-masked inputs are identical and same-noise deterministic inference gives $H^+=H^-$. The paired mechanism contributes nothing where the central hypothesis explicitly claims missing-observation recovery.

2. **The optimized rotation state is not operationally defined — `FATAL`.** `[VERIFIED]` Neither $\mathcal C_R$ nor $\mathcal C_L$ has a validated MANO/HandFlow-to-SMPL-X joint-order, local-parent-frame and parity contract. Until that is closed, the $SO(3)$ support distances can compare different physical rotations and the accepted hand block is not demonstrably a valid unified-SMPL-X state. The left README/demo behavior is additionally inconsistent.

3. **The surviving delta is an unresolved inference wrapper — `FATAL`.** `[INFERENCE]` Frozen generative-prior test-time refinement, conditional/unconditional paired queries, common-noise response comparison, projection, Pareto language, and baseline fallback all have close prior precedents. StableHand attacks state-selective bimanual hand quality more directly; ATDEdit already formalizes same-state condition sensitivity with common noise and projection. Because $\mathcal C_s$, physical-side identity, window merging, numerical acceptance, and solver selection are not operationally closed, CLRA-Dex does not preserve a distinct, executable scientific mechanism.

---

## 5. Targeted novelty-collision review

### 5.1 Search protocol

`[VERIFIED]` Search date: 2026-08-26. Sources were restricted to arXiv, PMLR, NeurIPS/CVF proceedings, author project pages, and official GitHub repositories. The search was deliberately narrow; it was not a broad literature review.

Representative queries:

- `common noise paired conditional counterfactual diffusion flow inference`;
- `same-state condition switch sensitivity common random numbers diffusion`;
- `masked condition consistency frozen generative model test-time optimization`;
- `Pareto common descent projection baseline fallback pose fitting`;
- `quality uncertainty temporal hand motion flow matching bimanual`;
- `diffusion prior SMPL-X MANO hand pose completion test-time optimization`;
- `counterfactual diffusion causal model abduction intervention`;
- `conservative safe model update abstention generative prior`.

Inclusion required a technical mechanism overlapping at least one claimed CLRA component. Works about recognition, generation without pose recovery, rendering, 2D-only estimation, or generic commentary were excluded. Six additional works were retained—below the maximum of ten—besides the six already audited in Steps 2–4.

### 5.2 Collision matrix

| Work | Actual mechanism and state | Frozen generative model? | Paired masked/counterfactual query? | Common-random control? | Projection / fallback / Pareto? | Collision with CLRA-Dex | Evidence |
|---|---|---:|---:|---:|---|---|---|
| **HandFlow: Fully Generative 4D Hand Recovery with Flow Matching** — Xu et al., 2026 | T=16 single-hand flow over MANO shape10, pose48 AA, translation3; continuous confidence replaces image and skeleton tokens with a mask token | Yes at released inference | Native cmask, but no paired comparison | No native paired control | No CLRA-style projection/fallback | **`PARTIAL COLLISION`**, high: temporal generative substrate and masking are inherited | [Paper §3, Eq. 5](https://arxiv.org/html/2607.11221v1); official repo commit `67fa7df…`, `condition_builder.py:29–66` |
| **StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video** — Zeng et al., 2026 | T=150 dual-hand state with four learned quality channels for left/right wrist/fingers; quality-aware flow training and ODE | No; new learned components | No paired target mask | No | Quality-conditioned generation, no CLRA fallback | **`PARTIAL COLLISION`**, very high on the scientific question; it is stronger on component-selective and true bimanual uncertainty | [Paper §§3–4](https://arxiv.org/html/2605.18553v1); [official repo](https://github.com/huajian-zeng/stablehand) was placeholder-only on access date |
| **HMP: Hand Motion Priors for Pose and Shape Estimation from Video** — Duran et al., WACV 2024 | Right-canonical motion VAE prior and latent fitting/infilling for MANO hand motion | Frozen prior during fitting | Masked observations, not paired counterfactual queries | No | Latent test-time optimization | **`PARTIAL COLLISION`**: prior-manifold projection and missing-data fitting precede CLRA | [CVF paper](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html); repo commit `35d799f…` |
| **Dyn-HaMR** — Yu et al., CVPR 2025 | Two side-indexed tracks, per-hand HMP latents, global/world optimization and interaction constraints | Uses frozen priors/estimators | No paired mask | No | Multi-term optimization; no exact baseline fallback | **`PARTIAL COLLISION`**: per-side temporal fitting, missingness, and abstention metadata are not new | [CVF paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html); repo commit `fa9cd741…` |
| **HaPTIC: Predicting 4D Hand Trajectory from Monocular Videos** — Ye et al., 3DV 2026 | T=8 temporal HaMeR adaptation; separate side streams; MANO pose/shape and trajectory | Checkpoint frozen at inference | No | No | No CLRA rule | **`PARTIAL COLLISION`**, low: temporal side-stream estimation only | [Paper §§3–4](https://arxiv.org/html/2501.08329v1); repo commit `f9362c1…` |
| **Hand4Whole++** — Moon, CVPR 2026 | Whole-body/hand conditional modulation; predicts SMPL-X but released final mesh also aligns/scatters MANO vertices | Frozen estimator at inference | No | No | No | **`NO COLLISION`** with the core; relevant only to wrist/body-hand consistency and representation constraints | [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html); repo commit `f81d35d…` |
| **ATDEdit: Diffusion Image Editing via Asynchronous Token Decoding** — Shi, Lu, Guo et al., ACM MM 2026 | Training-free frozen DiT; same-state condition switch, token-wise response magnitude, common-noise coupling, hard source-state projection | Yes | Yes, source/target condition pair | Yes | Hard projection; not Pareto/fallback | **`PARTIAL COLLISION`**, very high: common-noise conditional sensitivity plus projection is already explicit prior art, though for image tokens rather than hand pose | [Paper §§3.2–3.4](https://arxiv.org/html/2608.09322v1); official code **not found** on access date |
| **Classifier-Free Diffusion Guidance** — Ho & Salimans, 2022 | Joint conditional/unconditional score estimates combined during sampling | Yes at inference | Conditional/unconditional pair; not leave-one-frame-out | Shared noisy state in the standard score evaluation | Weighted guidance, no fallback/Pareto | **`PARTIAL COLLISION`**, foundational: condition ablation inside one generative model is not itself novel | [arXiv:2207.12598](https://arxiv.org/abs/2207.12598) |
| **Diffusion Causal Models for Counterfactual Estimation** — Sanchez & Tsaftaris, CLeaR 2022 | Known SCM; deterministic latent inference followed by intervention in reverse diffusion | Yes at inference | Causal intervention, not CLRA mask pair | Deterministic latent abduction | Counterfactual generation; no CLRA projection | **`NO COLLISION`** with pose fitting, but decisive terminology boundary: genuine counterfactual estimation assumes causal structure and abduction–intervention semantics absent in CLRA | [PMLR 177, §§1–3](https://proceedings.mlr.press/v177/sanchez22a.html); [official code](https://github.com/SANCHES-Pedro/Diff-SCM) |
| **ScoreHMR: Score-Guided Diffusion for 3D Human Recovery** — Stathopoulos et al., CVPR 2024 | Frozen conditional diffusion score guides test-time SMPL fitting to 2D, multiview, or temporal evidence | Yes | No | No | Guided iterative refinement; no baseline-dominance fallback | **`PARTIAL COLLISION`**: frozen generative-prior, observation-guided post-fit refinement is established | [Project/paper](https://statho.github.io/ScoreHMR/); [official MIT repo](https://github.com/statho/ScoreHMR) with checkpoints and train/eval paths |
| **DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior** — Lu et al., ICCV 2025 Oral | Unconditional diffusion pose prior; variational sampling and test-time optimization for completion, IK, HMR, hands and whole body | Yes at downstream use | Masked training/completion, not paired conditions | Uses diffusion noise, not CLRA common pair | Generative-prior regularization/projection; no fallback | **`PARTIAL COLLISION`**: pose-manifold TTO and missing-pose completion are established | [Paper §§2.2–2.5](https://arxiv.org/html/2508.00599v2); [official MIT repo](https://github.com/moonbow721/DPoser-X) with checkpoints/training |
| **Multi-Task Learning as Multi-Objective Optimization** — Sener & Koltun, NeurIPS 2018 | Gradient-based multi-objective optimization with Pareto-stationarity analysis | No | No | No | Explicit Pareto/common-descent machinery | **`PARTIAL COLLISION`** on terminology only: Pareto/common-descent is established, while CLRA does not implement MGDA or prove Pareto stationarity | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html); [official code](https://github.com/isl-org/MultiObjectiveOptimization) |

### 5.3 Novelty conclusion

**Level: `WEAK / ENGINEERING`.**

`[VERIFIED]` No inspected work contains the exact full tuple “HandFlow factual/target-masked pair + SignHPoser latent + two baseline-dominance inequalities on $SO(3)$ + exact Dex fallback.” Therefore this audit does **not** assert literal `CORE COLLISION` with one paper.

`[INFERENCE]` That tuple is nevertheless compositional rather than a new information source: HandFlow supplies masking and temporal generation; HMP/ScoreHMR/DPoser-X establish frozen-prior test-time projection; multi-objective literature supplies Pareto language; fallback is conservative selection; and ATDEdit already establishes common-noise same-state condition sensitivity plus projection. StableHand attacks the same hand-motion quality problem with finer state decomposition and actual bimanual modeling. Once the missing-case pair collapses and causal/safety wording is removed, the distinct scientific content is not strong enough to survive as more than a DexAvatar-specific inference wrapper.

---

## 6. Counterfactual and causal audit

### 6.1 What the intervention actually removes

`[VERIFIED]` HandFlow stacks one image token and one skeleton token per frame, expands the same scalar $m_t$ to both, and computes

\[
c'_t=m_t c_t+(1-m_t)c_{mask}.
\]

At $m_t=0$, both target image and skeleton condition tokens are replaced by the learned mask token (`condition_builder.py`, lines 29–66). This is a **model-input ablation**, not deletion of the physical observation, crop, target time state, model prior, neighboring evidence, or shared upstream biases. In the audited default denoiser path, the two target condition tokens are withheld from cross-conditioning; optional external reprojection guidance is not part of the frozen CLRA specification.

### 6.2 Evidence independence and the common-noise scope

| Question | Finding | Verdict |
|---|---|---|
| Are $H^+$ and $H^-$ two independent evidence sources? | No. They are two responses from one frozen model, one checkpoint, one temporal context, one HaMeR/WiLoR frontend, and one noise realization. | `[VERIFIED]` |
| What does common noise control? | It removes the difference due solely to choosing different initial Gaussian states. This is a common-random-number control. | `[VERIFIED]` HandFlow `inference_utils.py:206–212`; ATDEdit §3.3 supplies the relevant variance argument. |
| What does common noise not control? | Detector error, wrong crops, left/right mistakes, HaMeR feature/keypoint bias, HandFlow domain bias, model misspecification, ODE discretization bias, or SignHPoser manifold bias. | `[INFERENCE]`, directly implied by unchanged inputs/model. |
| Is the output difference used? | No. The objective uses separate distances to $H^+$ and $H^-$; it never uses $H^+-H^-$, a calibrated uncertainty, or response magnitude. | `[VERIFIED]` Step 4 §10.5. |
| Does agreement imply correctness? | No. Both responses can agree because they share the same model and context and can agree on the same wrong mode. Step 4 itself records this failure. | `[VERIFIED]` Step 4 §§9.1, 17. |

### 6.3 Confidence is not local-finger uncertainty

`[VERIFIED]` The released online path gets `valid_confs` from the selected WiLoR YOLO bounding box and stores that scalar as `hamer_confidence` (`online_hamer.py`, lines 223–255, 298–309). It is shared across the two tokens of the frame. It is neither per-finger nor per-joint, and no calibration to 3D finger error is documented. HandFlow's own confidence ablation was reported mainly to stabilize translation rather than isolate local articulation (HandFlow §4.5, Table 4; Step 3 §§1–2).

- **High confidence, wrong fingers.** `[INFERENCE]` CLRA can only change the state if the masked and factual model outputs plus the surrogate constraints jointly admit a new latent. It has no explicit signal identifying a high-confidence local-finger failure, so intervention is possible but not targeted or guaranteed.
- **Low confidence, correct observation.** `[INFERENCE]` As $m$ approaches zero, factual and masked conditions become similar. Both may favor the temporal prior over a brief correct target detail; 2D non-worsening cannot protect monocular depth/articulation. The fallback may reject the change, but does not prove it will.

### 6.4 Permissible terminology

`[VERIFIED]` Diff-SCM performs counterfactual estimation with an explicit causal structure and latent abduction/intervention. CLRA-Dex has neither an SCM nor an identified physical intervention. ATDEdit is closer terminologically: it calls the same-state condition-switch response a model-derived heuristic and explicitly denies that it is a correctness certificate (§3.4).

Accordingly:

- **Allowed:** “paired common-noise condition ablation,” “frozen-model response consistency,” or “model-conditioning intervention.”
- **Not supported:** “causal isolation,” “causal counterfactual of the hand,” “two independent supports,” “local-finger uncertainty estimate,” or “agreement certificate of correctness.”

**Audit result:** the narrow model-conditioning interpretation survives; the causal and uncertainty interpretations do not.

---

## 7. Missing-observation collapse test

This is the decisive kill test.

For a target whose released factual confidence is already zero,

\[
m_{t,s}=0,
\qquad
m^{\setminus t}_{t,s}=0.
\]

All other confidence values, observations, checkpoint parameters, normalization, and initial noise are held equal by construction. Therefore

\[
(O_s,m_s;\epsilon_s)=(O_s,m_s^{\setminus t};\epsilon_s).
\]

Because the released Euler/ODE inference is deterministic conditional on these quantities,

\[
F_\phi(O_s,m_s;\epsilon_s)
=F_\phi(O_s,m_s^{\setminus t};\epsilon_s),
\qquad
H^+_{t,s,j}=H^-_{t,s,j}\quad\forall j
\]

in exact arithmetic. Any measured discrepancy would be numerical nondeterminism or an uncontrolled implementation difference, not counterfactual information.

The support costs become identical:

\[
\mathcal D^+_{t,s}(z)=\mathcal D^-_{t,s}(z)=\mathcal D^0_{t,s}(z),
\]

so the two support constraints reduce to one, and the minimax objective reduces to $\mathcal D^0$. The frozen mechanism is then simply:

> project one HandFlow temporal infill onto the SignHPoser latent while constraining a truncated DexAvatar surrogate, or retain the baseline.

| Kill-test question | Answer |
|---|---|
| Does the factual/masked pair add information at $m=0$? | **No — exactly none in the specified model.** `[VERIFIED]` |
| Does dual support remain dual? | **No.** The two feasible inequalities and two objective terms are duplicates. `[VERIFIED]` |
| Can HandFlow still infer a pose from neighboring context? | Possibly; HandFlow was trained with masking and uses a full window. That is **HandFlow infilling**, not the paired CLRA mechanism. `[VERIFIED]` mechanism, usefulness `[UNRESOLVED]`. |
| Does this contradict the frozen central hypothesis? | Yes. Missing observations are explicitly part of the reduced PRIMARY and Step 4 hypothesis, while the claimed novel component disappears there. `[VERIFIED]` |
| What happens on a long missing span? | Step 4 requires unambiguous same-side context on both sides inside a 16-frame window and refuses unanchored boundaries/whole-window gaps. Eligible coverage therefore contracts as spans lengthen; actual coverage is `[UNRESOLVED]`. |
| Does CLRA recover evaluator-missing frames? | No. Step 4 restricts intervention to $\mathcal T_E$, frames already retained by the output/evaluator contract. “Missing” can only mean missing/unreliable hand observation within a retained frame. `[VERIFIED]` |
| Recovery or abstention? | The specification may accept a single-support HandFlow projection or abstain. It provides no evidence on the ratio; recovery mass is `[UNRESOLVED]`. |

**Kill-test decision: `FATAL`.** Restricting the claim to $0<m<1$ would remove the literal collapse but would abandon the frozen missing-observation claim. Creating a distinct missing-case query or a new information source would change the central mechanism and is outside a Step 5 revision.

---

## 8. Mathematical audit

### 8.1 Validity of the idealized optimization

| Item | Audit | Status |
|---|---|---|
| Baseline feasibility | Substituting $z=z^D$ satisfies all three non-strict inequalities by equality. Thus $\mathcal Z_{t,s}\neq\varnothing$. | `[VERIFIED]` valid. |
| Existence of an argmin | If the retained latent weight is strictly positive, $\mathcal E^{safe}$ contains $\lambda\|z\|^2$, making its baseline sublevel bounded; continuity makes the feasible set closed, so Weierstrass gives existence. Dex final stages use hand-prior weight 4.78, but stage 1 uses zero (`fit_smplx_vposer_x.yaml:90–93`). Step 4 does not name the arbitration-stage weight precisely. | `[CONDITIONAL]`; not guaranteed by the frozen text as written. |
| Compactness/coercivity | Holds under a positive latent coefficient and continuous decoder/projection/robustifier. It is not implied if the selected weight is zero. | `[CONDITIONAL]`. |
| Uniqueness | A nonlinear 23-D decoder need not be injective; non-convex preimages and tied minimax solutions can exist. No uniqueness theorem or deterministic tie-break is supplied. | `[UNRESOLVED]`; uniqueness cannot be assumed. |
| Local-frame validity | $d_{SO(3)}$ is meaningful only after $H^\pm$ and $R(z)$ refer to the same physical joint, parent, side, pose mean, and local frame. That is exactly the unresolved duty of $\mathcal C_s$. | `[CONDITIONAL]`; otherwise the objective is semantically undefined. |
| Axis-angle periodicity | Applying $\operatorname{Exp}$ before the trace angle correctly compares rotations rather than raw axis-angle vectors, so $2\pi$-equivalent encodings do not create a false Euclidean difference. | `[VERIFIED]` mathematical merit. |
| Cut locus near $\pi$ | The geodesic is valid but non-smooth at the cut locus; `arccos` is ill-conditioned near 0 and $\pi$, and clipping can yield poor/zero gradients outside the interval. No numerical treatment is specified. | `[VERIFIED]` numerical risk. |

### 8.2 Strict acceptance and “threshold-free” behavior

`[VERIFIED]` “At least one strict inequality” is well-defined over real numbers. It is not a robust finite-precision decision rule. Solver tolerance, approximate feasibility, decoder/SMPL-X floating-point error, and repeated HandFlow inference can create infinitesimal sign changes. Without a predeclared numerical margin, two hardware/library configurations can accept different frames.

`[INFERENCE]` A tolerance would be an arithmetic/solver contract, not a confidence threshold. It could be fixed from numerical precision, deterministic-repeatability checks, and non-test validation without SGNify test GT. But adding such a margin contradicts the literal “threshold-free” operational claim unless that wording is narrowed. The frozen specification provides neither a value nor a selection rule.

### 8.3 Is this Pareto/common-descent arbitration?

The feasible set enforces weak dominance of the **baseline** for the two support distances; admission adds one strict support improvement. This is a valid baseline-dominance partial order if all quantities are defined.

It is not equivalent to a common descent direction or MGDA:

- no gradient-combination/common-descent condition is imposed;
- $\max(\mathcal D^+,\mathcal D^-)$ is a Chebyshev compromise, not a proof of Pareto stationarity;
- a minimizer can be dominated by another minimizer when the maximum ties;
- $\mathcal E^{safe}$ is only a feasibility constraint, so the chosen point need not be Pareto-efficient in the three-objective vector $(\mathcal E^{safe},\mathcal D^+,\mathcal D^-)$.

**Permissible wording:** “baseline-dominance-constrained dual-support minimax projection.”  
**Unsupported wording:** “Pareto-optimal solution,” “common-descent certificate,” or “multi-objective safety certificate.”

### 8.4 Correctness and metric safety

`[VERIFIED]` Nothing in the inequalities refers to 3D ground truth or official TR-V2V. A latent can be closer to two correlated but wrong HandFlow outputs, retain equal/better 2D projection, and have lower latent norm while producing worse centered 3D vertices. This follows directly from monocular 2D depth ambiguity and shared-model error; Step 4 already acknowledges the shared-wrong-mode case (§17).

The four layers must therefore be separated:

| Layer | Finding |
|---|---|
| Mathematical validity | Baseline feasibility is valid; existence is conditional on a positive latent term; no uniqueness is established. |
| Numerical implementability | Underspecified because strict comparison, cut-locus gradients, solver convergence, tie-breaking, and reproducibility tolerance are absent. |
| Empirical usefulness | Entirely `[UNRESOLVED]`; no candidate or score was run. |
| Metric safety | **Not provided.** Surrogate non-worsening does not imply non-worsening LHand/RHand/UBody(-F) TR-V2V. |

**Mathematical verdict:** partially valid as an ideal baseline-dominance projection, but the “Pareto/common-descent,” “threshold-free,” and metric-safety readings do not survive.

---

## 9. Safe-energy and baseline-confounding audit

### 9.1 The objective is not the released DexAvatar objective

`[VERIFIED]` Released DexAvatar uses robust 2D reprojection (`fitting.py:520–525`), SignHPoser latent regularization, and direct decoded-axis-angle supervision toward the HaMeR-derived hand target (`fitting.py:543–590`). The default config assigns the latter weight 1200 in every fitting stage (`fit_smplx_vposer_x.yaml:66–73`). CLRA's target trial removes that direct term while retaining only the robust hand reprojection and latent penalty.

Therefore the post-fit optimization changes two factors simultaneously:

1. it removes a dominant baseline target term; and
2. it adds HandFlow support constraints and a new selection rule.

`[VERIFIED]` Even with no HandFlow information, reoptimizing the truncated energy can change the latent. Any gain could come from releasing the direct AA anchor, from an additional solver pass, or from the latent mean—not from the factual/masked pair.

### 9.2 Correlated evidence and ambiguity

- `[VERIFIED]` HandFlow conditions use frozen HaMeR features/keypoints, with $m$ supplied by the selected WiLoR/YOLO box. DexAvatar's non-class-`0` hand path inserts HaMeR hand estimates and sets inserted hand-keypoint confidences to one (Step 4 §7.1; Dex `data_parser.py:397–425`). The sources are correlated, not independent.
- `[INFERENCE]` Retaining 2D evidence while adding two HandFlow distances may reuse upstream hand evidence in different transforms. It is not a probabilistic double likelihood because CLRA does not claim independence, but it is still a confound and cannot be called independent corroboration.
- `[VERIFIED]` A latent Gaussian penalty can prefer the SignHPoser mean when 2D residuals are flat or ambiguous. Equal/better 2D projection does not resolve finger depth, self-occlusion, or articulation around the camera ray.

### 9.3 What “safe” can and cannot mean

| Claim | Audit verdict |
|---|---|
| Same stored upstream arrays are used | `[VERIFIED]` by formulation. |
| Truncated surrogate does not increase in exact arithmetic | `[PROPOSED]`, mathematically true for a feasible accepted point. |
| Full released DexAvatar objective does not increase | **False**, because the direct AA term is removed and not checked. |
| 3D hand geometry does not worsen | **Unsupported.** |
| Official TR-V2V does not worsen | **Invalidated as a guarantee.** |
| Observations are independent/calibrated | **False/unsupported.** |

The only defensible phrase is **“non-worsening under the specified truncated fixed-array surrogate.”** “Safety certificate,” “fixed-observation accuracy safety,” and “metric-safe update” exceed the evidence.

### 9.4 Required causal separation

The minimum controls that isolate these confounds are stated in §16. Without them, an observed aggregate change cannot be attributed to the counterfactual pair, the target-masked branch, or Pareto/refusal behavior. This is a causal-attribution issue, not an evaluator artifact.

---

## 10. Temporal/window operational audit

`[VERIFIED]` The pinned HandFlow release uses $T=16$, overlap 2, three Euler steps, and default velocity blending. It creates one global pose-noise vector per sequence frame, assembles every overlapping window from those shared frame states, predicts window velocities, and blends velocities before each global state update (`configs/inference.yaml:6–15`; `inference_utils.py:195–278`). The alternative path independently integrates windows and averages their final normalized states (`inference_utils.py:280–317`). CLRA freezes the default `vblend` contract.

### 10.1 What Step 4 fixes and what it does not

| Operational question | Frozen Step 4 status | Review classification |
|---|---|---|
| Which windows must see the target mask? | It says “set only the target side–frame cmask to zero” at sequence level. Logically every overlapping window containing that global frame must receive the zero. | Meaning inferable, but not stated as an explicit window contract: **clarification needed**. |
| Is $H^-_t$ taken before or after overlap blending? | The notation treats $F_\phi$ as a sequence function, but does not explicitly bind $H^\pm$ to the release's final post-`vblend` state. | **Major underspecification** because pre-window and post-blend responses are different objects. |
| How is common noise shared? | “One predeclared draw per physical-side sequence” and reuse across the pair is specified. The release internally samples global pose/root/shape arrays per call; the frozen text does not define an operational interface that guarantees byte-identical arrays across target-specific reruns. | Conceptually clear; operationally **major underspecification**. |
| Does masking target (t) change neighboring outputs? | Yes in principle: full-window attention and blended velocities couple frames. Step 4 discards non-target outputs but does not specify whether their changed state participates in later target queries. | `[INFERENCE]`; target queries must be independent relative to one fixed baseline, otherwise order dependence appears. Frozen semantics are incomplete. |
| Can adjacent accepted latents conflict? | Each side–frame latent is optimized independently after temporal support generation. No sequence-level constraint or joint selection exists. | `[VERIFIED]` by formulation; temporal incoherence risk `[INFERENCE]`. |
| Boundary/skipped indices | HandFlow windows operate on input sequence positions; Step 4 preserves retained timestamps but does not define resampling/FPS or adjacency after skipped source indices. | **Major underspecification**, already unresolved in Step 3 §3. |
| Cost of one masked query per side–frame | The frozen text implies a distinct full paired sequence response for every tested target. Because one target affects all containing windows, responses cannot be read from one jointly masked pass. No cost/coverage bound is supplied. | Feasibility `[UNRESOLVED]`; potentially substantial, not measured. |

### 10.2 Consequences

- `[INFERENCE]` Independent accept/reject decisions can alternate between DexAvatar and HandFlow-supported states even when the underlying HandFlow trajectory is smooth. The official evaluator has no temporal smoothness term, so flicker is not directly penalized unless it changes per-frame centered geometry.
- `[INFERENCE]` A target-specific masked rerun changes the denoising context for other frames in all overlapping windows. Although only $H^-_t$ is consumed, the final value depends on the exact global merge semantics.
- `[VERIFIED]` A missing span near a boundary often violates Step 4's “valid context on both temporal sides” rule and triggers refusal; this does not improve the baseline.

**Classification:** **`MAJOR UNDERSPECIFICATION`**, not independently fatal. A unique mathematical target response can be defined without changing the high-level mechanism, but Step 4 has not done so. In combination with the missing-case collapse and unresolved $\mathcal C_s$, the candidate is not ready for an empirical specification.

---

## 11. SMPL-X and handedness audit

### 11.1 What is established

- `[VERIFIED]` HandFlow pose48 is `global_aa(3) + joints_aa(45)`; CLRA discards the first three values and all non-finger state (HandFlow `configs/model.yaml:1–6`; Step 4 §§2, 6–7).
- `[VERIFIED]` HandFlow's MANO runtime and DexAvatar's audited neutral SMPL-X runtime both use `flat_hand_mean=True` and `use_pca=False` (HandFlow `mano_utils.py:20–29`; Dex `main.py:144–151`). This closes one pose-mean precondition only for that audited neutral runtime.
- `[VERIFIED]` DexAvatar already decodes a 23-D SignHPoser latent to a 45-D local hand block, and final vertices can be regenerated through one SMPL-X forward pass (`fit_single_frame.py:237–239, 476–503, 627–647`). The intended final topology is therefore conceptually valid.

### 11.2 What $\mathcal C_s$ leaves unresolved

`[VERIFIED]` Equal dimensions and flat-hand means do not prove equal joint order, local parent frames, axis conventions, or parity. DexAvatar's left HaMeR parser flips the $y/z$ components of axis-angle values (`data_parser.py:397–425`), but that rule is specific to the audited HaMeR conversion and is not evidence for HandFlow.

`[VERIFIED]` HandFlow's README says left videos must be mirrored to the right-canonical training side. The demo accepts `--side left`, but the inspected crop path does not perform that mirror and feeds predicted pose directly to a left MANO layer (`scripts/demo.py:125–220`; `online_hamer.py:190–314`; `mano_utils.py:14–62`). These are inconsistent operational cues, not a validated left conversion.

| Question | Finding |
|---|---|
| Is $\mathcal C_s$ merely independent engineering validation? | It could become one **after** an authoritative convention proof. Before that, every $d_{SO(3)}(R,H^\pm)$ may compare different physical rotations; it is a mathematical precondition, not cosmetic plumbing. |
| Is right mapping closed? | No. It is dimensionally plausible, but exact joint ordering/local-parent convention has not been verified. `[UNRESOLVED]` |
| Is left mapping closed? | No; right-canonical mirroring, parity, and local-frame conversion are additionally unresolved. `[UNRESOLVED]` |
| Does symbolic notation hide a blocker? | Yes. The optimization objective and final state are not semantically reviewable for a side until $\mathcal C_s$ is fixed. `[VERIFIED]` logical consequence. |
| Can novelty be discussed abstractly? | Yes, as in §5. Feasibility and correctness cannot be accepted. |
| If only right works, is the frozen PRIMARY preserved? | No. The frozen hypothesis is two persistent physical-side sequences and claims LHand/RHand leverage. A right-only system would be a narrowed candidate, not the frozen CLRA-Dex. |

**Audit result:** **`FATAL OPERATIONAL BLOCKER` while unresolved.** A validated mapping might close this blocker without adding a scientific module, but it cannot be presumed. No mapping is constructed or guessed here.

---

## 12. Side-identity feasibility audit

`[VERIFIED]` The Step 4 “physical-side identity contract” is a representation and refusal contract, not an implemented tracking algorithm. It demands a persistent left/right label, separates observation validity from identity validity, and refuses conflicts, but does not specify how those variables are produced (§12.3).

| Required operation | Evidence in the frozen candidate | Consequence |
|---|---|---|
| Initialize a physical track | No input/association rule is specified beyond side-labelled detections. | `[UNRESOLVED]` which detection begins and retains a track. |
| Detect a side conflict or crossing | HandFlow selects the highest-confidence box of a requested detector class and can reject heavy opposite-side overlap (`online_hamer.py:242–255`); this is not persistent identity tracking. | Crossing detection and side-swap detection are `[UNRESOLVED]`. |
| Bridge a missing interval | The label is normatively held fixed, but the evidence that establishes continuity is unspecified. | A declared contract cannot create identity information. |
| Reacquire after occlusion | No reacquisition criterion is stated. | `[UNRESOLVED]`; possible silent swap or indefinite refusal. |
| Use Dyn-HaMR metadata | Step 4 borrows only the concept. The exact released tracker/submodule and crossing policy were not inspectable at commit `fa9cd741…`. | It cannot be treated as a verified dependency. |

`[INFERENCE]` Refusing every ambiguous crossing may be conservative relative to overwriting the pose, but can drive coverage toward the easiest intervals—where DexAvatar may already have reliable hands. The actual coverage and metric mass are unknown. Refusal is not accuracy improvement; it merely returns a possibly wrong baseline.

**Audit result:** the identity contract is scientifically sensible but operationally unsupported. It is a **major** blocker for low-confidence observations and a **fatal** dependency if CLRA claims two-side physical continuity without an admissible source for $(\iota_s,v^{id})$.

---

## 13. Sign-language and official-metric relevance

### 13.1 Metric linkage that survives review

`[VERIFIED]` The official evaluator independently centroid-aligns each evaluated vertex subset (`transl_point_error`, lines 159–169), omits LHand for class-`0`, removes left-hand vertices from other class-`0` regions (lines 380–395), and concatenates per-frame vertex errors before one mean (lines 432–461). Therefore:

- changes in local finger rotation can change centered LHand/RHand geometry and survive alignment;
- changed retained hand vertices can affect UBody(-F), subject to class-`0` masking;
- pure global hand/body translation does not improve the isolated hand metrics;
- visual temporal smoothness is not itself scored;
- metric mass is vertex-frame weighted, not sign balanced;
- class-`0` left updates would have no official leverage, so Step 4 correctly freezes them.

This establishes **potential leverage**, not expected improvement or safety.

### 13.2 Domain and semantic risks

| Issue | Evidence-backed fact | Scientific implication |
|---|---|---|
| Training domain | `[VERIFIED]` HandFlow was trained primarily on DexYCB/HOT3D hand–object motion, not sign-language motion (HandFlow Appendix B; Step 3 §4). | Sign articulation generalization is `[UNRESOLVED]`. |
| Brief meaningful articulation | `[VERIFIED]` DexAvatar supplementary shows motion blur and occlusion failures (PDF pp. 16–17, Figs. S8–S9). `[INFERENCE]` A masked temporal prior may smooth or miss short handshape transitions/peaks. | Qualitative smoothness can conflict with semantic fidelity and centered vertex accuracy. |
| Eligible metric mass | Step 4 requires retained evaluator frames, valid identity/interface, valid temporal context, and surrogate-feasible improvement. No leakage-safe prevalence estimate exists. | Overall TR-V2V leverage is `[UNRESOLVED]`; subset relevance cannot be extrapolated. |
| Reliable frames | CLRA tests every metric-eligible, identity-valid frame without a confidence threshold. | `[INFERENCE]` Reliable frames can change; surrogate fallback is not metric protection. Reliable-regime regression must be measured. |
| Class-`0` metadata | `[VERIFIED]` DexAvatar already reads `sign_class` from its configuration/signs file (`fit_smplx_vposer_x.yaml:19`; `main.py:85–103`). | It is baseline pipeline metadata, not newly imported SGNify test GT. Its provenance/general-deployment availability remains `[UNRESOLVED]`. |
| Sign validation | `[INFERENCE]` A non-test, signer-disjoint sign validation source could evaluate operational tolerances without test-GT tuning, but Step 4 does not pin such a manifest. | No test-GT use is permitted; leakage-safe selection remains unresolved. |

### 13.3 Four claims that must not be conflated

1. **Regime-specific hypothesis:** local rotations may help retained frames with poor hand observation and valid context — plausible but untested.
2. **Overall official TR-V2V claim:** depends on eligible vertex-frame mass, accepted-update quality, and reliable-frame regression — unsupported.
3. **Qualitative smoothness:** not directly measured by the locked evaluator.
4. **Sign-semantic preservation:** neither trained nor evaluated by CLRA's geometric surrogate.

**Audit result:** official-metric alignment survives only as a potential causal path. Sign-language relevance and aggregate benefit remain hypotheses, and the missing-case collapse removes CLRA's claimed distinctive mechanism from an important part of that path.

---

## 14. Reviewer-objection ledger

Severity follows the Step 5 definition. A fallback does not reduce severity because retaining a bad baseline is not an accuracy result.

| ID | Reviewer objection | Evidence | Affected claim/component | Severity | Resolvable without changing central mechanism? | Required resolution | Consequence if unresolved |
|---|---|---|---|---|---|---|---|
| O01 | At $m=0$, factual and masked inputs are identical, hence $H^+=H^-$. | HandFlow cmask equation/code; same-noise deterministic inference; §7 derivation. | Core paired mechanism; missing recovery | **`FATAL`** | **No** for the frozen missing-regime claim | Would require a genuinely distinct information-bearing intervention or removal of missing recovery from the central hypothesis—both change the frozen mechanism/scope. | CLRA contribution vanishes in its central missing regime. |
| O02 | The pair measures condition sensitivity, not physical causality or correctness. | One model/checkpoint/context; Diff-SCM causal requirements; ATDEdit §3.4 limitation. | Counterfactual/causal interpretation | **`MAJOR`** | Yes, wording only | Restrict language to paired model-conditioning ablation/consistency. | Causal claims are rejected. |
| O03 | $m$ is box confidence, not local-finger uncertainty. | `online_hamer.py:223–255,298–309`; one scalar shared by both tokens. | State-selective uncertainty | **`MAJOR`** | No for the uncertainty claim; yes for wording | Remove local-uncertainty/calibration language and establish only regime-stratified empirical association. | Mechanism has no verified trigger for local-finger failure. |
| O04 | Common noise controls only draw mismatch. | `inference_utils.py:206–212`; ATDEdit §3.3. | Causal isolation | **`MAJOR`** | Yes | State its limited variance-control role; do not claim detector/model-bias isolation. | “Causal isolation” remains false. |
| O05 | The optimization never consumes $H^+-H^-$. | Step 4 §10.5 uses separate distances and a max. | Counterfactual difference / novelty | **`MAJOR`** | No without changing the objective; wording can be narrowed | Describe dual-support consistency, not response-difference inference. | Claimed counterfactual diagnostic is not the operative signal. |
| O06 | $\mathcal C_R$ and especially $\mathcal C_L$ are not validated. | Step 4 §§2,12; README/demo left-hand inconsistency; no authoritative local-frame map. | All $SO(3)$ costs and final SMPL-X hand state | **`FATAL`** | Yes in principle, via independent convention validation | Authoritative joint order, parent frames, parity, pose mean, and round-trip validation for each side. | Distances may compare different rotations; output may be invalid for a side. |
| O07 | Side identity is a declaration, not an operational source. | Step 4 §12.3; HandFlow per-frame detector selection; Dyn tracker not inspected. | Physical-side continuity / two-side claim | **`MAJOR`** | Possibly, if a verified admissible metadata source already exists | Close initialization, conflict, missing-span continuity, and reacquisition provenance without importing a new optimizer. | Swaps or widespread refusal; two-side result not attributable. |
| O08 | Removing the direct AA anchor changes the baseline objective. | Dex `fitting.py:543–590`; config weight 1200; Step 4 §10.4. | Attribution to CLRA | **`MAJOR`** | Yes, by controls | Separate repeated optimization, AA-anchor removal, and HandFlow additions (§16). | Any gain can be caused by a new baseline rather than CLRA. |
| O09 | Latent mean and 2D evidence can favor the wrong 3D articulation. | Safe energy is 2D plus $\|z\|^2$; monocular ambiguity. | Safety / accuracy | **`MAJOR`** | No guarantee possible from the frozen surrogate; empirical measurement can delimit claim | Restrict safety wording and report 3D metric regressions. | Accepted states can worsen hand TR-V2V. |
| O10 | Upstream evidence is correlated. | Both paths use HaMeR-derived hand information; $m$ comes from WiLoR detector; Dex inserted hand confidence often equals 1. | Independent support / fixed-evidence safety | **`MAJOR`** | Yes, wording and attribution controls | Treat responses as correlated model outputs, not independent likelihoods. | Agreement may reproduce one shared error. |
| O11 | Argmin existence depends on the positive latent weight, but stage is not pinned. | Config hand-prior weights `0, 4.78, 4.78`. | Mathematical well-posedness | **`MAJOR`** | Yes | Bind arbitration to a strictly positive frozen weight and state continuity assumptions. | Compactness/existence is not guaranteed as written. |
| O12 | Non-unique non-convex solutions and solver tie-break are unspecified. | Nonlinear SignHPoser decoder; max objective; no uniqueness/tie rule. | Determinism / reproducibility | **`MAJOR`** | Yes, operational specification | Declare convergence, feasibility, and deterministic tie/refusal behavior without test tuning. | Same input can yield different accepted mesh. |
| O13 | Strict inequality is unstable in finite precision; “threshold-free” is non-operational. | Exact vs approximate solver arithmetic; no margin specified. | Acceptance/refusal | **`MAJOR`** | Yes, but wording must narrow | Predeclare numerical tolerance from arithmetic/repeatability and non-test validation. | Infinitesimal numerical noise changes coverage and result. |
| O14 | “Pareto/common descent” is stronger than the formulation. | Baseline-dominance constraints + Chebyshev max; no MGDA/Pareto proof. | Mathematical contribution | **`MINOR`** | Yes, terminology | Use “baseline-dominance-constrained minimax projection.” | Overclaimed theory; formulation itself can remain. |
| O15 | Target response after overlapping-window inference is not uniquely specified. | HandFlow T=16 `vblend`; target can occur in multiple windows; Step 4 abstracts $F$. | $H^\pm$, common noise, reproducibility | **`MAJOR`** | Yes | Bind masking to all target-containing windows, response to final merge, and noise to one global sequence state. | Different legitimate readings produce different supports. |
| O16 | Independent side–frame decisions can break temporal consistency. | One optimized latent per frame; no joint sequence constraint. | Temporal claim / sign semantics | **`MAJOR`** | Not without changing optimization; can delimit claim | Do not claim output temporal consistency; measure per-frame metric and semantic/temporal regressions separately. | A temporal substrate can yield a temporally discontinuous selected sequence. |
| O17 | One target-specific masked full-sequence rerun per tested state may be impractical. | Global attention/windows imply target-specific responses; no cost bound. | Training-free feasibility | **`MAJOR`** | Possibly via feasibility evidence, not method changes | Establish deterministic cost/coverage for the frozen semantics. | Candidate may be operationally unusable or silently test only a subset. |
| O18 | HandFlow's hand–object prior may suppress sign-specific fast transitions. | Training domain DexYCB/HOT3D; Dex supplementary blur/occlusion; no sign-domain validation. | Sign relevance | **`MAJOR`** | Yes empirically, without test GT | Use leakage-safe, signer-disjoint non-test sign evidence to test short transitions and occlusions. | Smooth but semantically/geometrically wrong hands. |
| O19 | Eligible metric mass is unknown. | Multiple eligibility/refusal conditions; locked vertex-frame aggregation. | Overall TR-V2V relevance | **`MAJOR`** | Yes empirically | Report eligible vertex-frame mass and regime-stratified coverage without test-guided selection. | Subset behavior cannot support an aggregate claim. |
| O20 | Reliable frames can regress despite fallback. | Every eligible frame is tested; safe surrogate is not metric safe. | Non-worsening / overall score | **`MAJOR`** | Yes empirically | Measure accepted/rejected reliable frames and their centered metric deltas. | Low-confidence gains may be canceled by reliable-frame regressions. |
| O21 | The exact component tuple is novel only as composition; central ingredients have close prior art. | HandFlow, StableHand, ATDEdit, ScoreHMR, DPoser-X, MGDA (§5). | Publication contribution | **`FATAL`** | **No**, not while the frozen pair collapses and no new information source remains | A stronger scientific mechanism would be required; Step 5 forbids replacing the frozen candidate. | Contribution remains a generic inference wrapper/loss formulation. |
| O22 | Class-`0` is not newly justified as deployment metadata. | Dex already reads `sign_class`; external provenance is not documented. | Scope/generalization | **`MINOR`** | Yes | State that class labels are inherited baseline inputs and delimit deployment scope. | No leakage claim, but general-use scope remains unclear. |

---

## 15. Claim-by-claim verdict

| # | Frozen claim | Verdict | Evidence-based reason | Wording permitted after review |
|---:|---|---|---|---|
| 1 | Counterfactual interpretation | **`SURVIVES WITH NARROWER WORDING`** | It is a condition intervention inside one frozen model, not a physical/causal counterfactual. | “Target-condition ablation response of frozen HandFlow.” |
| 2 | State-selective uncertainty | **`UNSUPPORTED`** | Box confidence is frame scalar; the output pair is not calibrated uncertainty and is not part/joint selective. | “A side–frame condition-sensitivity probe,” without uncertainty/calibration claims. |
| 3 | Common-noise causal isolation | **`INVALIDATED`** | Same noise removes draw mismatch only; all upstream/model biases remain. | “Common-random-number control for paired inference.” |
| 4 | Dual-support/Pareto arbitration | **`SURVIVES WITH NARROWER WORDING`** | It weakly dominates the baseline in two support distances when $m>0$ and $\mathcal C_s$ is valid; it is not MGDA/Pareto-certified and collapses at $m=0$. | “Baseline-dominance-constrained dual-support minimax projection.” |
| 5 | Safety/non-worsening | **`INVALIDATED`** as 3D/metric safety | Only a truncated 2D-plus-latent surrogate is constrained; the full Dex objective and TR-V2V can worsen. | “Exact-arithmetic non-increase of the specified truncated surrogate,” conditional on solver feasibility. |
| 6 | Missing-frame recovery by CLRA | **`INVALIDATED`** | $H^+=H^-$ at factual $m=0$; any infill is HandFlow single-support behavior. | None for the paired mechanism; only “HandFlow temporal infill projected to SignHPoser” describes the degenerate case. |
| 7 | Physical-side continuity | **`UNSUPPORTED`** | The identity variables are a contract without verified generation/tracking/reacquisition semantics. | “Requires externally validated persistent-side metadata; otherwise refuses.” |
| 8 | SMPL-X compatibility | **`UNSUPPORTED`** as an operational claim | Unified final topology is sound in principle, but $\mathcal C_R,\mathcal C_L$ are not validated, so $SO(3)$ costs/final rotations are not yet meaningful. | “Conditionally SMPL-X-parametric after side-specific convention validation.” |
| 9 | Official-metric alignment | **`SURVIVES`** | Local finger rotations change centered evaluated vertices; non-local HandFlow outputs and evaluator changes are excluded. | “Has a direct potential path to centered hand/retained-hand TR-V2V.” |
| 10 | Training-free feasibility | **`SURVIVES WITH NARROWER WORDING`** | No weights are trained, and checkpoints exist; target-specific reruns, mapping, identity, and solver semantics remain unresolved. | “No learned parameters are added; end-to-end operational feasibility is conditional.” |
| 11 | Novelty beyond HandFlow replacement | **`UNSUPPORTED`** | The distinctive tuple is compositional, high-overlap prior art exists, and the pair adds no information at $m=0$. | “A DexAvatar-specific inference wrapper combining established components,” not a validated new scientific mechanism. |
| 12 | Overall official TR-V2V relevance | **`SURVIVES WITH NARROWER WORDING`** | Metric leverage exists, but eligible mass, acceptance quality, domain generalization, and reliable-frame regression are unknown. | “Potential metric relevance; no improvement or non-worsening claim.” |

---

## 16. Minimum decisive controls required for Step 6

Because the verdict is `REJECT`, this section is a **falsifiability ledger**, not authorization to proceed with CLRA-Dex or an experimental plan. It records the minimum logical comparisons that would have been necessary to attribute any observation. No hyperparameter, runtime, score, dataset split, or implementation procedure is specified.

| Control / reporting stratum | Logical question isolated |
|---|---|
| **Released DexAvatar** | Establish the unchanged comparator and verify that all reported deltas originate after baseline fitting. |
| **Post-fit repeat with the full released hand objective** | Separate an additional SignHPoser optimization/solver pass from all objective changes. |
| **Post-fit reoptimization with CLRA's safe energy but no HandFlow** | Measure the joint effect of removing the direct AA anchor and optimizing the truncated 2D-plus-latent energy. |
| **Factual-only HandFlow local projection** | Isolate ordinary frozen HandFlow finger support from the masked branch and dual arbitration. This is the true engineering replacement baseline. |
| **Target-masked-only projection** | Determine whether temporal masked support alone explains behavior, especially at low confidence and $m=0$. |
| **Factual + masked supports without Pareto/refusal** | Separate the presence of both supports from baseline-dominance selection and abstention. |
| **Full frozen CLRA-Dex** | Test the indivisible claimed tuple against every constituent control. |
| **Same-noise versus independent-noise pair** | Establish whether common random numbers change decision stability rather than accuracy; no causal interpretation follows. |
| **Exact $m=0$ equality check** | Confirm the analytical collapse $H^+=H^-$ and prevent numerical noise from being misreported as counterfactual signal. |
| **Reliable / low-confidence / missing strata** | Determine where effects occur and whether the central regime is actually addressed. Regimes must be defined without SGNify test-GT tuning. |
| **Centered LHand, centered RHand, retained-hand UBody(-F)** | Link changed local rotations only to official metric-preserved regions, respecting class-`0` masks. |
| **Eligibility and metric mass** | Report tested, accepted, refused, mapping-invalid, identity-invalid, boundary, and missing counts as vertex-frame mass—not just frame or sign percentages. |
| **Reliable-frame regression** | Falsify the surrogate-safety claim by checking whether accepted reliable frames worsen official centered geometry. |
| **Right-valid subset versus two-side-valid subset** | Prevent a conditional/invalid $\mathcal C_L$ from being hidden inside an aggregate and test whether the two-side claim is operational. |
| **Final-parameter regeneration equality** | Verify that the reported mesh is generated solely from the final unified SMPL-X parameter vector, with no MANO scatter or post-mesh correction. |

Attribution logic:

- full objective repeat → safe-only isolates **anchor removal/truncated objective**;
- safe-only → factual-only isolates **factual HandFlow support**;
- factual-only and masked-only → dual no-refusal isolates **paired support**;
- dual no-refusal → full CLRA isolates **baseline-dominance/refusal**;
- same-noise → independent-noise isolates **random-number coupling**;
- $m=0$ equality falsifies any claim that the paired mechanism adds missing-case information.

These controls could distinguish causes, but they cannot repair the analytical missing-case collapse or manufacture novelty. Consequently they do not change the `REJECT` verdict.

---

## 17. Minimal mandatory revisions

**Not applicable.** The Step 5 contract permits revisions only under `MAJOR REVISION`. The selected verdict is `REJECT` because preserving a meaningful missing-observation mechanism and a defensible scientific contribution would require changing the central frozen mechanism. No replacement method or revision is proposed here.

---

## 18. Final frozen/rejected status

| Decision item | Final status |
|---|---|
| Core factual/target-masked mechanism | **Rejected**: exactly degenerates at $m=0$. |
| Counterfactual/causal interpretation | **Rejected** beyond narrow model-condition ablation wording. |
| State-selective uncertainty claim | **Rejected**: $m$ is not local-finger uncertainty. |
| Dual-support mathematics | **Conditionally meaningful** only for $m>0$, validated $\mathcal C_s$, positive latent weight, deterministic solver/tolerance; not Pareto-certified. |
| Safety claim | **Rejected** for 3D and TR-V2V; only truncated-surrogate non-increase is expressible. |
| Missing-observation contribution | **Rejected**; remaining infill is ordinary HandFlow behavior. |
| SMPL-X final representation | **Conceptually admissible**, operationally blocked by unresolved side mappings. |
| Physical-side continuity | **Unsupported** as an actual pipeline. |
| Metric relevance | **Potential path verified**, benefit/non-worsening unverified. |
| Novelty | **`WEAK / ENGINEERING`**; no exact one-paper core collision, but no surviving strong scientific delta. |
| Step 6 admission | **Denied for frozen CLRA-Dex.** No final research/evaluation specification should be built for this rejected candidate. |

**Final verdict: `REJECT`.** This verdict is driven by an analytical failure of the central paired mechanism in the missing regime, not by absent empirical gains, evaluator quirks, implementation inconvenience, or unavailable test GT.

---

## 19. Primary-source manifest

### 19.1 Frozen Step artifacts

| Artifact | Scope inspected | SHA-256 |
|---|---|---|
| `DexAvatar_Baseline_and_TR-V2V_Evaluation_Dossier.md` | Full, 804 lines | `715e36be9bf892386f78fa2833c981b4e6485b74cca4a07afa685e8d179d44b5` |
| `DexAvatar_Step2_Bottleneck_Prioritization_and_Targeted_Literature_Review.md` | Full, 442 lines | `f80d840fd5b4d1595bdbd9e67fffac795571bb8e82a74119aefced8bc6f58a9f` |
| `DexAvatar_Step3_Feasibility_Compatibility_and_Novelty_Gate.md` | Full, 328 lines | `f578280f4c9668a9d6b2ac30b985d40e849aecdb9e3fcbcf1c3343a966fdae84` |
| `DexAvatar_Step4_Hypothesis_to_Method_Formulation.md` | Full, 610 lines | `a50960bb8bf22e1d2eb6ec905df6845070734676a3f5dc60474ace2f00d032bb` |
| Step 5 specification `Đã dán markdown (1)(3).md` | Full, 493 lines | `b2f1904fa455b9aec0d84a705c9e7d6277c0a3fe555ef5c763f311dbd20c2ebc` |

### 19.2 DexAvatar, paper, and evaluator

- Saandeep Kundu et al., [*DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors*](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html), WACV 2026 / arXiv 2025. Attached PDF, pp. 1–21, including all supplementary; main §§1–5, Eqs. 1–12, Figs. 1–5, Tables 1–3; supplementary §§A–H, Figs. S1–S9, Tables S1–S5. Visual inspection included PDF pp. 15–17 (Figs. S7–S9). PDF SHA-256 `49d44f0d03c8d5a98e23594c2f43d8c4f9e7c07eeb701d99bac341e929077ed3`.
- [Official DexAvatar repository](https://github.com/kaustesseract/DexAvatar), branch `main`, commit `a0dfd427f60f5811aadb35c8657b3856d47f56b5`, accessed 2026-08-26. Step 1 full manifest governs all baseline files. Re-inspected for Step 5: `dexavatar_fitting/smplifyx/fitting.py:515–664`; `fit_single_frame.py:223–243,476–503,611–660`; `main.py:85–103,144–151`; `data_parser.py:397–425`; `cfg_files/fit_smplx_vposer_x.yaml:19,60–99`.
- Attached `evaluate_new_fitting(2).py`, static inspection: `transl_point_error` lines 159–169; topology/NaN lines 356–370; class masks and regional scoring lines 380–411; frame summaries lines 417–449; global concatenate/mean lines 455–461. SHA-256 `2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300`.

### 19.3 Mandatory prior works and official repositories

1. Mingxi Xu, Bowen Duan, Yi Gu, Zhengyang Shen, Renjing Xu, Yutao Yue. [*HandFlow: Fully Generative 4D Hand Recovery with Flow Matching*](https://arxiv.org/html/2607.11221v1), 2026. Main §§3–4, Appendix B/D. [Repository](https://github.com/mxxu00/HandFlow), `main`, commit `67fa7df536db233408fe6270ca5d2de28d5959c3`; `README.md` full; `condition_builder.py:20–67`; `online_hamer.py:190–314`; `inference_utils.py:195–325`; `scripts/demo.py:125–220`; `mano_utils.py:14–62`; configs. Accessed 2026-08-26.
2. Huajian Zeng, Chaohua Yao, Yuantai Zhang, Jiaqi Yang, Rolandos A. Potamias, Xingxing Zuo. [*StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video*](https://arxiv.org/html/2605.18553v1), 2026 preprint, §§3–4. [Official repository](https://github.com/huajian-zeng/stablehand): README/assets placeholder and “code after acceptance”; exact commit SHA **`UNRESOLVED`** from the inspected public page. No code/checkpoint used.
3. Enes Duran, Muhammed Kocabas, Vasileios Choutas, Zicong Fan, Michael J. Black. [*HMP: Hand Motion Priors for Pose and Shape Estimation from Video*](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html), WACV 2024. [Repository](https://github.com/enesduran/HMP), `main`, commit `35d799f76b2b2bc1d1e945117b021014b099e7e6`; README/license and `src/datasets/amass.py:96–164` inspected.
4. Zhengdi Yu, Stefanos Zafeiriou, Tolga Birdal. [*Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera*](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html), CVPR 2025. [Repository](https://github.com/ZhengdiYu/Dyn-HaMR), `main`, commit `fa9cd7412c205fd15ee4139c8caacf79bf6167e6`; README, config, and dataset side metadata inspected. Missing tracker submodule **NOT INSPECTED**.
5. Yufei Ye, Yao Feng, Omid Taheri, Haiwen Feng, Shubham Tulsiani, Michael J. Black. [*Predicting 4D Hand Trajectory from Monocular Videos*](https://arxiv.org/html/2501.08329v1) (HaPTIC), 3DV 2026. [Repository](https://github.com/JudyYe/haptic), `main`, commit `f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df`; README, `seq2clip.py:97–224`, `demo.py:342–394` inspected.
6. Gyeongsik Moon. [*Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator*](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html), CVPR 2026. [Repository](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE), `main`, commit `f81d35ddd2b74206c40142243eb62b6d64ce0d65`; README/license, `main/model.py:42–126,160–222`, `common/nets/wilor.py:79–124` inspected.

### 19.4 Additional targeted collision sources

1. Yang Shi, Liangsi Lu, Minzhe Guo, Yifeng Xie, Yanhui Chen, Jingchao Wang, Xuhang Chen. [*Diffusion Image Editing via Asynchronous Token Decoding*](https://arxiv.org/html/2608.09322v1), ACM MM 2026, §§3.2–3.4, Eqs. 4–10. Official repository **NOT FOUND** on access date.
2. Jonathan Ho, Tim Salimans. [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598), 2022, method/Algorithms 1–2. Author-official standalone repository **NOT FOUND**; technical claim taken from the paper only.
3. Pedro Sanchez, Sotirios A. Tsaftaris. [*Diffusion Causal Models for Counterfactual Estimation*](https://proceedings.mlr.press/v177/sanchez22a.html), CLeaR/PMLR 177, 2022, §§1–3. [Official code](https://github.com/SANCHES-Pedro/Diff-SCM).
4. Anastasis Stathopoulos, Ligong Han, Dimitris Metaxas. [*Score-Guided Diffusion for 3D Human Recovery*](https://statho.github.io/ScoreHMR/), CVPR 2024, method and video fitting. [Official MIT repository](https://github.com/statho/ScoreHMR), README/inference/training/evaluation manifest and checkpoint links inspected through the public page.
5. Junzhe Lu et al. [*DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior*](https://arxiv.org/html/2508.00599v2), ICCV 2025 Oral, §§2.2–2.5 and appendices. [Official MIT repository](https://github.com/moonbow721/DPoser-X), README, checkpoint and train/demo manifests inspected through the public page.
6. Ozan Sener, Vladlen Koltun. [*Multi-Task Learning as Multi-Objective Optimization*](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html), NeurIPS 2018. [Official MIT code](https://github.com/isl-org/MultiObjectiveOptimization).

### 19.5 Explicitly not inspected or executed

- SGNify test ground-truth meshes/data: **NOT INSPECTED; NOT USED**.
- DexAvatar, HandFlow, SignHPoser, SMPL-X/MANO checkpoints: **NOT EXECUTED**.
- Any optimization, mesh export, metric evaluation, reproduction, runtime, memory, coverage, or numerical repeatability test: **NOT EXECUTED**.
- StableHand implementation/checkpoint: **NOT AVAILABLE / NOT INSPECTED**; this does not reduce its prior-art relevance.
- Exact Dyn-HaMR tracker implementation at the pinned audit state: **NOT INSPECTED**.
- Exact $\mathcal C_R,\mathcal C_L$ convention proof: **NOT AVAILABLE**.
- No score, improvement percentage, or “reproduced” result is asserted in this Step 5 report.

STEP 5 COMPLETE — REJECT; CLRA-DEX DOES NOT SURVIVE ADVERSARIAL SCIENTIFIC REVIEW.

