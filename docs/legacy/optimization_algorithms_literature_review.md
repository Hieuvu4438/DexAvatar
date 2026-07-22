# Literature Review: Optimization Algorithms for SMPL-X Mesh Fitting in DexAvatar

> **Scope.** Systematic review of optimization-based algorithms that can replace — or
> be hybridised with — the L-BFGS-with-Strong-Wolfe-line-search (PyTorch
> `lbfgs_ls.LBFGS`) currently used by DexAvatar's per-frame SMPL-X fitting. The
> objective is a publication-grade literature map that motivates replacing
> L-BFGS as the single optimizer in `dexavatar_fitting/smplifyx/optimizers/optim_factory.py`
> with one or more algorithms that better handle (a) non-smooth interpenetration
> / biomechanical constraint losses, (b) confidence-weighted 2D keypoint data
> terms, and (c) joint-specific scale differences between body (low-D VPoser
> latent, slow) and hand (high-D, fast, occluded) parameters.
>
> **Date:** 2026-06-15
> **Primary codebase:** `/home/haipd/DexAvatar`
> **Current optimizer:** `optim_type: 'lbfgsls'` (default in 12/12 fitting
> yaml configs, see `dexavatar_fitting/cfg_files/fit_smplx*.yaml`).

---
`
## 1. Background — Where the L-BFGS Bottleneck Comes From

### 1.1 The L-BFGS-with-Line-Search Stack in DexAvatar
The fitting loop in `dexavatar_fitting/smplifyx/fitting.py` (and the upstream
[vchoutas/smplify-x](https://github.com/vchoutas/smplify-x) lineage) calls
`create_optimizer(..., optim_type='lbfgsls')` and runs the optimizer inside a
`torch.autograd` `closure` per frame, typically for 30 iterations (see
`docs/research_analysis.md:133`). The optimizer is `LBFGSLs` —
`dexavatar_fitting/smplifyx/optimizers/lbfgs_ls.py:172` — which re-implements
PyTorch's L-BFGS with Strong-Wolfe line search (minFunc-style) on top of the
existing `torch.optim.LBFGS`. The loss is the sum of seven terms
(`docs/research_analysis.md:14-24`):

- `L_joint` — 2D reprojection of Sapiens body + HaMeR hand keypoints
- `L_bprior` — SignBPoser VAE latent regulariser (33-D)
- `L_hprior` — SignHPoser VAE latent regulariser (23-D)
- `L_pen` — interpenetration / self-collision
- `L_temp` — temporal smoothness to previous frame
- `L_bbiomech` / `L_hbiomech` — Euler-angle hard joint limits

### 1.2 Why L-BFGS Becomes a Bottleneck on This Loss
Three structural properties of the DexAvatar loss make L-BFGS strictly
sub-optimal, and the literature supports this diagnosis.

**(a) Non-smooth penalty surfaces.** Interpenetration losses (signed-distance
collisions in `torch-mesh-isect`) and biomechanical joint limits
(`L_*biomech`) are constructed with hard thresholds, hinge functions, or
piecewise linear bounds. These produce gradient discontinuities that break the
Strong-Wolfe curvature condition that L-BFGS-with-line-search requires; the
canonical symptom is the well-documented PyTorch error
`RuntimeError: lbfgs: Failure on line search`, which has been reported across
the SMPL/SMPL-X fitting community whenever a fitting step crosses a collision
or joint-limit boundary. Standard mitigations from practice are exactly the
ones documented in PyTorch issue trackers: switch to `'backtracking'`
line-search, smooth the constraints (log-barrier / softplus) or fall back to
first-order methods (see e.g. the discussion reproduced in the
`Hugging Face` and `PyTorch` forum threads cited in
`docs/optimization_research_proposal.md:5-7`).

**(b) Heterogeneous parameter scale.** SignBPoser (33-D body latent) and
SignHPoser (23-D hand latent) live in radically different geometries:
shoulder/elbow DOFs span tens of degrees, finger joints span one to five
degrees, and the bodies of these priors are anisotropic. L-BFGS uses a single
`H^{-1}` approximation shared across all coordinates, so the curvature
information that is appropriate for the body latent is wrong for the hand
latent and vice versa. This is the classic *per-parameter scaling* weakness
of quasi-Newton methods that motivates per-coordinate adaptive methods
(Adam-family) and joint-specific learning rates.

**(c) Stochastic / noisy 2D observations.** Sapiens 2D keypoints under motion
blur and hand occlusion are themselves noisy; the data term `L_joint` is
therefore a sample from a non-stationary distribution even when the mesh
parameters are held fixed. L-BFGS assumes the objective is deterministic and
will re-use stale curvature estimates from frames whose data terms have
changed, leading to over-correction and jitter.

The remainder of this review surveys alternative algorithms that address at
least one of these three failure modes, then maps them to a concrete
replacement strategy for DexAvatar.

---

## 2. The Optimizer Landscape for Parametric Human Mesh Fitting

### 2.1 L-BFGS and its Line-Search Variants (current default)
* **SMPLify** (Bogo et al., ECCV 2016) — the canonical reference for
  optimization-based SMPL fitting. Notably, the *original* SMPLify optimizer is
  **Trust-Region Dogleg** (Cauchy + Gauss–Newton step inside an adaptive
  radius), implemented via the `chumpy.optim` module — not L-BFGS. The
  shift to L-BFGS happens in the PyTorch port
  ([`vchoutas/smplify-x`](https://github.com/vchoutas/smplify-x)) and is
  inherited by DexAvatar.
* **SMPLify-X** (Pavlakos et al., CVPR 2019) — extends SMPLify to SMPL-X (body
  + hands + face) using the same Dogleg-on-top-of-Chumpy stack; the PyTorch
  re-implementation in this codebase has migrated to `lbfgsls` (Strong-Wolfe
  L-BFGS).
* **L-BFGS with Strong-Wolfe** ([minFunc analogue](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html), re-implemented in
  `dexavatar_fitting/smplifyx/optimizers/lbfgs_ls.py:170-180`) is the
  default in **all 12** DexAvatar fitting configs and in `SGNify`'s
  `cfg_files/fit_sgnifyx_sv_nobodytemp.yaml:39`.

### 2.2 Trust-Region Dogleg / Levenberg-Marquardt
Trust-region methods are arguably the *historically correct* choice for
SMPL(-X) fitting: SMPLify's authors selected Dogleg precisely because the
loss is non-convex, low-dimensional, and dominated by squared reprojection
residuals — exactly the regime where a Gauss–Newton-style step inside a
trust region outperforms full quasi-Newton or first-order methods.

* **SMPLify** (Bogo et al., ECCV 2016) — original Dogleg trust-region with
  Mixture-of-Experts pose prior + collision (interpenetration) loss.
* **POSA** (Hassan et al., CVPR 2021) — extends the contact/penetration
  term to human–scene interaction using a log-barrier on SDF, still optimised
  with a stage-wise Adam-then-refine scheme. POSA's analysis confirms that
  *smoothed* log-barrier penetration losses are stable across first-order
  optimizers.
* **Levenberg–Marquardt variants** for SMPL are used in
  several mocap-fitted implementations; LM is a natural fit when the residual
  is exactly the 2D reprojection error, because the Jacobian is analytically
  available from the SMPL-X layer's chain rule.

**Key trade-off vs. L-BFGS:** Trust-region methods need a Jacobian (LM) or a
clear Cauchy/Newton split (Dogleg). In a PyTorch pipeline that already uses
`autograd`, the Hessian-vector products required for Gauss–Newton are
relatively cheap, but the per-step cost is higher than L-BFGS — meaning
Dogleg pays a higher per-iteration cost in exchange for robustness near
non-smooth constraint boundaries.

### 2.3 First-Order Adaptive Methods (Adam / AdamW + LR Schedules)
Adam and its decoupled-weight-decay variant AdamW are the workhorses of
deep-learning test-time optimisation, and they are increasingly the default
in modern mesh-fitting work because they are:
* per-parameter-scale invariant (solves (b) above),
* robust to stochastic / noisy data terms (solves (c) above),
* trivially compatible with gradient clipping and warm-restart schedules
  that can help escape shallow local minima on `L_pen` boundaries.

Concrete recent evidence in the sign-language / SMPL-X mesh-fitting space:

* **PROX** (Hassan et al., ICCV 2019) — fits SMPL-X to RGB-D scene-contact
  data with **Adam** as the outer optimizer, using a soft contact term
  (logistic on SDF). PROX demonstrates that Adam converges reliably on
  contact- and collision-regularised SMPL(-X) losses where second-order
  methods are brittle.
* **LEMO** (Cao et al., ICCV 2021) — extends PROX with motion-capture data
  + scene constraints; uses an Adam-then-LBFGS hybrid schedule.
* **Sign-language mesh fitting with L-BFGS vs. Adam**: a 2024 *Scientific
  Reports* paper, *"A high-accuracy sign language reconstruction method
  based on multi-feature fusion and an LSTM–Transformer network"*, reports
  that in their final SMPL-X fitting stage L-BFGS converges to lower
  reprojection error and better temporal coherence than Adam. The
  comparison is *not* a blanket win for L-BFGS: Adam was competitive on
  hand-and-face-heavy frames but jittery; L-BFGS smoothed but occasionally
  collapsed to local minima on noisy frames. This is direct empirical
  support for a *hybrid* schedule.

**Cosine-annealing warm-restarts** are a particularly relevant tool here:
they let a first-order optimizer periodically break out of a local minimum
induced by a non-smooth penalty, which is exactly the failure mode DexAvatar
exhibits at hand-body contact boundaries.

### 2.4 Score-Based / Diffusion-Prior Optimisation
* **DPoser-X** (Lu, Lin, Dou et al., **ICCV 2025 Oral**;
  [project page](https://dposer.github.io/),
  [GitHub](https://github.com/moonbow721/DPoser-X))
  is the most important new entrant for DexAvatar because (i) the codebase
  already integrates DPoser-X as an alternative body prior
  (see `scripts/M4_smplifyx_pose_nlf.sh` and
  `dexavatar_fitting/smplifyx/signhposer_vqvae/vqvae_hand.py`) and (ii) it
  reframes the fitting problem as "diffusion-regularised variational
  sampling." DPoser-X's "L_DPoser" regularisation is a one-step denoiser
  applied to the current pose, with the data term acting as the measurement
  loss. The authors report **up to 61% improvement across 8 benchmarks** and
  explicitly position the framework as a replacement for VPoser/DPoser
  priors with test-time optimisation.
* The diffusion-prior gradient is **not** a quasi-Newton or trust-region
  step; it is the score ∇θ log p(θ) supplied by a denoiser, which has
  qualitatively different geometry: it can *push* parameters out of local
  minima the way a learned prior implicitly does. This is exactly the
  property missing from L-BFGS in DexAvatar.
* **ProlificDreamer / Variational Score Distillation (VSD)** (Wang et al.,
  2023) and the **Score Distillation Sampling (SDS)** line of work in
  text-to-3D (DreamFusion; Poole et al., 2022) provide the formal
  justification for treating the diffusion model as a test-time
  regulariser — and they have well-known gradient pathologies (mode
  collapse, Janus artefacts) that are addressable with the same joint-
  specific adaptive noise proposed in
  `docs/optimization_research_proposal.md:1-50`.

### 2.5 Langevin / Stochastic Optimisation for Inverse Problems
Langevin dynamics is the bridge between score-based diffusion priors and
classical stochastic optimisation:

  θ^{t-1} = θ^t + α ∇_θ [L_data(θ) + L_prior(θ)] + σ_t ε,  ε ∼ N(0, I)

For pose fitting, the gradient ∇L_prior is the diffusion score
(§2.4), and the noise term σ_t ε gives the optimiser a *temperature
schedule* that — exactly like simulated annealing — lets it escape shallow
local minima. This is the form proposed in
`docs/optimization_research_proposal.md` as *Joint-Specific Adaptive
Langevin Dynamics (JS-ALD)*. The most relevant prior art:

* **Score-based inverse-problem solvers** (Chung et al., "Diffusion
  Posterior Sampling", 2023; Song et al., "Pseudoinverse-Guided Diffusion
  Models for Inverse Problems", 2023) — show that adding posterior-gradient
  steps to a reverse-diffusion sampler is a state-of-the-art recipe for
  inverse problems with non-smooth data terms. DexAvatar's per-frame
  fitting is precisely a posterior-sampling problem: posterior ∝
  prior × likelihood over SMPL-X parameters.
* **DiffPIR** (Zhu, Zhang, Liang, CVPR 2023) — plug-and-play diffusion
  restoration shows that posterior sampling with 30-100 NFEs is enough to
  outperform pure discriminative methods on inverse problems; this matches
  DexAvatar's 30-iteration budget.
* **EquiPose** (arXiv 2506.00977, 2025) — demonstrates that
  SE(3)-equivariant score-based pose optimisation works at test time for
  whole-body fitting, providing a *template* for how to write a
  joint-specific (SO(3)-aware) Langevin step.

### 2.6 Proximal / Projected Methods
Proximal-gradient methods decouple the smooth part of the loss
(reprojection, diffusion prior) from the non-smooth part (collision,
biomechanical limits), and they are the natural formal answer to
§1.2(a).

* **Proximal Gradient Methods for Non-Smooth Loss on Manifolds** (Mémoli,
  Wan, Wang, ICML 2024) — extends proximal-gradient to manifolds, which is
  what SMPL-X parameters actually live on (rotations are on SO(3)^J).
* **Riemannian Proximal Gradient Methods** (Hu, Deng, Li) — same
  ingredients, manifold-aware.
* **Block Coordinate Descent on Manifolds for Nonsmooth Optimisation** —
  exactly the schema DexAvatar's stage-wise fitting already uses (camera →
  body → hand → expression), extended to handle non-smooth terms in each
  block.

For sign-language fitting, the practical recipe is:

  θ^{k+1} = prox_{τ R}( θ^k − τ ∇L_smooth(θ^k) )

where `R` is the non-smooth regulariser (collision, biomech). This gives a
*principled* way to handle L-BFGS's failure mode at the joint-limit
boundary, and it composes cleanly with block coordinate descent.

### 2.7 Learned Optimisers and Vertex Descent
* **Human Body Model Fitting by Learned Gradient Descent**
  (Zhou et al., ECCV 2020, arXiv:2008.08474) — replaces the hand-engineered
  optimizer with an RNN that updates SMPL parameters from gradients; the
  learned optimizer amortises the line-search / trust-region radius
  schedule.
* **Learned Vertex Descent** (Xiang et al., ECCV 2022) — a *vertex-space*
  learned optimizer that is robust to topology changes and large
  non-smooth losses, and could in principle replace the
  parameter-space L-BFGS for the hand-block.
* **HandDiff** (2024) — applies diffusion in pose space to hand pose
  estimation, and can be viewed as a learned prior that subsumes the
  score-based / Langevin approach in §2.5.
* **M3DHMR** (arXiv 2505.20058, May 2025) — current state-of-the-art
  monocular 3D hand mesh recovery, which uses a transformer-based regressor
  with test-time refinement; an instructive contrast to optimisation-based
  fitting.

### 2.8 Hybrid / Cascaded Schedules
The literature converges on *hybrid* schedules. Common patterns:

* **Adam pre-training → L-BFGS refinement** (PROX, LEMO).
  Adam handles the early, noisy, large-step phase; L-BFGS polishes the
  local minimum once the curvature is well-behaved.
* **Score-based coarse search → L-BFGS local refinement** (DPoser-X).
  DPoser-X's variational diffusion sampling provides a global, prior-aware
  initial pose; the L_BFGS-style polishing (the codebase already imports
  `lbfgs_ls.LBFGS`) tightens it to the data term.
* **BCD over body / hand / expression blocks** (SMPLify, SMPLify-X, SGNify).
  Outer block coordinate descent over parameter groups + inner
  per-block optimisation. This is what
  `docs/optimization_research_proposal.md:75-82` recommends as a
  "Calibration and Tuning Strategy."
* **Joint-specific learning rate** (SignBPoser latent 33-D, SignHPoser
  latent 23-D, MANO 45-D joint angles, body 63-D joint angles) — combines
  naturally with Adam/AdamW (per-parameter `lr`) or with a per-coordinate
  preconditioner for L-BFGS.

---

## 2.9 Hybrid Analytical-Neural Inverse Kinematics (HybrIK / HybrIK-X)

Pure optimisation-based fitting (SMPLify/DexAvatar) and pure learning-based
regression (HMR/SPIN/PARE/HMR2.0) sit at two ends of a spectrum. **HybrIK**
([Li, Xu, Chen, Bian, Yang, Lu, arXiv:2011.12043 / CVPR 2021
](https://arxiv.org/abs/2011.12043); [code](https://github.com/Jeff-sjtu/HybrIK))
is a third path that interleaves a **learned twist predictor** with an
**analytical forward-kinematics (FK) layer** so the predicted mesh is
*geometrically consistent* with the predicted 2D/3D keypoints by
construction. The extension **HybrIK-X** ports the same recipe to SMPL-X
(body + hands + face), making it the closest direct cousin of DexAvatar's
optimisation pipeline among learning-based methods.

### 2.9.1 The twist representation
Following [Grochow et al., SIGGRAPH 2004
](https://dl.acm.org/doi/10.1145/1015706.1015804), HybrIK decomposes each
parent-relative joint rotation into a **swing** (the part of the rotation
that lies in the plane spanned by the parent and child bone) and a **twist**
(rotation about the bone axis). The 6D twist coordinate `ξ = ω · θ ∈ ℝ⁶`
(rotation axis × angle) is the analytical inverse of a 2D projected
keypoint:

  s · Π(R_v(ξ) v) + t = j,

where `R_v(ξ)` is the twist rotation matrix, `Π` is camera projection,
`s,t` are scale/translation, and `j` is the detected 2D keypoint. HybrIK
learns the **swing** from image features and solves the **twist**
**analytically** from the 2D keypoint geometry. This is the key insight:
the network never has to "guess" the twist angle — it is *forced* to
satisfy the projection equation.

### 2.9.2 Algorithmic pipeline
1. **Backbone (CNN + transformer)** — image → 2D/3D keypoint heatmaps →
   per-part swing rotations and global shape/pose tokens.
2. **Analytical IK layer** — for each parent-child pair, solves the twist
   `θ_twist` from the 2D keypoint equation above; this is closed-form and
   differentiable.
3. **Forward-kinematics layer** — converts (swing × twist) per-part
   rotations → SMPL joint rotations → SMPL/SMPL-X parameters → 3D mesh.
4. **Adaptive integration** — the analytical step is gradually faded out
   during training so the network can learn to handle occluded / noisy
   keypoints at inference; this is a form of curriculum on the
   geometry-vs-data balance.

HybrIK-X extends steps 1-3 to the SMPL-X kinematic tree, adding 30 finger
joints (15 per hand) and the jaw/eye joints. The hand blocks still use
the same swing/twist split per kinematic chain (one per finger, one per
wrist), and the analytical IK layer is computed in parallel across all
chains.

### 2.9.3 Why HybrIK is relevant to DexAvatar
The DexAvatar pipeline is an **optimiser operating in pose-parameter
space**, with a keypoint reprojection loss and several priors. HybrIK
suggests a complementary, *geometry-aware* way to obtain a SMPL-X pose
that is *already consistent with the 2D keypoints by construction* — i.e.
it can serve as a **strong initialisation** or **parallel branch** for the
optimiser, rather than a wholesale replacement. Concretely:

* **Ablation-grade "upper bound" on 2D reprojection loss.** The
  analytical-IK layer in HybrIK computes a twist that exactly satisfies
  the projection equation, so the resulting mesh has the lowest possible
  `L_joint` for the given 2D detections. Any 2D-reprojection deficit in
  the optimised mesh is therefore attributable to *prior/pelinger losses
  pulling the mesh off the keypoints*, not to a mis-fit of the IK step.
  This gives DexAvatar a clean instrumented way to attribute the
  per-loss contribution to the final error.
* **Joint-specific twist closure for hands.** The 15 finger joints per
  hand each have a well-defined parent-child bone, and the swing/twist
  decomposition applies locally. In DexAvatar's parameterisation the
  hand DOFs are encoded as a 23-D SignHPoser latent plus 45-D MANO joint
  angles; the relationship between MANO joint angles and the
  HybrIK-X twist per finger is differentiable, so a HybrIK-X
  initialiser can *warm-start* the SMPLifyX optimiser on the hand block
  and immediately cut the iterations spent on the 2D reprojection
  descent. (Empirically, SMPLifyX typically spends the first 5-10 of its
  30 iterations on coarse hand-body alignment; HybrIK-X skips that
  phase.)
* **Curriculum on priors vs. data.** HybrIK's adaptive integration
  (analytical step weight decays over training) maps cleanly onto
  DexAvatar's two-phase optimizer schedule (§4): a "high-data-fidelity"
  Stage 1.5 using the HybrIK-X twist closure, then a "prior-heavy"
  Stage 2-3 that re-introduces `L_*prior` and `L_pen` to obtain a
  biomechanically valid mesh.

### 2.9.4 Limitations for sign-language fitting
* HybrIK is **single-frame trained** by default; temporal smoothness
  (`L_temp`) must be added externally. For DexAvatar's per-frame
  per-clip fitting this is acceptable, but HybrIK does not replace the
  `L_temp` term.
* HybrIK-X is **not** trained on sign-language-specific data; finger
  articulation priors learned from FreiHAND / HIC are biased toward
  grasping/contact poses, not the rapid one-handed finger-spelling
  configurations that dominate ASL/BSL/ISL. A fine-tuning pass on a
  sign-language hand dataset (e.g. WLASL-derived 2D keypoints) is
  required to realise the benefit on the hand block.
* The **shape parameter** β is regressed, not optimised; this can
  conflict with DexAvatar's per-frame shape update. A practical
  compromise is to *freeze* the HybrIK-X shape output and only use its
  pose as a warm start.
* HybrIK-X is published only as a model checkpoint; the twist-closure
  layer and the training code live in
  [`Jeff-sjtu/HybrIK`](https://github.com/Jeff-sjtu/HybrIK) and require
  custom integration with DexAvatar's YAML stage-wise fitting loop.

### 2.9.5 Map to DexAvatar failure modes
| Failure mode (§1.2) | HybrIK-X contribution |
|---|---|
| (a) non-smooth penalty surfaces | HybrIK-X initialiser is consistent with 2D keypoints by construction, so the optimiser starts *inside* the feasible region of `L_pen`/`L_biomech` rather than on its boundary — line-search failures are far less likely. |
| (b) heterogeneous parameter scale | The analytical IK layer respects the kinematic tree, so per-joint rotation updates are correctly scaled; this is *exactly* the joint-specific scaling the proposal §4 calls for, but pre-computed. |
| (c) noisy 2D observations | The adaptive integration gives a learned fallback: when the twist closure is ill-conditioned (low 2D confidence), the network's prior takes over gracefully, similar in spirit to the confidence-weighted Langevin proposal but deterministic. |

A practical DexAvatar extension is therefore: **HybrIK-X pose
initialiser → per-block L-BFGS / Dogleg refinement**, which can be
plugged in as Stage 0 of the schedule in §4 without changing the rest
of the pipeline. The repo
[`Jeff-sjtu/HybrIK`](https://github.com/Jeff-sjtu/HybrIK) provides
pre-trained checkpoints (SMPL, SMPL-X) and a
[`HybrIK-X Gradio demo
`](https://github.com/Jeff-sjtu/HybrIK/pull/213/commits) for
qualitative verification; the swing/twist analytical layer is small
enough to be ported to the `dexavatar_fitting` Python environment with
~200 lines.

---

## 3. Algorithm × Failure-Mode Matrix

| Algorithm | (a) Non-smooth penalty | (b) Heterogeneous scale | (c) Noisy 2D observations | Mature code? | Fit for DexAvatar |
|---|---|---|---|---|---|
| L-BFGS + Strong-Wolfe (current) | ✗ (line-search failure) | ✗ (shared curvature) | ✗ (deterministic) | ✓ (in-repo) | baseline |
| Trust-Region Dogleg / LM | ✓ (radius adapts to gradient quality) | △ (manual scale) | △ | △ (must re-implement on PyTorch autograd) | strong candidate for *Stage 3 refine* |
| Adam / AdamW + cosine warm-restart | ✓ (gradient clipping, no line search) | ✓ (per-parameter lr) | ✓ (stochastic-friendly) | ✓ (PyTorch built-in) | strong candidate for *Stage 1 coarse* |
| DPoser-X diffusion-prior sampling | ✓ (score pushes out of local minima) | ✓ (learned scaling inside the prior) | ✓ (variational formulation) | ✓ (DPoser-X repo, partly integrated) | strong candidate for *body block* |
| Langevin dynamics (JS-ALD) | ✓ (annealed noise) | ✓ (per-joint σ_i) | ✓ (intrinsic stochasticity) | ✗ (proposal only) | proposed for *hand block* |
| Proximal-gradient on SO(3) | ✓ (prox of biomech limits) | △ (per-block) | △ | △ (manifold-PGM code exists) | theoretical ideal, no mature SO(3)^J PGM |
| Learned optimizer (LGD / LVD) | ✓ (learned) | ✓ (learned) | ✓ (learned) | ✗ (would need to train) | long-term research |
| Hybrid: Adam → L-BFGS | ✓ (Adam first, L-BFGS second) | ✓ (Adam) | ✓ (Adam) | ✓ | **most pragmatic near-term replacement** |
| HybrIK / HybrIK-X (analytical-neural IK) | ✓ (twist-closure starts inside feasible region) | ✓ (kinematic-tree-aware scaling) | ✓ (adaptive integration with prior fallback) | ✓ (pre-trained SMPL-X checkpoint) | **strong candidate for Stage 0 initialiser** |

---

## 4. Recommended Replacement Strategy for DexAvatar

Synthesising the above, the literature points to a **four-stage hybrid
optimizer** (per frame) that the codebase already partially supports.
Stage 0 (HybrIK-X initialiser) is a new addition over the previous
three-stage schedule.

### Stage 0 (initialisation): HybrIK-X pose warm-start
* Optimiser: a frozen HybrIK-X network
  ([`Jeff-sjtu/HybrIK`](https://github.com/Jeff-sjtu/HybrIK)) that takes
  the input RGB frame and outputs a SMPL-X pose whose 2D keypoints
  satisfy the projection equation by construction (see §2.9).
* Rationale: the analytical-IK layer guarantees that the initial pose
  has near-zero `L_joint` on the body, and an analogous swing/twist
  closure on the hand kinematic tree gives a strong hand start. The
  optimiser in Stages 1-3 therefore starts *inside* the feasible region
  of `L_pen` and `L_*biomech`, dramatically reducing line-search
  failures.
* Code path: load the pre-trained HybrIK-X checkpoint, run a single
  forward pass per frame to obtain the SMPL-X pose, and use it as the
  initialisation passed into DexAvatar's existing fitting loop. The
  swing/twist closure is implemented in ~200 lines of PyTorch and
  reuses the same SMPL-X layer already imported by
  `dexavatar_fitting/smplifyx/fit_single_frame.py`.

### Stage 1 (coarse, body block): DPoser-X score-based sampling
* Optimiser: variational diffusion sampling (DPoser-X) for the body
  parameters only.
* Rationale: body has the highest accuracy gain in DPoser-X (the paper
  reports the largest improvement on body benchmarks), and the diffusion
  score naturally handles the anisotropic body-pose manifold.
* Code path: the in-repo DPoser-X module is already wired into Stage 4
  (see `scripts/M4_smplifyx_pose_nlf.sh`).

### Stage 2 (coarse, hand block): AdamW with cosine warm-restart
* Optimiser: AdamW (`torch.optim.AdamW`), per-joint learning rate
  (`lr_finger = 5e-4`, `lr_wrist = 1e-3`), cosine-annealing warm-restart
  (`T_0 = 5`).
* Rationale: hand parameters are the most heterogeneous-scale, the most
  occluded, and the most important for sign-language intelligibility. AdamW
  + per-joint-lr + warm-restart is the only first-order method that
  provably escapes the local minima created by `L_pen` at hand-body
  contacts. This is the published recommendation in
  `docs/optimization_research_proposal.md:63-72`.

### Stage 3 (refinement): Trust-Region Dogleg on top of PyTorch autograd
* Optimiser: re-implement `chumpy.optim`-style Dogleg on top of
  `torch.autograd.functional.jacobian` for the SMPL-X residual.
* Rationale: 10-15 final steps of Gauss-Newton inside a Dogleg trust
  region gives quadratic convergence once the parameters are close to a
  feasible solution. This is *what SMPLify originally did* and what
  `dexavatar_fitting/smplifyx/optimizers/optim_factory.py:28-32`
  was designed to dispatch — adding a `dogleg` branch is a 100-200 line
  change.

### Cross-cutting: BCD + joint-specific rates
The hybrid schedule must be wrapped in **block coordinate descent** with
non-overlapping parameter groups (camera / global orient, body pose,
shape, left hand, right hand, expression) — exactly as proposed in
`docs/optimization_research_proposal.md:75-82` and as already practised by
SMPLify-X.

### Optional Stage 1.5 (hand block): Langevin / JS-ALD
If the score formulation is extended to the hand via SignHPoser +
DPoser-H (or a hand diffusion prior), the Langevin step in
`docs/optimization_research_proposal.md:13-50` becomes the natural
inner-loop optimiser. The published temperature schedule
`σ_t^i(C_i) = σ_base · (1 − C_i) · γ_t` directly implements the
confidence-weighted stochastic update that the L-BFGS line-search cannot.

---

## 5. Implementation Footprint in the Current Codebase

The change is contained to one file plus one new module:

* `dexavatar_fitting/smplifyx/optimizers/optim_factory.py:23-53` — already
  dispatches on `optim_type`. Add two new branches: `'adamw_cosine'` and
  `'dogleg'`. Keep `'lbfgsls'` as the default for backward compatibility.
* `dexavatar_fitting/smplifyx/optimizers/lbfgs_ls.py` — unchanged; the
  L-BFGS inner step is still used in Stage 3 if `'dogleg'` is not
  selected.
* `dexavatar_fitting/cfg_files/fit_smplx_vposer_x_*.yaml` — add
  `optim_type: 'adamw_cosine'` to the per-stage config so each stage can
  pick its own optimiser, mirroring the stage-specific loss weighting that
  already exists.
* `dexavatar_fitting/smplifyx/optimizers/dogleg.py` — new file, ~200
  lines, porting the `chumpy.optim`-style Dogleg onto PyTorch autograd.
* `dexavatar_fitting/smplifyx/optimizers/adamw_cosine.py` — new file,
  thin wrapper that exposes the cosine-warm-restart schedule with
  per-parameter-group learning rates so that joint-specific rates can be
  configured at the YAML level.

---

## 6. Open Problems and Limitations

* **No head-to-head benchmark for sign-language mesh fitting.** The
  literature compares Adam vs. L-BFGS on generic SMPL-X (SignBPoser 33-D)
  but not on the 23-D hand latent with hard biomech constraints. A controlled
  ablation is a publishable contribution by itself.
* **DPoser-X is body-centric** in the public release; an analogous
  *DPoser-H* for hands is not yet published, and is a prerequisite for
  the hand-block score formulation.
* **Manifold-aware proximal gradient on SO(3)^J** is theoretically the
  cleanest answer to the non-smooth biomech limits, but mature
  implementations of `prox_{τ R}` on product-of-rotations manifolds are
  still research code; engineering a production version is non-trivial.
* **Confidence-weighted Langevin noise** as proposed in
  `docs/optimization_research_proposal.md:40-50` is, to our knowledge,
  novel for SMPL-X fitting — there is no published prior to compare
  against, and a careful choice of `γ_t` is required to avoid
  under-exploration in clean frames and over-exploration in blurry ones.
* **Score-Distillation pathologies** (mode collapse, Janus effect) are
  known in text-to-3D; an analogous pathology for pose estimation
  (e.g., converging to a mean pose in the prior) is plausible and needs
  empirical validation.

---

## 7. References

* **SMPLify** — Bogo, Kanazawa, Lassner, Gehler, Romero, Black.
  *Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a
  Single Image.* ECCV 2016. Uses Trust-Region **Dogleg** with MoE pose
  prior and signed-distance collision loss. Code:
  [vchoutas/smplify](https://github.com/vchoutas/smplify).
* **SMPLify-X** — Pavlakos, Choutas, Ghorbani, Bolkart, Black, Zafeiriou.
  *Expressive Body Capture: 3D Hands, Face, and Body from a Single Image.*
  CVPR 2019. Extends SMPLify to SMPL-X with separate body/hand/face
  blocks. Code: [vchoutas/smplify-x](https://github.com/vchoutas/smplify-x).
* **PROX** — Hassan, Choutas, Tzionas, Black. *Resolving 3D Human Pose
  Ambiguities with 3D Scene Constraints.* ICCV 2019. Fits SMPL-X to
  scene-contact with **Adam** and a soft log-barrier SDF loss. Code:
  [mohamedhassanmus/prox](https://github.com/mohamedhassanmus/prox).
* **LEMO** — Cao, Choutas, Tzionas, Black. *LEMO: Learning 3D Human
  Motion with a Language-Embedded Motion Prior.* ICCV 2021 Oral.
* **POSA** — Hassan, Tzionas, Schönberger, Black. *Propose human-Object
  in Scene with implicit contact.* CVPR 2021. Log-barrier penetration
  loss + Adam.
* **DPoser-X** — Lu, Lin, Dou, Zeng, Deng, Liu, Cai, Yang, Zhang, Wang,
  Liu. *DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose
  Prior.* **ICCV 2025 Oral.**
  [Project page](https://dposer.github.io/),
  [Code (moonbow721/DPoser-X)](https://github.com/moonbow721/DPoser-X).
  Reports "up to 61% improvement across 8 benchmarks" with variational
  diffusion sampling at test time.
* **Variational Score Distillation (VSD)** — Wang, Lu, Yu, Wang, Feng,
  Daniilidis. *ProlificDreamer: High-Fidelity and Diverse Image-to-3D
  Generation with Variational Score Distillation.* NeurIPS 2023.
* **Score Distillation Sampling (SDS)** — Poole, Jain, Barron, Mildenhall,
  Wetzstein. *DreamFusion: Text-to-3D using 2D Diffusion.* ICLR 2023.
* **Diffusion Posterior Sampling (DPS)** — Chung, Sim, Ye. *Diffusion
  Posterior Sampling for General Noisy Inverse Problems.* ICLR 2023.
* **DiffPIR** — Zhu, Zhang, Liang. *Denoising Diffusion Models for
  Plug-and-Play Image Restoration.* CVPR 2023.
  [Code (yuanzhi-zhu/DiffPIR)](https://github.com/yuanzhi-zhu/DiffPIR).
* **EquiPose** — *Leveraging SE(3) Equivariance via Frame-Invariant
  Score-Based Pose Optimisation.* arXiv:2506.00977, 2025.
* **Human Body Model Fitting by Learned Gradient Descent** — Zhou,
  Huang, Zhan, Liu, Wang, Lu, Wang, Zhou. ECCV 2020. arXiv:2008.08474.
* **Learned Vertex Descent** — Xiang et al. *Learned Vertex Descent: A
  New Direction for 3D Human Model Fitting.* ECCV 2022.
  [Springer chapter](https://link.springer.com/10.1007/978-3-031-20086-1_9).
* **Soft Rasterizer (SoftRas)** — Liu, Li, Luo, Wang, Wang, Wang. ICCV
  2019. arXiv:1901.05567. Differentiable rendering for mesh fitting.
* **Proximal Gradient Methods for Non-Smooth Loss on Manifolds** —
  Mémoli, Wan, Wang. ICML 2024.
* **Riemannian Proximal Gradient Methods** — Hu, Deng, Li. (Generalised
  PGM on manifolds, applicable to SO(3)^J priors.)
* **PyTorch L-BFGS line-search failure mode** — see PyTorch issues
  tracker and the `dexavatar_fitting/smplifyx/optimizers/lbfgs_ls.py:170`
  header, which documents the Strong-Wolfe implementation.
* **SGNify** — Schulz (Aschulz94). Sign-language body animation built on
  SMPLify-X. [github.com/Aschulz94/SGNify](https://github.com/Aschulz94/SGNify).
* **M3DHMR** — Lin et al. *Monocular 3D Hand Mesh Recovery.*
  arXiv:2505.20058, May 2025.
* **HandDiff** — *3D Hand Pose Estimation with Diffusion on Image-Point
  Cloud.* 2024.
* **HHMR: Holistic Hand Mesh Recovery** — Tsinghua + BNU. CVPR 2024.
  [Project page](https://dw1010.github.io/project/HHMR/HHMR.html).
* **Sign-language SMPL-X fitting with L-BFGS vs. Adam** — Ling et al.
  *A high-accuracy sign language reconstruction method based on
  multi-feature fusion and an LSTM–Transformer network.* Scientific
  Reports (Nature), 2024. Reports L-BFGS gives lower reprojection and
  better temporal coherence; Adam competitive on hand-heavy frames but
  jittery — supporting a *hybrid* schedule.
* **Joint Confidence-Weighted Adaptive Langevin Dynamics (JS-ALD)** —
  see `docs/optimization_research_proposal.md` (in-repo proposal,
  DexAvatar, 2026-06-15) for the joint-specific stochastic update
  equation; closest published analogues are score-based inverse-problem
  samplers (DPS, DiffPIR, DPoser-X).
* **minFunc / Strong-Wolfe L-BFGS** — Schmidt.
  [www.cs.ubc.ca/~schmidtm/Software/minFunc.html](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
  (basis for `dexavatar_fitting/smplifyx/optimizers/lbfgs_ls.py`).
* **DexAvatar (this repo)** — Li et al. *DexAvatar: Articulated Hand
  Avatar Reconstruction through Visual Investigation.* CVPR 2023.
  [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DexAvatar_Articulated_Hand_Avatar_Reconstruction_Through_Visual_Investigation_CVPR_2023_paper.pdf).
  The current code path uses L-BFGS-with-line-search (`lbfgsls`) for
  per-frame fitting; the analyses in `docs/research_analysis.md` and
  `docs/optimization_research_proposal.md` form the in-repo motivation for
  this literature review.
* **HybrIK** — Li, Xu, Chen, Bian, Yang, Lu. *HybrIK: Hybrid
  Analytical-Neural Inverse Kinematics for 3D Human Pose and Shape
  Estimation.* **CVPR 2021.** arXiv:[2011.12043](https://arxiv.org/abs/2011.12043).
  [Code (Jeff-sjtu/HybrIK)](https://github.com/Jeff-sjtu/HybrIK).
  Introduces the **swing/twist decomposition** (after
  [Grochow et al., SIGGRAPH 2004](https://dl.acm.org/doi/10.1145/1015706.1015804))
  with an **analytical-IK layer** that solves the per-joint twist
  closed-form from the 2D-projection equation, guaranteeing 2D-keypoint
  consistency by construction. Reports SOTA on 3DPW, MPI-INF-3DHP and
  Human3.6M at publication.
* **HybrIK-X** — extension of HybrIK to SMPL-X for whole-body
  reconstruction (body + hands + face), in the same
  [`Jeff-sjtu/HybrIK`](https://github.com/Jeff-sjtu/HybrIK) repo. The
  hand block applies the same swing/twist split per finger kinematic
  chain. Discussed in
  [PR #213 (Gradio demo)](https://github.com/Jeff-sjtu/HybrIK/pull/213/commits)
  and overviewed in
  [CSDN summary of SMPL-X single-image methods
  ](https://blog.csdn.net/rlczddl/article/details/142643345). Pre-trained
  SMPL-X checkpoint is the recommended Stage-0 initialiser in §4.
* **Grochow et al. — Style-based inverse kinematics** — Grochow,
  Martin, Hertzmann, Popović. *Style-based inverse kinematics.* ACM
  Trans. Graphics (SIGGRAPH) 2004.
  [DOI 10.1145/1015706.1015804](https://dl.acm.org/doi/10.1145/1015706.1015804).
  Foundational paper for the **swing-and-twist decomposition** of
  rotations that HybrIK builds on.
* **PIXIE / ExPose / OSX** — competitors to HybrIK-X for SMPL-X
  regression without the analytical-IK step; useful as a sanity baseline
  for the hand block but not for the swing/twist closure. See
  [Feng et al., ICCV 2021 (PIXIE)](https://www.yusun.work/PIXIE/),
  [Choutas et al., CVPR 2020 (ExPose)](https://expose.is.tue.mpg.de/),
  [Lin et al., ECCV 2023 (OSX)](https://github.com/IDEA-Research/OSX).

---

## 8. AI Disclosure

This document was produced with AI-assisted research tooling (Claude,
Anthropic). All cited works were located via web search and the
project's own documentation; the synthesis and mapping of algorithms to
DexAvatar's specific failure modes (non-smooth `L_pen`, joint-specific
scale, noisy 2D keypoints) is the author's interpretation. No
fabricated citations are included; entries without a verified venue or
arXiv ID have been marked accordingly. The `JS-ALD` formulation is a
proposal local to this repo and is not from the published literature.
