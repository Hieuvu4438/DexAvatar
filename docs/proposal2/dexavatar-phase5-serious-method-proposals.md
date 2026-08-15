# Phase 5 — Five Serious Method Proposals

**Target:** monocular RGB video → accurate SMPL-X reconstruction of sign language  
**Primary evaluation:** SGNify TR-V2V — upper body excluding face, left hand, right hand  
**Evidence cut-off:** 10 August 2026  
**Status:** research proposals, not performance claims. No millimetre gain below is asserted without an experiment.

---

## Executive decision

Five technically distinct methods survive the Phase-4 hypotheses and the subsequent closest-work check:

1. **BiVis-Sign** — visibility-conditioned bidirectional sequence inference;
2. **CalibSign-K** — calibrated component-wise multi-hypothesis reconstruction;
3. **OrderHands** — discrete-continuous temporal hand–hand interaction reasoning;
4. **SignAnchor** — temporally indexed hand-part ↔ body-region anchoring;
5. **RCP-Sign** — reliability-certified pseudo-GT correction coupled to hybrid inference.

The recommended portfolio is:

| Role | Candidate | Why |
|---|---|---|
| **PRIMARY** | **BiVis-Sign** | Best balance of direct DexAvatar failure, likely hand TR-V2V impact, available training data, clean falsification, and manageable implementation. |
| **BACKUP** | **OrderHands** | More sign-specific and technically distinctive; pursue if an interaction-prevalence audit shows enough ordered overlap/contact frames and annotations can be obtained. |
| **LOW-RISK** | **RCP-Sign** | DexAvatar's own filtering ablation already proves label quality affects upper-body error. The scientific path is clear, although novelty is less secure. |
| **HIGH-RISK / HIGH-REWARD** | **SignAnchor** | Anatomical contact can resolve otherwise unobservable depth and orientation, but nearby work is strong, new labels are needed, and aggregate TR-V2V may underweight the affected frames. |

**CalibSign-K** remains a valuable probabilistic alternative. It should be promoted only if a cheap pilot shows both a substantial oracle best-of-​K gap and a selector that closes most of it. Generating diverse samples without automatic selection is not a useful reconstruction contribution.

---

## 1. Evidence and claim boundaries

### 1.1 Facts inherited from the DexAvatar deconstruction

- **FACT:** DexAvatar uses Sapiens body/face observations, HaMeR hand observations, a SMPLer-X initialization, frozen SignBPoser/SignHPoser VAE decoders, and per-sequence latent optimization.
- **FACT:** its released temporal regularizer is first-order and body-only; it has no learned sequence-level hand prior.
- **FACT:** the released fitting objective does not explicitly represent observation uncertainty, visibility, ordered hand–hand interaction, intended contact, or sign semantics.
- **FACT:** camera, shape, global orientation, and several other initialization variables are fixed during the principal released optimization.
- **FACT:** the 3D HaMeR depth term has zero weight in the released objective, while initialization matching is strong.
- **FACT:** filtering the SignBPoser pseudo-ground truth improves the reported upper-body-excluding-face TR-V2V from 34.06 to 30.28 mm. That 3.78 mm result is evidence that label quality matters; it is not a forecast for RCP-Sign.
- **FACT:** the primary SGNify evaluation has 57 isolated DGS signs, one signer, and 2,872 central frames. It therefore provides weak evidence about cross-signer generalization and may underrepresent long occlusion.

### 1.2 Closest-work constraints added after Phase 4

- [StableHand](https://arxiv.org/abs/2605.18553), [HandFlow](https://arxiv.org/abs/2607.11221), [MoRo](https://arxiv.org/abs/2601.16079), [DanceHMR](https://arxiv.org/abs/2605.18102), [RoHM](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RoHM_Robust_Human_Motion_Reconstruction_via_Diffusion_CVPR_2024_paper.html), and [Dyn-HaMR](https://arxiv.org/abs/2412.12861) mean that temporal completion, generative hand recovery, confidence-aware masking, and bidirectional context are not novel on their own.
- [ProHMR](https://arxiv.org/abs/2108.11944), [POCO](https://arxiv.org/abs/2308.12965), [MaskHand](https://arxiv.org/abs/2412.13393), uncertainty-aware probabilistic HMR at [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Wehrbein_Utilizing_Uncertainty_in_2D_Pose_Detectors_for_Probabilistic_3D_Human_WACV_2025_paper.html), and [VLM-guided group preference alignment](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_VLM-Guided_Group_Preference_Alignment_for_Diffusion-based_Human_Mesh_Recovery_CVPR_2026_paper.html) mean that uncertainty, sampling, and hypothesis ranking are established adjacent ideas.
- [TUCH](https://arxiv.org/abs/2104.03176), [Goliath-SC](https://arxiv.org/abs/2509.23393), and monocular [head-avatar capture with hand contacts](https://arxiv.org/abs/2510.17181) mean that self-contact, shape-conditioned contact priors, and hand–face contact/depth-order losses are not novel in isolation.
- [BioTUCH](https://arxiv.org/abs/2512.04862) directly narrows both contact and pseudo-GT claims: it uses sensed self-contact timing to refine human-pose pseudo-ground truth and reports an 11.7% improvement in its own setting. RCP-Sign therefore cannot claim “contact-aware pseudo-GT correction” as a new generic idea.
- No verified primary source in the completed direct search was found to implement the exact target combinations: visibility-conditioned bidirectional **full SMPL-X sign reconstruction**; component-calibrated selected-K **sign reconstruction**; a temporal ordered bimanual state for sign SMPL-X; or a sign-specific temporal articulator-to-anatomical-region anchor. This is an absence-of-evidence statement, not proof of novelty.

### 1.3 Common notation

For frame (t\in\{1,\ldots,T\}):

- (x_t=(\theta_t,\beta,c)) denotes SMPL-X pose, clip-level shape, and camera;
- (V_t=M(x_t)\in\mathbb{R}^{10475\times3}) is the SMPL-X mesh;
- (J_t=R_J V_t) are model joints;
- (y_{t,j}\) is a 2D observation for joint/landmark (j\);
- (q_{t,c}\in[0,1]\) is reliability for component (c\), where components include torso/arms, left/right wrist, and left/right fingers;
- (\Pi_c\) is projection under camera (c\);
- (d_R\) is a geodesic rotation distance, not Euclidean distance between axis-angle vectors;
- (\rho\) is a robust penalty.

All five methods should be evaluated with one frozen evaluator and manifest. None should be allowed to drop difficult frames that a baseline retains.

---

# Proposal 1 — BiVis-Sign

## Visibility-Conditioned Bidirectional Sign Reconstruction

### 1. Method name

**BiVis-Sign: Visibility-Conditioned Bidirectional Full-SMPL-X Sign Reconstruction.**

### 2. Research hypothesis

**HYPOTHESIS:** when hand or wrist observations disappear or become unreliable, a model that explicitly uses reliable observations both before and after the gap will reconstruct the hidden SMPL-X trajectory more accurately than a framewise fit or an equal-capacity forward-only temporal model, without attenuating fast sign motion.

### 3. DexAvatar limitation addressed

DexAvatar has a body-only first-order temporal term and no learned hand dynamics. It cannot use a hand's later reappearance to correct the hidden interval. Its fixed strong initialization can preserve a detector's occlusion error.

### 4. Core technical insight

Visibility is not merely a loss weight. It determines **where information should flow from**. Reliable future frames can constrain hidden wrist orientation, relative hand depth, finger trajectory, and recovery timing. The proposed contribution is the coupling of:

1. calibrated component visibility/reliability;
2. bidirectional, full-SMPL-X sign-motion inference; and
3. an observation-preserving MAP refinement that avoids converting fast motion into smooth but incorrect motion.

The novelty claim is not “use a Transformer,” “use masking,” or “use bidirectionality.”

### 5. Architecture

1. **Frozen observation front end:** Sapiens body/face observations, a strong hand estimator such as HaMeR/WiLoR, and a full-body SMPL-X initializer. Using a newer detector is a baseline improvement, not the contribution.
2. **Component-quality encoder (Q_\phi\):** consumes detector heatmaps/scores, crop truncation, optical-flow consistency, blur statistics, left/right disagreement, mesh reprojection residuals, and temporal re-detection consistency. It predicts (q_{t,c}\) and calibration temperature.
3. **Bidirectional residual sequence model (S_\psi\):** consumes initialized rotations, joints, image/crop features, and (q_{1:T}\). It predicts a corrected sequence mean (\mu_{1:T}\) and diagonal/block covariance (\Sigma_{1:T}\). It must jointly represent body, wrists, and fingers.
4. **Structured SMPL-X MAP layer:** refines body/hand pose and, only if an oracle study supports it, camera variables. It trusts observations where (q\) is high and the sequence posterior where (q\) is low.

### 6. Input/output

- **Input:** a monocular RGB clip, detector outputs and confidences, initialized SMPL-X sequence, and optional masks/flow computed from the same video.
- **Output:** one deterministic SMPL-X sequence, component uncertainties, and a reliability trace. The primary result is single-output reconstruction, not oracle best-of-​K.

### 7. Mathematical formulation

The sequence model estimates

\[
(\mu_{1:T},\Sigma_{1:T})=S_\psi(x^0_{1:T}, f_{1:T}, q_{1:T}),
\]

where (x^0\) is the framewise initialization and (f\) contains visual/observation features. The final sequence minimizes

\[
\begin{aligned}
E_{\text{BiVis}}(x_{1:T})=
&\sum_{t,j} q_{t,c(j)}\,\rho\!\left(\|\Pi_c(J_{t,j})-y_{t,j}\|_2^2\right) \\
&+\lambda_s\sum_t d_{\Sigma_t}(x_t,\mu_t)^2
+\lambda_v\sum_t w_t^v\|\Delta J_t-\Delta\mu^J_t\|_1 \\
&+\lambda_a\sum_t w_t^a\|\Delta^2 J_t-\Delta^2\mu^J_t\|_1
+\lambda_{\mathrm{ROM}}E_{\mathrm{ROM}}
+\lambda_{\mathrm{pen}}E_{\mathrm{pen}}.
\end{aligned}
\]

(d_{\Sigma}\) is a covariance-weighted pose distance using geodesic rotation residuals. Velocity/acceleration weights are reduced at genuine high-speed transitions and raised inside unsupported gaps; they are not uniform smoothness penalties.

Training uses synthetic and real missingness:

\[
\mathcal L_{\text{seq}}=
\lambda_R\sum d_R(\hat\theta,\theta^*)
+\lambda_V\|\hat V-V^*\|_1
+\lambda_{vel}\|\Delta\hat J-\Delta J^*\|_1
+\lambda_{acc}\|\Delta^2\hat J-\Delta^2J^*\|_1
+\lambda_{nll}\sum_c\left(\frac{e_c^2}{\sigma_c^2}+\log\sigma_c^2\right).
\]

The quality head is calibrated against whether each component's initialization error exceeds a pre-registered threshold, using Brier score/expected calibration error in addition to classification loss.

### 8. Trainable modules

- component-quality encoder (Q_\phi\);
- bidirectional sequence model (S_\psi\);
- optional small residual visual-feature adapters;
- uncertainty calibration temperatures.

### 9. Frozen modules

- 2D/body/hand detectors and the base SMPL-X initializer for the main controlled comparison;
- SMPL-X model and joint regressors;
- DexAvatar's SignBPoser/SignHPoser in the first experiment, so gains can be attributed to temporal evidence rather than a new local prior.

An end-to-end variant may follow only after the causal result is established.

### 10. Loss functions

- rotation-geodesic and vertex losses;
- joint velocity and acceleration **fidelity** losses;
- heteroscedastic component NLL;
- reliability calibration loss;
- masked-span reconstruction loss;
- standard ROM and interpenetration terms;
- optional silhouette/reprojection consistency, gated by confidence.

No recognition or gloss supervision is required.

### 11. Training strategy

1. Train (Q_\phi\) on clean 3D sequences degraded with detector-like noise, crop truncation, blur, hand overlap, and contiguous masks.
2. Train (S_\psi\) with a curriculum: isolated random components, short contiguous gaps, then coupled hand–wrist–arm gaps and motion blur.
3. Include clean visible sequences so the model learns an identity mapping when observations are reliable.
4. Fine-tune on sign pseudo-GT only with reliability weighting; do not expose SGNify test sequences.
5. Freeze the network and tune MAP hyperparameters on a separate validation signer or held-out vocabulary.

### 12. Inference procedure

Run frozen observations and initialization over the full clip; estimate component reliability; perform one bidirectional pass; initialize the MAP layer with (\mu\); optimize a restricted set of body, wrist, and hand variables for a fixed iteration budget; output the final sequence and quality trace. For streaming use, report a separate fixed-lag version rather than conflating it with offline reconstruction.

### 13. Required data

- DexAvatar/SignAvatars/How2Sign pseudo-SMPL-X sequences for sign-motion coverage;
- SignHPoser mocap and/or [3D-LEX](https://arxiv.org/abs/2409.01901) for cleaner hand/phonetic motion where licensing permits;
- synthetic occlusion/blur/degradation generated from clean sequences;
- a small accurate multi-view or mocap validation set for calibration and gap recovery.

### 14. Required annotations

No new linguistic annotations. Required supervision is accurate or pseudo 3D pose/mesh, known synthetic visibility masks, and a modest clean subset with per-component 3D error for calibration. Real visibility labels can be weakly derived from rendered z-buffers when accurate meshes/cameras are available, then manually audited.

### 15. Compute cost

**Moderate.** One sequence model plus frozen front ends; training is materially cheaper than a large diffusion model. Inference adds one bidirectional pass and a short structured optimization. Report wall-clock time and matched-compute baselines.

### 16. Expected TR-V2V effect

**INFERENCE, not a result:** the largest expected effect is on left/right hand TR-V2V during contiguous occlusion and immediately after reappearance. Upper-body-excluding-face may improve through wrist/forearm corrections. Visible, slow frames should remain approximately unchanged. Aggregate gain is conditional on the benchmark's hard-frame prevalence.

### 17. Expected hand improvement

Fewer depth/identity switches; improved hidden finger pose and wrist orientation; lower recovery latency; less frame dropping; reduced post-occlusion discontinuity. The method must not merely reduce acceleration by shrinking true motion.

### 18. Expected body improvement

Small-to-moderate arm/shoulder benefit when hand evidence constrains the kinematic chain. Little torso benefit is expected unless camera/body initialization is also erroneous.

### 19. Expected robustness improvement

High expected robustness to short/medium hand occlusion and detector dropout; moderate robustness to motion blur; uncertain robustness to gaps longer than the training distribution or videos with no reliable observation on either side.

### 20. Main implementation risk

Pseudo-ground-truth temporal bias may teach oversmoothing. Quality estimates may be correlated with the same detector failure rather than independent evidence. Full-sequence memory can also become expensive at long duration.

### 21. Main reviewer objection

“This is StableHand/MoRo-style masked temporal completion applied to signing.” The response must be empirical and narrow: full SMPL-X hand–wrist–arm coupling, sign-specific missingness, future-reappearance causality, matched-capacity baselines, and motion-fidelity metrics—not architectural branding.

### 22. Falsification experiment

Use matched framewise, forward-only, backward-only, and bidirectional models with identical observations, capacity, loss, and compute. Test real and synthetic contiguous masks stratified by length, blur, speed, and overlap. Report TR-V2V, MPJPE/MPVPE, depth order, recovery latency, velocity/acceleration error, temporal lag, and visible-frame regression.

**Reject BiVis-Sign** if bidirectional inference does not significantly improve paired hidden-span geometry over forward-only inference, if gains disappear on real occlusions, or if lower temporal error is explained by attenuated peak velocity/amplitude.

---

# Proposal 2 — CalibSign-K

## Calibrated Component-Wise Multi-Hypothesis SMPL-X Reconstruction

### 1. Method name

**CalibSign-K: Calibrated Component-Wise K-Hypothesis Sign Reconstruction.**

### 2. Research hypothesis

**HYPOTHESIS:** monocular sign frames contain component-local multimodality—especially wrist twist, finger articulation, and relative hand depth—that cannot be represented by one deterministic fit; branching only uncertain components and selecting hypotheses with future/image/interaction evidence will reduce single-output TR-V2V.

### 3. DexAvatar limitation addressed

DexAvatar emits a single prior- and initialization-dominated solution. A wrong hand initialization can be preserved even when the body is correct. Detector score is not calibrated 3D error and does not distinguish wrist, fingers, arms, and camera.

### 4. Core technical insight

Ambiguity is structured, not global. If the left fingers are uncertain but torso, right hand, and left wrist are reliable, the model should branch the left fingers only. This keeps hypotheses coherent, controls combinatorial growth, and makes automatic selection diagnosable.

The scientific contribution must be **calibration plus non-oracle selection**, not sampling itself.

### 5. Architecture

1. Frozen frame/sequence initializer.
2. Component-error calibrator for {body/arms, L/R wrist, L/R fingers, camera}.
3. Conditional structured residual flow or mixture model that samples only low-reliability components while conditioning on reliable components and the whole clip.
4. Evidence selector that scores each complete SMPL-X hypothesis using image reprojection/silhouette evidence, future reappearance, temporal consistency, and interaction plausibility.
5. Optional restricted MAP refinement for the selected mode only.

### 6. Input/output

- **Input:** RGB video, initial SMPL-X sequence, detector features/heatmaps, masks where available, and component reliabilities.
- **Output:** (K\) coherent sequence hypotheses with probabilities, one automatically selected final sequence, and calibrated uncertainty/coverage estimates.

### 7. Mathematical formulation

Let (m_{t,c}=\mathbb 1[q_{t,c}<\tau_c]\) be a soft/hard branching mask. For (k=1,\ldots,K\):

\[
x^{(k)}_{1:T}=x^0_{1:T}\oplus G_\psi(z_k,o_{1:T},q_{1:T},m_{1:T}),\qquad z_k\sim\mathcal N(0,I),
\]

where (\oplus\) composes rotations on (SO(3)\); reliable components are copied or tightly regularized. A learned selector and explicit evidence energy define

\[
p_\omega(k\mid o)\propto
\exp\left[-E_\omega(x^{(k)},o)\right],
\]

\[
E_\omega=
E_{2D}+\lambda_{sil}E_{sil}+\lambda_{re}E_{reappearance}
+\lambda_tE_{temporal}+\lambda_pE_{sign\ prior}
+\lambda_{int}E_{penetration}+E_{rank,\omega}.
\]

Training separates coverage from selection:

\[
\mathcal L=
\underbrace{\min_k d(x^{(k)},x^*)}_{\text{coverage}}
+\lambda_{nll}\mathcal L_{NLL}
+\lambda_{rank}\max(0,m+E_\omega(k^*)-E_\omega(k^-))
+\lambda_{cal}\mathcal L_{cal}
+\lambda_{div}\mathcal L_{div}.
\]

Diversity is applied only in uncertain components and must be paired with image/sequence consistency. The selected result is (\hat k=\arg\min_k E_\omega\), never an oracle (\arg\min_k d(x^{(k)},x^*)\) at test time.

### 8. Trainable modules

- component-wise error calibrator;
- structured conditional distribution (G_\psi\);
- evidence/ranking model (E_{rank,\omega}\);
- optional uncertainty-aware residual refiner.

### 9. Frozen modules

- base detectors and SMPL-X initializer in controlled experiments;
- SMPL-X;
- image encoders may initially be frozen to prevent the selector from becoming an opaque second reconstructor.

### 10. Loss functions

- best-of-​K coverage and conditional likelihood;
- rotation/vertex/joint reconstruction;
- pairwise/listwise selector ranking;
- reliability calibration (Brier, NLL, risk–coverage);
- component-local diversity with global coherence;
- temporal, reprojection, silhouette, ROM, and penetration consistency.

### 11. Training strategy

1. Calibrate component error on a clean held-out 3D set and synthetic perturbations.
2. Train the hypothesis generator with controlled ambiguity: delete or corrupt specific components rather than whole frames.
3. Train the selector on held-out hypotheses using only evidence available at test time.
4. Evaluate selector generalization on unseen signers and degradation types; do not tune selection on SGNify test labels.
5. Start with (K\in\{4,8\}\); scale only if oracle coverage continues to improve and selection remains tractable.

### 12. Inference procedure

Estimate (q\); branch low-confidence components; generate (K\) full-sequence hypotheses; compute test-time evidence; select one hypothesis; optionally refine that hypothesis with a fixed-budget optimizer. Report both selected-​K and oracle best-of-​K, clearly separated.

### 13. Required data

The same sign sequences as BiVis-Sign, but with a stronger requirement for accurate 3D validation and diverse ambiguity. Synthetic occlusion/blur/overlap can create known modes; accurate multi-view/mocap is needed to verify calibration. Interacting-hand data may pretrain hand-depth alternatives but is not a substitute for sign evaluation.

### 14. Required annotations

Accurate component-wise 3D error on a calibration subset; no semantic labels. Silhouettes are helpful but optional. Contact/order annotations can improve selection but would confound H2 with H3, so the first version should treat them as an ablation.

### 15. Compute cost

**Moderate-to-high.** Memory and optimization scale approximately with (K\). A flow/mixture implementation is cheaper than iterative diffusion. The paper must compare at matched wall-clock/parameter budgets and report the accuracy–cost frontier.

### 16. Expected TR-V2V effect

**INFERENCE:** potentially high left/right hand benefit on ambiguous frames, but only if selected-​K closes much of the oracle gap. Expected upper-body benefit is small-to-moderate. If only best-of-​K improves, the method has not solved the practical task.

### 17. Expected hand improvement

Lower catastrophic wrist-twist, finger-mode, left/right-depth, and identity errors; better risk–coverage behavior; fewer instances where a poor hand estimate contaminates the full body.

### 18. Expected body improvement

Limited direct torso effect. Arm benefit may arise when wrist/arm components branch jointly. Camera branching should be disabled unless Phase-3 camera headroom is demonstrated.

### 19. Expected robustness improvement

Potentially strong under occlusion, truncation, and detector disagreement; less certain under blur that destroys all discriminating evidence, because the selector may have no basis for choosing among plausible modes.

### 20. Main implementation risk

Mode collapse, combinatorial component mixtures, and a large oracle–automatic selection gap. The selector may simply learn dataset/style bias and prefer plausible but image-incorrect signing.

### 21. Main reviewer objection

“Probabilistic HMR and multi-hypothesis hand reconstruction already exist, and your improvement is oracle best-of-​K.” This objection is valid unless the work demonstrates calibrated sign-specific component uncertainty and single-output selected-​K improvement against ProHMR/POCO/MaskHand/StableHand-style baselines.

### 22. Falsification experiment

Compare deterministic, global-​K, and component-​K models. Report calibration error, NLL, coverage, diversity, risk–coverage, oracle best-of-​K, automatically selected-​K, and compute. Stratify by which component is occluded.

**Reject CalibSign-K** if component uncertainty is not better calibrated than raw detector confidence, if sampling adds no oracle coverage, if selected-​K fails to beat the deterministic baseline, or if selection gains vanish on unseen signers/degradations.

---

# Proposal 3 — OrderHands

## Temporally Ordered Bimanual Interaction Reconstruction

### 1. Method name

**OrderHands: Discrete-Continuous Temporal Hand–Hand State Reconstruction for Signing.**

### 2. Research hypothesis

**HYPOTHESIS:** during bimanual overlap, explicit temporal inference of left/right depth order, interaction phase, and surface-pair identity provides geometric information that independent hand estimation and collision-only fitting lack, thereby reducing relative depth and contact errors.

### 3. DexAvatar limitation addressed

DexAvatar fits the hands without an explicit bimanual state. Interpenetration discourages invalid overlap but cannot distinguish intended contact from collision, determine which hand is in front, preserve identity through crossing, or model approach/touch/separate timing.

### 4. Core technical insight

The missing variable is a temporally persistent **interaction topology**, not another hand feature encoder. A discrete state constrains a continuous mesh differently in approach, touch, cross, and separation phases. This makes failure visible and ablatable.

### 5. Architecture

1. Frozen body/hand observation and SMPL-X initialization.
2. Bimanual crop encoder using both image evidence and projected initialized meshes.
3. Semi-Markov or structured temporal state model that predicts phase, depth order, contact probability, duration, and candidate hand-surface pairs.
4. Discrete decoding (dynamic programming/beam search) over the state sequence.
5. Continuous joint optimizer over both hands, wrists, and optionally forearms, conditioned on the decoded state.

### 6. Input/output

- **Input:** RGB clip, left/right hand crops and features, 2D keypoints/masks, initial SMPL-X meshes, and component reliability.
- **Output:** final SMPL-X sequence plus interpretable per-frame interaction state: phase, order, contact pair, and confidence.

### 7. Mathematical formulation

Define

\[
s_t=(p_t,o_t,a_t,b_t),
\]

where (p_t\in\{\text{separate, approach, touch, cross, separate-after}\}\), (o_t\in\{L\prec R,R\prec L,\text{unresolved}\}\), and (a_t,b_t\) index coarse hand surface regions. The state sequence is decoded by

\[
s^*_{1:T}=\arg\max_s\left[\sum_t u_\phi(s_t;f_t)+\sum_t A_\phi(s_{t-1},s_t)+D_\phi(s)\right],
\]

where (D\) is a duration potential. Continuous fitting minimizes

\[
\begin{aligned}
E_{HH}=&E_{base}
-\lambda_s\log p_\phi(s_{1:T}\mid f_{1:T}) \\
&+\lambda_o\sum_t\mathbb 1[o_t\ne\mathrm{unresolved}],
\mathrm{ReLU}(m-\delta_{o_t}(z^L_t-z^R_t))^2 \\
&+\lambda_c\sum_{t:p_t=\mathrm{touch}}
\left[d(V^L_{t,a_t},V^R_{t,b_t})^2
+\eta(1+n^L_{t,a_t}\cdot n^R_{t,b_t})\right] \\
&+\lambda_{sep}\sum_{t:p_t=\mathrm{separate}}
\mathrm{ReLU}(\tau-d_{min}(V^L_t,V^R_t))^2
+\lambda_{pen}E_{pen}
+\lambda_{kin}E_{relvel}.
\end{aligned}
\]

(\delta_o\) selects the signed front/back convention. (E_{relvel}\) penalizes normal relative velocity during stable touch but allows tangential sliding. Constraints are confidence-gated; an uncertain discrete state must not force false contact.

### 8. Trainable modules

- bimanual image/mesh encoder;
- state unary, transition, duration, and surface-pair heads;
- optional learned weights that calibrate when each geometric factor is active.

### 9. Frozen modules

- observation front ends and initial SMPL-X estimator for the primary causal comparison;
- SMPL-X and collision/contact geometry routines;
- local SignHPoser prior initially.

### 10. Loss functions

- phase/order/contact-state cross-entropy or structured NLL;
- surface-pair classification/contrastive loss;
- state-boundary timing loss;
- relative hand depth and 3D hand vertex losses;
- contact distance/normal and nonpenetration losses;
- temporal state consistency/duration loss;
- standard reprojection, pose prior, and ROM losses.

### 11. Training strategy

1. Create a compact controlled annotation vocabulary rather than dense vertex correspondences for every frame.
2. Pretrain bimanual spatial reasoning on verified interacting-hand datasets where compatible, but learn sign phase/duration on sign sequences.
3. Train the discrete state model first, then the continuous optimizer weights, then optionally fine-tune jointly.
4. Oversample crossings and order changes; include near-contact negatives that look like touch in 2D.
5. Keep annotator-held-out and signer-held-out tests.

### 12. Inference procedure

Initialize both hands; extract bimanual features; decode a small number of state sequences; optimize both hands jointly under the top state; if state confidence is low, back off to observation/prior-only fitting. An optional two-hypothesis order branch can be used internally, but final output remains one automatically selected reconstruction.

### 13. Required data

- multi-view or mocap sign clips containing overlap, crossing, touch, and near-touch;
- SGNify/other sign clips for manual state annotation and evaluation only under correct split discipline;
- interacting-hand data for pretraining relative order/surface features;
- synthetic rendered bimanual occlusion for balanced order cases.

### 14. Required annotations

Per-frame or interval labels for interaction phase, left/right front order where visible in multi-view, touch/no-touch, and coarse hand surface-pair identity. Accurate dense contact can be generated from synchronized meshes and manually audited. This annotation burden is the main gate.

### 15. Compute cost

**Moderate.** The structured state model is small; joint mesh distance calculations and optimization dominate. It is substantially cheaper than a large generative sequence model but more expensive than independent framewise hands.

### 16. Expected TR-V2V effect

**INFERENCE:** moderate left/right hand improvement on overlap/cross/contact frames; likely only small-to-moderate aggregate hand change because these frames may be a minority. Upper-body-excluding-face improvement should be small. Relative-depth, contact, and penetration metrics should improve more strongly than mean TR-V2V.

### 17. Expected hand improvement

Correct left/right identity through crossing, correct front/back order, lower interpenetration, fewer false separations, better finger placement at touch, and more accurate contact onset/offset.

### 18. Expected body improvement

Mostly forearm/wrist kinematic adjustments; little torso/shoulder effect unless both hands jointly constrain arm position.

### 19. Expected robustness improvement

Strongest for hand–hand overlap and partial bimanual occlusion; some motion-blur robustness from temporal state persistence; weak for a single hand fully hidden without interaction evidence.

### 20. Main implementation risk

Rare or ambiguous state transitions, annotation inconsistency, unstable contact correspondences, and discrete prediction errors that force the optimizer into a worse continuous mode.

### 21. Main reviewer objection

“Interacting-hand methods, Dyn-HaMR, TUCH, and contact-aware HOI already model joint hands/contact.” The defensible difference is the explicit sign temporal state combining order, pair identity, phase, and duration, plus proof that each state component improves measured geometry over collision-only and unordered contact baselines.

### 22. Falsification experiment

On an interaction-enriched, multi-view test subset, compare: independent hands; joint features only; collision-only; unordered contact; order-only; state without duration; full ordered temporal state; and oracle state. Report TR-V2V, relative-depth sign accuracy, contact precision/recall, contact transition timing, penetration volume, identity switches, and non-contact false positives.

**Reject OrderHands** if ordered/temporal states do not improve geometry beyond unordered contact at matched compute, if oracle state has negligible headroom, or if the benchmark contains too few relevant frames to support a meaningful claim.

---

# Proposal 4 — SignAnchor

## Temporally Indexed Hand–Body Anatomical Anchoring

### 1. Method name

**SignAnchor: Temporally Indexed Articulator-to-Body-Region Anchors for Monocular Sign Reconstruction.**

### 2. Research hypothesis

**HYPOTHESIS:** when a hand touches or passes very near a known anatomical region, the body surface supplies a metrically useful 3D anchor for otherwise ambiguous hand depth and orientation; explicitly inferring the hand part, body region, and contact phase will improve geometry beyond repulsion or generic contact.

### 3. DexAvatar limitation addressed

DexAvatar treats the body mainly as an articulated mesh and collision obstacle. It does not represent that a fingertip/palm is intended to touch the cheek, chin, chest, shoulder, neck, or opposite arm, nor does it distinguish touch from a near-body pass.

### 4. Core technical insight

The anchor identity converts a 2D overlap into a constrained 3D relation. The key candidate contribution is a temporal **articulator ↔ anatomical-region** variable used to resolve depth/orientation, not a generic self-contact loss. Normal relative velocity should be constrained during touch while tangential motion remains possible, preserving brushing/sliding signs.

### 5. Architecture

1. Frozen RGB observation and SMPL-X initialization.
2. Semantic surface partition of SMPL-X hands and upper body/face into coarse anatomical regions.
3. Temporal anchor predictor consuming full-frame, hand-crop, projected-mesh, and reliability features.
4. Heads for contact probability, hand part, body region/triangle, barycentric anchor, phase/duration, and front/back order.
5. Confidence-gated optimizer over hand, wrist, forearm, and—only when supported—nearby upper-body joints.

### 6. Input/output

- **Input:** monocular RGB clip, initial SMPL-X mesh, image/mask/keypoint evidence, visibility/reliability, and semantic surface maps.
- **Output:** final SMPL-X sequence plus contact intervals, articulator/body-region labels, continuous body-surface anchors, and confidence.

### 7. Mathematical formulation

For predicted hand vertex/part (h_t\) and body triangle (r_t=(i,j,k)\), define a barycentric anchor

\[
a_t=\alpha_iV^B_{t,i}+\alpha_jV^B_{t,j}+\alpha_kV^B_{t,k},
\quad \alpha\ge0,\quad \sum\alpha=1.
\]

With contact probability (p_t\), phase (g_t\), and normals (n_h,n_a\):

\[
\begin{aligned}
E_{HB}=&E_{base}
+\lambda_a\sum_t p_t\|V^H_{t,h_t}-a_t\|_2^2
+\lambda_n\sum_t p_t(1+n^H_{t,h_t}\cdot n^B_{t,a_t}) \\
&+\lambda_v\sum_{t:g_t=\mathrm{touch}}p_t
\left[(v^H_t-v^B_t)\cdot n^B_{t,a_t}\right]^2 \\
&+\lambda_o E_{depth-order}
+\lambda_{persist}E_{anchor-transition}
+\lambda_{neg}\sum_{t:p_t\approx0}\mathrm{ReLU}(\tau-d_{HB,t})^2
+\lambda_{pen}E_{pen}.
\end{aligned}
\]

The non-contact term is used only for high-confidence negatives; otherwise it would incorrectly repel legitimate near-body articulation. (E_{anchor-transition}\) allows transitions between adjacent triangles and does not freeze tangential sliding.

### 8. Trainable modules

- temporal contact/phase classifier;
- hand-part and body-region/triangle predictor;
- barycentric anchor regressor;
- depth-order/confidence head;
- optional adaptive constraint-weight predictor.

### 9. Frozen modules

- primary body/hand detectors and initializer;
- SMPL-X and semantic surface partition;
- generic contact prior, if used for pretraining, must remain separate from the sign-specific anchor predictor in ablations.

### 10. Loss functions

- contact/phase and anatomical-region classification;
- surface geodesic error for anchor location;
- contact distance and normal alignment;
- normal relative-velocity compatibility;
- depth-order loss;
- transition/duration consistency;
- non-contact hard-negative loss;
- SMPL-X vertex/joint, reprojection, ROM, and penetration losses.

### 11. Training strategy

1. Pretrain generic surface/contact geometry with sources such as Goliath-SC where permitted, without claiming sign specificity.
2. Fine-tune on sign clips with anatomical contact intervals and near-contact negatives.
3. Train region classification before triangle regression; add continuous optimization only after state accuracy is adequate.
4. Balance contact and matched near-contact examples by region, speed, and visibility.
5. Keep face-contact evaluation even though face is excluded from primary UBody TR-V2V, because the face can anchor the evaluated hand.

### 12. Inference procedure

Predict contact intervals and candidate anchors; decode temporally consistent anchors; optimize hand/wrist/forearm under high-confidence constraints; jointly refine nearby body joints only when the body anchor itself is uncertain. Back off to baseline fitting on uncertain/non-contact frames.

### 13. Required data

- sign-specific multi-view/mocap clips with hand-to-face/torso/shoulder/arm interactions;
- generic self-contact data such as Goliath-SC for pretraining if available;
- BioTUCH as methodological precedent, not directly sufficient RGB sign training data;
- near-contact negative examples and synthetic renders.

### 14. Required annotations

Contact interval, touching hand part, anatomical body region, and ideally mesh-derived closest surface/triangle and relative depth order. A practical label pipeline can derive dense anchors from accurate multi-view SMPL-X and ask annotators only to verify contact and coarse regions.

### 15. Compute cost

**Moderate.** Surface-nearest-neighbour/contact calculations add optimization cost but are localized. Annotation and accurate data capture, not GPU compute, are the principal expense.

### 16. Expected TR-V2V effect

**INFERENCE:** moderate hand improvement on body-contact frames and possible arm/upper-body improvement via the kinematic chain; small-to-moderate aggregate effect unless contact prevalence is high. Contact-location error and relative hand-to-body error should be more sensitive than standard TR-V2V.

### 17. Expected hand improvement

Correct absolute depth and orientation relative to the body; reduced penetration/hovering; improved fingertip/palm placement; better wrist pose; fewer false front/behind solutions near face/torso.

### 18. Expected body improvement

Potential forearm/elbow/shoulder improvement when hand position constrains the chain. Body-region identity should prevent the optimizer from compensating with an incorrect torso/shoulder location. Large torso gains are not expected.

### 19. Expected robustness improvement

High potential for hand–body occlusion and long contact holds; moderate motion-blur robustness because the body anchor persists; low benefit for mid-air signing with no body relation. False contact under near-body motion is the central robustness failure.

### 20. Main implementation risk

Insufficient sign-specific contact data, inaccurate body/face surfaces, ambiguity between contact and hover, and contact constraints that canonicalize motion or suppress valid sliding. Face expression/shape errors can also shift the anchor.

### 21. Main reviewer objection

“TUCH, Goliath-SC, BioTUCH, and hand–face reconstruction already use self-contact, contact priors, or depth order.” This is the strongest novelty risk in the portfolio. A defensible claim requires a temporal sign articulator-to-anatomical-region representation, matched near-contact negatives, and evidence that the anchor resolves observed hand geometry—not merely collision or plausibility.

### 22. Falsification experiment

First perform a prevalence/error audit. Then compare collision-only, generic contact, coarse body-region contact, continuous anchor without temporal state, and full temporal SignAnchor. Evaluate hand TR-V2V/MPVPE, relative hand-to-body surface error, region classification, contact F1, depth-order accuracy, penetration, hover distance, and performance on matched non-contact frames.

**Reject SignAnchor** if oracle contact/anchor information has negligible geometric headroom, if inferred anchors do not beat generic contact, if false positives harm near-body non-contact motion, or if relevant frames are too sparse for aggregate or stratified statistical power.

---

# Proposal 5 — RCP-Sign

## Reliability-Certified Pseudo-GT Correction and Hybrid Reconstruction

### 1. Method name

**RCP-Sign: Reliability-Certified Pseudo-Ground-Truth Correction for Hybrid Sign SMPL-X Reconstruction.**

### 2. Research hypothesis

**HYPOTHESIS:** systematic component errors in pseudo-SMPL-X labels are learned by sign pose priors and regressors; independently identifying and correcting/downweighting unreliable body, wrist, finger, and camera components will improve a fixed downstream reconstructor more than hard filtering while retaining rare valid signing motion.

### 3. DexAvatar limitation addressed

DexAvatar demonstrates that filtering SignBPoser training labels improves upper-body TR-V2V, but hard filtering discards data and its correction process does not provide a calibrated, component-wise uncertainty that is used consistently at training and inference. Strong initialization matching can then preserve remaining label bias.

### 4. Core technical insight

Pseudo-label trust should be **component-wise and evidence-certified**. A rare pose should not be rejected merely because one teacher dislikes it; a label should be corrected only when several partly independent tests identify the same geometric inconsistency. The final experiment fixes the reconstructor so that downstream gain can be causally attributed to label quality.

### 5. Architecture

1. **Offline evidence bank:** multiple frozen estimators/fits, 2D reprojection/heatmap evidence, silhouette where available, temporal forward/backward consistency, left/right/kinematic consistency, contact/penetration checks, and multi-view/mocap truth on a small subset.
2. **Reliability model (R_\phi\):** predicts per-frame, per-component reliability and error covariance.
3. **Correction model (C_\psi\):** proposes a residual pose correction only for components that fail independent consistency tests; abstains otherwise.
4. **Corrected sign prior/regressor:** trained using corrected targets and uncertainty weights.
5. **Hybrid inference:** regressor supplies mean/covariance; a fixed structured optimizer combines it with video observations.

### 6. Input/output

- **Offline input:** sign videos, existing pseudo-SMPL-X, frozen teacher outputs, 2D/mask/temporal evidence, and a small accurate 3D subset.
- **Offline output:** corrected pseudo labels (\tilde x\), per-component covariance (\Sigma\), abstention flags, and an audit log.
- **Online input/output:** RGB clip → initial SMPL-X mean/covariance → optimized final SMPL-X sequence.

### 7. Mathematical formulation

Let teachers/evidence sources provide (x^{(r)}_{t,c}\). Reliability uses features (e_{t,c}\) that include disagreement, reprojection residual, temporal cycle error, and biomechanical violations:

\[
(w_{t,c},\Sigma_{t,c})=R_\phi(e_{t,c}),\qquad w_{t,c}\in[0,1].
\]

The corrected label solves or amortizes

\[
\begin{aligned}
\tilde x_{1:T}=\arg\min_x
&\sum_{t,c,r}\omega^{(r)}_{t,c}\,d_R(x_{t,c},x^{(r)}_{t,c})^2
+\lambda_{2D}E_{2D}(x) \\
&+\lambda_t E_{motion}(x)
+\lambda_bE_{biomech}(x)
+\lambda_{sil}E_{sil}(x)
+\lambda_{chg}\sum_{t,c} w_{t,c}\,d_R(x_{t,c},x^0_{t,c})^2.
\end{aligned}
\]

The last term protects reliable/rare valid components from unnecessary correction. A regressor (F_\eta\) learns heteroscedastically:

\[
\mathcal L_{reg}=\sum_{t,c}\left[
d_R(F_\eta(I)_{t,c},\tilde x_{t,c})^2/\sigma_{t,c}^2
+\log\sigma_{t,c}^2\right]
+\lambda_V\|\hat V-\tilde V\|_1
+\lambda_v\|\Delta\hat J-\Delta\tilde J\|_1.
\]

At inference,

\[
E_{hybrid}(x)=E_{obs}(x)+\sum_{t,c}d_{\Sigma_{t,c}}(x_{t,c},\mu_{t,c})^2
+\lambda_tE_{motion}+\lambda_bE_{biomech}+\lambda_{pen}E_{pen}.
\]

### 8. Trainable modules

- reliability/covariance model (R_\phi\);
- abstaining correction model (C_\psi\) or correction amortizer;
- compact sign SMPL-X regressor/prior (F_\eta\);
- optional adaptive inference-weight predictor.

### 9. Frozen modules

- all teacher estimators/fitting pipelines used to build the evidence bank;
- SMPL-X;
- one downstream evaluator and, for the causal label experiment, the entire downstream reconstructor except the training data/weights being compared.

### 10. Loss functions

- reliability classification/regression and calibration;
- clean-subset correction geodesic/vertex loss;
- evidence consistency: 2D, silhouette, temporal cycle, multi-view where available;
- abstention/selective-risk loss;
- heteroscedastic pose/mesh/velocity training;
- ROM and penetration as weak evidence, not absolute truth;
- distribution-preservation loss/statistical audit for rare poses.

### 11. Training strategy

1. Build splits before correction; no SGNify test leakage.
2. Train (R_\phi\) on a clean 3D subset plus synthetic corruptions modeled after actual teacher failure.
3. Cross-fit reliability/correction so a clip is never corrected by a model trained on its clean labels.
4. Produce raw, hard-filtered, weighted-only, corrected, and corrected-plus-weighted datasets.
5. Train identical priors/regressors from identical seeds and budgets on each data version.
6. Freeze the downstream optimizer and compare causal downstream effects.
7. Audit diversity/tails to ensure correction has not removed uncommon valid signs.

### 12. Inference procedure

Run the trained sign regressor once to obtain (\mu,\Sigma\); initialize SMPL-X; perform covariance-weighted optimization against observations and standard constraints; output one sequence. The expensive multi-teacher bank is offline only.

### 13. Required data

- raw SignAvatars/How2Sign sign videos and existing pseudo-SMPL-X;
- multiple frozen teacher outputs from established body/hand estimators/fitting pipelines;
- SignHPoser/3D-LEX or newly captured multi-view/mocap for a small clean subset;
- synthetic corruptions matched to real teacher disagreement.

### 14. Required annotations

A modest clean 3D subset with synchronized RGB is essential. No gloss/contact labels are required for the core method. If contact evidence is used, it must remain one feature among several; otherwise BioTUCH becomes an especially close precedent and contact errors can dominate the correction.

### 15. Compute cost

**High offline, moderate online.** Generating several teacher/fitter outputs is expensive but parallelizable and performed once. Training the reliability/correction/regressor stack is moderate. Online inference is comparable to a hybrid regressor plus short optimization.

### 16. Expected TR-V2V effect

**EVIDENCE-BASED INFERENCE:** upper-body-excluding-face has the clearest headroom because DexAvatar's filtering ablation already improves it. Hand benefit is plausible only if clean hand supervision is sufficient to certify finger/wrist corrections. No new millimetre gain is predicted.

### 17. Expected hand improvement

Reduced systematic wrist/finger bias, better initialization, calibrated hand trust, and fewer frames dominated by a poor hand teacher. The method may fail to improve fine fingers if the clean subset or teachers lack that resolution.

### 18. Expected body improvement

Potentially broad arm/upper-body improvement through cleaner prior/regressor training, better rare-pose retention than hard filtering, and less test-time pull toward biased pseudo-label modes.

### 19. Expected robustness improvement

Better domain/teacher robustness if evidence sources fail differently; improved reliability under detector disagreement. Limited benefit when all teachers share the same systematic bias or the clean subset does not cover a new domain.

### 20. Main implementation risk

Correlated teacher errors can create false consensus. Correction can erase rare but valid signing, and “biomechanical plausibility” can be culturally/linguistically wrong. Dataset construction may consume more time than method development.

### 21. Main reviewer objection

“This is dataset cleaning plus EFT/POCO-style pseudo-label weighting and standard hybrid fitting; BioTUCH already refines pseudo-GT with contact.” The paper must show independently calibrated component reliability, correction rather than selection alone, rare-motion preservation, and a fixed-downstream causal improvement. Otherwise the novelty is weak.

### 22. Falsification experiment

Train identical models on: raw pseudo-GT; DexAvatar-style hard filtering; reliability weighting; correction only; correction plus weighting; and an equal-size clean subset. Use fixed inference and held-out signers/domains. Report clean-subset label MPVPE/MPJPE, downstream TR-V2V, calibration/selective risk, tail-pose coverage, contact/biomechanical violations, and data retention.

**Reject RCP-Sign** if corrected labels do not beat hard filtering on independent 3D truth and fixed downstream TR-V2V, if gains vanish under cross-fitting, if teachers share unrecoverable bias, or if tail-sign performance/diversity worsens.

---

## 2. Cross-proposal scorecard

Scores are research-triage judgements, not measurements.

- For **Novelty, Expected performance, Technical contribution, Feasibility, Data availability, Compute efficiency, Ablatability, Generalization, and Conference potential**, 10 is better.
- For **Reviewer risk**, 10 means **higher risk** and is worse.
- “Compute efficiency” is used instead of raw compute burden so the direction is unambiguous: 10 means cheap/efficient.

| Candidate | Novelty | Expected performance | Technical contribution | Feasibility | Data availability | Compute efficiency | Ablatability | Generalization | Reviewer risk ↑ | Conference potential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **BiVis-Sign** | 7 | 8 | 8 | 8 | 8 | 7 | 9 | 8 | 4 | **9** |
| **CalibSign-K** | 6 | 8 | 9 | 6 | 5 | 4 | 8 | 8 | 7 | **8** |
| **OrderHands** | 8 | 7 | 9 | 5 | 4 | 6 | 9 | 7 | 6 | **9** |
| **SignAnchor** | 7 | 7 | 9 | 4 | 3 | 6 | 8 | 7 | 8 | **8** |
| **RCP-Sign** | 6 | 8 | 8 | 7 | 6 | 5 | 10 | 8 | 6 | **8** |

### Why the rankings are not a simple sum

- **BiVis-Sign** is primary because its failure is common, the oracle test is cheap, relevant sign sequences exist, and the hidden-span causal claim is easy to falsify. Its novelty is moderate, so disciplined comparison to 2026 temporal hand work is essential.
- **OrderHands** has the strongest sign-specific variable but needs an interaction-enriched dataset. It is the backup only after a prevalence audit and oracle-state experiment pass.
- **RCP-Sign** is low-risk scientifically because label-quality evidence already exists and every stage has a fixed-control experiment. It is not low-risk for novelty.
- **SignAnchor** has high geometric leverage on genuinely anchored frames, yet strong nearby contact work and sparse annotations create both reviewer and implementation risk.
- **CalibSign-K** could outperform all candidates if ambiguity is common and selection works. The field is crowded, and selected-​K—not oracle-​K—is the decisive gate.

---

## 3. Recommended staged research program

The proposals should not be merged into a large architecture before these gates are run.

| Stage | Cheapest decisive experiment | Decision |
|---|---|---|
| 1. Observation audit | Oracle 2D/visibility and frame/component error stratification | Establish how much error is recoverable from better temporal/ambiguity reasoning. |
| 2. Primary gate | Forward vs bidirectional with identical masked sequence model | Continue BiVis-Sign only if future context improves real hidden-span geometry without motion attenuation. |
| 3. Interaction audit | Measure prevalence/error of order, hand–hand contact, and hand–body contact | Decide whether OrderHands or SignAnchor has enough benchmark/data support. |
| 4. Ambiguity gate | Deterministic vs oracle best-of-​K vs automatically selected-​K | Continue CalibSign-K only if the selection gap is tractable. |
| 5. Data-quality gate | Raw vs hard-filtered vs corrected labels under fixed inference | Continue RCP-Sign only if correction beats filtering on independent truth. |

### Minimal primary-paper scope

If the gates support BiVis-Sign, the strongest disciplined paper is likely:

1. a quantified sign-specific occlusion/blur and recovery audit;
2. visibility-conditioned bidirectional full-SMPL-X inference;
3. an observation-preserving sequence MAP layer;
4. matched temporal baselines and real/synthetic hidden-span tests;
5. TR-V2V plus motion-fidelity, recovery, and robustness metrics.

Do **not** add contact, diffusion, recognition, signer adaptation, and pseudo-GT correction merely to make the model look larger. Each extra mechanism creates a confound and a prior-work burden. OrderHands or SignAnchor should become a separate primary contribution only if their oracle studies show more headroom than bidirectional inference.

---

## 4. Reviewer-validation checklist

Before claiming SOTA or novelty, the eventual paper must satisfy all of the following:

- run DexAvatar and newer directly comparable methods through the same evaluator, manifest, frames, alignment, and missing-frame policy;
- separate directly comparable TR-V2V from results transferred across protocols;
- report confidence intervals and paired per-sequence/per-sign tests, not only mean millimetres;
- report SGNify overall and hard strata: occlusion length, blur, speed, hand–hand overlap, hand–body overlap, and one-/two-handed signs;
- report MPJPE/MPVPE, velocity/acceleration error, lag/recovery, penetration, contact/order where applicable, and visible-frame regression;
- evaluate at least one multi-signer/domain-shift set because SGNify's primary test is one signer;
- prevent test-video/pseudo-label/recognition leakage;
- compare parameter count, training data, inference time, and optimization iterations;
- distinguish an architectural improvement from an upgraded detector or additional data;
- include oracle experiments that establish headroom before presenting the learned mechanism;
- state failed hypotheses and negative results;
- conduct one final forward/recency/code search immediately before submission.

---

## Final conclusion

**FACT:** all five proposals address real limitations in DexAvatar, but their generic ingredients already have substantial adjacent prior art.

**EVIDENCE:** DexAvatar directly supports the pseudo-label-quality failure; temporal hand reasoning, uncertainty, and contact are supported mainly by adjacent reconstruction literature and by DexAvatar's missing mechanisms, not yet by a causal sign-specific oracle study.

**INFERENCE:** BiVis-Sign currently has the best realistic balance of expected TR-V2V impact, feasibility, and conference potential. OrderHands may offer a stronger sign-specific contribution if interaction labels and oracle headroom exist. RCP-Sign is the safest route to measurable improvement but carries a “data cleaning” reviewer risk. SignAnchor has the highest annotation/novelty risk. CalibSign-K should live or die by automatic selection.

**DECISION:** begin with the five decisive pilot experiments above. Do not implement a combined final architecture until at least the primary and interaction/ambiguity/data-quality gates have been measured.

