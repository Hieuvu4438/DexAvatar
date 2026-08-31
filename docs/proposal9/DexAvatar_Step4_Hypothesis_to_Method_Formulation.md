# DexAvatar Step 4 — Hypothesis-to-Method Formulation Under a Locked Feasibility Contract

**Audit/formulation date:** 2026-08-26 (Asia/Bangkok)  
**Mode:** paper, supplementary, official documentation, and static-source inspection only. No repository, checkpoint, dataset, optimizer, or evaluator was executed. No result is reproduced or forecast.  
**Evidence labels:** `[VERIFIED]`, `[INFERENCE]`, `[CONDITIONAL]`, `[UNRESOLVED]`, and `[PROPOSED]`. A statement tagged `[PROPOSED]` is part of the method candidate, not an observed fact.  
**Decision vocabulary:** `PASS`, `CONDITIONAL`, `FATAL` for preconditions; `FROZEN`, `OPTIMIZED`, `METADATA`, `DISCARDED` for system state.

For compactness, when a paragraph immediately preceding an equation, list, or table begins with `[PROPOSED]`, that label governs every normative method element in that block unless a cell or sentence carries a different evidence label.

---

## 1. Step 3 contract restatement

**Locked verdict:** `GO WITH SCOPE REDUCTION`.

`[VERIFIED]` The official evaluator independently centroid-aligns UBody(-F), LHand, and RHand per frame. It therefore removes pure region translation but retains errors caused by local articulation, rotation, scale, and shape. It aggregates over concatenated vertex–frame samples, not by first averaging each sign. For class-`0`, LHand is not accumulated and left-hand vertices are removed from UBody(-F) (`evaluate_new_fitting(2).py`, `transl_point_error`, lines 159–169; region/class rules, lines 380–395; aggregation, lines 432–461; Step 1 dossier §§6 and 8).

`[VERIFIED]` DexAvatar's final fitting variables are a 33-D SignBPoser body latent and one or two 23-D SignHPoser hand latents; the frozen SignHPoser decoder produces a 45-D axis-angle hand block which is passed to `left_hand_pose` or `right_hand_pose` in one SMPL-X forward pass ([`fit_single_frame.py`, lines 220–243 and 476–503](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py#L220-L243); [`fitting.py`, lines 258–277](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fitting.py#L258-L277); [`fit_single_frame.py`, lines 627–647](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py#L627-L647)).

`[VERIFIED]` The reduced PRIMARY is limited to two physical-side-indexed sequences of 15 local finger rotations. DexAvatar body pose, shoulder–elbow–wrist chain, wrist/root orientation, SMPL-X shape, camera, translation, upstream observations, evaluator, frame eligibility, masks, and class-`0` rules are locked. The two streams remain independent; they are not described as a learned joint-bimanual model.

`[VERIFIED]` HandFlow is the frozen technical substrate. Only its finger45 portion may enter the proposed intervention. Its MANO root/global orientation, shape, translation, camera/world trajectory, vertices, and faces are inadmissible. Dyn-HaMR contributes at most a side-indexed identity/observation-validity contract—not SLAM, HMP, world motion, full optimization, biomechanics, penetration, root, shape, or mesh.

The Step 3 hypothesis is retained without changing its scientific scope:

> Trong các frame ký hiệu có quan sát tay bị thiếu hoặc có độ tin cậy thấp nhưng còn ngữ cảnh temporal lân cận, sử dụng suy luận temporal sinh có conditioning theo trạng thái confidence/missing để ước lượng, cho từng side-track vật lý cố định, hai chuỗi 15 local finger-joint rotations tương thích SMPL-X sẽ giảm centroid-aligned LHand/RHand TR-V2V và phần lỗi của các hand vertices còn được giữ trong UBody(-F) so với DexAvatar, khi giữ cố định body pose, wrist/root orientation, SMPL-X shape, camera, translation, upstream per-frame observations và official evaluator.

`[INFERENCE]` Step 4 formulates a falsifiable mechanism for that hypothesis. It does not assert that the hypothesis is true.

---

## 2. Precondition closure table

Repository evidence is pinned to the states listed in §20. “Dimensionally compatible” is not treated as “semantically compatible.”

| Preconditions required by Step 4 | Official evidence | Closure | Consequence for formulation |
|---|---|---:|---|
| HandFlow canonical training side | `[VERIFIED]` The paper says training crops are right-hand regions (Appendix B); the README says left videos must first be mirrored because the FM model was trained on right hands ([paper Appendix B](https://arxiv.org/html/2607.11221v1#A2); [`README.md`, line 154](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/README.md#L154)). | `PASS` for **right-canonical training** | The checkpoint is never assumed natively side-symmetric. |
| Meaning of `--side left` versus README mirroring | `[VERIFIED]` The demo accepts `--side left`; `process_sequence` selects the detector class but its crop path performs no horizontal flip; the predicted 48-D pose is then fed directly to a MANO layer selected by the side label ([`scripts/demo.py`, lines 125–220](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/scripts/demo.py#L125-L220); [`online_hamer.py`, lines 190–314](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/online_hamer.py#L190-L314); [`mano_utils.py`, lines 14–62](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/mano_utils.py#L14-L62)). This does not implement the README's stated mirror-to-right contract. | `CONDITIONAL` | Exact left conversion cannot be inferred from `--side left`; it remains symbolic as \(\mathcal C_L\). |
| \(\theta^{48}\) split and representation | `[VERIFIED]` Paper §3 defines \(\theta_i\in\mathbb R^{48}\) in axis-angle; release config states `global_aa(3) + joints_aa(45)`, and MANO utilities use the same layout ([paper §3, Eq. 1](https://arxiv.org/html/2607.11221v1#S3); [`configs/model.yaml`, lines 1–6](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/configs/model.yaml#L1-L6); [`mano_utils.py`, lines 1–5](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/mano_utils.py#L1-L5)). | `PASS` | The first 3 values are always discarded; only the final 45 values are eligible. |
| Joint order and local-parent frames of finger45 | `[VERIFIED]` Release passes the 48-D vector directly to `manopth.ManoLayer(use_pca=False, flat_hand_mean=True)`. `[UNRESOLVED]` Neither paper nor release supplies an authoritative 15-joint order/parent-frame table tied to DexAvatar's SMPL-X blocks. | `CONDITIONAL` | \(\mathcal C_R\) and \(\mathcal C_L\) must validate permutation and local-frame semantics; equal dimension is insufficient. |
| Pose-mean convention | `[VERIFIED]` HandFlow's MANO FK is `flat_hand_mean=True`, `use_pca=False`; DexAvatar's neutral SMPL-X runtime is also explicitly `flat_hand_mean=True`, `use_pca=False` ([HandFlow `mano_utils.py`, lines 20–29](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/mano_utils.py#L20-L29); [DexAvatar `main.py`, lines 144–151](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/main.py#L144-L151)). `[CONDITIONAL]` This match is established only for the neutral runtime actually hard-coded there, not arbitrary gender-specific SMPL-X construction. | `PASS` for audited neutral runtime; otherwise `CONDITIONAL` | Method contract is tied to the audited neutral model unless a separate exact convention check is supplied. |
| HandFlow finger45 → DexAvatar SMPL-X hand blocks | `[VERIFIED]` DexAvatar already converts HaMeR's 15 matrices to 45-D Rodrigues vectors; its left path flips axis-angle y/z signs, but that is a HaMeR-specific parser path and is not evidence for HandFlow ([`data_parser.py`, lines 397–425](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L397-L425)). | `CONDITIONAL` | No copied sign rule. The method is admissible only through validated \(\mathcal C_s\). |
| Checkpoint and normalization availability/identity | `[VERIFIED]` The official README links both artifacts. Hugging Face pins `handflow_denoiser.pt` at commit `3ca50e4…`, SHA-256 `2fbc4e1f…fcda`, and `normalization_stats.npz` at commit `fc35519…`, SHA-256 `7313334e…e92c4` ([checkpoint](https://huggingface.co/mxxu00/HandFlow/commit/3ca50e4afececc8a7bc361b74954c77307bd0a5f); [normalization](https://huggingface.co/mxxu00/HandFlow/blob/fc35519962867acdf834ccef13b9a2814cbbd15d/normalization_stats.npz)). `[UNRESOLVED]` The 21-byte model card does not bind these hashes to a detailed training-data/config manifest. | `PASS` for immutable artifact identity; provenance `CONDITIONAL` | Step 5 can pin exact bytes, but must not infer undocumented provenance. |
| Window, ODE, frontend contract | `[VERIFIED]` Paper Appendix B and release specify \(T=16\), three Euler steps, frozen HaMeR features/keypoints/confidence; default overlap is 2 with velocity blending ([paper §§3 and Appendix B](https://arxiv.org/html/2607.11221v1); [`configs/inference.yaml`, lines 6–15](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/configs/inference.yaml#L6-L15); [`configs/model.yaml`, lines 27–29](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/configs/model.yaml#L27-L29)). | `PASS` | These values are frozen artifact contract, not tunable method claims. |
| Confidence can be counterfactually masked at inference | `[VERIFIED]` `ConditionBuilder.forward` consumes a \((B,T)\) confidence tensor directly and applies \(m_t c_t+(1-m_t)c_{mask}\); random masking is training-only ([`condition_builder.py`, lines 29–66](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/model/feature_extractors/condition_builder.py#L29-L66); paper §3.2, Eq. 5). | `PASS` | A paired factual/masked query is possible without modifying or retraining the denoiser. |
| Frozen release supports stochastic inference | `[VERIFIED]` Inference starts from Gaussian noise; code creates fresh `torch.randn` pose/root/shape state. Appendix D reports ten seeded passes and finds very small pose dispersion ([paper §3.1 and Appendix D](https://arxiv.org/html/2607.11221v1#A4); [`inference_utils.py`, lines 206–212](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/inference_utils.py#L206-L212)). | `PASS`, with a negative design implication | Flow matching is not called a calibrated posterior. The proposed method uses one predeclared common noise draw, not best-of-\(K\), sample averaging, or sample-spread uncertainty. |
| Frozen SignHPoser projects to valid DexAvatar hand state | `[VERIFIED]` DexAvatar instantiates one frozen 23-D-latent SignHPoser and calls `decode(..., output_type='aa')` for both hands before SMPL-X ([`test_hposer.py`, lines 21–43](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/test_hposer.py#L21-L43); [`fit_single_frame.py`, lines 630–640](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py#L630-L640)). | `PASS` as a released baseline dependency | New optimization can remain on the already-used decoded SMPL-X hand manifold. |

**Gate result:** no item is `FATAL`. Exact HandFlow-to-left-SMPL-X semantics and the complete cross-model joint-frame contract remain `CONDITIONAL`; therefore a fully unconditional Step 4 verdict is not permitted.

---

## 3. Method-formulation verdict

### `CONDITIONAL METHOD CANDIDATE`

`[PROPOSED]` Exactly one method candidate is formulated below. It is training-free, keeps HandFlow and SignHPoser frozen, changes only decoded local finger rotations, and has one post-fitting intervention point.

`[CONDITIONAL]` Scientific formulation is complete, but admission of right and especially left HandFlow local states requires an authoritative or independently validated \(\mathcal C_s\). There is no evidence that such a mapping is impossible, so the appropriate decision is conditional rather than blocked.

---

## 4. One-sentence method claim

`[PROPOSED]` **A common-noise factual/counterfactual query of frozen HandFlow can arbitrate each physical-side local-finger state: a frozen-SignHPoser update is admitted only when it moves the unified SMPL-X hand pose toward both the observed-confidence and target-masked temporal responses on \(SO(3)\), without worsening DexAvatar's fixed-observation hand-fitting evidence; otherwise the released DexAvatar state is retained.**

This is a falsifiable claim about an intervention, not a claim of measured improvement.

---

## 5. Provisional method name

`[PROPOSED]` **CLRA-Dex — Counterfactual Local-Rotation Arbitration for DexAvatar.**

The name is provisional and descriptive. It does not assert novelty, accuracy, or superiority.

---

## 6. System boundary and frozen-state table

`[PROPOSED]` The disposition column below defines the method boundary; factual source-state descriptions retain their explicit evidence status from §§1–2.

| State/component | Released source state | Step 4 disposition | May change? | Scientific role |
|---|---|---|---:|---|
| RGB sequence and retained timestamps | DexAvatar input | `FROZEN` | No | Observation source; no frame addition/deletion. |
| Sapiens/OpenPose/HaMeR detections, keypoints, confidence, crops | DexAvatar/HandFlow frontends | `FROZEN` | No | Fixed evidence. A confidence value may be counterfactually masked **inside a diagnostic HandFlow query**; its stored upstream value is not edited. |
| Physical-side identity \(\iota_s\), observation validity \(v^{obs}_{t,s}\), identity validity \(v^{id}_{t,s}\) | Dyn-HaMR-style contract only | `METADATA` | No pose state | Prevents side reassignment and defines refusal. Not claimed as learned bimanual inference. |
| Frozen HandFlow denoiser and normalization | HandFlow | `FROZEN` | No | Produces two controlled temporal responses per target state. |
| HandFlow root3, \(\beta^{MANO}\), translation, camera/world state, vertices/faces | HandFlow | `DISCARDED` | Never enters DexAvatar | Explicitly outside metric/state boundary. |
| Converted HandFlow finger rotations \(H^+_{t,s,j},H^-_{t,s,j}\) | HandFlow + conditional \(\mathcal C_s\) | `FROZEN CANDIDATE INFORMATION` | No | Temporal support only; not final pose and not an independent likelihood. |
| Released DexAvatar full parameters \(\Theta^D_t\) | DexAvatar fitting | `FROZEN REFERENCE` | No, except candidate hand block below | Defines the comparator and safe fallback. |
| Body pose, shoulder–elbow–wrist chain, global orientation | DexAvatar | `FROZEN` | No | Prevents wrist/body-chain confounding. |
| SMPL-X shape, expression, jaw/eyes | DexAvatar | `FROZEN` | No | Preserves unified body identity and face state. |
| Camera and translation | DexAvatar | `FROZEN` | No | Prevents metric-removed translation gain from being misattributed. |
| Frozen SignHPoser decoder \(D_H\) | DexAvatar | `FROZEN` | No | Maps 23-D latent to a 45-D local axis-angle hand block already accepted by DexAvatar. |
| Baseline latents \(z^D_{t,s}\in\mathbb R^{23}\) | DexAvatar | `FROZEN REFERENCE` | No | Default final state and comparator. |
| Candidate latents \(z^*_{t,s}\in\mathbb R^{23}\) | `[PROPOSED]` CLRA-Dex | `OPTIMIZED` only when admitted | Yes | The only continuous state optimized by the new intervention. |
| Final local hand blocks | `[PROPOSED]` selected decoded latent or baseline | `OPTIMIZED/UNCHANGED` | Only 15 finger joints of an admitted side-frame | Direct LHand/RHand and hand-vertex UBody(-F) leverage. |
| Full final mesh | SMPL-X | `REGENERATED` | Only through admitted hand rotations | One unified SMPL-X forward pass; no MANO vertex transfer. |
| Evaluator, regions, frames, class labels, centering, aggregation | Supplied SGNify protocol | `FROZEN` | No | Locked infrastructure. |

`[PROPOSED]` No HandFlow parameter, SignHPoser parameter, body parameter, shape coefficient, root/wrist rotation, camera parameter, translation, mesh vertex, or evaluator state is a new optimization variable.

---

## 7. Exact input/output/state definition

### 7.1 Input

`[VERIFIED]` Let \(\mathcal T_E\) be exactly the set of frames retained by the existing DexAvatar output/evaluator contract. CLRA-Dex does not reconstruct an evaluator-missing frame and does not alter positional pairing or frame eligibility.

`[PROPOSED]` For each \(t\in\mathcal T_E\) and physical side \(s\in\{L,R\}\), the method consumes:

- the unchanged RGB sequence and HandFlow's frozen HaMeR-derived image/skeleton conditions;
- frame-level HandFlow detection confidence \(m_{t,s}\in[0,1]\);
- fixed DexAvatar 2D hand observations \(y_{t,s,\ell}\in\mathbb R^2\) and their joint-level confidence \(q_{t,s,\ell}\in[0,1]\);
- physical-side label \(\iota_s\), observation validity \(v^{obs}_{t,s}\), and identity validity \(v^{id}_{t,s}\);
- released DexAvatar parameters \(\Theta_t^D\) and released hand latent \(z^D_{t,s}\);
- the frozen HandFlow and SignHPoser checkpoints.

`[PROPOSED]` Frame confidence \(m\) and joint confidence \(q\) are not conflated: \(m\) controls HandFlow's released frame-wise cmask; \(q\) remains in the released robust 2D reprojection evidence. Neither is asserted to be calibrated 3D-pose probability.

`[VERIFIED]` The HandFlow paper calls \(m\) “HaMeR detection confidence,” but the released online path obtains `valid_confs` from the WiLoR YOLO detector's selected bounding box and writes that scalar into `hamer_confidence` ([paper §3.2, Eq. 5](https://arxiv.org/html/2607.11221v1#S3.SS2); [`online_hamer.py`, lines 223–255 and 298–309](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/online_hamer.py#L223-L309)). The method contract follows released code: \(m\) is a frame-level selected-box confidence, not a per-joint HaMeR uncertainty.

`[VERIFIED]` In the non-class-`0` parser path, DexAvatar writes the inserted HaMeR hand-keypoint confidences to 1 (`data_parser.py`, lines 408–413). Thus \(q\) is often only a fixed fitting weight for those hand joints, not an informative local-hand uncertainty signal. `[PROPOSED]` CLRA-Dex never promotes it to one; the uncertainty intervention uses HandFlow's frame-level detector confidence \(m\), while \(q\) only preserves the released reprojection contract.

### 7.2 Output

`[PROPOSED]` The output is a standard DexAvatar SMPL-X parameter vector \(\Theta_t^{\star}\), not a MANO mesh. It is identical to \(\Theta_t^D\) except that an admitted `left_hand_pose` or `right_hand_pose` is the 45-D output of the same frozen SignHPoser decoder evaluated at \(z^*_{t,s}\). A single SMPL-X forward pass produces all final vertices.

### 7.3 Allowed and forbidden state

`[PROPOSED]` The intervention unit is one **side–frame local-pose block** of 15 finger joints. All 15 decoded rotations remain coupled through the 23-D SignHPoser latent. CLRA-Dex does not pretend to estimate finger joints independently when the decoder does not.

`[PROPOSED]` HandFlow supplies temporal candidate information only. Its root3, shape10, translation3, MANO vertices, camera, and world trajectory are removed before arbitration. No MANO state is written into SMPL-X outside \(\mathcal C_s\)'s local finger interface.

### 7.4 Metric eligibility

`[PROPOSED]` An update is eligible only where its changed vertices are retained by the official metric. In class-`0` signs, the left stream remains exactly DexAvatar because LHand and left-hand UBody(-F) vertices are excluded; the right stream remains eligible under the original class rule. No metric mask is changed.

---

## 8. Single intervention point

**Selected point: after released DexAvatar fitting, before the final unified SMPL-X forward output.**

`[PROPOSED]` DexAvatar first produces the complete baseline state \(\Theta^D_t,z^D_{t,L},z^D_{t,R}\) unchanged. CLRA-Dex then freezes every baseline state, performs only local-hand latent arbitration, substitutes an admitted decoded 45-D block, and invokes the normal SMPL-X forward model.

`[INFERENCE]` This point gives the cleanest causal isolation: upstream observations, initialization, body fitting, wrist chain, shape, camera, translation, mesh topology, and evaluator remain exactly the baseline. Any geometric difference is caused by the accepted local hand block. Acting before or jointly with body fitting would allow body/wrist/camera compensation and would weaken this attribution.

`[PROPOSED]` This is the sole intervention point. There is no pre-fitting estimator replacement and no post-mesh correction.

---

## 9. Proposed state-selective uncertainty mechanism

### 9.1 Mechanism: a counterfactual evidence lens

`[PROPOSED]` CLRA-Dex queries the same frozen HandFlow sequence twice with the same initial Gaussian noise:

1. a **factual response**, using the unchanged released confidence sequence \(m_s\);
2. a **target-masked counterfactual response**, setting only the target side–frame's cmask value to zero while keeping RGB/features, keypoints, intrinsics, all other confidences, checkpoint, normalization, and noise identical.

`[VERIFIED]` The release directly treats the supplied \((B,T)\) confidence tensor as \(m_t\) in Eq. 5, so zeroing it invokes the learned mask token rather than deleting or fabricating an observation (`ConditionBuilder.forward`, lines 29–66). `[PROPOSED]` The common noise is a control variable: differences between the two outputs are attributed to the confidence intervention, not to a different random draw. The draw is predeclared and never selected by SGNify GT.

`[PROPOSED]` “Counterfactual” here means a counterfactual of the **frozen model's conditioning input**. It is not asserted to be a causal counterfactual of the physical scene, a calibrated uncertainty estimate, or a second independent observation.

`[PROPOSED]` The two outputs form an **evidence lens**, not two independent measurements and not a calibrated posterior. A candidate SignHPoser state must move no farther from either response than the DexAvatar baseline and must preserve the released local fitting evidence. If no common improvement exists, the method abstains.

`[PROPOSED]` The temporal information added beyond the target frame's HaMeR evidence is precisely the remaining full-window image/skeleton condition and the frozen learned sequence distribution: in the target-masked response, HandFlow cannot consume that target's image or skeleton token because both are replaced by the learned mask token. This does not make neighboring HaMeR evidence statistically independent; it only isolates whether the frozen temporal model supports the same local state when the disputed target observation is withheld.

### 9.2 State selection without a confidence threshold

`[PROPOSED]` CLRA-Dex introduces no rule of the form \(m_{t,s}<\tau\). Every metric-eligible side–frame with valid identity can be tested; the accepted action is determined by Pareto feasibility in §10. Continuous \(m\) remains inside HandFlow's released cmask, and joint-level \(q\) remains inside the fixed robust reprojection evidence. Thus confidence changes the evidence presented to the arbitration but is not used to linearly blend rotations or to select a hard cutoff.

`[VERIFIED]` `configs/inference.yaml` names `conf_threshold: 0.1`, but the inspected default demo inference path passes the continuous `hamer_confidence` tensor directly and does not consume that configuration field (`scripts/demo.py`, lines 186–212; `ConditionBuilder.forward`, lines 29–66). `[PROPOSED]` CLRA-Dex does not rely on the unused named threshold.

`[PROPOSED]` The action variable has exactly three semantic outcomes:

- \(a_{t,s}=D\): evidence is valid but no certified common improvement exists; retain DexAvatar;
- \(a_{t,s}=H\): admit the SignHPoser-manifold projection supported by both HandFlow responses;
- \(a_{t,s}=\varnothing\): identity/interface/context is invalid; refuse inference and retain DexAvatar while recording abstention.

The output for \(D\) and \(\varnothing\) is deliberately identical; their diagnostic meanings differ.

### 9.3 Missing spans

`[PROPOSED]` Missing observation means \(v^{obs}_{t,s}=0\), not missing physical identity. Such a frame may be considered only if \(v^{id}_{t,s}=1\) and the 16-frame HandFlow window contains at least one unambiguous observation of the same physical side outside the missing span on both temporal sides. The factual HandFlow confidence is already zero there, so factual and target-masked responses may coincide; the temporal output then acts as a single frozen support, still constrained by SignHPoser and baseline evidence.

`[PROPOSED]` If the entire window lacks same-side context, if the missing interval reaches an unanchored sequence boundary, or if side identity through the interval is ambiguous, \(a_{t,s}=\varnothing\). The method does not extrapolate a side identity or pose mode.

### 9.4 Why this is not ordinary smoothing or confidence fusion

`[PROPOSED]` No Dex/HandFlow rotation is linearly interpolated; no axis-angle or matrix average is formed; no temporal finite-difference term is the decision mechanism. The HandFlow trajectory supplies context, but acceptance is a counterfactual common-descent certificate on \(SO(3)\). A visually smoother state that fails the fixed-evidence and dual-support inequalities is rejected.

---

## 10. Mathematical formulation

### 10.1 Notation and fixed state

Let \(t\in\mathcal T_E\), \(s\in\{L,R\}\), and \(j\in\{1,\ldots,15\}\). Let

\[
\Xi_t^D=\{R^{body,D}_t,R^{wrist,D}_{t,L},R^{wrist,D}_{t,R},\beta_t^D,\psi_t^D,\Pi_t^D,\gamma_t^D\}
\]

collect the frozen body/global rotations, wrists, SMPL-X shape, expression, camera, and translation. `[PROPOSED]` Every element of \(\Xi_t^D\) is held equal to DexAvatar.

`[VERIFIED]` DexAvatar's frozen decoder \(D_H:\mathbb R^{23}\rightarrow\mathbb R^{45}\) returns axis-angle values. `[PROPOSED]` Define the valid local SMPL-X rotation block generated from latent \(z\) as

\[
R_{t,s,j}(z)=\operatorname{Exp}\!\left(\left[D_H(z_{t,s})\right]_{j}^{\wedge}\right)\in SO(3),
\qquad
R^D_{t,s,j}=R_{t,s,j}(z^D_{t,s}),
\]

where \((\cdot)^\wedge\) maps a 3-vector to \(\mathfrak{so}(3)\). This is a representation of the decoder output, not an average in axis-angle space.

### 10.2 Frozen HandFlow responses

Let \(F_\phi(O_s,m_s;\epsilon_s)\) be frozen HandFlow with observations \(O_s\), confidence sequence \(m_s\), and one predeclared initial noise draw \(\epsilon_s\). Let \(P_f\) discard the first three pose values and all non-pose outputs. `[PROPOSED]` For target \((t,s)\), define

\[
\begin{aligned}
H^+_{t,s,1:15}
&=\mathcal C_s\!\left(P_f F_\phi(O_s,m_s;\epsilon_s)\right)_t,\\
H^-_{t,s,1:15}
&=\mathcal C_s\!\left(P_f F_\phi(O_s,m_s^{\setminus t};\epsilon_s)\right)_t,
\end{aligned}
\]

with \(m^{\setminus t}_{u,s}=m_{u,s}\) for \(u\neq t\) and \(m^{\setminus t}_{t,s}=0\). Each \(H^{\pm}_{t,s,j}\in SO(3)\). \(\mathcal C_s\) is the conditional interface in §12; it may permute joints and transform handed local frames, but its exact operation is not guessed here.

`[PROPOSED]` There is no hypothesis index \(k\). Although stochastic inference is verified, CLRA-Dex does not sample, average, rank, or select multiple noises.

### 10.3 Rotation geometry

`[PROPOSED]` Rotation discrepancy is the intrinsic angle

\[
d_{SO(3)}(A,B)
=\cos^{-1}\!\left(
\operatorname{clip}\!\left(\frac{\operatorname{tr}(A^\top B)-1}{2},-1,1\right)
\right)\in[0,\pi].
\]

For a latent \(z\), define factual and counterfactual local-support costs

\[
\mathcal D^{\pm}_{t,s}(z)
=\sum_{j=1}^{15} d_{SO(3)}^2\!\left(R_{t,s,j}(z),H^{\pm}_{t,s,j}\right).
\]

No vertices, axis-angle vectors, or rotation matrices are averaged.

### 10.4 Fixed-observation safety energy

`[VERIFIED]` Released DexAvatar weights a robust 2D reprojection residual with squared joint confidence and a hand latent Gaussian penalty; it also directly anchors decoded axis-angle hand pose to the HaMeR-derived initialization ([`fitting.py`, lines 520–525 and 543–588](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fitting.py#L520-L588)).

`[PROPOSED]` For arbitration only, define \(\mathcal E^{safe}_{t,s}(z;\Xi^D)\) as the released robust hand-joint reprojection term plus the released SignHPoser latent regularizer, with all of \(\Xi^D\) fixed and their released weights unchanged. The direct axis-angle initialization term is excluded for the target trial because it is the same HaMeR-derived estimate whose reliability is under intervention; treating it as independent evidence would double-count the disputed source. Outside admitted target states, the exact baseline rotation is equality-locked.

`[PROPOSED]` \(\mathcal E^{safe}\) is a consistency certificate, not a probabilistic likelihood and not a new loss contribution.

### 10.5 Counterfactual Pareto set and arbitration

For each identity-valid, metric-eligible target, let the feasible set be

\[
\begin{split}
\mathcal Z_{t,s}=\big\{z\in\mathbb R^{23}:\;&
\mathcal E^{safe}_{t,s}(z;\Xi^D)
\leq \mathcal E^{safe}_{t,s}(z^D_{t,s};\Xi^D),\\
&\mathcal D^+_{t,s}(z)\leq\mathcal D^+_{t,s}(z^D_{t,s}),\\
&\mathcal D^-_{t,s}(z)\leq\mathcal D^-_{t,s}(z^D_{t,s})\big\}.
\end{split}
\]

The baseline \(z^D_{t,s}\) is always feasible. `[PROPOSED]` The counterfactual local-state projection is

\[
z^*_{t,s}
\in\arg\min_{z\in\mathcal Z_{t,s}}
\max\{\mathcal D^+_{t,s}(z),\mathcal D^-_{t,s}(z)\}.
\]

`[PROPOSED]` Action \(a_{t,s}=H\) is admitted only if at least one of the two support inequalities is strict at \(z^*\), neither is worse, the safe-energy inequality holds, \(v^{id}_{t,s}=1\), required temporal context exists, and \(\mathcal C_s\) is validated. If the candidate is merely smoother, improves only one response while worsening the other, or worsens the fixed evidence, \(a_{t,s}=D\). If identity, context, or interface is invalid, \(a_{t,s}=\varnothing\).

This is a threshold-free partial order: it does not require a newly selected confidence cutoff or a numeric acceptance margin.

### 10.6 Final state and unified mesh

`[PROPOSED]` The final rotations are

\[
R^{\star}_{t,s,j}=
\begin{cases}
R_{t,s,j}(z^*_{t,s}), & a_{t,s}=H,\\
R^D_{t,s,j}, & a_{t,s}\in\{D,\varnothing\}.
\end{cases}
\]

The actual `left_hand_pose`/`right_hand_pose` written to SMPL-X is the decoder's 45-D axis-angle output corresponding to that selected latent; \(R^\star\) is used to define valid geometry and comparisons. `[PROPOSED]` The full parameter vector is

\[
\Theta_t^\star=
\big(\Xi_t^D,\,D_H(z^{final}_{t,L}),\,D_H(z^{final}_{t,R})\big),
\qquad
V_t^\star=M_{SMPL-X}(\Theta_t^\star),
\]

with class-`0` left state fixed to DexAvatar. This is one SMPL-X model evaluation with its original topology; there is no vertex stitch, MANO scatter, Procrustes correction, or evaluator-side transformation.

---

## 11. End-to-end information flow

`[PROPOSED]` The following table fixes the single end-to-end information path; rows explicitly attributed to released systems are reused facts rather than new contributions.

| Block | Input | Output | Source/status | New contribution? |
|---|---|---|---|---:|
| 1. Released observations | RGB sequence and existing detections/keypoints/confidences | Unchanged DexAvatar and HaMeR observations | DexAvatar/HandFlow, `FROZEN` | No |
| 2. Physical-side contract | Side-labelled detections and temporal presence | \((\iota_s,v^{obs}_{t,s},v^{id}_{t,s})\) | Dyn-HaMR-style contract, `METADATA` | No; only admissibility metadata |
| 3. Released DexAvatar fitting | Original inputs/config/prior | \(\Theta_t^D,z^D_{t,s}\) | DexAvatar, `FROZEN REFERENCE` | No |
| 4. Factual temporal response | Side stream, actual \(m_s\), fixed noise | HandFlow \(\theta^{48},\beta,\tau\) sequence | HandFlow, `FROZEN` | No |
| 5. Target-masked temporal response | Same stream/noise; target \(m_{t,s}=0\) only | Paired HandFlow response | HandFlow queried by `[PROPOSED]` intervention | Part of the single CLRA mechanism |
| 6. State exclusion | Both HandFlow outputs | finger45 only | Contract, `DISCARDED` for root/shape/trans/mesh | No |
| 7. Local-state conversion | HandFlow finger45 + physical side | \(H^+,H^-\in SO(3)^{15}\) | \(\mathcal C_s\), `CONDITIONAL` | No scientific claim; compatibility requirement |
| 8. Counterfactual arbitration | \(H^+,H^-\), \(z^D\), frozen observations and \(\Xi^D\) | \(a_{t,s}\), possibly \(z^*_{t,s}\) | `[PROPOSED]` CLRA-Dex, `OPTIMIZED` | **Yes—the sole proposed intervention** |
| 9. Unified parameter assembly | Frozen \(\Xi^D\), selected frozen-decoder hand blocks | \(\Theta_t^\star\) | `[PROPOSED]`, standard SMPL-X state | No separate contribution |
| 10. Mesh and official scoring | \(\Theta_t^\star\) | SMPL-X vertices and locked TR-V2V | SMPL-X/evaluator, `FROZEN` | No |

`[PROPOSED]` The left and right streams share frozen model weights and arbitration rules but exchange no learned state. Their only relationship is that each is bound to a persistent physical-side label and inserted into one body parameter vector.

---

## 12. SMPL-X and handedness compatibility contract

### 12.1 Formal duties of \(\mathcal C_s\)

`[CONDITIONAL]` \(\mathcal C_s:SO(3)^{15}_{HandFlow/MANO}\rightarrow SO(3)^{15}_{Dex/SMPL-X}\) is admissible only if all five duties below are closed by authoritative source or convention validation:

1. exact 15-joint permutation;
2. equality or explicit conversion of each local parent frame;
3. right-canonical-to-physical-side parity, including image unmirroring for \(s=L\);
4. equality or explicit conversion of pose-mean convention;
5. a round-trip check showing that the converted local block, with DexAvatar's unchanged wrist/body chain, produces the intended physical-side articulation in the audited neutral SMPL-X layer.

`[PROPOSED]` \(\mathcal C_s\) may not copy root orientation, translation, shape, or vertices. It may not be instantiated by reusing DexAvatar's HaMeR left y/z sign flip without proof.

### 12.2 Right and left status

- **Right:** `[CONDITIONAL]` The checkpoint is right-canonical; the 45-D local/non-PCA/flat-mean structure is compatible in dimension and broad semantics. Exact permutation/local frames remain unclosed.
- **Left:** `[CONDITIONAL]` README mirroring and demo `--side left` behavior conflict. No axis sign, conjugation, inverse mirror, or pose permutation is specified here.

### 12.3 Physical-side identity

`[VERIFIED]` Dyn-HaMR represents handedness as \(h\in\{l,r\}\), keeps side-specific tracks, and stores per-track visibility/side metadata; its released dataset path asserts side constancy across a track ([paper §3.1](https://arxiv.org/html/2412.12861v3#S3.SS1); [`dataset.py`, lines 213–305](https://github.com/ZhengdiYu/Dyn-HaMR/blob/fa9cd7412c205fd15ee4139c8caacf79bf6167e6/dyn-hamr/data/dataset.py#L213-L305)). `[CONDITIONAL]` The exact tracker/crossing policy behind the release was not available at the Step 3 commit.

`[PROPOSED]` CLRA-Dex therefore uses only this contract:

- a track starts at an unambiguous detector-side observation;
- its physical label does not change through missing frames;
- overlap/crossing with conflicting side evidence sets \(v^{id}=0\), not a guessed swap;
- the method refuses updates until an unambiguous same-side track is reacquired;
- missing observation and missing identity are separate variables;
- class-`0` masking is applied exactly as the evaluator already defines it.

No claim is made that Dyn-HaMR's released tracker solves crossings.

---

## 13. Training/inference regime

### `TRAINING-FREE / OPTIMIZATION-BASED`

`[PROPOSED]` HandFlow, its HaMeR frontend, normalization statistics, and SignHPoser decoder are frozen. CLRA-Dex introduces no learned parameter, quality network, calibration network, retraining, fine-tuning, or SGNify-derived supervision.

`[PROPOSED]` The only new optimization variable is \(z^*_{t,s}\) for a candidate side–frame. The optimization uses released Dex observation/prior evidence and two frozen HandFlow responses. No SGNify test GT, mesh, TR-V2V value, class-conditioned error, or test-derived threshold participates in inference or selection.

`[PROPOSED]` Stochastic inference is controlled by one predeclared common noise draw per physical-side sequence. The draw is reused for factual and counterfactual queries and never selected from multiple runs. This avoids both best-of-\(K\) and an unsupported posterior-calibration claim. `[VERIFIED]` HandFlow Appendix D reports extremely small pose variation across ten seeds, which further weakens—not strengthens—the case for sample-dispersion uncertainty.

`[CONDITIONAL]` Feasibility depends on validating \(\mathcal C_s\) and on pinning the exact checkpoint/normalization bytes already identified. It does not depend on unavailable HandFlow training code.

---

## 14. Novelty delta against nearest prior works

All cells describing CLRA-Dex are `[PROPOSED]`. “Novelty” below is a targeted technical distinction, not a certification of publication novelty.

| Prior work | Component already exists | Reused here | Proposed component | Scientific difference | Independently ablatable? | Engineering-integration risk | Falsification of the claimed delta |
|---|---|---|---|---|---:|---|---|
| **DexAvatar** | Per-frame SignHPoser-latent fitting, robust 2D evidence, direct HaMeR/SMPLer-X initialization, unified SMPL-X output | Frozen baseline state, decoder, local evidence, final model | `[PROPOSED]` Common-noise factual/masked arbitration and Pareto-constrained local projection | Replaces unreliable direct AA anchoring only at admitted states with an interventional support test on \(SO(3)\); all non-hand state is locked | Yes | Medium: could look like a loss wrapper | Fails if the mechanism cannot outperform or behave differently from unchanged Dex under the locked state boundary |
| **HandFlow** | Full-window generative MANO inference; frame-scalar cmask inside the denoiser; random training masks; overlap inference | Frozen factual and masked responses; finger45 only | `[PROPOSED]` Post-denoiser counterfactual **comparison**, manifold projection, and abstention | cmask changes the condition for one generative pass and jointly predicts root/pose/shape/trans; CLRA compares two controlled passes, acts only on final local SMPL-X finger state, and never modifies the denoiser | Yes: factual HandFlow local replacement is the ablated baseline | High: without counterfactual arbitration it is only estimator replacement | Fails if factual-only replacement is equivalent to the full method or if masking changes only discarded translation/root state |
| **HMP** | Right-canonical learned hand motion prior and test-time latent fitting/infilling | Nothing | `[PROPOSED]` Frozen image-conditioned HandFlow support projected through DexAvatar's already-used SignHPoser manifold | No HMP latent, AMASS prior, direct trajectory optimization, or reflected HMP stream; uncertainty is diagnosed by paired condition intervention | Yes | Low–medium | Fails if the method reduces mathematically to ordinary prior fitting without a distinct counterfactual decision |
| **Dyn-HaMR** | Side-indexed tracks, missingness, per-hand HMP latent fitting, SLAM/world optimization, interaction/biomechanical terms | Side/validity metadata contract only | `[PROPOSED]` Local-only counterfactual arbitration after a unified-body fit | No camera/world/root/shape/HMP/penetration state; two streams are not called a joint learned prior | Yes | Medium because per-side optimization is already prior art | Fails if the only effective part is the reused track mask or ordinary per-hand prior fitting |
| **HaPTIC** | Deterministic temporal attention over short clips; right-canonical side processing; trajectory emphasis | Nothing | `[PROPOSED]` Frozen generative paired-condition query plus abstaining SMPL-X local projection | No temporal backbone training, trajectory transfer, crop-to-world claim, or root change | Yes | Low | Fails if the intervention contributes only smoother/world motion and no centered local geometry change |
| **Hand4Whole++** | Body-conditioned wrist prediction (CHAM) and aligned MANO vertex insertion/smoothing | Nothing | `[PROPOSED]` Wrist/body remain immutable; only SignHPoser-decoded SMPL-X local rotations may change | Opposite state boundary: no wrist modulation and no MANO vertex scatter; one regenerable SMPL-X vector is mandatory | Yes | Low | Fails if final vertices cannot be regenerated solely from \(\Theta^\star\) or any wrist/body state changes |
| **StableHand** (2026 preprint; adversarial addendum) | Learned four-channel wrist/finger, left/right quality network and retrained bimanual quality-aware flow process; world-space state | Nothing | `[PROPOSED]` Training-free paired confidence intervention on frozen single-hand HandFlow, followed by local-only SMPL-X Pareto arbitration | StableHand learns GT-derived quality and modifies flow training/ODE for coupled world-space hands; CLRA learns nothing, changes no flow weights, excludes wrist/world state, and uses counterfactual consistency rather than predicted quality | Yes | **High conceptual proximity** on state-selective quality | Fails if paired masking is merely an inferior surrogate for existing quality conditioning or offers no distinct behavior beyond frozen cmask |

Primary evidence: HandFlow §§3–4 and Appendix D; HMP §§3–4 ([official paper](https://arxiv.org/html/2312.16737v1)); Dyn-HaMR §§3.1–3.3; HaPTIC §§3.3–3.4 ([paper](https://arxiv.org/html/2501.08329v1)); Hand4Whole++ §3 and released `combine_smplx_mano` ([paper](https://arxiv.org/html/2603.14726v1); [`main/model.py`, lines 42–126](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE/blob/f81d35ddd2b74206c40142243eb62b6d64ce0d65/main/model.py#L42-L126)); StableHand pp. 1–4 ([preprint](https://arxiv.org/pdf/2605.18553), [official repository](https://github.com/huajian-zeng/stablehand)). `[VERIFIED]` StableHand's official repository contained only project assets/README and stated that code would be released after acceptance on the access date; it is frontier evidence, not a public substrate.

### Required six-way novelty questions

1. **Beyond HandFlow cmask:** `[PROPOSED]` CLRA holds noise fixed, intervenes on one target's condition, compares both outputs after the frozen denoiser, projects only local SMPL-X state, and can abstain. Cmask alone neither performs that comparison nor enforces the SMPL-X state boundary.
2. **Beyond HMP infilling:** `[PROPOSED]` no HMP model or latent is used; image-conditioned temporal support comes from frozen HandFlow and is accepted only through a counterfactual common-descent condition.
3. **Beyond Dyn-HaMR:** `[PROPOSED]` side identity is metadata only; there is no world/camera/root/shape or coupled interaction optimization.
4. **Beyond confidence weighting/smoothing:** `[PROPOSED]` confidence is not a rotation weight; the output is a discrete retain/admit/refuse decision with intrinsic-rotation Pareto constraints.
5. **Metric-preserved state only:** `[PROPOSED]` candidate root, translation, camera, shape, and mesh are discarded; only decoded local fingers change.
6. **Ablated baseline:** `[PROPOSED]` removing the counterfactual query, dual-support constraints, and abstaining arbitration yields **frozen HandFlow factual finger45 replacement/projected into the Dex hand block with all other state fixed**. That is an engineering baseline, not CLRA-Dex.

---

## 15. Component-to-contribution mapping

`[PROPOSED]` The contribution assignments and removal interpretations in this table are part of the candidate's falsifiability contract.

| Component | Status | Necessary role | Claimed contribution? | What its removal means |
|---|---|---|---:|---|
| Released DexAvatar | `[VERIFIED]`, reused | Baseline, observations, SignHPoser manifold, SMPL-X output | No | No valid comparator/system boundary |
| Frozen HandFlow factual pass | `[VERIFIED]`, reused | Temporal local candidate | No | No selected technical base |
| Frozen HandFlow target-masked pass with common noise | `[PROPOSED]` | Measures response to removing the disputed frame condition | Part of the **single** CLRA mechanism | Method collapses toward factual HandFlow replacement |
| \(SO(3)\) dual-support/Pareto arbitration | `[PROPOSED]` | Admits only common improvement and supplies refusal | Part of the **single** CLRA mechanism | No state-selective interventional decision; only replacement/regularization remains |
| Frozen SignHPoser projection | `[VERIFIED]` decoder; `[PROPOSED]` use as projection variable | Keeps output in DexAvatar-compatible decoded hand state | No standalone novelty | Direct candidate insertion would leave the audited Dex manifold boundary |
| Dyn-HaMR-style side/validity contract | `[VERIFIED]` concept, reused as metadata | Prevents silent side switching | No | Side ambiguity becomes an uncontrolled confound |
| \(\mathcal C_s\) | `[CONDITIONAL]` interface | Ensures hand order/frame/parity/mean compatibility | No | Method is blocked for that side |
| Refusal/fallback | `[PROPOSED]` consequence of Pareto/identity constraints | Prevents unsupported overwrite | Not a separate module | Unsafe forced update |
| Single SMPL-X forward | `[VERIFIED]` representation requirement | Produces evaluator-compatible mesh | No | Violates locked topology/parameterization |

`[INFERENCE]` The only scientific contribution candidate is the **counterfactual local-rotation arbitration as one indivisible mechanism**. HandFlow replacement, side tracking, SignHPoser decoding, and SMPL-X assembly are substrates or contracts.

---

## 16. Internal causal and compatibility audit

`[PROPOSED]` PASS/CONDITIONAL judgments below audit the formulated candidate; cited implementation facts remain `[VERIFIED]`.

| Audit question | Status | Evidence/reasoning |
|---|---:|---|
| What is the system when only HandFlow replaces HaMeR? | `PASS` | `[PROPOSED]` It is the explicit ablated baseline: frozen factual HandFlow finger45 local replacement/projection, with all other Dex state fixed. It is not called CLRA-Dex. |
| Can the proposed mechanism be ablated independently? | `PASS` | Remove the masked pass and Pareto/refusal rule while keeping the factual pass, \(\mathcal C_s\), decoder, frozen state, and evaluator. |
| Can wrist, body, shape, camera, or translation change accidentally? | `PASS` by formulation | They are equality-fixed in \(\Xi^D\); HandFlow non-local outputs are discarded before optimization. |
| Are the same HaMeR signals counted twice as independent evidence? | `CONDITIONAL` | HandFlow and Dex inputs are correlated. `[PROPOSED]` The two HandFlow responses are explicitly treated as paired interventions, not independent likelihoods; the direct target AA initialization term is excluded during an admitted trial. The fixed 2D term remains a consistency constraint and may still share upstream information. |
| Could visual smoothness improve while centered hand geometry does not? | `CONDITIONAL` | Yes. The evaluator has no temporal term. The claim is falsified if only smoothness/world motion changes; local rotations must change centered vertices beneficially. |
| Low confidence but correct DexAvatar state | `PASS` for safe behavior, not accuracy | `[PROPOSED]` No update occurs unless a non-worsening common support state exists; the baseline remains feasible. There is no guarantee numerical optimization finds the global certificate. |
| High confidence but wrong pose/handedness mode | `CONDITIONAL` | The paired mask may expose condition dependence, but high-confidence correlated evidence can also block correction. Identity ambiguity forces refusal; a coherent wrong mode can survive all constraints. |
| Can temporal processing hide a side swap? | `CONDITIONAL` | `[PROPOSED]` Ambiguous crossing sets \(v^{id}=0\) and refuses. This depends on an externally valid side contract; the full released Dyn tracker/crossing behavior was not verified. |
| Are final vertices entirely generated from SMPL-X parameters? | `PASS` | Final hand pose comes from SignHPoser into `left/right_hand_pose`; one SMPL-X pass creates the mesh. No MANO vertices enter. |
| Does the method need a new label or unavailable training artifact? | `PASS` | It is training-free and uses released baseline/checkpoint inputs. Exact \(\mathcal C_s\) documentation remains a compatibility unknown, not a learned artifact. |
| Does the method claim a calibrated HandFlow posterior or useful sample diversity? | `PASS` | No. One common draw is used; Appendix D's low seed sensitivity is disclosed. |
| Are class-`0` semantics preserved? | `PASS` | Left is unchanged/excluded exactly as audited; no frame or vertex mask changes. |
| Does every allowed change have official TR-V2V leverage? | `PASS` | Local finger rotations deform evaluated hand vertices after centering. Pure translation, face, and candidate shape/root states are excluded. |
| Is novelty merely “sign-language application”? | `PASS` at formulation level | The proposed delta is paired counterfactual arbitration and Pareto projection under a unified-state boundary; dataset domain is not the novelty argument. Publication novelty remains `[UNRESOLVED]`. |

---

## 17. Known failure modes and refusal/fallback behavior

`[PROPOSED]` Every behavior in the CLRA-Dex column is normative method behavior, not a released-system fact.

| Condition | What is known | `[PROPOSED]` CLRA-Dex behavior | Residual scientific risk |
|---|---|---|---|
| \(\mathcal C_L\) or \(\mathcal C_R\) unvalidated | Equal dimension does not prove convention equivalence | Refuse all updates for the affected side; output DexAvatar | Method remains conditional and may have no two-side coverage |
| Detector-side conflict or hand crossing | HandFlow release can veto overlapping opposite-side boxes; exact persistent tracker policy is unverified | Set \(v^{id}=0\), retain baseline, do not smooth across the ambiguity | Long refusals can eliminate the intended benefit |
| Missing target with valid context on both sides | HandFlow was trained with random masks and whole-window context | Permit arbitration; if factual/counterfactual coincide, use that temporal support under safe constraints | Long sign-specific missing motion may be outside the learned prior |
| Entire 16-frame window missing or unanchored edge span | HandFlow training discarded windows with fewer than 50% valid frames (paper Appendix B) | Refuse; no temporal hallucination is installed | Dex baseline may itself be poor, but unsupported overwrite is prohibited |
| Factual and counterfactual responses point to incompatible pose modes | Monocular ambiguity is real; no calibrated posterior is available | If no common Pareto improvement exists, retain baseline | A valid alternative may be rejected |
| Both responses agree on the same wrong mode | Shared frozen model and shared evidence can correlate errors | The certificate can admit a wrong state; there is no internal detector for this case | Core hypothesis may be falsified by sign-domain prior mismatch |
| Low confidence but baseline is correct | Detector confidence is not calibrated local-pose correctness | Safe-energy and dual-support non-worsening conditions preserve baseline unless certified | Tiny numerical “improvement” remains an optimization concern, not resolved here |
| High confidence but wrong local pose | Confidence can be overconfident | No forced update; counterfactual conflict may cause retention/refusal | Method is conservative and may miss correctable errors |
| Sign motion outside HandFlow's object-manipulation training distribution | HandFlow trained on DexYCB/HOT3D, not sign data | No adaptation; reject when no common safe projection exists | Factual and masked outputs may be jointly biased |
| SignHPoser manifold cannot represent HandFlow-supported state | Decoder is frozen and lower-dimensional | No strict feasible improvement implies baseline fallback | Manifold projection can suppress useful candidate articulation |
| Frame absent from DexAvatar/evaluator pairing | Locked evaluator cannot be altered | Do not create or insert a scored frame | Missing-frame coverage is not a contribution |
| Only trajectory/smoothness improves | Those effects are not direct centered local geometry gains | Non-local HandFlow outputs are discarded; claim is considered falsified if local metrics do not improve | Visual quality can still mislead qualitative inspection |

No fallback changes the evaluator, frame set, vertex region, centering, alignment, aggregation, camera, wrist, or topology.

---

## 18. Remaining assumptions and blockers

1. **Exact right conversion.** `[CONDITIONAL]` A definitive HandFlow/MANO-to-Dex/SMPL-X 15-joint order and local-parent-frame mapping is still required for \(\mathcal C_R\).
2. **Exact left conversion.** `[CONDITIONAL]` Author documentation or convention validation must reconcile README mirroring, `--side left`, native-image crops, and direct MANO-left FK. No sign pattern is inferred here.
3. **Pose-mean/runtime scope.** `[CONDITIONAL]` The audited neutral Dex model and HandFlow FK are flat-mean/no-PCA; other Dex gender-model branches require a separate contract.
4. **Checkpoint provenance.** `[CONDITIONAL]` Artifact bytes are pinned, but the minimal Hugging Face model card does not bind them to a detailed paper split/frontend/training manifest.
5. **Common-noise query interface.** `[INFERENCE]` Paper and release establish seeded random initialization and arbitrary confidence tensors. The public function does not expose an explicit initial-noise argument; exact common-random-number control is an inference-wrapper contract, not a weight change.
6. **Physical track continuity.** `[CONDITIONAL]` Side-indexed metadata exists, but the exact released Dyn-HaMR tracker and crossing policy at the audited commit were not inspectable. CLRA's refusal semantics are proposed, not attributed to Dyn-HaMR.
7. **Local-articulation sensitivity.** `[UNRESOLVED]` HandFlow's confidence ablation mainly supports translation/global coherence; it does not prove that target masking alters correct finger articulation under sign blur/occlusion.
8. **Domain transfer.** `[UNRESOLVED]` DexYCB/HOT3D hand-object dynamics may not support sign-language articulations; no outcome is assumed.
9. **Eligible-regime prevalence.** `[UNRESOLVED]` The frequency and metric mass of missing/low-confidence but temporally anchored states must be established without SGNify test GT.
10. **Correlated observation sources.** `[CONDITIONAL]` HaMeR-derived HandFlow conditions and Dex hand evidence are not independent. The method avoids a false likelihood product, but residual correlation can still make both support responses wrong.
11. **Optimization attainability.** `[UNRESOLVED]` The formulation defines a Pareto certificate on the nonlinear SignHPoser manifold; static inspection cannot establish that nontrivial feasible improvements exist or are practically reachable.
12. **Contemporary novelty pressure.** `[VERIFIED]` StableHand already proposes learned per-hand/per-component quality-aware bimanual flow matching. `[INFERENCE]` CLRA remains technically distinct because it is training-free, counterfactual, frozen-model, local-SMPL-X-only, and abstaining; whether that delta is publishably novel must be adversarially reviewed.

None of these unknowns is hidden by a guessed conversion, new training claim, or evaluator change. Items 1–2 are the reason for the conditional verdict.

---

## 19. Frozen method specification for Step 5

The following is the only method candidate passed forward. Changing any bold item reopens Step 4.

`[PROPOSED]` Every specification in the following table is frozen for adversarial review.

| Decision | Frozen specification |
|---|---|
| Candidate | `[PROPOSED]` **CLRA-Dex** only; no alternative branch |
| Verdict | `CONDITIONAL METHOD CANDIDATE` |
| Scientific hypothesis | Step 3 reduced PRIMARY, unchanged |
| Regime | **`TRAINING-FREE / OPTIMIZATION-BASED`** |
| Intervention point | **After DexAvatar fitting, before final SMPL-X forward output** |
| Technical base | **Frozen HandFlow**, exact checkpoint and normalization hashes in §§2/20 |
| Optional auxiliary | **Physical-side identity and validity contract only**; no Dyn-HaMR pose, SLAM, HMP, root, shape, camera, mesh, biomechanics, or optimizer |
| HandFlow state admitted | **finger45 candidate information only** |
| HandFlow state rejected | root3, \(\beta\), translation, camera/world trajectory, vertices/faces |
| New optimized variable | **23-D DexAvatar SignHPoser latent \(z_{t,s}\)** for an admitted side–frame only |
| Frozen variables | Body/global pose, shoulder–elbow–wrist chain, wrist/root orientation, shape, expression, camera, translation, upstream observations, all network weights |
| Novel mechanism | **One common-noise factual/target-masked HandFlow pair + \(SO(3)\) dual-support Pareto arbitration + refusal**, treated as one mechanism |
| Rotation operation | Intrinsic geodesic comparison; no AA/matrix/vertex averaging |
| Compatibility | Conditional symbolic \(\mathcal C_s\); no guessed left parity/order/mean |
| Uncertainty semantics | Interventional response consistency; **not** calibrated posterior and **not** sample dispersion |
| Multiple samples | Excluded; one predeclared common draw, no best-of-\(K\) |
| Safe evidence | Existing robust 2D hand reprojection + latent prior with fixed states/weights; disputed direct AA target excluded only for the target trial |
| Acceptance | Non-worsening on fixed evidence and both HandFlow supports, with at least one strict support improvement |
| Refusal/fallback | Any invalid identity/interface/context or failed certificate returns exact DexAvatar state |
| Final representation | One unified SMPL-X parameter vector and one SMPL-X mesh forward pass |
| Metric | Official evaluator locked; no GT optimization, alignment change, frame change, class change, region change, or aggregation change |
| Class-`0` | Left state unchanged; original evaluator semantics preserved |
| Ablated integration baseline | Frozen factual HandFlow finger45 replacement/projection with all other state fixed |
| Falsification | No centered local-hand/retained-hand-vertex gain over DexAvatar in the stated unreliable-evidence regime, or no gain beyond factual-only replacement, falsifies the intervention claim even if motion looks smoother |

`[PROPOSED]` Step 5 may adversarially challenge compatibility, novelty, causal isolation, and falsifiability. It must not silently substitute StableHand, HMP, a new tracker, a trainable quality model, wrist correction, mesh fusion, multiple-sample selection, or an evaluator modification into this candidate.

---

## 20. Primary-source manifest

### 20.1 Supplied artifacts and integrity

All were read in full unless a range is stated.

| Artifact | Inspection record |
|---|---|
| `DexAvatar_Baseline_and_TR-V2V_Evaluation_Dossier.md` | SHA-256 `715e36be9bf892386f78fa2833c981b4e6485b74cca4a07afa685e8d179d44b5`; full, 804 lines |
| `DexAvatar_Step2_Bottleneck_Prioritization_and_Targeted_Literature_Review.md` | SHA-256 `f80d840fd5b4d1595bdbd9e67fffac795571bb8e82a74119aefced8bc6f58a9f`; full, 442 lines |
| `DexAvatar_Step3_Feasibility_Compatibility_and_Novelty_Gate.md` | SHA-256 `f578280f4c9668a9d6b2ac30b985d40e849aecdb9e3fcbcf1c3343a966fdae84`; full, 328 lines |
| Step 4 specification, `Đã dán markdown (1)(2).md` | SHA-256 `b4e3c27a5663b4d549127caf523c75bb7deeaf19d3509d3884e29c42a9ecbcb1`; full |
| DexAvatar PDF, Kundu et al. 2025 | Main and supplementary, PDF pages 1–21; read in full in Step 1 and claims rechecked through the exact-page dossier manifest |
| `evaluate_new_fitting(2).py` | Static inspection only: `transl_point_error` lines 159–169; mesh/topology lines 356–370; class/region rules 380–395; aggregation 432–461; CLI/assets 479–567. Not executed. |

### 20.2 Repositories inspected

Access date for every repository: **2026-08-26**.

| Repository | Branch / commit | Files used in Step 4 |
|---|---|---|
| [DexAvatar](https://github.com/kaustesseract/DexAvatar) | `main` / `a0dfd427f60f5811aadb35c8657b3856d47f56b5` | `README.md` full via Step 1; `dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml` full; `smplifyx/main.py` L120–165; `data_parser.py` L390–430; `fit_single_frame.py` L220–250, 443–570, 620–655; `fitting.py` L250–290, 510–600; `test_hposer.py` full. The broader full/partial source inventory is in Step 1 §12. |
| [HandFlow](https://github.com/mxxu00/HandFlow) | `main` / `67fa7df536db233408fe6270ca5d2de28d5959c3` | `README.md` full; `configs/model.yaml` and `configs/inference.yaml` full; `model/feature_extractors/condition_builder.py` full; `model/flow_matching/denoiser.py` relevant normalization/state path; `utils/inference_utils.py` L180–325; `utils/mano_utils.py` L1–125; `utils/online_hamer.py` L190–320 and output assembly; `utils/dual_conf.py` full; `scripts/demo.py` L120–225. |
| [Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR) | `main` / `fa9cd7412c205fd15ee4139c8caacf79bf6167e6` | `README.md` full; `dyn-hamr/confs/config.yaml` full; `dyn-hamr/data/dataset.py` L205–315; tracking submodule absent/not inspected. |
| [HMP](https://github.com/enesduran/HMP) | `main` / `35d799f76b2b2bc1d1e945117b021014b099e7e6` | `README.md` and license full; `src/datasets/amass.py` L85–150; documented training/fitting paths. |
| [HaPTIC](https://github.com/JudyYe/haptic) | `main` / `f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df` | `README.md` full; `haptic/datasets/seq2clip.py` L85–235; `demo.py` L342–394; root license not found at audited state. |
| [Hand4Whole++](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE) | `main` / `f81d35ddd2b74206c40142243eb62b6d64ce0d65` | `README.md` and license full; `main/model.py` L35–135 and L155–230; `common/nets/wilor.py` L79–124. |
| [StableHand](https://github.com/huajian-zeng/stablehand) | `main`, one-commit project repository as displayed on access date; exact SHA not exposed by the indexed page | README/project assets only. README line 175 says code will be released after acceptance. No implementation/checkpoint was available. |
| [Official SMPL-X Python package](https://github.com/vchoutas/smplx) | current public source inspected via official GitHub page | `smplx/body_models.py` documentation/source for left/right hand pose, non-PCA axis-angle/matrix interfaces, and flat-hand-mean semantics; exact HandFlow cross-model mapping still unresolved. |

### 20.3 Papers and supplementary

1. **Kundu et al.**, *DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors*, 2025. Supplied PDF pages 1–21, including supplementary; all sections/tables/equations inspected in Step 1.
2. **Mingxi Xu, Bowen Duan, Yi Gu, Zhengyang Shen, Renjing Xu, Yutao Yue**, [*HandFlow: Fully Generative 4D Hand Recovery with Flow Matching*](https://arxiv.org/html/2607.11221v1), arXiv:2607.11221v1, 2026. Main §§1–5, Tables 1–4, Appendices A–G; especially §3, Eq. 1–9, Appendix B (T/training/frontend), Appendix D (noise sensitivity).
3. **Zhengdi Yu, Stefanos Zafeiriou, Tolga Birdal**, [*Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera*](https://arxiv.org/html/2412.12861v3), CVPR 2025. Main §§3.1–3.3 and supplementary/appendices.
4. **Enes Duran, Muhammed Kocabas, Vasileios Choutas, Zicong Fan, Michael J. Black**, [*HMP: Hand Motion Priors for Pose and Shape Estimation from Video*](https://arxiv.org/html/2312.16737v1), WACV 2024. Main and supplementary, representation, training, latent optimization, left reflection.
5. **Yufei Ye, Yao Feng, Omid Taheri, Haiwen Feng, Shubham Tulsiani, Michael J. Black**, [*Predicting 4D Hand Trajectory from Monocular Videos*](https://arxiv.org/html/2501.08329v1), HaPTIC, arXiv 2025 / publication metadata rechecked in Step 3. Main §§3–4 and appendices.
6. **Gyeongsik Moon**, [*Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator*](https://arxiv.org/html/2603.14726v1), Hand4Whole++, CVPR 2026. Main §§3–5, Table 2, supplementary, and official code mesh path.
7. **Huajian Zeng, Chaohua Yao, Yuantai Zhang, Jiaqi Yang, Rolandos Alexandros Potamias, Xingxing Zuo**, [*StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video*](https://arxiv.org/pdf/2605.18553), arXiv:2605.18553v1, 2026. Pages 1–4 and method/quality sections inspected as an adversarial novelty addendum; [official project page](https://huajian-zeng.github.io/projects/stablehand/).

### 20.4 Checkpoint manifest

| Artifact | Official location | Immutable identity |
|---|---|---|
| HandFlow denoiser | [Hugging Face commit](https://huggingface.co/mxxu00/HandFlow/commit/3ca50e4afececc8a7bc361b74954c77307bd0a5f) | commit `3ca50e4afececc8a7bc361b74954c77307bd0a5f`; SHA-256 `2fbc4e1fa7a60f469a6ac94933a6e6dc8a86a0e1fc13bd7cd81c430c79acfcda`; 667,907,131 bytes |
| HandFlow normalization | [Hugging Face artifact](https://huggingface.co/mxxu00/HandFlow/blob/fc35519962867acdf834ccef13b9a2814cbbd15d/normalization_stats.npz) | commit `fc35519962867acdf834ccef13b9a2814cbbd15d`; SHA-256 `7313334e6b9537fa57ec9763e83f36dfd4998e7d1de09aa3fb21c5bfca8e92c4`; 2,008 bytes |

### 20.5 Not inspected / not executed

- SGNify test GT: **NOT INSPECTED** and unused.
- Every repository, checkpoint, dataset, optimization, mesh export, and evaluator: **NOT EXECUTED**.
- HandFlow complete training and dataset-construction code: **NOT AVAILABLE / NOT INSPECTED** at the audited commit.
- HandFlow left conversion author clarification: **NOT AVAILABLE**.
- Exact HandFlow 15-joint-to-Dex SMPL-X conversion table: **NOT AVAILABLE**.
- Dyn-HaMR tracking submodule/crossing policy at the pinned Step 3 state: **NOT INSPECTED**.
- StableHand code/checkpoint: **NOT RELEASED** at access date.
- No number in this dossier is a reproduced score. No score improvement is predicted.

### 20.6 Final quality check

- Exactly one method candidate: **yes**.
- New mechanism beyond HandFlow replacement: **yes—paired counterfactual local-state arbitration**.
- HandFlow retraining/fine-tuning: **no**.
- New code, pseudocode, implementation plan, experiment protocol, or hyperparameter sweep: **none**.
- Direct AA/matrix/vertex averaging: **none**.
- MANO root/shape/translation/mesh transfer: **none**.
- Wrist/body/camera/translation change: **none**.
- Learned joint-bimanual claim: **none**.
- SGNify test-GT use or evaluator exploit: **none**.
- Exact left mapping guessed: **no; explicitly conditional**.
- Reported result called reproduced: **no**.
- Contemporary nearest-prior risk hidden: **no; StableHand is explicitly audited**.

STEP 4 COMPLETE — CONDITIONAL METHOD CANDIDATE FORMULATED; INTERFACE RISKS MUST BE RED-TEAMED BEFORE ACCEPTANCE.
