# DexAvatar3D Research Analysis from Recent Sign-Language Production Papers

**Goal.** This note analyzes how three recent sign-language production papers can be used to improve **DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors** and to define a new research-paper direction.

Papers reviewed:

1. **Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator** / SOKE, ICCV 2025, arXiv:2411.17799.
2. **M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production**, arXiv:2603.23617.
3. **SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning**, arXiv:2603.10446.

---

## 1. DexAvatar3D: what the current repository does

DexAvatar is an **optimization-based monocular 3D sign-language reconstruction pipeline**. It does not directly generate motion from text; instead, it reconstructs SMPL-X motion from input image frames.

### 1.1 Current pipeline

The repository pipeline is visible in `Full_running_command.sh`:

```text
Input frames
  -> Sapiens 2D whole-body keypoints
  -> aggregate_sapiens.py -> sapiens.pkl
  -> SMPLer-X initial SMPL-X parameters
  -> HaMeR hand reconstruction
  -> SMPLify-X / DexAvatar fitting
  -> SMPL-X meshes, parameters, overlays
```

Important source locations:

- `run_dexavatar.py:13-30`: loops over input subfolders and launches the shell pipeline.
- `Full_running_command.sh:3-14`: defines the ordered execution stages.
- `README.md:120-142`: paper motivation: monocular sign videos suffer from self-occlusion, noise, motion blur, and missing accurate 3D annotations.
- `dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml:10-20`: enables SMPL-X, hands, face, sign classes, and sign segments.
- `dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml:117-120`: enables `use_signbposer` and `use_hposer3d`.
- `dexavatar_fitting/smplifyx/fitting.py:520-664`: combines 2D reprojection, 3D hand supervision, body/hand priors, biomechanical losses, temporal loss, face/jaw priors, and collision loss.

### 1.2 Current strengths

DexAvatar is already strong because it combines multiple complementary sources:

- **Sapiens** gives dense 2D whole-body keypoints.
- **SMPLer-X** gives initial SMPL-X body, hand, face, camera, shape, and expression estimates.
- **HaMeR** improves 3D hand estimates.
- **SignBPoser** and **SignHPoser** add sign-language-specific body and hand priors.
- SMPLify-style optimization gives a controllable way to add new losses and priors.

### 1.3 Main weaknesses / research gaps

From the source code and paper motivation, the most useful research gaps are:

| Gap | Current DexAvatar behavior | Why it matters for sign language |
|---|---|---|
| Weak temporal modeling | Temporal term is mainly a local smoothness loss; `fitting.py:499` uses current pose vs. previous-frame body pose. | Signing has long-range co-articulation, holds, transitions, repeated handshape patterns, and phrase-level timing. |
| Limited non-manual modeling | SMPL-X face is enabled, but face/expression weights are weak; config uses expression and jaw priors but no sign-specific facial grammar prior. | Mouth, eyebrows, gaze, head motion, and facial expressions can change meaning. |
| Heavy dependence on pseudo-observations | Sapiens, SMPLer-X, and HaMeR are off-the-shelf components. Missing or wrong estimates can propagate into fitting. | Fast hand motion, occlusion, and blur are common in signing. |
| Per-frame optimization cost | Pipeline fits frames with iterative LBFGS-style optimization. | Long signing videos are slow; dense frame fitting wastes effort on redundant transition frames. |
| Hard-coded sign segmentation/class assumptions | Config uses `sign_class` and `sign_segment`; source has dataset-specific assumptions. | Generalization to continuous signing and multilingual data is limited. |
| Body/hand priors are continuous VAE-style priors | Current priors regularize pose but do not explicitly use discrete sign-motion units. | Sign language has reusable lexical/phonological motion primitives that can be tokenized or retrieved. |

---

## 2. Literature review of the three requested papers

## 2.1 Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator

**Task.** Text-to-sign-language production.

**Core idea.** Convert sign motion into discrete body-part tokens and generate those tokens with a multilingual language model, enhanced by retrieval from sign dictionaries.

**Main modules.**

- **Decoupled motion tokenizer.** The paper uses body-part-specific token streams instead of a single flattened pose stream. Local notes describe DETO-style tokenization with separate streams for upper body, left hand, and right hand.
- **Autoregressive multilingual generator.** A language-model-style generator predicts motion tokens from spoken-language text.
- **Retrieval enhancement.** Retrieved sign exemplars or sign dictionary entries provide lexical grounding.
- **Multi-body-part decoding.** Body-part tokens can be generated in a structured way instead of treating the body as one unstructured vector.

**Representation.**

- Discrete sign-motion tokens.
- Part-factorized streams: upper body, left hand, right hand.
- 3D avatar/sign pose output.

**Strengths.**

- The token representation is closer to language modeling than raw continuous angles.
- Body/hand factorization matches sign structure: hands carry much lexical content; torso, arms, shoulders, and head carry support and grammar.
- Retrieval can reduce hallucination and improve rare lexical signs.
- Multilingual setup suggests that some signing primitives transfer across sign languages.

**Weaknesses / limitations for reconstruction use.**

- It is designed for **production**, not monocular reconstruction.
- Discrete tokens may lose subtle finger articulation if codebooks are too coarse.
- Separate streams may desynchronize body and hands unless cross-part coupling is explicit.
- Retrieval quality depends on the dictionary/database coverage.

**What DexAvatar can borrow.**

1. **Part-token priors.** Replace or complement continuous SignBPoser/SignHPoser priors with discrete token priors for upper body, left hand, and right hand.
2. **Retrieval-guided fitting.** Retrieve similar sign-motion segments and use them as initialization or soft constraints during SMPL-X fitting.
3. **Token confidence.** If a frame is visually ambiguous, token-level priors can regularize the pose toward plausible sign-language motion.
4. **Cross-lingual training data.** Multilingual tokens could help DexAvatar learn a broader signing manifold than SGNify-only data.

---

## 2.2 M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production

**Task.** Text-to-sign-language production with manual and non-manual features.

**Core idea.** Use discrete motion tokens for multiple modalities: body, hands, and face. The paper introduces **SMPL-FX**, combining SMPL-X body capacity with FLAME-style facial expression modeling.

**Main modules.**

- **SMPL-FX representation.** SMPL-X body/hands plus richer FLAME facial expression capacity.
- **Modality-specific FSQ-VAE tokenizers.** Separate discrete tokenizers for body, hands, and face.
- **Autoregressive transformer.** Generates multi-modal motion tokens.
- **Auxiliary translation objective.** Helps the motion embeddings stay semantically grounded.

**Representation.**

- Discrete body tokens.
- Discrete hand tokens.
- Discrete face/non-manual tokens.
- Hybrid body-face parametric representation.

**Strengths.**

- Directly addresses a major weakness in many sign pipelines: **non-manual features**.
- Face, mouth, eyebrow, gaze, and head motion are treated as meaningful sign-language channels, not rendering details.
- Discrete multi-modal tokens can make temporal modeling easier and more interpretable.
- Modality-specific tokenization avoids one codebook being dominated by large body motion while ignoring subtle face motion.

**Weaknesses / limitations for reconstruction use.**

- It is a production/generation model, not an observation-driven reconstruction optimizer.
- Reliable face fitting from monocular in-the-wild videos is difficult under low resolution, motion blur, occlusion, and camera viewpoint changes.
- Adding face tokens increases synchronization complexity: mouth, head, hands, and torso must align temporally.
- SMPL-FX may require additional implementation effort because DexAvatar currently outputs SMPL-X-centered parameters.

**What DexAvatar can borrow.**

1. **Non-manual feature prior.** Add a sign-specific face/head/mouth prior, analogous to SignBPoser and SignHPoser.
2. **SMPL-X -> SMPL-FX upgrade.** Use FLAME-style facial expression modeling for richer sign-language reconstruction.
3. **Multi-modal token constraints.** During fitting, constrain body, hands, and face with separate learned token likelihoods.
4. **Semantic face evaluation.** Evaluate not only MPJPE/mesh error but also whether non-manual signs are preserved.

This is probably the **most important paper for improving DexAvatar's linguistic completeness**, because DexAvatar already focuses on body and hands but does not strongly model sign-language-specific facial grammar.

---

## 2.3 SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning

**Task.** Efficient multilingual sign-language production.

**Core idea.** Instead of generating every frame equally, identify sparse keyframes and generate dense motion from them using Conditional Flow Matching.

**Main modules.**

- **FAST sign segmentation/keyframe extraction.** Automatically identifies important temporal anchors.
- **Sparse keyframe learning.** Focuses modeling capacity on important signing frames.
- **Conditional Flow Matching generation.** Generates dense pose trajectories in SMPL-X and MANO space.
- **Keyframe-to-pose editing.** Keyframes can guide controllable motion generation.
- **3D Gaussian Splatting rendering path.** Used for photorealistic output in the production setting.

**Representation.**

- Sparse keyframes.
- Continuous SMPL-X/MANO motion space.
- Conditional dense motion generated from sparse anchors.

**Strengths.**

- Efficient: does not waste equal compute on redundant frames.
- Keyframes align well with signing structure: holds, lexical peaks, handshape changes, contact frames, and phrase boundaries.
- Flow matching can generate smooth dense trajectories from sparse constraints.
- Keyframes are interpretable and useful for editing/debugging.

**Weaknesses / limitations for reconstruction use.**

- Keyframe extraction mistakes can miss semantically important transitions.
- Sparse methods may under-model subtle finger motion between keyframes.
- MANO and SMPL-X alignment must be handled carefully if both are used.
- Designed for production, not fitting to noisy monocular observations.

**What DexAvatar can borrow.**

1. **Keyframe-first fitting.** Fit high-confidence semantic frames first, then propagate to dense frames.
2. **Conditional flow temporal prior.** Replace simple local smoothness with a learned trajectory prior in SMPL-X parameter space.
3. **Compute reduction.** Spend full optimization budget only on keyframes; use cheaper refinement for in-between frames.
4. **Robustness to missing observations.** If Sapiens/HaMeR fail on some frames, reconstruct them using neighboring keyframes and a learned motion prior.

This is probably the **most practical paper for improving DexAvatar runtime and temporal stability**.

---

# 3. Can modules from these papers be applied to DexAvatar3D?

Yes. They cannot be copied directly because the three papers are mainly **sign-language production** systems, while DexAvatar is a **video-to-3D reconstruction** system. However, their modules can be reformulated as **priors, initializers, temporal constraints, and missing-frame reconstruction modules** inside DexAvatar.

## 3.1 Direct module transfer matrix

| Paper module | Directly usable in DexAvatar? | How to adapt it | Expected benefit |
|---|---:|---|---|
| SOKE decoupled body/hand tokenizer | Yes | Train tokenizers on SMPL-X sequences from SGNify, How2Sign-derived pseudo-3D, or DexAvatar outputs. | Better body/hand priors; interpretable discrete sign units. |
| SOKE retrieval enhancement | Yes | Retrieve similar sign segments from a motion database using text/gloss, keypoints, or visual embeddings. | Better initialization and fallback when detectors fail. |
| SOKE multilingual generator | Partially | Use generated/retrieved token sequences as priors, not as final output. | Cross-lingual sign-motion manifold. |
| M3T SMPL-FX | Yes, but larger implementation | Add FLAME-style expression parameters or a richer face head to SMPL-X fitting. | Non-manual features and facial grammar. |
| M3T body/hand/face FSQ-VAE tokens | Yes | Train separate token priors and add token likelihood loss to fitting. | Better multi-modal temporal regularization. |
| M3T auxiliary translation objective | Partially | If gloss/text is available, align pose tokens with sign labels or glosses. | Semantically grounded reconstruction. |
| SignSparK FAST keyframes | Yes | Select reliable/semantic frames for high-budget fitting. | Faster and more stable reconstruction. |
| SignSparK Conditional Flow Matching | Yes | Train a flow prior over SMPL-X trajectories conditioned on sparse fitted keyframes. | Smooth dense motion; missing-frame recovery. |
| SignSparK 3DGS rendering | Not core | Optional visualization layer after reconstruction. | Better visual presentation, not core scientific contribution. |

---

# 4. Strong candidate research directions

## Direction A: Sparse Token-Guided DexAvatar

**Short name:** `STG-DexAvatar`

**Main question.** Can sparse keyframe fitting plus discrete sign-motion token priors improve monocular 3D sign reconstruction under blur, self-occlusion, and detector failure?

### Method

1. **Keyframe selection.**
   - Select frames with high Sapiens/HaMeR confidence, handshape changes, large wrist velocity, sign holds, or contact likelihood.
   - Inspired by SignSparK's sparse keyframe idea.

2. **High-budget DexAvatar fitting on keyframes.**
   - Run the current DexAvatar optimization strongly on selected keyframes.
   - Save fitted SMPL-X body, left hand, right hand, and face parameters.

3. **Part-wise discrete token encoding.**
   - Train or use VQ/FSQ tokenizers for upper body, left hand, right hand, and optionally face.
   - Inspired by SOKE and M3T.

4. **Dense trajectory reconstruction.**
   - Use a temporal transformer or Conditional Flow Matching model to fill in non-keyframes.
   - Condition on sparse keyframe tokens/poses plus noisy observations from Sapiens/SMPLer-X/HaMeR.

5. **Final refinement.**
   - Run a light SMPLify refinement on all frames with token prior and flow prior losses.

### New loss terms

Current DexAvatar has terms like:

```text
L = L_2D + L_hand3D + L_SignBPoser + L_SignHPoser + L_shape + L_collision + L_temp + L_biomech
```

A new version could add:

```text
L_new = L_original
      + lambda_tok  * L_token_prior(body, left_hand, right_hand)
      + lambda_flow * L_flow_trajectory(theta_1:T | keyframes)
      + lambda_sync * L_cross_part_sync(body_tokens, hand_tokens)
```

### Why this can become a paper

**Novelty.** Previous SLP papers use tokens/keyframes for generation. This direction uses them for **monocular 3D sign reconstruction**.

**Feasibility.** It keeps the current DexAvatar pipeline and adds modules around it.

**Possible title.**

> Sparse Token-Guided Priors for Robust Monocular 3D Sign Language Reconstruction

**Expected contributions.**

1. First sparse-keyframe + token-prior framework for monocular 3D sign-language reconstruction.
2. Part-wise discrete priors for body, left hand, and right hand reconstruction.
3. Improved robustness under motion blur, occlusion, and missing detector outputs.
4. Lower runtime by optimizing keyframes more heavily than redundant frames.

**Risk.** Medium. Requires training a motion prior, but can start with pseudo-labels from DexAvatar/SMPLer-X/HaMeR.

---

## Direction B: Non-Manual-Aware DexAvatar with SMPL-FX

**Short name:** `NMF-DexAvatar`

**Main question.** Can sign-language-specific facial/non-manual priors improve 3D sign reconstruction beyond body and hands?

### Method

1. **Upgrade output representation.**
   - Keep SMPL-X body and hands.
   - Add stronger FLAME-style face/expression modeling, inspired by M3T's SMPL-FX.

2. **Face and head observation extraction.**
   - Use Sapiens/face landmarks, SMPLer-X expression estimates, or a dedicated face model.
   - Estimate confidence for mouth, eyebrows, eyes, jaw, and head pose.

3. **Train a SignFacePrior.**
   - Similar role to SignBPoser and SignHPoser, but for signing facial grammar.
   - Could use discrete FSQ/VQ tokens as in M3T.

4. **Add face-aware fitting terms.**

```text
L_new = L_original
      + lambda_face_token * L_face_token_prior
      + lambda_nmf_sync   * L_sync(face, hands, head)
      + lambda_face_obs   * L_face_landmark_reprojection
```

5. **Evaluate on non-manual-sensitive subsets.**
   - Signs differing mainly by mouth/eyebrow/head motion.
   - If no ground truth exists, create a small annotated subset.

### Why this can become a paper

**Novelty.** DexAvatar is hand/body-prior-focused. M3T shows that non-manual features are central for sign production. Bringing that into **reconstruction** is a clear gap.

**Possible title.**

> Non-Manual-Aware 3D Sign Language Reconstruction with Multi-Modal Motion Priors

**Expected contributions.**

1. First non-manual-aware extension of DexAvatar-style 3D sign reconstruction.
2. A sign-specific face/head/mouth prior for monocular fitting.
3. A new evaluation protocol for non-manual reconstruction quality.
4. Better linguistic completeness, not just lower joint error.

**Risk.** Medium-high. Requires reliable face observations and appropriate evaluation data.

---

## Direction C: Retrieval-Augmented DexAvatar

**Short name:** `RA-DexAvatar`

**Main question.** Can retrieved sign-motion exemplars help DexAvatar recover from ambiguous monocular frames?

### Method

1. Build a database of fitted or mocap sign-motion clips.
2. For a query video segment, retrieve similar clips using:
   - 2D keypoint trajectory similarity,
   - visual embeddings,
   - sign class/gloss if available,
   - handshape/contact descriptors.
3. Use retrieved SMPL-X segments as:
   - initialization,
   - soft pose prior,
   - temporal template,
   - missing-frame fallback.
4. Add a retrieval consistency loss:

```text
L_retrieval = min_k distance(theta_pred, theta_retrieved_k)
```

or attention-weighted retrieval:

```text
theta_prior = sum_k attention_k * theta_retrieved_k
L_retrieval = || theta_pred - theta_prior ||
```

### Why this can become a paper

**Novelty.** SOKE uses retrieval for text-to-sign generation. DexAvatar can use retrieval for observation-constrained 3D reconstruction.

**Possible title.**

> Retrieval-Augmented Monocular 3D Sign Reconstruction under Occlusion and Motion Blur

**Expected contributions.**

1. Retrieval-augmented sign-motion prior for DexAvatar-style fitting.
2. Strong improvements in hard cases: self-occlusion, hand blur, missing detections.
3. Interpretable retrieved exemplars for debugging and qualitative analysis.

**Risk.** Low-medium if a motion database is available. Higher if database construction is required.

---

## Direction D: Cross-Lingual Part-Factorized Sign Motion Priors

**Short name:** `XL-Prior-DexAvatar`

**Main question.** Can multilingual sign-language production datasets improve reconstruction priors for languages or signs with little 3D supervision?

### Method

1. Collect pseudo-3D or mocap sequences from multiple sign languages.
2. Train part-wise token priors:
   - body,
   - left hand,
   - right hand,
   - face/head.
3. Learn shared universal motion tokens plus language-specific adapters.
4. Use these priors inside DexAvatar fitting.

### Why this can become a paper

**Novelty.** The three SLP papers are multilingual or multi-dataset. DexAvatar reconstruction remains tied to limited benchmarks. A cross-lingual reconstruction prior is a natural extension.

**Possible title.**

> Cross-Lingual Discrete Motion Priors for 3D Sign Language Reconstruction

**Risk.** High. Needs more data and careful evaluation across languages.

---

# 5. Recommended best research direction

## Best balance: Sparse Token-Guided DexAvatar with Optional Non-Manual Extension

The strongest and most feasible direction is:

> **Sparse Token-Guided Monocular 3D Sign Reconstruction**

It combines:

- **SignSparK:** sparse keyframe learning and Conditional Flow Matching.
- **Signs as Tokens:** part-wise discrete token priors and retrieval.
- **M3T:** optional face/non-manual token stream.
- **DexAvatar:** strong optimization baseline with SignBPoser/SignHPoser.

### Why this is the best direction

1. **It directly addresses DexAvatar's weaknesses.**
   - Weak temporal modeling -> use flow/transformer trajectory prior.
   - Slow dense optimization -> keyframe-first fitting.
   - Detector failure -> token/retrieval prior and keyframe interpolation.
   - Body/hand-only focus -> optional face token stream.

2. **It is novel but not too risky.**
   - You do not need to replace DexAvatar entirely.
   - You can treat DexAvatar as a baseline and improve it modularly.

3. **It creates clear ablations.**
   - Baseline DexAvatar.
   - + keyframe fitting.
   - + token prior.
   - + retrieval prior.
   - + flow temporal prior.
   - + optional face/non-manual prior.

4. **It has a clear paper story.**
   - Existing reconstruction methods optimize frame-wise with weak temporal priors.
   - Existing production methods show that sign motion has reusable sparse/tokenized structure.
   - The proposed method brings sparse/tokenized generative structure into monocular reconstruction.

---

# 6. Proposed paper blueprint

## 6.1 Tentative title

**Sparse Token-Guided Priors for Robust Monocular 3D Sign Language Reconstruction**

Alternative titles:

- **TokenDex: Discrete Motion Priors for Monocular 3D Sign Reconstruction**
- **SparseDex: Keyframe-Guided 3D Sign Language Reconstruction with Motion Tokens**
- **Retrieval- and Token-Guided DexAvatar for Robust 3D Signing Reconstruction**

## 6.2 Abstract idea

Current monocular 3D sign reconstruction systems such as DexAvatar depend on frame-wise pseudo-observations and local smoothness, making them vulnerable to fast hand motion, self-occlusion, and detector failures. Inspired by recent sign-language production models, we propose a sparse token-guided reconstruction framework. The method first identifies reliable semantic keyframes, fits SMPL-X parameters at those frames, encodes body and hand motion into part-wise discrete tokens, and reconstructs dense trajectories using a learned temporal prior conditioned on sparse keyframes and noisy observations. The framework improves robustness, temporal consistency, and runtime while preserving DexAvatar's biomechanical fitting advantages.

## 6.3 Main contributions

1. **Sparse keyframe-first 3D sign reconstruction.**
   - A keyframe selection strategy for signing videos based on detector confidence, motion saliency, handshape change, and temporal boundaries.

2. **Part-wise discrete sign-motion priors.**
   - Body, left-hand, and right-hand token priors trained from SMPL-X sign motion.

3. **Trajectory reconstruction prior.**
   - A transformer or Conditional Flow Matching model that reconstructs dense SMPL-X motion from sparse fitted keyframes.

4. **Retrieval-guided robustness.**
   - Retrieved sign-motion exemplars provide initialization and constraints for ambiguous frames.

5. **Comprehensive evaluation.**
   - Compare against DexAvatar on normal, motion-blur, self-occlusion, and noisy conditions.

## 6.4 Method diagram

```text
Input video frames
      |
      v
Sapiens / SMPLer-X / HaMeR observations
      |
      v
Keyframe selector --------------+
      |                         |
      v                         |
High-budget DexAvatar fitting   |
      |                         |
      v                         |
Sparse fitted SMPL-X keyframes  |
      |                         |
      v                         |
Part-wise token encoder          |
      |                         |
      v                         |
Retrieval + temporal prior <----+
      |
      v
Dense SMPL-X trajectory
      |
      v
Light final SMPLify refinement
      |
      v
Output meshes / parameters / evaluation
```

## 6.5 Implementation plan using this repository

### Stage 1: Establish baseline

- Run current DexAvatar as-is with:

```bash
python run_dexavatar.py \
  --input_img_folder ./data/images_sgnify/[SIGN_NAME]/images \
  --output_path ./output/[SIGN_NAME] \
  --fitting_experiment ./dexavatar_fitting
```

- Save outputs from `smplifyx/results/*.pkl`.
- Measure baseline pose error, hand error, jitter, runtime, and failure cases.

### Stage 2: Keyframe selector

Add a preprocessing script that computes a keyframe score:

```text
score_t = alpha * detector_confidence_t
        + beta  * hand_motion_saliency_t
        + gamma * handshape_change_t
        + delta * segmentation_boundary_t
        - eta   * blur_or_occlusion_t
```

Select top-K frames or segment-level local maxima.

### Stage 3: Keyframe-first fitting

Modify the fitting schedule so that:

1. Keyframes receive full optimization.
2. Non-keyframes initialize from nearest keyframes or interpolated SMPL-X parameters.
3. Non-keyframes use fewer iterations.

### Stage 4: Token prior

Train tokenizers on SMPL-X pose sequences:

- body pose tokens,
- left hand tokens,
- right hand tokens,
- optional face/head tokens.

Possible tokenizers:

- VQ-VAE, inspired by SOKE.
- FSQ-VAE, inspired by M3T.

Add a token-prior loss in `dexavatar_fitting/smplifyx/fitting.py` near the existing prior losses.

### Stage 5: Dense temporal model

Train one of:

- temporal transformer over tokens,
- sequence VAE,
- Conditional Flow Matching model in SMPL-X parameter space.

Use it to predict or regularize non-keyframe poses.

### Stage 6: Optional non-manual branch

If time/data allows:

- Add face/head/mouth token stream.
- Add stronger face landmark/expression supervision.
- Evaluate non-manual-sensitive signs.

---

# 7. Evaluation design

## 7.1 Quantitative metrics

Use standard reconstruction metrics if ground truth is available:

| Metric | Meaning |
|---|---|
| MPJPE body | Mean per-joint position error for body. |
| MPJPE hands | Mean per-joint position error for hands. |
| PA-MPJPE | Procrustes-aligned joint error. |
| Vertex error | SMPL-X mesh vertex distance. |
| Acceleration error | Temporal smoothness / jitter. |
| Handshape error | Finger articulation quality. |
| Runtime per frame | Efficiency. |
| Failure rate | Missing/invalid output frames. |

## 7.2 Stress-test metrics

Because DexAvatar's README explicitly discusses motion blur, self-occlusion, and noise, evaluate under:

- clean frames,
- synthetic motion blur,
- Gaussian noise,
- hand self-occlusion,
- missing Sapiens keypoints,
- missing HaMeR outputs,
- fast signing segments.

## 7.3 Ablation table

| Model | Keyframes | Token prior | Retrieval | Flow prior | Face/NMF | Expected result |
|---|---:|---:|---:|---:|---:|---|
| DexAvatar baseline | No | No | No | No | Weak | strong but jitter/failures remain |
| + keyframe-first | Yes | No | No | No | Weak | faster, more stable |
| + token prior | Yes | Yes | No | No | Weak | better hand/body plausibility |
| + retrieval | Yes | Yes | Yes | No | Weak | better occlusion recovery |
| + flow prior | Yes | Yes | Yes | Yes | Weak | best temporal smoothness |
| + NMF branch | Yes | Yes | Yes | Yes | Yes | best linguistic completeness |

## 7.4 Qualitative analysis

Show side-by-side videos for:

1. fast hand motion,
2. hand-hand contact,
3. hand-body contact,
4. occluded hand,
5. blurred hand,
6. sign transition between keyframes,
7. non-manual expression if implemented.

---

# 8. Research gaps and paper positioning

## 8.1 Gap against DexAvatar

DexAvatar introduced sign-specific body and hand priors for reconstruction, but it is still mainly an optimization pipeline guided by per-frame pseudo-observations and local smoothness. It does not fully exploit sparse temporal structure, discrete sign units, retrieval, or non-manual grammatical features.

## 8.2 Gap against the three SLP papers

The three reviewed papers are primarily **production/generation** methods. They generate signing from text, keyframes, or tokens. They do not solve the same problem as DexAvatar: robust reconstruction from noisy monocular video observations.

Therefore, the research gap is:

> Recent SLP models show that signing has sparse, tokenized, multi-modal structure, but this structure has not been fully used as a reconstruction prior for monocular 3D sign-language fitting.

## 8.3 Novel research claim

A strong claim for a new paper:

> We are the first to integrate sparse keyframe learning, part-wise discrete sign-motion tokens, and retrieval/flow priors into an observation-driven SMPL-X optimization framework for monocular 3D sign-language reconstruction.

---

# 9. Practical recommendation

If the goal is to write a publishable follow-up paper from DexAvatar, I recommend this sequence:

## Phase 1: Low-risk paper core

Implement:

1. keyframe-first fitting,
2. part-wise body/hand token prior,
3. temporal transformer or flow prior for dense reconstruction.

This is enough for a clear paper if results improve temporal consistency and robustness.

## Phase 2: Add retrieval

Add retrieval from a database of DexAvatar/SGNify/How2Sign-style motion clips. This improves the story for occlusion and rare signs.

## Phase 3: Add non-manual features

Add M3T-inspired face/head/mouth prior. This can become either:

- a stronger extension of the same paper, or
- a second paper focused on non-manual-aware reconstruction.

---

# 10. Final answer to the main question

## Can modules from the three papers be applied to DexAvatar3D?

**Yes.** The best modules to apply are:

1. **From Signs as Tokens:**
   - decoupled body/left-hand/right-hand motion tokens,
   - retrieval-enhanced sign-motion priors.

2. **From M3T:**
   - multi-modal body/hand/face tokenization,
   - SMPL-FX-style non-manual feature modeling,
   - face/head/mouth priors for linguistic completeness.

3. **From SignSparK:**
   - sparse keyframe extraction,
   - keyframe-conditioned dense motion reconstruction,
   - Conditional Flow Matching trajectory prior.

## Best paper idea

The strongest research direction is:

> **Sparse Token-Guided Monocular 3D Sign Language Reconstruction**

It improves DexAvatar by adding:

- keyframe-first optimization,
- discrete part-wise sign-motion priors,
- retrieval-guided initialization/regularization,
- learned temporal trajectory prior,
- optional non-manual face/head stream.

This direction is stronger than simply replacing HaMeR/SMPLer-X or tuning weights, because it changes the scientific contribution: DexAvatar becomes not only an optimization pipeline, but a **sign-structured reconstruction framework** that uses the sparse, tokenized, multi-modal nature of sign language.

---

# 11. Sources

- DexAvatar repository README and source files in this project.
- DexAvatar paper page: https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html
- Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator: https://arxiv.org/abs/2411.17799
- SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning: https://arxiv.org/abs/2603.10446
- M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production: https://arxiv.org/abs/2603.23617
- Existing local note consulted: `research_analysis.md`
