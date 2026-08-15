# DexAvatar research program — Phase 2: comprehensive literature and SOTA audit

**Target task:** monocular RGB video → accurate sign-language SMPL-X reconstruction  
**Primary metric:** SGNify TR-V2V, upper body excluding face / left hand / right hand  
**Search cut-off:** 10 August 2026  
**Seed:** Kundu *et al.*, “DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors,” WACV 2026

## 0. Executive conclusion

There is **no defensible single current SOTA** unless the alignment and evaluation implementation are fixed first.

- **FACT — protocol-clean upper-body result.** DexAvatar reports **30.13 mm** upper-body TR-V2V on SGNify, the lowest result found that is explicitly reported under DexAvatar’s translation-only protocol.
- **FACT — best published hand numbers on the SGNify table.** SOKE reports **10.55 / 8.94 mm** left/right-hand mean per-vertex error, below DexAvatar’s **13.53 / 13.08 mm**. Its supplementary table reproduces all earlier SGNify baseline values exactly, but does not restate translation-only alignment. Thus it is the strongest **TR-V2V-like hand result**, not yet a protocol-certified TR-V2V winner.
- **FACT — newest direct claim.** The May–June 2026 Tamaththul3D preprint reports **29.28 / 10.65 / 8.90 mm**, but explicitly calls its metric **PA-MPVPE**, while copying rows that originated as translation-only errors. A Procrustes-aligned score cannot be ranked against translation-only TR-V2V. Its claimed overall SOTA is therefore **not protocol-valid as published**.
- **INFERENCE — component progress is real even where ranking is not.** SOKE and Tamaththul3D independently show that replacing older hand initialization with WiLoR-class hand reconstruction produces a large hand-error reduction. The evidence does **not** show that either method solves occlusion, contact, or sign semantics.
- **FACT — post-DexAvatar adjacent progress is substantial.** Hand4Whole++ (CVPR 2026) improves wrist/hand integration within whole-body SMPL-X; DanceHMR (May 2026 preprint) introduces temporally coherent hand-aware video SMPL-X; MaskHand models hand ambiguity and occlusion generatively; and temporal low-resolution hand reconstruction explicitly addresses tiny hand crops. None has been evaluated on SGNify TR-V2V.
- **INFERENCE — the field’s primary bottleneck is now evaluation as much as architecture.** One small isolated-sign benchmark, inconsistent metric naming, no released frame manifest/evaluator, and little contact/occlusion stratification prevent a reliable global leaderboard.

This report identifies gaps and novelty constraints. It deliberately **does not propose a final method**.

---

## 1. Epistemic labels and notation

- **FACT:** directly stated in a primary paper, official project page, proceedings record, or official repository.
- **EVIDENCE:** the primary-source observation supporting a fact or inference.
- **INFERENCE:** a conclusion supported by multiple observations but not explicitly established by an experiment.
- **HYPOTHESIS:** a falsifiable claim for a later phase. No method hypothesis is advanced in this report.
- **SPECULATION:** a plausible but presently unsupported possibility.
- **NR:** not reported or not verifiable in the checked primary source.
- **Code: partial:** official code exists, but the reconstruction/evaluation component required to reproduce the reported geometric result was not found.

All errors are millimetres unless stated otherwise. “Body” in the SGNify tables means the evaluated upper-body vertex subset; DexAvatar explicitly describes this as upper body excluding the face.

---

## 2. Search and verification protocol

### 2.1 Search routes

The search covered:

1. **Direct-title and keyword search:** 3D sign-language reconstruction, signing-avatar reconstruction, sign-language HMR, SMPL-X sign language, monocular sign reconstruction.
2. **Seed backward search:** the reference and dependency chains of DexAvatar, SGNify, Neural Sign Actors, EVA, SignAvatars, OSX, and SMPLify-X.
3. **Seed forward search:** exact-title searches, arXiv bibliographic links, Semantic Scholar pages, Google Scholar-indexed author/title pages where accessible, proceedings records, and citations in later primary papers.
4. **Recency search:** work published or posted after DexAvatar’s 24 December 2025 arXiv release, with particular attention to 2026 proceedings/preprints.
5. **Adjacent search:** image/video SMPL-X recovery, hand mesh recovery, interacting hands, hand/body integration, occlusion, blur/low resolution, temporal and generative motion priors, uncertainty, fitting, contact, biomechanics, and pseudo-ground-truth curation.

### 2.2 Source hierarchy

Technical claims were checked against, in priority order: conference/journal proceedings, arXiv full text, official project page, official repository. Semantic Scholar and Google Scholar were used as discovery/citation-graph aids, not as the sole source for method or result claims.

### 2.3 Coverage limitations

- Citation indices lag for very recent work. Exact-title forward search found **Tamaththul3D** as a direct post-DexAvatar citing reconstruction work; this is not a guarantee that no unindexed citation exists.
- Google Scholar pages can be incomplete or access-limited. Citation counts were therefore not used as scientific evidence.
- The accessible IEEE record for **FusePose** exposed metadata and abstract-level claims, but not enough full-text detail to verify its complete quantitative reconstruction protocol.
- No public SGNify evaluator or canonical frame manifest was found in the checked releases. This prevents independent proof that papers using the same-looking table used the same exact vertices, frames, centering, and units.

---

## 3. Backward, forward, and recency map

### 3.1 Backward dependency chain

| Work | Important reconstruction ancestors / components | What was inherited |
|---|---|---|
| SMPLify-X (2019) | SMPL-X, VPoser, 2D body/hand/face detections | Differentiable expressive model fitting, pose prior, collision loss |
| SGNify (2023) | SMPLify-X, SMPL-X, OpenPose/MediaPipe-style observations | Optimization framework plus sign symmetry and hand-pose invariance priors |
| Neural Sign Actors (2024) | OSX, MediaPipe, AMASS, ARCTIC, SGNify benchmark | Whole-body initialization, part refinement, mesh/joint temporal regularization |
| SignAvatars (2024) | OSX, ACR, PARE, ViTPose, MediaPipe, biomechanical hand constraints | Hierarchical pseudo-GT fitting at large scale |
| SOKE (2025) | OSX, WiLoR, MediaPipe, Neural Sign Actors curation idea | Strong hand substitution, arm reprojection refinement, first-order temporal mesh/joint term |
| DexAvatar (2026) | SMPLify-X, SMPLer-X, HaMeR, Sapiens, SGNify, Neural Sign Actors, SignAvatars, EVA | Sign-trained VAE priors plus framewise latent optimization |
| Tamaththul3D (2026 preprint) | SMPLer-X, WiLoR, MediaPipe, VPoser, analytic IK/swing–twist | Modular body/hand fusion, geometric wrist alignment, post-hoc derivative smoothing |

### 3.2 Forward search from the direct lineage

| Cited seed | Verified forward works relevant to RGB→3D sign reconstruction |
|---|---|
| SGNify | Neural Sign Actors; SignAvatars; SOKE; DexAvatar; Tamaththul3D |
| Neural Sign Actors | SOKE; DexAvatar; Tamaththul3D; later sign-generation/dataset works |
| SignAvatars | DexAvatar and later sign-motion dataset/generation work; not all are reconstruction methods |
| DexAvatar | Tamaththul3D was the only clearly verified direct reconstruction paper found by the cut-off |

### 3.3 Techniques appearing after DexAvatar’s arXiv release

| Work | Date / venue status | New evidence relevant to this program |
|---|---|---|
| PEAR | Jan 2026; SIGGRAPH 2026 | Single-image SMPL-X/EHM-s with differentiable pixel-level training supervision and >100 FPS claim |
| Hand4Whole++ | Mar 2026; CVPR 2026 | Frozen whole-body + frozen hand expert; learned conditional hand modulation; differentiable MANO-to-body alignment |
| Tamaththul3D | May/Jun 2026; arXiv preprint | SMPLer-X + WiLoR + closed-form forearm IK + shoulder fitting + derivative smoothing |
| DanceHMR | May 2026; arXiv preprint | Unified temporal SMPL-X with body/hand feature fusion, visibility-aware supervision, upper-body close-up augmentation |
| FusePose dataset/benchmark | FG 2026 | Large-scale continuous-sign 3D representations and body/hand fusion claims; geometric details not fully verifiable from accessible primary text |

**FACT:** None of PEAR, Hand4Whole++, or DanceHMR reports SGNify TR-V2V. **INFERENCE:** they are transfer candidates, not direct SOTA evidence.

---

## 4. Evaluation-protocol ledger

### 4.1 Metrics are not interchangeable

| Metric | Alignment | What it retains | Main risk |
|---|---|---|---|
| TR-V2V | Per-frame translation/centering only | Rotation, articulation, relative placement and scale errors | Requires identical topology, vertex subsets, frames, and centering rule |
| MPVPE | Paper-dependent: raw, root-aligned, wrist-aligned, or otherwise | Depends on stated alignment | The name alone is insufficient |
| PA-MPVPE | Similarity Procrustes alignment | Local shape/articulation after removing global rigid/similarity mismatch | Cannot be directly compared with TR-V2V |
| MPJPE | Paper-dependent root/alignment convention | Joint error rather than dense surface error | Joint set and alignment vary |
| PA-MPJPE | Procrustes-aligned joints | Relative articulated structure | Removes important global/wrist orientation error |
| Jitter / jerk | Third temporal derivative, implementation-dependent | High-frequency instability | Can be reduced by over-smoothing accurate fast motion |
| Acceleration error | Prediction vs GT acceleration | Temporal fidelity when GT exists | Rarely reported for sign reconstruction |
| RTE | Definition varies; often root/wrist trajectory error | Global or inter-frame trajectory stability | Tamaththul3D uses it without 3D GT on its temporal sequence |

### 4.2 SGNify benchmark facts

- **FACT:** 57 isolated German Sign Language signs, one signer, synchronized RGB and commercial motion capture.
- **FACT:** SGNify reports 2,872 evaluated central RGB frames after temporal synchronization/downsampling.
- **FACT:** TR-V2V uses translation-only alignment and region-specific same-topology vertex distances.
- **FACT:** the public benchmark is small, isolated-sign, lab-captured, and single-signer.
- **FACT:** no direct paper reports a sign-reconstruction benchmark with ground-truth contact labels, blur strata, occlusion strata, or continuous-sentence semantic correctness paired with accurate SMPL-X ground truth.

### 4.3 Protocol anomaly that blocks a clean leaderboard

The following rows originated in SGNify/Neural Sign Actors/DexAvatar tables:

| Method | Upper body | Left hand | Right hand |
|---|---:|---:|---:|
| FrankMoCap | 78.07 | 20.47 | 19.62 |
| PIXIE | 60.11 | 25.02 | 22.42 |
| PyMAF-X | 68.61 | 21.46 | 19.19 |
| SMPLify-X / SMPLify-SL label variant | 56.07 | 22.23 | 18.83 |
| SGNify | 55.63 | 19.22 | 17.50 |
| OSX | 47.32 | 18.34 | 18.12 |
| Neural Sign Actors | 46.42 | 16.17 | 15.23 |

SOKE reproduces these numbers and labels its table “mean per vertex error.” Tamaththul3D reproduces most of them and labels the table “PA-MPVPE,” while giving a different OSX row. Exact equality across many rows is strong evidence of inheritance, not evidence that every paper re-evaluated every baseline under its newly named metric.

**INFERENCE:** at least one later table has an alignment-description inconsistency. Until evaluation code is run on released predictions, Tamaththul3D’s ranking cannot be compared scientifically with DexAvatar/SOKE.

---

## 5. A. Directly comparable or near-direct methods

### 5.1 Requested comparison matrix

| Paper | Year | Venue | Input | Representation | Temporal | Sign-specific | Hand prior | Body prior | Contact | Occlusion | Dataset | TR-V2V UBody | LHand | RHand | Code |
|---|---:|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| [SGNify](https://arxiv.org/abs/2304.10482) | 2023 | CVPR | Monocular RGB video | SMPL-X | Previous-frame init / sequence extensions | Yes: symmetry, hand invariance, sign class | Hand pose prior + linguistic constraints | VPoser / SMPLify-X | No dense contact model | Indirect only | SGNify mocap; fluid DGS supplement | 55.63 | 19.22 | 17.50 | [Yes](https://github.com/MPForte/SGNify) |
| [Neural Sign Actors — fitting pipeline](https://arxiv.org/abs/2312.02702) | 2024 | CVPR | Monocular RGB video | SMPL-X | Adjacent mesh + joint loss | No reconstruction semantics; used to curate ASL | PCA prior from AMASS/ARCTIC | OSX + PCA prior | No | Not explicit | SGNify; How2Sign pseudo-GT | 46.42† | 16.17† | 15.23† | Reconstruction code unavailable; data/project page only |
| [SOKE — pose-fitting appendix](https://arxiv.org/abs/2411.17799) | 2025 | ICCV | Monocular RGB video | SMPL-X subset (133-D motion) | Adjacent mesh + joint loss | No semantics in fitting; semantics in later generator | WiLoR | OSX + L2 pose regularizer | No | No explicit visibility model | SGNify; CSL-Daily; Phoenix-2014T | 46.73† | **10.55†** | **8.94†** | [Generation code/data yes](https://github.com/2000ZRL/SOKE); fitter/evaluator not found |
| [DexAvatar](https://arxiv.org/abs/2512.21054) | 2026 | WACV | Monocular RGB video | SMPL-X | Previous-frame body-latent term in release | Yes: SignBPoser / SignHPoser | SignHPoser; HaMeR observations | SignBPoser; SMPLer-X init | No explicit contact | Confidence weighting only; no explicit latent visibility | SGNify; qualitative in-the-wild | **30.13** | 13.53 | 13.08 | [Yes, partial vs paper objective](https://github.com/kaustesseract/DexAvatar) |

† The paper calls the values MPVPE/mean per-vertex error and reproduces the SGNify baseline table, but does not explicitly restate translation-only alignment. These are **TR-V2V-like**, not protocol-certified TR-V2V.

### 5.2 Direct paper evidence cards

#### SGNify — Forte *et al.*, 2023, CVPR

- **Authors:** Maria-Paola Forte, Peter Kulits, Chun-Hao P. Huang, Vasileios Choutas, Dimitrios Tzionas, Katherine J. Kuchenbecker, Michael J. Black.
- **Method:** SMPL-X fitting from monocular videos with hand symmetry and within-sign hand-pose invariance informed by sign categories.
- **Training/data:** learned priors from motion sources; evaluation mocap corpus contains 57 isolated DGS signs.
- **Evaluation:** TR-V2V on body/left/right hand, plus perceptual studies. Main quantitative result 55.63/19.22/17.50; fluid-sentence supplement 54.72/20.28/17.44.
- **Relevance:** establishes the only public sign-specific geometric benchmark and the direct ancestor of the later numeric table.
- **Limit:** isolated signs, one signer, coarse sign-class semantics, no dense contact model, and no explicit probabilistic occlusion model.

#### Neural Sign Actors — Baltatzis *et al.*, 2024, CVPR

- **Authors:** Vasileios Baltatzis, Rolandos Alexandros Potamias, Evangelos Ververas, Guanxiong Sun, Jiankang Deng, Stefanos Zafeiriou.
- **Primary contribution:** text-to-3D sign generation using an anatomical graph diffusion model.
- **Reconstruction subpipeline:** OSX initialization; MediaPipe arm/hand 2D observations; PCA mesh prior trained on AMASS and ARCTIC; adjacent-frame vertex and joint consistency.
- **Evaluation:** SGNify mean per-vertex table 46.42/16.17/15.23. The paper does not restate the translation-alignment implementation.
- **Code:** the fitting pipeline was later reported by Tamaththul3D as unavailable following author correspondence. Treat its reconstruction result as non-reproducible externally.
- **Relevance:** first large How2Sign SMPL-X curation pipeline to beat SGNify without sign-specific fitting losses.

#### Signs as Tokens (SOKE) — Zuo *et al.*, 2025, ICCV

- **Authors:** Ronglai Zuo, Rolandos Alexandros Potamias, Evangelos Ververas, Jiankang Deng, Stefanos Zafeiriou.
- **Primary contribution:** multilingual text-to-sign generation through part-wise VQ tokenization, multi-head autoregressive decoding, and dictionary retrieval.
- **Reconstruction subpipeline:** OSX rough body; WiLoR hand pose plus global wrist orientation substitution; MediaPipe 2D shoulder/arm reprojection fitting; adjacent mesh/joint temporal term; L2 pose regularization.
- **Datasets:** How2Sign, CSL-Daily, Phoenix-2014T; quantitative fitting check on SGNify.
- **Evaluation:** SGNify mean per-vertex error 46.73/10.55/8.94. The supplementary explicitly says 57 annotated signs but does not state translation-only alignment.
- **Code:** generator/tokenizer code and processed poses are public. The checked repository contains training/inference for the generator; the appendix fitter and SGNify evaluator were not found.
- **Critical implication:** SOKE predates DexAvatar’s publication and has lower reported hand errors. Any claim that DexAvatar was unambiguously hand SOTA at publication must address SOKE.

#### DexAvatar — Kundu *et al.*, 2026, WACV

- **Authors:** Kaustubh Kundu, Hrishav Bakul Barua, Lucy Robertson-Bell, Zhixi Cai, Kalin Stefanov.
- **Method:** SMPLer-X body and HaMeR hand initialization/observations; sign-trained hand/body VAE priors; optimization of latent pose codes.
- **Datasets:** SignHPoser motion-capture fingerspelling corpus; SignBPoser pseudo-GT derived from sign video; evaluation on SGNify.
- **Evaluation:** explicit translation-only TR-V2V 30.13/13.53/13.08.
- **Code audit carried from Phase 1:** released fitting optimizes body and hand latents while fixing several variables described in the paper; no explicit hand-body/hand-hand contact loss; hand 3D depth term has zero weight; temporal term is previous-frame body-latent consistency.
- **Strength:** strongest explicitly documented upper-body TR-V2V and a large body improvement.
- **Limit:** its hand result is not the lowest published SGNify-table hand result; claimed biomechanical/contact/occlusion behavior is not matched by explicit released mechanisms.

---

## 6. B. Partially comparable sign-specific / expressive methods

| Paper | Authors | Year | Venue | Input / task | Representation | Temporal / sign-specific | Contact / occlusion | Dataset and evaluation | Quantitative result relevant here | Code |
|---|---|---:|---|---|---|---|---|---|---|---|
| [SignPose](https://openaccess.thecvf.com/content/ICCV2021W/XSAnim/papers/Krishna_SignPose_Sign_Language_Animation_Through_3D_Pose_Lifting_ICCVW_2021_paper.pdf) | Shyam Krishna, Vijay Vignesh P, Dinesh Babu J | 2021 | ICCV XSAnim workshop | 2D keypoints → custom 3D avatar pose | Custom quaternion skeleton, not SMPL-X | Frame-level; trained on synthetic ISL animations | No / no | Synthetic data; MPJPE/PCK | Body MPJPE 27.09; hand 11.7 L / 17.1 R on its own incompatible protocol | NR |
| [SignAvatar](https://arxiv.org/abs/2405.07974) | Lu Dong, Lipisha Chaudhary, Fei Xu, Xiao Wang, Mason Lary, Ifeoma Nwogu | 2024 | FG | SMPL-X motion autoencoding/generation from curated WLASL | SMPL-X motion | Transformer CVAE; sign semantics via CLIP | No / masked-motion training, not visual occlusion | ASL3DWord; recognition, FID, diversity vs pseudo-pose | Reconstruction accuracy 0.952 and FID 40.637 on 103-word setting; not geometric GT | [Yes](https://github.com/dongludeeplearning/SignAvatar) |
| [SignAvatars](https://arxiv.org/abs/2310.20436) | Zhengdi Yu, Shaoli Huang, Yongkang Cheng, Tolga Birdal | 2024 | ECCV | RGB sign videos → pseudo-GT dataset; sign generation benchmark | SMPL-X + MANO | Sequence fitting; sign corpus | Biomechanical hand constraints; no explicit uncertainty | 70K clips / 8.34M frames; reconstruction evaluated on EHF, not SGNify | With biomechanics: EHF holistic MPVPE 20.1, hands 9.7; PA-MPVPE 12.9/4.7 | [Yes](https://github.com/ZhengdiYu/SignAvatars) |
| [EVA](https://arxiv.org/abs/2407.03204) | Hezhen Hu, Zhiwen Fan, Tianhao Wu, Yihan Xi, Seoyoung Lee, Georgios Pavlakos, Zhangyang Wang | 2024 | NeurIPS | Monocular video → expressive Gaussian avatar; SMPL-X alignment substage | SMPL-X + 3D Gaussians | Video avatar; fitting lacks an explicit temporal pose loss | Confidence-aware rendering, not hand occlusion state | XHumans/UPB rendering metrics; DexAvatar’s EVA* SGNify result is author-modified | EVA* 40.38/13.73/13.68 only in DexAvatar, not original EVA | [Yes](https://github.com/evahuman/EVA_Official) |
| [Tamaththul3D](https://arxiv.org/abs/2605.05367) | Eyad Alghamdi, Sattam Altuuaim, Obay Ghulam, Abdulrahman Qutah, Yousef Basoodan | 2026 | arXiv preprint | Monocular RGB video → SMPL-X | SMPL-X + MANO hand source | Derivative smoothing; Saudi-sign application | No contact; WiLoR fallback on severe occlusion | SGNify labelled PA-MPVPE; Ishara-500 qualitative | 29.28/10.65/8.90 **not comparable to TR-V2V as stated** | No official code/release found by cut-off |
| [FusePose / large-scale 3D representation dataset](https://doi.org/10.1109/FG67764.2026.11557028) | Lipisha Chaudhary, Enjamamul Hoq, Lu Dong, Henry Adler, Ifeoma Nwogu | 2026 | FG | Continuous sign video → 3D representation dataset/benchmarks | Claimed 3D mesh representations | Continuous sign-specific | Details NR in accessible primary text | 250+ h claimed; translation/generation benchmarks | Geometric reconstruction values NR | No official repository found |

### 6.1 Tamaththul3D evidence card

- **Authors:** Eyad Alghamdi, Sattam Altuuaim, Obay Ghulam, Abdulrahman Qutah, Yousef Basoodan.
- **Status:** arXiv v1 6 May 2026, v2 4 June 2026; no peer-reviewed venue was listed.
- **Method:** SMPLer-X body; WiLoR hand; MANO-to-SMPL-X conversion; closed-form elbow/forearm kinematics using swing–twist wrist alignment; shoulder-only reprojection fitting; post-hoc velocity/acceleration/jerk smoothing.
- **Data:** SGNify for quantitative evaluation; Ishara-500 for Saudi Sign Language annotations/qualitative results. No ground-truth 3D exists for Ishara-500.
- **Reported SGNify result:** PA-MPVPE 29.28/10.65/8.90.
- **Ablation:** direct hand-coordinate substitution already yields 10.71/9.03; geometric alignment changes this to 10.68/8.95; the full smoother reaches 10.65/8.90. Thus most hand gain comes from the WiLoR substitution, while geometric alignment mainly addresses kinematic coherence.
- **Temporal evidence:** lower jerk/RTE than DexAvatar on 560 frames, but without ground-truth motion accuracy. This shows smoothness, not necessarily fidelity.
- **Failure disclosure:** severe inter-hand occlusion/extreme side views can make WiLoR fail; the system falls back to SMPLer-X.
- **Protocol failure:** its paper defines PA-MPVPE as Procrustes aligned, then inserts earlier translation-only table values. The claim of state-of-the-art cannot be accepted without re-evaluation.

### 6.2 Why these are not direct winners

- SignPose uses another skeleton, synthetic training/evaluation, and incompatible joint definitions.
- SignAvatar reconstructs already-extracted pose sequences and evaluates against pseudo poses, not RGB→mocap geometry.
- SignAvatars evaluates its annotator on EHF, not SGNify TR-V2V.
- EVA’s primary task is novel-view avatar rendering; DexAvatar’s EVA* modification is not an original EVA benchmark.
- Tamaththul3D changes the alignment definition.
- FusePose’s accessible primary record does not expose a reproducible geometric comparison.

---

## 7. C. Transferable adjacent methods

### 7.1 Whole-body expressive SMPL-X recovery

| Paper | Authors | Year / venue | Input / representation | Core method | Datasets / protocol / verified result | Contact / occlusion / temporal | Code |
|---|---|---|---|---|---|---|---|
| [SMPLify-X](https://arxiv.org/abs/1904.05866) | Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, Michael J. Black | 2019 CVPR | Image → SMPL-X | 2D fitting + VPoser + interpenetration | EHF pseudo-GT; foundational, not SGNify SOTA | Collision yes; temporal no; visibility via detector confidence only | [Project/code](https://smpl-x.is.tue.mpg.de/) |
| [OSX](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.pdf) | Jing Lin, Ailing Zeng, Haoqian Wang, Lei Zhang, Yu Li | 2023 CVPR | Image → SMPL-X | Component-aware one-stage transformer | AGORA/EHF/UBody-style whole-body tests; SGNify 47.32/18.34/18.12 in NSA/Dex tables | Framewise; no contact; no explicit uncertainty | [Yes](https://github.com/IDEA-Research/OSX) |
| [SMPLer-X](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2614947a25d7c435bcd56c51958ddcb1-Abstract-Datasets_and_Benchmarks.html) | Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei, Qingping Sun, Yanjun Wang, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, Chen Change Loy, Lei Yang, Ziwei Liu | 2023 NeurIPS Datasets & Benchmarks | Image → SMPL-X | ViT scaling over 32 datasets / 4.5M instances | AGORA 107.2 NMVE; UBody 57.4 PVE; EgoBody 63.6; EHF 62.3 without finetune | Framewise; broad-data robustness | [Yes](https://github.com/MotrixLab/SMPLer-X) |
| [AiOS](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_AiOS_All-in-One-Stage_Expressive_Human_Pose_and_Shape_Estimation_CVPR_2024_paper.html) | Qingping Sun, Yanjun Wang, Ailing Zeng, Wanqi Yin, Chen Wei, Wenjia Wang, Haiyi Mei, Chi Sing Leung, Ziwei Liu, Lei Yang, Zhongang Cai | 2024 CVPR | Full image → multi-person SMPL-X | DETR-style progressive body/part tokens | AGORA plus UBody/EHF/ARCTIC/EgoBody; no SGNify | Framewise; part localization, no contact | [Yes](https://github.com/SMPLCap/AiOS) |
| [SMPLest-X](https://arxiv.org/abs/2501.09782) | Wanqi Yin, Zhongang Cai, Ruisi Wang, Ailing Zeng, Chen Wei, Qingping Sun, Haiyi Mei, Yanjun Wang, Hui En Pang, Mingyuan Zhang, Lei Zhang, Chen Change Loy, Atsushi Yamashita, Lei Yang, Ziwei Liu | 2025 TPAMI accepted / 2026 volume | Image → SMPL-X | Minimal architecture, data/model scaling to 40 datasets and 10M instances | Reports SOTA on seven expressive benchmarks; no SGNify | Framewise; scale/data robustness | [Yes](https://github.com/MotrixLab/SMPLest-X) |
| [PEAR](https://arxiv.org/abs/2601.22693) | Jiahao Wu, Yunfei Liu, Lijian Lin, Ye Zhu, Lei Zhu, Jingyi Li, Yu Li | 2026 SIGGRAPH | Image → SMPL-X/EHM-s | Unified ViT + training-time differentiable pixel alignment + modular annotations | Multiple expressive benchmarks; >100 FPS claim; no SGNify | Framewise; demonstrated robustness to diverse crops; no contact | [Yes](https://github.com/Pixel-Talk/PEAR) |
| [Hand4Whole++](https://arxiv.org/abs/2603.14726) | Gyeongsik Moon | 2026 CVPR | Image → SMPL-X | Frozen body/hand experts + Conditional Hands Modulator + differentiable rigid alignment | AGORA full/hands 76.84/49.71; ARCTIC 45.95/25.03; EHF 61.24/33.43 MPVPE | Framewise; hand-body kinematic integration; no dense contact | [Yes](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE) |

**Key transferable fact:** Hand4Whole++ explicitly shows that copying a hand-only wrist orientation into a body model can produce anatomically invalid wrists. It preserves the whole-body model’s wrist/upper-body reasoning while transferring finger articulation and shape. This is the closest verified prior art to any future “specialist hand + whole-body fusion” novelty claim.

### 7.2 Hand reconstruction, interacting hands, occlusion, blur, and low resolution

| Paper | Authors | Year / venue | Representation / temporal | Mechanism | Dataset / verified quantitative evidence | Explicit uncertainty / occlusion / contact | Code |
|---|---|---|---|---|---|---|---|
| [HaMeR](https://arxiv.org/abs/2312.05251) | Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, Jitendra Malik | 2024 CVPR | MANO; framewise | Large ViT + scaled multi-dataset training | FreiHAND PA-MPJPE/PA-MPVPE 6.0/5.7; HO3D and HInt tests | HInt reports visible/occluded PCK; no probabilistic output | [Yes](https://github.com/geopavlakos/hamer) |
| [WiLoR](https://arxiv.org/abs/2409.12259) | Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, Stefanos Zafeiriou | 2025 CVPR | MANO; nominally framewise | Fast hand detector + ViT reconstruction + multi-scale refinement; >2M in-the-wild hand images | FreiHAND full ablation 5.5/5.1 PA-MPJPE/PA-MPVPE; dynamic tracking evaluated despite no temporal module | Training includes diverse occlusion; no explicit latent visibility | [Yes](https://github.com/rolpotamias/WiLoR) |
| [MaskHand](https://arxiv.org/abs/2412.13393) | Muhammad Usama Saleem, Ekkasit Pinyoanuntapong, Mayur Jagdishbhai Patel, Hongfei Xue, Ahmed Helmy, Srijan Das, Pu Wang | 2025 ICCV | VQ-MANO; iterative generative decoding | Masked pose tokens + confidence-guided sampling + image/2D context | FreiHAND PA-MPJPE/PA-MPVPE 5.5/5.4 at 5 iterations; large HInt occluded-joint PCK gains | **Yes**, token confidence and occlusion-oriented masking; contact not modeled | [Project](https://m-usamasaleem.github.io/publication/MaskHand/MaskHand.html) |
| [Pose-Guided Temporal Enhancement](https://cv.nirc.top/2025/temp-lowres-hand/) | Kaixin Fan, Pengfei Ren, Jingyu Wang, Haifeng Sun, Qi Qi, Zirui Zhuang, Jianxin Liao | 2025 CVPR | MANO; video | Temporal joint features → triplane dense features → current visual enhancement | DexYCB, HanCo, H2O; competitive at much lower hand-crop resolution | Targets low resolution; uses past frames; no contact | [Repository](https://github.com/NewbieFan/Temp-LowRes-hand) is sparse |
| [IntagHand](https://arxiv.org/abs/2203.09364) | Mengcheng Li, Liang An, Hongwen Zhang, Lianpeng Wu, Feng Chen, Tao Yu, Yebin Liu | 2022 CVPR | Two MANO hands; framewise | Interacting-attention graph across two hands | InterHand2.6M and two-hand tests; protocol differs from SGNify | Interaction features and mutual occlusion; no temporal/contact physics | [Yes](https://github.com/Dw1010/IntagHand) |
| [Dyn-HaMR](https://arxiv.org/abs/2412.12861) | Zhengdi Yu, Stefanos Zafeiriou, Tolga Birdal | 2025 CVPR | Two MANO hands in world space; video optimization | SLAM + hierarchical tracking + interacting-hand generative infilling | HOI4D, FPHA, EgoDexter, InterHand2.6M; improves 4D global hand recovery | Explicit generative infilling for (self-)occlusion and interacting-hand prior | [Yes](https://github.com/ZhengdiYu/Dyn-HaMR) |
| [KNOWN-Hand](https://arxiv.org/abs/2407.12307) | Yufei Zhang, Jeffrey O. Kephart, Qiang Ji | 2024 ECCV | 3D hand / MANO-compatible constraints; framewise | Differentiable biomechanics, functional anatomy, physics + NLL uncertainty | Nearly 21% improvement over weakly supervised SOTA on FreiHAND | **Yes**, observation uncertainty; biomechanical constraints | [Yes](https://github.com/zhangy76/KNOWN-Hand) |

### 7.3 Video body motion, temporal priors, and world grounding

| Paper | Authors | Year / venue | Representation | Mechanism | Datasets / result role | Contact / occlusion / uncertainty | Code |
|---|---|---|---|---|---|---|---|
| [VIBE](https://arxiv.org/abs/1912.05656) | Muhammed Kocabas, Nikos Athanasiou, Michael J. Black | 2020 CVPR | SMPL video | Temporal regressor + AMASS adversarial motion discriminator | 3DPW/H36M/MPI-INF-3DHP; foundational sequence HMR | Temporal prior; no explicit contact or occlusion state | [Yes](https://github.com/mkocabas/VIBE) |
| [GLAMR](https://arxiv.org/abs/2112.01524) | Ye Yuan, Umar Iqbal, Pavlo Molchanov, Kris Kitani, Jan Kautz | 2022 CVPR oral | Global SMPL motion | Generative motion infiller + global trajectory predictor + camera/human optimization | Dynamic-camera and occlusion benchmarks; no hands | **Explicit long-occlusion infilling**; no hand contact | [Yes](https://github.com/NVlabs/GLAMR) |
| [WHAM](https://arxiv.org/abs/2312.07531) | Soyong Shin, Juyong Kim, Eni Halilaj, Michael J. Black | 2024 CVPR | World-grounded SMPL | 2D-to-3D motion lifting + image features + camera angular velocity + contact-aware trajectory refinement | 3DPW/RICH/EMDB/global-motion tests | Foot-ground contact; temporal; no articulated hands | [Yes](https://github.com/yohanshin/WHAM) |
| [GVHMR](https://arxiv.org/abs/2409.06662) | Zehong Shen, Huaijin Pi, Yan Xia, Zhi Cen, Sida Peng, Zechen Hu, Hujun Bao, Ruizhen Hu, Xiaowei Zhou | 2024 SIGGRAPH Asia | World-grounded SMPL | Gravity-view coordinate prediction + camera transform | In-the-wild world-motion benchmarks; body only | Temporal/world consistency; no hands/contact output | [Yes](https://github.com/zju3dv/GVHMR) |
| [DanceHMR](https://arxiv.org/abs/2605.18102) | Wenhao Shen, Ming Zhou, Hengyuan Zhang, Siyuan Bian, Youjiang Xu, Xi Lin | 2026 arXiv preprint | **SMPL-X video** | Body features + hand expert features; residual hand→body fusion; temporal transformer; close-up/truncation augmentation | ARCTIC: hand PA-PVE 8.5, hand PVE 24.8, hand jitter 4.2; UBody hand PVE 22.2, PA-MPJPE 4.8 | Visibility-aware 2D loss; learned missing-joint embeddings; wrist/foot static contact, not hand contact | Code not located by cut-off |

DanceHMR is the closest adjacent match to the target input/output pair: monocular video → temporally stable whole-body SMPL-X with detailed hands. It is not sign-specific and has no SGNify result, so it cannot establish sign-reconstruction SOTA.

### 7.4 Generative priors, uncertainty, fitting, contact, and pseudo-ground truth

| Paper | Authors | Year / venue | Mechanism | Evaluation evidence | Transfer relevance | Code |
|---|---|---|---|---|---|---|
| [ProHMR](https://arxiv.org/abs/2108.11944) | Nikos Kolotouros, Georgios Pavlakos, Dinesh Jayaraman, Kostas Daniilidis | 2021 ICCV | Conditional normalizing flow over plausible SMPL poses; mode/sample/likelihood; image-conditioned fitting prior | Standard HMR and fitting benchmarks | Represents 2D→3D ambiguity rather than a single deterministic pose | [Yes](https://github.com/nkolot/ProHMR) |
| [POCO](https://arxiv.org/abs/2308.12965) | Sai Kumar Dwivedi, Cordelia Schmid, Hongwei Yi, Michael J. Black, Dimitrios Tzionas | 2024 3DV | Per-sample variance via dual conditioning; confidence-aware pseudo-label selection and video inpainting | Consistent modest accuracy gain; uncertainty correlates with error | Direct evidence that uncertain/occluded frames can be identified and inpainted | [Yes](https://github.com/saidwivedi/POCO) |
| [ScoreHMR](https://arxiv.org/abs/2403.09623) | Anastasis Stathopoulos, Ligong Han, Dimitris Metaxas | 2024 CVPR | Conditional diffusion score guides latent fitting; task-specific keypoint/temporal guidance | Single-frame, multi-view, and video inverse problems; beats optimization baselines | Differentiable/test-time fitting with a learned generative prior | [Yes](https://github.com/statho/ScoreHMR) |
| [Pose-NDF](https://arxiv.org/abs/2207.13807) | Garvita Tiwari, Dimitrije Antic, Jan Eric Lenssen, Nikolaos Sarafianos, Tony Tung, Gerard Pons-Moll | 2022 ECCV | Differentiable distance to a pose manifold on SO(3)^K | Denoising, occluded pose recovery, image reconstruction | A prior that penalizes distance to valid poses without Gaussian latent assumption | [Yes](https://github.com/garvita-tiwari/PoseNDF) |
| [On Self-Contact and Human Pose](https://arxiv.org/abs/2104.03176) | Lea Müller, Ahmed A. A. Osman, Siyu Tang, Chun-Hao P. Huang, Michael J. Black | 2021 CVPR | Contact-pose datasets; SMPLify-X contact losses; TUCH regressor | Improves contact and non-contact 3DPW/test poses | Proves explicit self-contact labels/losses can improve HMR; not hand-sign-specific | [Yes](https://github.com/muelea/selfcontact) |
| [Weakly Supervised Hand Pose via Biomechanics](https://arxiv.org/abs/2003.09282) | Adrian Spurr, Umar Iqbal, Pavlo Molchanov, Otmar Hilliges, Jan Kautz | 2020 ECCV | Bone-length, palm-structure, joint-angle soft constraints | Additional 2D data reduces FreiHAND depth error by 50% with constraints vs 15% without | Direct biomechanical precedent; novelty cannot be claimed for generic constraints | Primary paper/project available |
| [EFT](https://arxiv.org/abs/2004.03686) | Hanbyul Joo, Natalia Neverova, Andrea Vedaldi | 2021 3DV | Exemplar fine-tuning of a pretrained regressor to 2D evidence for pseudo-GT | Pseudo-labels train competitive HMR models; human-quality assessment | Image-conditioned pseudo-GT refinement alternative to generic fitting | [Yes](https://github.com/facebookresearch/eft) |
| [ARCTIC](https://arctic.is.tue.mpg.de/) | Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, Otmar Hilliges | 2023 CVPR | 2.1M frames with MANO/SMPL-X, articulated objects, dense dynamic contact | Consistent motion and interaction-field benchmarks | High-quality bimanual/contact source, but object manipulation differs from signing | [Yes](https://github.com/zc-alexfan/arctic) |

---

## 8. What is genuinely supported about failure modes

| Failure mode | Direct sign evidence | Strong adjacent evidence | What remains unmeasured |
|---|---|---|---|
| Fast finger articulation | DexAvatar qualitative/ablation; SOKE and Tamaththul3D hand gains | HaMeR/WiLoR/MaskHand; hand-focused scaling | Per-phoneme or per-handshape error on sign GT |
| Inter-hand occlusion | Mentioned by SGNify/DexAvatar; Tamaththul3D discloses WiLoR failures | IntagHand, Dyn-HaMR, MaskHand | SGNify error stratified by inter-hand overlap |
| Hand-body contact | Linguistically important in papers | Self-contact/TUCH; ARCTIC contact labels | Sign-specific hand-body contact precision/recall, penetration, relative location |
| Motion blur | DexAvatar qualitative corruption video | Robust/low-resolution hand work; data scaling | Controlled blur-severity curves with 3D GT |
| Tiny hand crops / distance | Common in continuous sign corpora | Pose-Guided Temporal Enhancement directly targets <64×64 crops | Continuous-sign GT benchmark with crop-size bins |
| Temporal jitter | NSA/SOKE local temporal losses; Tamaththul3D reports jerk | VIBE/WHAM/GVHMR/DanceHMR | GT acceleration/velocity fidelity for sign hands; oversmoothing penalty |
| Depth ambiguity | Motivates all fitting methods | ProHMR, ScoreHMR, MaskHand, KNOWN-Hand | Calibrated hand/body uncertainty on sign videos |
| Body–hand kinematic break | Tamaththul3D qualitative IK evidence | Hand4Whole++ quantitative evidence | Wrist-orientation/forearm-twist metric on SGNify |
| Pseudo-GT bias | SignAvatars/NSA/SOKE/DexAvatar rely on fitted data | EFT/POCO expose curation/confidence strategies | Independent mocap validation across signers/languages |
| One-/two-handed semantics | SGNify/DexAvatar use sign-class logic | SOKE models body parts separately for generation | Automatic uncertainty-aware dominance/interaction state and geometric evaluation |

---

## 9. Dominant research directions

### 9.1 Specialist hand models are being fused into whole-body recovery

**EVIDENCE:** SOKE substitutes WiLoR; Tamaththul3D compares HaMeR/Hamba/WiLoR across body backbones; Hand4Whole++ learns body-conditioned wrist integration; DanceHMR fuses hand and body features temporally.

**INFERENCE:** a monolithic whole-body estimator is no longer the strongest default for fine hands. The dominant pattern is specialist part evidence plus global-body consistency.

### 9.2 Scale—data and model capacity—has displaced many bespoke image architectures

**EVIDENCE:** HaMeR, WiLoR, SMPLer-X, and SMPLest-X attribute large gains to broader training data and larger ViTs. PEAR then reduces complexity while retaining broad annotations/pixel supervision.

**INFERENCE:** a new sign method cannot claim architectural novelty or superiority without controlling for newer initialization and data scale.

### 9.3 Temporal modeling is moving from smoothing to learned evidence fusion and infilling

**EVIDENCE:** VIBE uses a learned motion discriminator; GLAMR performs generative occlusion infilling; POCO selects uncertain frames; the low-resolution hand method transfers past joint evidence into dense current-frame features; Dyn-HaMR infills interacting-hand motion; DanceHMR jointly models body and hands.

**INFERENCE:** first-order adjacent-frame penalties, as used by NSA/SOKE and partially by DexAvatar, are behind the adjacent temporal frontier.

### 9.4 Ambiguity is increasingly explicit

**EVIDENCE:** ProHMR models a pose distribution, POCO predicts variance, MaskHand uses confidence-guided sampling, KNOWN-Hand uses uncertainty in weak supervision, ScoreHMR uses a conditional diffusion distribution.

**INFERENCE:** deterministic confidence-weighted fitting is no longer the only credible treatment of occluded monocular pose.

### 9.5 Evaluation is becoming part-specific and interaction-aware outside sign language

**EVIDENCE:** Hand4Whole++ avoids PA hand metrics because they erase wrist orientation; ARCTIC provides contact; DanceHMR separates all-body and hand temporal metrics; HInt separates visible and occluded joints.

**INFERENCE:** the sign-reconstruction benchmark is under-instrumented relative to adjacent fields.

---

## 10. Underexplored directions — gap statements only

These are **research gaps**, not proposed solutions.

### Gap 1 — No protocol-stable post-DexAvatar leaderboard

- **FACT:** TR-V2V, unspecified MPVPE, and PA-MPVPE are mixed across otherwise identical tables.
- **FACT:** exact frame/vertex/evaluator code is not public in the checked direct releases.
- **INFERENCE:** the first scientific requirement is reproducible metric equivalence, not a new architecture.

### Gap 2 — No sign-specific occlusion benchmark with 3D ground truth

- **FACT:** direct papers discuss occlusion, but report only aggregate errors.
- **FACT:** HInt, MaskHand, GLAMR, and Dyn-HaMR demonstrate visibility-stratified or infilling evaluations in adjacent tasks.
- **INFERENCE:** inter-hand, hand-face, and hand-torso occlusion remain empirically unresolved for signing.

### Gap 3 — Contact is linguistically important but not evaluated

- **FACT:** DexAvatar’s introduction emphasizes hand-body contact, but the released objective has no explicit contact state/loss.
- **FACT:** SGNify/SOKE/Tamaththul3D do not report sign contact precision/recall or penetration.
- **FACT:** TUCH and ARCTIC show contact labels and losses are technically feasible elsewhere.
- **INFERENCE:** contact correctness is a high-value, weakly occupied sign-specific axis.

### Gap 4 — No calibrated uncertainty for sign SMPL-X

- **FACT:** direct methods produce one pose and use detector confidence or fallback heuristics.
- **FACT:** ProHMR, POCO, MaskHand, and KNOWN-Hand explicitly model distributions or variance.
- **INFERENCE:** the field lacks evidence that a reconstructed sign is aware of when its hand depth/articulation is ambiguous.

### Gap 5 — Temporal metrics can reward oversmoothing

- **FACT:** Tamaththul3D reports much lower jerk after derivative smoothing without ground-truth dynamic accuracy on that sequence.
- **FACT:** sign meaning can depend on rapid motion and timing.
- **INFERENCE:** jerk alone is not a valid temporal-quality target; velocity/acceleration error, event timing, and semantic preservation are missing.

### Gap 6 — Continuous signing and cross-signer generalization lack mocap validation

- **FACT:** SGNify’s primary quantitative set is isolated signs from one signer.
- **FACT:** How2Sign, CSL-Daily, Phoenix, SignAvatars, Ishara-500, and FusePose provide scale or diversity mostly through pseudo-3D labels.
- **INFERENCE:** gains on pseudo-GT corpora may reproduce annotator bias and cannot establish geometric SOTA.

### Gap 7 — Hand/body integration is studied, but sign-aware integration is not

- **FACT:** Hand4Whole++ and DanceHMR study specialist-hand/whole-body integration in general datasets.
- **FACT:** SOKE and Tamaththul3D use substitution/alignment, not sign-aware learned integration.
- **INFERENCE:** the combination of sign kinematics, two-hand interaction, and globally coherent SMPL-X remains under-tested. Any novelty claim here must distinguish itself from Hand4Whole++ and DanceHMR.

### Gap 8 — Biomechanics is claimed more often than it is measured

- **FACT:** direct work uses priors or qualitative plausibility language; no common joint-limit, bone-length, palm-structure, penetration, or torque/physics metric is reported.
- **FACT:** Spurr *et al.*, SignAvatars, KNOWN-Hand, Pose-NDF, and TUCH provide existing biomechanical/contact mechanisms.
- **INFERENCE:** “biomechanically plausible” is not currently a validated SOTA dimension in sign reconstruction.

### Gap 9 — Semantics is separated from reconstruction quality

- **FACT:** SGNify uses coarse linguistic sign classes; DexAvatar learns pose distributions from sign data; NSA/SOKE use semantics mainly for generation after fitting.
- **FACT:** no direct benchmark ties 3D reconstruction error to handshape/orientation/location/movement confusions judged by signers.
- **INFERENCE:** lower vertex error is not yet shown to preserve lexical meaning or grammatical non-manual signals.

---

## 11. Answers to the five requested questions

### 1. What is the true current SOTA?

There is no single protocol-independent answer.

- **Upper-body SGNify TR-V2V:** DexAvatar, **30.13 mm**, is the best explicitly documented translation-only result found.
- **Hands on the inherited SGNify table:** SOKE, **10.55/8.94 mm**, is the best published result under a table that strongly appears to inherit the TR-V2V protocol, but the alignment is not restated.
- **Newest direct paper’s own metric:** Tamaththul3D, **29.28/10.65/8.90 PA-MPVPE**, is lowest within its stated PA protocol, but its baseline table mixes earlier TR values; the ranking is not valid until recomputed.
- **Adjacent temporal whole-body SMPL-X:** DanceHMR is the closest recent hand-aware video method, but it is not a sign benchmark result.

The scientifically correct statement is therefore: **DexAvatar remains the verified upper-body TR-V2V leader; SOKE is the strongest hand-result candidate; Tamaththul3D is an unverified post-DexAvatar claimant because of protocol inconsistency.**

### 2. Which methods are directly comparable?

- **Strictly protocol-clean:** SGNify and DexAvatar, plus the baselines each explicitly evaluates under its own documented TR-V2V implementation.
- **Near-direct but requiring evaluator confirmation:** Neural Sign Actors and SOKE, because they use SGNify and reproduce the same baseline rows but call the metric generic mean per-vertex error.
- **Not directly comparable despite the same dataset:** Tamaththul3D, because it states Procrustes alignment.
- **Partially comparable only:** SignAvatars, SignAvatar, EVA, SignPose, FusePose.
- **Transferable, not comparable:** the Category C methods.

### 3. Which techniques appeared after DexAvatar?

- Conditional specialist-hand modulation within frozen whole-body SMPL-X (Hand4Whole++).
- Temporally unified body/hand video SMPL-X with visibility-aware supervision and close-up augmentation (DanceHMR).
- Training-time dense pixel alignment in a fast expressive regressor (PEAR).
- A direct sign pipeline using SMPLer-X + WiLoR + analytic forearm alignment + derivative smoothing (Tamaththul3D).
- A large continuous-sign 3D representation benchmark/fusion claim (FusePose), with insufficient accessible geometric detail for ranking.

### 4. Which directions are becoming dominant?

1. Specialist hand experts integrated with global body reasoning.
2. Data/model scaling and strong generalist initialization.
3. Learned temporal fusion and generative infilling rather than local smoothness alone.
4. Explicit ambiguity/confidence modeling.
5. Part-specific and visibility-aware evaluation.
6. Modular pipelines that can swap body and hand backbones.

### 5. Which directions appear underexplored?

1. A reproducible SGNify evaluator and cross-paper protocol audit.
2. Sign-specific occlusion stratification with 3D GT.
3. Hand-hand and hand-body contact accuracy/penetration metrics.
4. Calibrated uncertainty for SMPL-X sign reconstruction.
5. Temporal fidelity metrics that penalize both jitter and oversmoothing.
6. Continuous, cross-signer, cross-language mocap evaluation.
7. Sign-aware hand/body integration beyond generic fusion or hard substitution.
8. Quantitative biomechanical plausibility.
9. Semantic intelligibility linked to geometric failure types.

---

## 12. Reviewer-style validity checks before Phase 3

The next phase should not advance a method hypothesis until the following questions are answered experimentally or through artifact inspection:

1. Can SOKE, DexAvatar, and Tamaththul3D predictions be evaluated by one independent TR-V2V script on the identical 2,872-frame manifest?
2. Does SOKE’s hand advantage survive translation-only alignment and identical hand vertex indices?
3. Does Tamaththul3D’s PA advantage survive TR-V2V, and does smoothing improve GT acceleration/velocity error rather than only jerk?
4. Which SGNify frames contain hand-hand, hand-face, and hand-torso occlusion/contact, and how do errors change by subset?
5. Are the direct methods’ reported improvements primarily initialization upgrades, learned priors, optimization terms, or different evaluation implementations?
6. Do modern adjacent methods—at minimum WiLoR, Hand4Whole++, DanceHMR, and MaskHand—improve the exact sign benchmark without sign-specific training?

These are falsification prerequisites. They are not a final method proposal.

---

## 13. Primary-source index

### Direct and sign-specific

- [DexAvatar paper](https://arxiv.org/abs/2512.21054) · [WACV proceedings](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html) · [code](https://github.com/kaustesseract/DexAvatar)
- [SGNify paper](https://arxiv.org/abs/2304.10482) · [project](https://sgnify.is.tue.mpg.de/) · [code](https://github.com/MPForte/SGNify)
- [Neural Sign Actors paper](https://arxiv.org/abs/2312.02702) · [project](https://baltatzisv.github.io/neural-sign-actors/)
- [SOKE paper](https://arxiv.org/abs/2411.17799) · [project](https://2000zrl.github.io/soke/) · [code](https://github.com/2000ZRL/SOKE)
- [SignAvatars paper](https://arxiv.org/abs/2310.20436) · [project](https://signavatars.github.io/) · [code](https://github.com/ZhengdiYu/SignAvatars)
- [SignAvatar paper](https://arxiv.org/abs/2405.07974) · [code](https://github.com/dongludeeplearning/SignAvatar)
- [Tamaththul3D paper](https://arxiv.org/abs/2605.05367)
- [EVA paper](https://arxiv.org/abs/2407.03204) · [code](https://github.com/evahuman/EVA_Official)

### Whole-body and hand recovery

- [SMPLify-X](https://arxiv.org/abs/1904.05866)
- [OSX](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.html)
- [SMPLer-X](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2614947a25d7c435bcd56c51958ddcb1-Abstract-Datasets_and_Benchmarks.html)
- [SMPLest-X](https://arxiv.org/abs/2501.09782)
- [AiOS](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_AiOS_All-in-One-Stage_Expressive_Human_Pose_and_Shape_Estimation_CVPR_2024_paper.html)
- [Hand4Whole++](https://arxiv.org/abs/2603.14726)
- [PEAR](https://arxiv.org/abs/2601.22693)
- [HaMeR](https://arxiv.org/abs/2312.05251)
- [WiLoR](https://arxiv.org/abs/2409.12259)
- [MaskHand](https://arxiv.org/abs/2412.13393)
- [Pose-Guided Temporal Enhancement](https://cv.nirc.top/2025/temp-lowres-hand/)
- [IntagHand](https://arxiv.org/abs/2203.09364)
- [Dyn-HaMR](https://arxiv.org/abs/2412.12861)
- [KNOWN-Hand](https://arxiv.org/abs/2407.12307)

### Temporal, uncertainty, fitting, contact, and datasets

- [VIBE](https://arxiv.org/abs/1912.05656)
- [GLAMR](https://arxiv.org/abs/2112.01524)
- [WHAM](https://arxiv.org/abs/2312.07531)
- [GVHMR](https://arxiv.org/abs/2409.06662)
- [DanceHMR](https://arxiv.org/abs/2605.18102)
- [ProHMR](https://arxiv.org/abs/2108.11944)
- [POCO](https://arxiv.org/abs/2308.12965)
- [ScoreHMR](https://arxiv.org/abs/2403.09623)
- [Pose-NDF](https://arxiv.org/abs/2207.13807)
- [On Self-Contact and Human Pose](https://arxiv.org/abs/2104.03176)
- [Biomechanical hand constraints](https://arxiv.org/abs/2003.09282)
- [EFT](https://arxiv.org/abs/2004.03686)
- [ARCTIC](https://arctic.is.tue.mpg.de/)

---

## Final Phase-2 conclusion

**FACT:** DexAvatar is not unambiguously the current SOTA on all three requested regions. It is the strongest explicit upper-body TR-V2V result, while SOKE has lower published hand errors on a near-identical SGNify table. Tamaththul3D is newer and reports still lower aggregate numbers in two regions, but its alignment inconsistency prevents a valid comparison.

**INFERENCE:** a strong next paper is more likely to emerge from resolving the evaluation and failure evidence—especially hand/body integration, occlusion, contact, uncertainty, temporal fidelity, and pseudo-GT bias—than from naming a new architecture before those failures are measured.

**No final method is proposed in this phase.**
