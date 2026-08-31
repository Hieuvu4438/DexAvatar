# CUSP-SL third-party source and checkpoint audit

Audit date: 2026-08-21 (Asia/Ho_Chi_Minh)

This document maps each inherited CUSP-SL method to a primary public source,
an executable artifact, its licensing constraints, and an explicit reuse
decision. A paper or project page is not treated as released source code. A
repository is not admitted merely because its name resembles a paper title.

## Decision matrix

| Method / role in CUSP-SL | Primary public artifact | Local status | License / release status | Decision |
|---|---|---|---|---|
| HandFlow: conditional rectified flow, confidence masking, normalized targets, velocity-blended overlapping windows | [project](https://mxxu00.github.io/HandFlow/), [official code](https://github.com/mxxu00/HandFlow), [official weights](https://huggingface.co/mxxu00/HandFlow) | `third_party/HandFlow`, commit `67fa7df536db233408fe6270ca5d2de28d5959c3`; submodules are deliberately uninitialized | Repository root declares MIT, but several flow/transformer helper files carry explicit CC BY-NC 4.0 notices. Public V1 contains inference and visualization, not training, data preprocessing, or evaluation. | Keep as an immutable external reference and possible pretrained hand-only adapter. Reuse the released inference contract and method ideas with attribution; do not copy mixed-license implementation files into native CUSP modules. CUSP training remains repository-native because the required HandFlow training pipeline is absent. |
| HandFlow checkpoint and normalization statistics | [official model repository](https://huggingface.co/mxxu00/HandFlow) | Not downloaded | Model repository revision `3ca50e4afececc8a7bc361b74954c77307bd0a5f`; `handflow_denoiser.pt` is 667,907,131 bytes, SHA-256 `2fbc4e1fa7a60f469a6ac94933a6e6dc8a86a0e1fc13bd7cd81c430c79acfcda`; `normalization_stats.npz` is 2,008 bytes, SHA-256 `7313334e6b9537fa57ec9763e83f36dfd4998e7d1de09aa3fb21c5bfca8e92c4`. Model card declares MIT but is otherwise empty. | Download only for a preregistered development comparison. It cannot directly replace CUSP's 51-joint SMPL-X residual generator: it predicts right-hand MANO pose/shape/translation and requires HaMeR features. |
| Foundational Flow Matching implementation | [Meta official library](https://github.com/facebookresearch/flow_matching) | Not cloned | CC BY-NC; provides algorithms/training examples but no pretrained models. | Do not add a runtime/vendor dependency. CUSP needs only the audited affine path, velocity target, and Euler solver already covered by native tests. Use as a mathematical/source reference. |
| SMPLer-X body frontend | [official repository](https://github.com/MotrixLab/SMPLer-X) | Existing `SMPLer-X/` snapshot and H32 checkpoint in the DexAvatar workspace | S-Lab License 1.0, non-commercial; official repository provides training, testing, inference, and checkpoints. | Reuse existing frozen frontend artifacts. Do not create a redundant second clone. Record checkpoint and source hashes in each run manifest. |
| WiLoR hand frontend | [official repository](https://github.com/rolpotamias/WiLoR) | Existing independent clone at commit `fcb911312a38fa8badd30d9656a167485d61b8f9`; repository-local exporter is an untracked integration utility, not upstream WiLoR source | Repository `license.txt` is CC BY-NC-ND 4.0; MANO and detector/model assets may impose additional terms. | Reuse only through a frozen adapter. Do not modify or redistribute weights. The local exporter now preserves the model's global wrist orientation and camera metadata needed by A1, but old caches remain incomplete. This remains the required strong A1 frontend/control. |
| SMPLer-X + WiLoR geometric fusion | [Tamaththul3D paper](https://arxiv.org/abs/2605.05367) | Paper available; no verified official implementation repository or checkpoint found | Preprint describes wrist alignment, swing-twist fusion, mirroring, and 2D-supervised optimization but does not link a public code package. | Do not claim code reuse or exact reproduction. Implement a tested adapter independently and label it an adaptation; require chirality, rest-pose, round-trip, and original-image overlay tests. |
| Sign-aware form representation | [SignDINO CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html) | Paper and supplement available; no verified official code/checkpoint package found | The CVF record exposes paper and supplement, not source or weights. Searches of the exact title/method and author publication pages did not yield a released model package. | Keep the form scorer disabled (`form_weight=0`). Do not relabel generic DINO/DINOv2 as SignDINO. Enable only after an official checkpoint with preprocessing and license appears, or after a separately documented reproduction passes signer/source-disjoint counterfactual probes. |
| Masked/confidence-guided hand generation | [MaskHand ICCV 2025 project](https://m-usamasaleem.github.io/publication/MaskHand/MaskHand.html), [CVF paper](https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html) | Paper/project page available | The author project page exposes no code or checkpoint. `m-usamasaleem/MaskHand` does not resolve publicly. The similarly named anonymous repository contains a website rather than model code and has no declared license. | Use only as motivation for corruption/missingness curricula and confidence-guided hypotheses. Do not clone the anonymous project or claim reproduction. |
| Heterogeneous 2D priors and penetration-free two-hand diffusion (A2P) | [official project](https://gaogehan.github.io/A2P/), [paper](https://arxiv.org/abs/2503.17788) | Project source repository exists at `gaogehan/A2P`, but contains only website assets (`index.html`, static files) | No model code, checkpoint, or code license is published by the project as of the audit date. | Do not clone the website repository. Use the paper only to motivate separation of observation alignment and physical validity. CUSP must not claim exact penetration-free diffusion; the present ROM proxy remains explicitly approximate. |
| DexAvatar and official evaluation lineage | [DexAvatar official code](https://github.com/kaustesseract/DexAvatar), [SGNify official code](https://github.com/MPForte/SGNify) | Existing workspace source/assets; CUSP evaluators are isolated under `cusp_sl/` | DexAvatar root MIT; SGNify is restricted to non-commercial scientific research and prohibits redistribution. | Preserve author-comparability evaluator behavior and hashes. Do not vendor or redistribute SGNify assets through the CUSP package. All post-v1 SGNify runs are declared test-exposed. |

## HandFlow compatibility verdict

The official HandFlow clone is useful, but it is not a drop-in implementation
of CUSP-SL. Its released checkpoint has four incompatible assumptions that
must be handled explicitly:

1. It predicts MANO parameters for one hand, whereas CUSP generates tangent
   residuals for 51 SMPL-X upper-body and hand joints.
2. The public model is trained for right hands; left-hand input must be mirrored
   and then transformed back with a chirality-safe adapter.
3. It conditions on frozen HaMeR patch features, 2D rays, and per-frame HaMeR
   confidence; current CUSP caches contain a different, targetless feature
   contract.
4. Its V1 release has no training/data/evaluation code, so using its checkpoint
   tests transfer, not reproduction of the CUSP residual training objective.

Accordingly, a pretrained HandFlow experiment, if run, will be registered as a
separate **hand-only proposal baseline**. It must not silently replace the CUSP
generator. Before that experiment, the HaMeR submodule/checkpoint, left-hand
round trip, MANO-to-SMPL-X mapping, camera convention, and development-only
headroom must all pass. Until then the cloned source remains a read-only method
reference.

## Search and admission policy

Searches covered exact paper titles, method names, author/project pages, GitHub
repository metadata, CVF records, arXiv records, and official model hosting.
Only author-, lab-, publisher-, or project-linked artifacts were accepted as
primary evidence. Community paper indexes were used only to discover candidate
URLs and never to establish code availability or licensing.

No additional clone was admitted in this audit: the useful official source
trees (DexAvatar, SGNify, SMPLer-X, WiLoR, and now HandFlow) are already present;
the remaining critical-path papers expose papers/project pages but no usable
official implementation package. This decision should be revisited before the
camera-ready prior-art/code freeze because several sources are recent 2026
releases.
