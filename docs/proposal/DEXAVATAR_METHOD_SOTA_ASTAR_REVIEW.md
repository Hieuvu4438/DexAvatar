# Final adversarial report: proposed 3D sign reconstruction direction

## Executive decision

**Do not begin implementing the full SignPosterior4D/SP4D stack.**  
**Do begin a narrow Phase 0–2 program now.**

The proposal identifies genuine DexAvatar weaknesses: independent body/hand priors, weak wrist–forearm coupling, frame-causal fitting, and poor use of hand geometry. The historical depth-only/disabled 3D hand supervision claim remains a sound motivation: the original-code discussion is at [`/home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md:114-122`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L114-L122), based on release commit `7e97916`.

The proposed full system, however, bundles an observation replacement, coordinate conversion, unified kinematics, temporal inference, uncertainty, relations/contact, diffusion, multi-sample inference, evidence selection, phonology, phase modeling, and final optimization. A positive result would not identify its cause. More seriously, neither the benchmark protocol nor the phonology supervision is ready enough to support the headline claims.

### Direct answers

1. **Is it good enough to start implementing now?**  
   **Yes, only as a constrained de-risking implementation. No, not as the proposed full paper or large-scale training plan.** Start evaluator/provenance work, a locally executable WiLoR/SMPLer-X fusion baseline, and a deterministic bidirectional refiner. Do not commit to diffusion, multi-hypothesis selection, contact, phase, or phonology as core contributions yet.

2. **Which methods should be implemented first?**  
   **First:** coordinate-correct SMPLer-X + WiLoR fusion with full XYZ, palm-normal, and wrist–forearm constraints, using a frozen cached observation bank.  
   **Second:** a matched-compute deterministic bidirectional sequence refiner trained on realistic estimator residuals and burst corruption.  
   These are the two mechanisms most likely to reveal whether there is recoverable signal beyond the initializer. They also establish a usable paper even if linguistic supervision cannot be secured.

3. **What must be clarified or changed before large-scale training?**  
   An independently audited benchmark manifest and alignment definition; reproducible initializer installation; licensed, independently annotated phonology data with a causal intervention protocol; source/signer/checkpoint provenance; and a calibrated-validation set. Until those exist, large-scale training would optimize an uncertain target and make causal claims that cannot be defended.

**Readiness:** Phase 0–2 engineering is **3/5**. A phonology-headline paper is **1/5** until a licensed, linguistically audited, source/signer-disjoint attribute set and intervention test exist. Readiness cannot become 4/5 through model engineering alone.

---

## 1. The first blocker is evaluation, not modeling

The proposal currently calls SGNify TR-V2V a fixed, standard protocol ([`lines 5, 17-21`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L5-L21)) and mixes DexAvatar with Tamaththul3D values ([`lines 50-69`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L50-L69)). That is not defensible with the repository’s current evidence.

### What the local code actually measures

The local metric independently mean-centers **each reported region** before V2V:

- [`/home/haipd/DexAvatar/evaluation/eval_mpvpe_regions.py:93-103`](file:///home/haipd/DexAvatar/evaluation/eval_mpvpe_regions.py#L93-L103)
- [`/home/haipd/DexAvatar/evaluation/evaluation_trv2v_wilor.py:57-69`](file:///home/haipd/DexAvatar/evaluation/evaluation_trv2v_wilor.py#L57-L69)

That is **repository-local centroid-aligned V2V**. It must not be described as SGNify-comparable TR-V2V merely because the local docstrings say it “matches” SGNify. SGNify’s stated translational mesh alignment has not been shown equivalent to independent upper-body/left-hand/right-hand centroid removal. The latter can remove different translations for parts that a single global mesh alignment would preserve.

Tamaththul3D v2 is an **arXiv preprint**, not a peer-reviewed paper, and reports regional **PA-MPVPE**. Procrustes alignment removes rotation, translation, and scale under its alignment convention; local centroid alignment removes only a separately estimated translation per region. Therefore:

- remove the current “strong modern baseline,” “paper-worthy,” and “stretch” numerical targets tied to `29.28 / 10.65 / 8.90`;
- remove the inferred percentage improvement target;
- call Tamaththul3D an **external PA-MPVPE reference**, not a directly beatable TR-V2V target;
- report local results only as repository-local centroid-aligned V2V until a common protocol is independently validated.

### Why output counts are not a minor bookkeeping issue

The local evaluator:

- maps a prediction `low_i` to GT frame `2i` ([`eval_mpvpe_regions.py:106-128`](file:///home/haipd/DexAvatar/evaluation/eval_mpvpe_regions.py#L106-L128));
- contains multiple central-frame mechanisms ([`131-147`](file:///home/haipd/DexAvatar/evaluation/eval_mpvpe_regions.py#L131-L147), [`197-218`](file:///home/haipd/DexAvatar/evaluation/eval_mpvpe_regions.py#L197-L218));
- accepts `segment_json` in the second evaluator but explicitly does not use it for pairing ([`evaluation_trv2v_wilor.py:167-180`](file:///home/haipd/DexAvatar/evaluation/evaluation_trv2v_wilor.py#L167-L180)).

Existing aggregate results cover 1,429–1,493 frames depending on method ([`/home/haipd/DexAvatar/outputs/all_methods_comparison_final.csv:2-6`](file:///home/haipd/DexAvatar/outputs/all_methods_comparison_final.csv#L2-L6)); the NLF-WiLoR summary covers 1,450 frames ([`benchmark_summary.csv:2`](file:///home/haipd/DexAvatar/outputs/method_nlf_wilor/benchmark_summary.csv#L2)). This is evidence of unverified pairing, temporal sampling, central-frame membership, or missing-output handling, not a valid comparative table.

### Phase-0 go/no-go

Build and validate a new evaluator; do not simply containerize the current scripts.

Its immutable manifest must enumerate, for every expected official evaluation frame:

- clip/source ID and official source-frame index;
- predicted frame index, prediction path, and content hash;
- GT path and content hash;
- inclusion/exclusion reason;
- missing/failure reason;
- central-frame decision and rule version;
- mesh topology, units, and region-index hashes;
- global-versus-region translation alignment outcome.

Every method must use the **same manifest** before any aggregate comparison. The first validation experiment is global translation alignment versus independent regional centering, checked against an official-protocol DexAvatar reproduction. If equivalence cannot be demonstrated, rename the local metric permanently.

---

## 2. Recommended paper scope

### Paper thesis, conditional on data

**Default thesis now:**

> A frozen-observation, coordinate-correct bidirectional refiner improves monocular upper-body, wrist, and two-hand reconstruction under transient observation failures.

This is coherent and implementable. Its novelty is moderate, so it must be supported by clean evaluation, strong geometric baselines, and targeted corruption tests.

**Conditional stronger thesis:**

> Independently validated structured sign attributes causally improve reconstruction decisions under visual ambiguity beyond the same generic sign-video context.

Use this only after the phonology prerequisites below are met. Until then, call the conditioning **sign-video contextual conditioning**, not phonology-conditioned reconstruction.

### Minimum viable method

1. **One locally executable frozen observation bank:** SMPLer-X body plus WiLoR hands, cached before refiner training.
2. **Unified coordinate-valid state:** upper body, wrists, both hands, shared shape, documented camera convention; full XYZ, 2D, fingertip, palm-normal, and wrist–forearm constraints.
3. **Deterministic bidirectional refiner:** 32–64 frame windows initially, realistic residuals and burst corruption.
4. **Calibration gate:** use learned reliability only if held-out NLL, coverage, selective-risk, and error-confidence tests pass. Otherwise use fixed, predeclared observation weights.
5. **Predefined tests:** clean validation, blur, occlusion, estimator disagreement, and interacting-hand subsets, with clip-clustered paired confidence intervals.

### Do not make core in version 1

- Hand4Whole++ as the primary baseline;
- FUSION initialization;
- conditional diffusion;
- `K>1` sampling or learned selection;
- dense contact constraints;
- phase-aware dynamics;
- semantic–kinematic cycle as evidence of linguistic correctness;
- phonology as a headline claim.

The existing proposal itself shows why this reduction is necessary: it currently introduces the full diffusion posterior before its proposed phonology/phase contribution ([`lines 718-739`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L718-L739)) and defines a minimum paper that already requires diffusion and phase ([`lines 980-991`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L980-L991)). That order makes both attribution and delivery risk unacceptable.

---

## 3. The phonology claim is not yet identified

The proposal correctly says HamNoSys should not be treated as frame-level phase supervision ([`/home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md:1201-1208`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L1201-L1208)). But its central phonology latent is still assumed rather than operationalized ([`lines 369-416`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L369-L416)).

There are two separate problems.

### 3.1 Circular supervision

Geometry-derived attributes can be useful auxiliary targets, but they cannot prove linguistic structure if they are generated from pseudo-mesh labels or the same reconstruction being evaluated. Likewise, agreement between a video attribute head and `F_geom` can be self-confirming. The cycle loss at [`lines 403-414`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L403-L414) must be auxiliary only; `F_geom` must be fixed or stop-gradient.

### 3.2 Generic-context confounding

A generic temporal model receiving the same RGB/SHuBERT/video features can internally learn the same visual signals. Parameter matching is necessary but not sufficient. “Predicted soft phonology” only earns a phonology claim if structured tokens have a demonstrated intervention effect.

### Mandatory phonology gate

Before retaining phonology as a paper-defining contribution, require:

1. **A formal ontology:** class inventory and counts; handshape/orientation/location/movement/symmetry/dominance definitions; frame/segment/clip resolution; unknown/ambiguous state; language-specific fields.
2. **Independent labels:** signer- and source-disjoint held-out linguistic annotations, separate from pseudo-mesh-derived labels and separate from the geometry extractor used in the cycle.
3. **License and access proof:** SignAvatars’ public project page supports the claimed scale and prompts, but its webpage’s CC BY-SA notice is not proof of dataset or annotation reuse rights.
4. **A strict generic control:** identical encoder, frozen observation bank, training clips, augmentations, temporal receptive field, optimization budget, and capacity.
5. **Structured-token interventions:** correct tokens; shuffled tokens among matched clips; confidence-matched random tokens; and masked/unknown tokens.
6. **Mechanism-specific analysis:** report whether perturbing a token changes reconstruction where that attribute is visually ambiguous, not merely average error.
7. **Held-out metrics:** attribute accuracy, calibration, and geometric effects with uncertainty.

If any of these fail, the correct claim is contextual sign-video conditioning. That is still useful, but it is not a phonology-conditioned reconstruction contribution.

---

## 4. Component decisions and evidence thresholds

| Component | Decision | Evidence threshold before retention |
|---|---|---|
| SMPLer-X + WiLoR observation bank | Implement first | Reproducible installation and common-manifest output |
| MANO/SMPL-X/wrist fusion and full geometry | Implement first | Paired per-clip improvement or no predefined regional regression versus frozen observations |
| Deterministic bidirectional refiner | Implement second | Improvement on clean validation and preregistered ambiguity subsets, not only lower jerk |
| Learned uncertainty | Gate | Held-out NLL, coverage, selective risk, error-confidence correlation, and subset reconstruction improvement |
| Minimal relation features | Later optional study | Isolated interacting-hand improvement without global regression |
| Contact/depth-order | Defer | Manually verified contact subset; compare no/predicted/oracle contact; no forced-contact damage |
| Diffusion, `K=1` | Defer | Beats deterministic refiner on hard subsets or calibrated likelihood under matched compute |
| `K>1` and selection | Defer | Deployed selector closes a preregistered meaningful fraction of the oracle gap |
| Phonology | Conditional headline | All causal-label and intervention gates above |
| Phase dynamics | Auxiliary later | Transition-subset spatial-error gain, not just smoother motion |
| DPoser-X body checkpoint | Integration artifact only | Run logs, data/split hashes, step, validation results, normalization artifact, and documented configuration provenance |

Do not use 1–5 “expected metric impact” scores. No pilot evidence justifies them. Use preregistered falsifiable thresholds: paired clip-level difference and 95% CI, median change, maximum allowable regression by region, and ambiguity-subset effect with uncertainty.

---

## 5. Dependency and repository reality

### Initializer choice

Use **locally executable WiLoR/SMPLer-X** as the Phase 0–1 baseline. The proposal currently describes SMPLer-X + WiLoR as the minimum reproducible setup ([`lines 342-350`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L342-L350)) but later calls Hand4Whole++ the primary initializer ([`lines 1083-1095`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L1083-L1095)). Resolve this contradiction in favor of the former.

Hand4Whole++ and SHuBERT are not present in this checkout. Hand4Whole++ is an external acquisition/integration gate, not a Phase-1 option. WiLoR weights have a restrictive CC-BY-NC-ND gate; distinguish that checkpoint license from the repository source-code license, do not fine-tune or redistribute derivative weights without permission, and record the exact terms.

### DPoser-X

The sign route is explicitly body-only: `pose_body.pt` has shape `(N, 63)` for 21 body joints, with no hands or face ([`/home/haipd/DexAvatar/docs/dposerx_sign_body_training.md:50-58`](file:///home/haipd/DexAvatar/docs/dposerx_sign_body_training.md#L50-L58)). The status document says the configured fine-tune had not launched as of 2026-06-24 ([`lines 13-28`](file:///home/haipd/DexAvatar/docs/dposerx_sign_body_training.md#L13-L28)); its planned run is 30,000 steps, batch size 512 ([`lines 143-151`](file:///home/haipd/DexAvatar/docs/dposerx_sign_body_training.md#L143-L151)).

A local ignored `last.ckpt` may now exist, but it is not evidence of a research-ready prior. Audit:

- provenance against the documented configuration;
- exact checkpoint step;
- logs and validation records;
- source video and split hashes;
- normalizer artifact;
- any deviation from How2Sign-only staging;
- whether the checkpoint was trained or merely copied/initialized.

It must not be described as a unified body–hand sequence prior.

### Selection protocol, if ever attempted

Withholding observations from the selector is necessary but not enough. Predefine disjoint observation channels or spatial/temporal masks before training; generate and refine candidates without that channel; freeze the holdout sampler; and evaluate selection on a separate test split. Candidate likelihood, selector labels, compute budget, refinement iterations, and candidate count must be identical for random, likelihood-only, learned, and oracle conditions.

---

## 6. Phased plan

### Phase 0 — authoritative evaluation and provenance

**Build:** evaluator, container, manifest, mesh/index checksums, alignment audit, and per-clip failure records.

**Go:** all methods share precisely the same complete frame manifest, and an official-protocol DexAvatar reproduction is explained against the published result.  
**No-go:** stop all SOTA comparisons if alignment, pairing, central-frame selection, or missing-output policy changes aggregate scores.

### Phase 1 — geometric fusion baseline

**Build:** frozen SMPLer-X + WiLoR observations; MANO-to-SMPL-X mapping; full XYZ/2D/fingertip/palm-normal observations; wrist–forearm consistency.

**Go:** maintains or improves frozen observation geometry on a clean held-out set without degrading arm/wrist attachment.  
**No-go:** repair coordinates, scale, global orientation, camera convention, or mapping before any temporal model.

### Phase 2 — deterministic bidirectional refiner

**Build:** cached tokens, 32–64-frame bidirectional model, estimator residuals, burst-corruption training, and identical generic-context baseline.

**Go:** statistically supported clip-level improvement on clean validation and predefined ambiguity subsets, with no unacceptable region regression.  
**No-go:** if only jerk improves, investigate alignment, target quality, and oversmoothing; do not add diffusion.

### Phase 3 — calibration gate

**Build:** estimator-specific reliability head and held-out calibration data.

**Go:** confidence predicts actual error; calibration improves selective risk and failure-subset reconstruction.  
**No-go:** retain fixed predeclared weights.

### Phase 4 — conditional phonology study

Run only after licensed audited annotations and the causal intervention protocol are available.

**Go:** correct structured tokens outperform all perturbation controls on target-attribute ambiguity cases and improve held-out attribute metrics.  
**No-go:** describe results as generic sign-video context or auxiliary supervision.

### Phase 5 — optional extensions

Order: minimal relations, phase, diffusion `K=1`, contact, then `K>1` selection. Each needs an isolated matched-compute mechanism test and no significant global regression.

---

## 7. Exact proposal changes

1. Replace the title typo at line 3 and remove “Posterior”/“diffusion” from the title until those mechanisms pass their gates.
2. Replace the claim that the SGNify protocol is fixed and directly reproduced ([`lines 5, 17-21`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L5-L21)) with a protocol-audit commitment.
3. Delete the Tamaththul3D-derived targets at [`lines 64-69`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L64-L69). Keep its values only as external PA-MPVPE reference values.
4. Replace the central-method claim at [`lines 38-41`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L38-L41) with a deterministic bidirectional refiner; classify diffusion and sampling as optional extensions.
5. Select SMPLer-X + WiLoR as the initial primary configuration and make Hand4Whole++ an external robustness gate.
6. Replace the semantic-cycle interpretation: it is auxiliary, `F_geom` is fixed or stop-gradient, and held-out linguistic annotation is the primary phonology evidence.
7. Remove unsupported “first” language at [`lines 880-883`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L880-L883).
8. Insert the formal phonology specification and intervention protocol before presenting phonology as a contribution.
9. Replace the minimum viable paper definition with frozen observations, coordinate-correct unified state, deterministic bidirectional refinement, and a calibration gate.
10. Add a release/provenance table: dependency; code terms; checkpoint terms; body-model/data terms; modification and redistribution permission; principal-configuration status; verification date.
11. Replace the final-system recommendation at [`lines 1010-1016`](file:///home/haipd/DexAvatar/docs/DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md#L1010-L1016) with:  
   **“Frozen local observations + coordinate-correct unified body/wrist/two-hand state + deterministic bidirectional refiner with a calibration gate. Add structured linguistic conditioning only after independently audited labels and causal intervention evidence; add relations, diffusion, contact, and sampling only after mechanism-specific gates.”**

---

## Verified primary sources

- Forte et al., **SGNify: Reconstructing Signing Avatars From Video Using Linguistic Priors**, CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html
- Kundu et al., **DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose**, WACV 2026. https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html
- Lu et al., **DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior**, ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html
- Yu et al., **SignAvatars**, ECCV 2024 project/data page. https://signavatars.github.io/
- Tamaththul3D v2, **arXiv preprint**. https://arxiv.org/html/2605.05367v2
- WiLoR official repository and model release. https://github.com/rolpotamias/WiLoR
- Hand4Whole++ official repository. https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE