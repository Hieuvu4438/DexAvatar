# DexAvatar Phase 2 Full-GO Recovery: Root Causes, Redesign, and Execution Record

**Status date:** 12 August 2026  
**Decision:** **FORMAL NO-GO / EXTERNALLY BLOCKED** — implementation remediation is
complete as far as the available inputs permit, but redesigned G2 and G3–G6 cannot
be executed to acceptance without eligible licensed nonbenchmark 3D sign targets
and an available GPU. The local SGNify meshes remain evaluation-only; no proxy or
benchmark target is promoted to training/validation evidence.

## Material Passport

| Field | Value |
|---|---|
| Material | Phase 2 root-cause audit, redesign, implementation, and execution record |
| Created by | OpenAI Codex, operating in the local DexAvatar workspace |
| Created | 12 August 2026 |
| Primary inputs | `DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md`; `DEXAVATAR_PHASE2_UNCERTAINTY_AWARE_WHOLE_SEQUENCE_REFINER.md`; all local Phase 2/Phase 2R gate artifacts, caches, logs, code, and datasets |
| External sources | Official SignAvatars project/repository/access form; official Hand4Whole++ repository/paper; primary papers listed in the SOTA context section |
| Transformations | Read-only evidence audit; full-cache provenance and completion audits; local dataset eligibility audit; code redesign; tests; CPU training smoke test; append-only exact-A1R/proxy experiment queues |
| Sensitive/external data | No local unpublished data or credentials were uploaded. SignAvatars access was not requested because its form requires the owner's identity, institute, email, and acceptance of non-commercial terms. |
| Evidence policy | Proxy, feasibility, and formal evidence are separated. Formal gates fail closed when provenance is incomplete. |

## 1. Executive finding

Phase 2 did not fail for one reason. It accumulated three different failure
classes that earlier reports sometimes mixed together:

1. **Evidence invalidity:** G4/G5 used the H32 initializer rather than the
   selected A1 domain, so strong proxy numbers could not close a formal gate.
2. **Objective/selection mismatch:** the temporal model was selected with
   rotation error, while release success is centered regional mesh error. The
   seed-42 Phase 2R checkpoint therefore achieved a 62.18% rotation-proxy gain
   but failed decoded mesh recoverability.
3. **Real transfer failure:** direct T2 and T5 did not improve the locked Lane-L
   benchmark safely. The already-strong right hand regressed, the hard subset
   regressed, and T5 required excessive fallback.

The redesign makes those three claims independently testable. A cache cannot
claim formal provenance without a portable frozen initializer and released or
multi-view 3D target evidence; a checkpoint cannot be selected without the
release-aligned mesh metric; and U1 cannot pass merely by correlating with
error—it must improve reconstruction when uncertainty feedback is switched on
in the same checkpoint.

## 2. Gate ledger and every observed NO-GO

| Gate/history | Observed NO-GO | Root cause | Present disposition |
|---|---|---|---|
| Early G2 | Only small local framewise/sign snippets were available | No sequence-scale eligible sign corpus with complete body/two-hand targets | How2Sign later closed volume, but its current targets are same-view 2D pseudo-targets; the local 57-sign SGNify GT is benchmark-locked, so formal nonbenchmark 3D target acquisition remains open |
| InterHand T1 / G3 | Rotation recovery passed but decoded left-hand recovery stayed below 30% | Hand-only/partial supervision and metric mismatch | ARCTIC complete-region T1 later passed historical G3; current Phase 2R model again fails mesh G3 |
| Old exact-A1 Stage 1 | `KeyError: 'images'`; zero fitted outputs although the shell printed success | Frozen fitter indexed a 57-German-sign name/class/segment table; How2Sign batch names violated the hidden contract; nested shell failure was masked | Replaced by A1R observation-derived per-clip contracts, exact image coverage, and fail-fast subprocess handling; GPU preflight pending |
| Formal G4 | Numerically strong proxy, formally false | 10,822/498/497 train/validation/calibration clips used frozen SMPLer-X H32, not the selected initializer | Formal audit now rejects all 11,817 clips and binds reports to exact manifest hashes |
| Formal split leakage | Source-video groups were disjoint, but signer identity was neither stored nor audited; filename preflight shows identities 1/2/3/5/8 reused across all old splits | Official-source splits were reused as model-development splits and source-group identity was incorrectly treated as sufficient | Formal audit now requires stable signer IDs and zero signer overlap; signer repartition preflight passes at 10,643/754/420 clips with zero signer/source overlap |
| Residual transfer | Lane A1 and H32 residual scales differ by about 2.35× body and 0.33–0.35× hands | Training learned an initializer-specific correction distribution | A1R must be run on the training/validation/calibration frames; no cross-initializer residual reuse is permitted |
| Phase 2R seed-42 mesh G3 | Clean decoded drift 7.76/9.51/9.90 mm; 4/8/16-frame recovery far below 30% | Rotation-selected checkpoint; mesh term underweighted; benefit labels were angular; validation did not use final selector or release centering | Mesh loss/checkpoint/benefit labels now share the centered regional release metric; T1 evaluator now executes the selector and the same centering |
| Direct T2 / G6 | Balanced gain only 0.0666%; right hand +0.1648 mm; hard left/right subsets regress | Domain shift, asymmetric baseline headroom, no learned regional abstention | Candidate-plus-benefit selection is trained per frame/region against mesh improvement; clean mesh validation selects checkpoints |
| T5 / G6 | Balanced gain −0.5934%; right hand +0.2465 mm; hard subset −5.25%; fallback 30.54% | Reprojection improvement is not equivalent to 3D regional mesh safety | T5 is no longer the primary remedy; final safety is learned against independent 3D mesh benefit and constrained by G6 fallback |
| Old U1 warm-up | Random variance predictions entered attention before calibration | Untrained U1 feedback perturbed the refiner | Existing staged reliability warm-up retained; U1 formal gate remains off until causal intervention passes |
| Old U1 objective | NLL did not optimize worst-decile ranking | Loss/gate mismatch | Regional worst-decile ranking retained |
| G5 causal validity | U1 vs U0 compared separately trained models | Improvement could not be attributed to uncertainty feedback | Residual exporter now evaluates the same U1 checkpoint with feedback on/off; calibration requires corrupt gain and ≤1% clean regression |
| Corruption features | Cached reprojection residuals remained after replacing/corrupting the initializer | Features described a stale pose and created a shortcut | Existing residual refresh/zeroing retained and tested |
| Cache semantics | Provider IDs/camera labels were truncated to one character | `np.full(..., dtype=str)` silently created a NumPy `U1` array | All Phase 2 writers now allocate full-width string arrays; regression test added |
| G7 | Earlier reports mentioned a 2,872-frame target | Population contract ambiguity | Resolved: author 57-sign/1,493-frame population is canonical by project scope |

## 3. New fail-closed method contract

The proposed final method is **A1R + Mesh-U AWSR**, with four separable parts:

1. **A1R portable frozen initializer.** The local SMPLer-X + WiLoR + Sapiens
   ensemble/fitter is retained, but sign class and active side are inferred
   only from the clip's observations. Every result PKL, configuration, provider
   source, and actual model weight is hashed. Folder aliases are per clip, so no
   benchmark sign name is consumed.
2. **Formal sequence targets.** Licensed SignAvatars SMPL-X annotations are the
   planned sign-domain 3D supervision. Exact How2Sign source frame numbers bind
   each target; source-video/cache FPS and full annotation/video frame counts
   must match; released validity masks bind body/left/right supervision. The
   released local body/hand rotations are decoded with common initializer
   shape, root, and translation so the centered mesh metric measures only the
   pose state the refiner can change. This is recorded as released 3D SMPL-X
   pose supervision, not mislabeled as a directly stored target mesh. A separate
   100-clip audit must pass below 10% catastrophic failures.
3. **Mesh-aligned temporal refiner.** Rotation remains an auxiliary stable
   objective, but balanced upper-body/left/right centered vertex loss is
   primary. Target quality weights both training and checkpoint validation.
   Checkpoints are selected on clean release-aligned mesh ratios, with a penalty
   for any region exceeding 1% regression.
4. **Selective benefit and causal U1.** A group/frame benefit head learns whether
   the candidate improves mesh error by a positive margin. At inference it can
   return a body/hand group to A1R. U1 must pass ranking/calibration and the same
   checkpoint must improve when uncertainty feedback is enabled versus disabled.

This design specifically avoids a dense reprojection skip. Reprojection remains
an observation, not a direct correction path, so matched framewise-versus-
temporal ablations can establish that useful improvement comes from sequence
context.

## 4. Implementation completed in this run

### 4.1 Release-metric training and selection

- `balanced_region_vertex_loss` now supports target-quality weighting and
  optional per-region translation centering.
- `regional_vertex_errors` provides the exact per-frame regional metric needed
  by mesh benefit labels and validation.
- the effective initializer after residual mixing is decoded for geometry loss;
  the original cached initializer is no longer decoded by mistake;
- `benefit_target: vertex` and `benefit_margin_mm` train abstention against mesh
  improvement rather than angular improvement;
- `checkpoint_metric: vertex` and `checkpoint_validation: clean` make model
  selection match the release metric and safety regime;
- the T1 vertex evaluator applies benefit selection and reports whether regional
  translation centering was used.

### 4.2 Formal provenance boundary

`phase2_refiner.data.audit_formal_phase2r` validates every clip and requires:

- frozen, portable, non-benchmark-conditioned initializer provenance;
- hashes for weights, configuration, and provider code;
- independent-from-initializer released 3D SMPL-X pose targets, decoded with
  shared non-pose geometry for the centered regional metric, plus source and
  target-audit hashes;
- no release-benchmark training leakage;
- exact frame/FPS bindings, stable licensed signer identities, exact manifest
  hashes, and both source-group- and signer-disjoint train/validation/calibration.

Training with `data.formal_evidence: true` refuses to start without a passing
report bound to the exact manifests in the configuration.

### 4.3 Data and initializer intake

- `materialize_signavatars_targets.py` is ready for the licensed download. It
  checks the owner's license-acceptance record, exact source video FPS and full
  frame count, per-frame bindings, released validity, annotation hashes, stable
  signer IDs, shared-geometry decoding, and the target audit before writing an
  append-only cache. Audit candidates are explicitly ineligible; a passing
  audit produces a fresh formal cache rather than promoting one in place.
- `sample_signavatars_target_audit.py` freezes a deterministic, source-disjoint
  100-clip sample balanced by signer, hand activity/size, truncation, and motion;
  `render_signavatars_target_audit.py` generates source-aligned and side-view
  evidence bound to the exact sample and target hashes.
- `repartition_phase2r_by_signer.py` creates append-only manifests and refuses
  either source-group or signer leakage across train/validation/calibration.
- `run_a1r_fitting.py` now verifies exact input image coverage, uses a per-clip
  observation-derived folder contract, rejects stale results, and writes a
  machine result record.
- `materialize_a1r_cache.py` binds every result and decision into the cache.
- `a1r_portable_ensemble_provider_v1.json` hashes the actual 7.9 GB SMPLer-X,
  2.5 GB WiLoR, 4.7 GB Sapiens, detector, and sign-prior weights in addition to
  code/configuration. Its local verification passes.
- `audit_completion.py` evaluates G0–G7 as one fail-closed chain. It requires
  the redesigned formal cache for G2/G4/G5, centered final-candidate evidence
  for G3, same-checkpoint causal U1 checks for G5, and exactly three seeds for
  G6; historical proxy evidence cannot satisfy those fields.

### 4.4 U1 causal gate

The residual exporter now runs a controlled intervention:

```text
same U1 weights + same inputs
        ├── uncertainty_feedback = on
        └── uncertainty_feedback = off
```

G5 additionally requires feedback-on to improve corrupted reconstruction and
stay within 1% of feedback-off on clean reconstruction. A separately trained
U0 remains the detector-confidence/NLL comparator but no longer substitutes for
the causal test.

### 4.5 Verification

- Ruff: **pass** over the complete `phase2_refiner` package.
- Tests: **94 passed** after adding strict target-audit, deterministic sampler,
  audit-candidate isolation, signer/source repartition, and A1R runner integration
  coverage, plus complete-chain audit tests.
- CPU end-to-end mesh-aligned training smoke: **pass**, one synthetic batch,
  finite loss and gradients.
- A1R provider manifest including large checkpoints: **pass**.
- Exact A1R preflight image/provider audit: **pass**, 32/32 retained How2Sign
  images and frozen-expert products; fitting is queued for free GPU memory.
- Current full-cache formal audit: **expected fail**, 11,817/11,817 clips
  rejected; source-group disjointness **passes**.
- Complete G0–G7 machine audit: **expected fail**, with only G0/G1/G7 passing;
  historical G2 volume/integrity passes but redesigned formal G2 does not.

## 5. Current measured evidence

### 5.1 Existing Phase 2R seed-42 proxy

The rotation proxy is strong but not formal:

| Metric | Value |
|---|---:|
| Validation frames | 15,936 / 15,936 |
| Upper-body relative gain | 50.14% |
| Left-hand relative gain | 70.52% |
| Right-hand relative gain | 65.88% |
| Equal-region gain | 62.18% |
| Hard-subset equal-region gain | 65.88% |
| Formal real-residual audit | **false** |

The same checkpoint was rerun over all 15,936 validation frames with the
corrected centered regional metric and final benefit selector. Centering lowers
clean errors from the old absolute 7.76/9.51/9.90 mm to **5.90/4.55/5.00 mm**
for upper-body/left/right, but does not rescue G3:

| Burst | Upper-body recovery | Left recovery | Right recovery |
|---:|---:|---:|---:|
| 4 frames | 2.77% | 20.27% | 18.36% |
| 8 frames | 2.25% | 20.63% | 18.37% |
| 16 frames | 1.13% | 17.58% | 17.21% |

All are below 30%. Clean-to-injected ratios are 7.9–8.8% for body and
23.5–25.9% for the hands, also failing the 2% ceiling. The machine report is
`outputs/phase2r/domain_aligned_v1_seed42/vertex_proxy_release_aligned_v2.json`.
This is direct evidence that objective/checkpoint redesign is necessary; metric
correction alone cannot promote the old checkpoint.

### 5.2 Full formal preflight of the current cache

| Split | Clips | Frames | Source groups |
|---|---:|---:|---:|
| Train | 10,822 | 346,304 | 2,128 |
| Validation | 498 | 15,936 | 57 |
| Calibration | 497 | 15,904 | 57 |

There is no source-group overlap. Formal eligibility is nevertheless **false**
for every clip because the cache lacks portable initializer contracts, lacks
released/multi-view 3D target contracts, explicitly labels its target as a
same-view proxy, stores no initializer result hashes, and does not carry licensed
signer identities. The machine evidence is
`outputs/phase2r/formal_preflight_current_proxy.json`.

The old split population was not signer-disjoint: filename-identity preflight
finds identities 1/2/3/5/8 in train, validation, and calibration. The new
append-only signer repartition preflight assigns 3/5/8 to train, 1/2 to
validation, and 4/9/11 to calibration. Its executable report passes all checks:

| Split | Clips | Frames | Source groups | Identity groups |
|---|---:|---:|---:|---:|
| Train | 10,643 | 340,576 | 2,054 | 3 |
| Validation | 754 | 24,128 | 104 | 2 |
| Calibration | 420 | 13,440 | 84 | 3 |

The report is
`cache/phase2r/signer_disjoint_preflight_v1/repartition_report.json`. This is
still a filename-derived **preflight**, not formal signer evidence; the exact
mapping must be cross-checked against licensed metadata before target
materialization.

### 5.3 Consolidated completion and blocker audit

`phase2_refiner.audit_completion` was executed against the current authoritative
artifacts. Its machine report is
`outputs/phase2r/full_go_completion_audit_v1.json`, SHA-256
`4473c62011419813cf22e7713eee3cb9983a8cadecb85067b088356d98e0ffc9`.

| Gate | Current final-chain result | Decisive evidence |
|---|:---:|---|
| G0 | GO | Locked 57-sign/1,493-frame evaluator, manifest, and regional baseline |
| G1 | GO | Same-manifest A1 improves every region with clustered CI below zero |
| G2 | NO-GO | Historical volume/integrity passes, but the formal A1R/3D-target cache does not exist |
| G3 | NO-GO | Current centered final-candidate proxy fails every regional recovery threshold |
| G4 | NO-GO | No passing formal external-validation report bound to an eligible cache |
| G5 | NO-GO | No passing formal calibration report with same-checkpoint feedback intervention |
| G6 | NO-GO | Existing candidates fail and no accepted three-seed frozen candidate exists |
| G7 | GO | Author 57-sign/1,493-frame population accepted by project scope |

The audit records four unavailable target-intake prerequisites: verified
SignAvatars license, annotations, licensed true signer map, and a stratified
target audit. The latter two are downstream of receiving the first two. It also
records A1R as `WAITING_FOR_GPU` and mesh training as `MISSING`, because the
ordered queue deliberately cannot start the second stage before the first.

## 6. Ordered execution to full GO

The order is mandatory; later stages do not reinterpret failures in earlier
stages.

### Stage A — close external acquisition and A1R preflight

1. The project owner registers for SignAvatars non-commercial research access.
2. Store the received annotations outside Git and create a local license record
   containing dataset, registrant, grant date, and accepted terms.
   Start from `phase2_refiner/configs/signavatars_license_record.example.json`;
   it intentionally defaults acceptance to `false`.
3. Run A1R on one existing 32-frame How2Sign preflight only after at least
   24 GB GPU memory is free. The fitter is sequential; this preserves more than
   twice the observed model headroom. Require 32/32 schema-valid result PKLs.
4. Render the initializer overlay and inspect wrist attachment, handedness,
   camera, and active-side decisions. Any mismatch stops the full run.

### Stage B — target audit and formal cache

1. Bind SignAvatars annotations to exact How2Sign source frames with
   `materialize_signavatars_targets --audit-candidate`. This cache is explicitly
   marked `audit_passed=false` and cannot pass the formal provenance audit.
2. Use the licensed source-clip-to-signer map to rerun
   `repartition_phase2r_by_signer`; require zero source-group and signer overlap
   before any target-quality sampling. The checked-in assignment preserves G2
   training volume while placing every multi-identity source group in one split.
3. Freeze 100 source-group-disjoint clips with
   `sample_signavatars_target_audit`, using an explicit source-clip-to-signer
   map. The deterministic greedy sampler balances signer, one/two-hand activity,
   hand size, truncation, and motion and emits the frozen-manifest hash.
4. Render the frozen sample with `render_signavatars_target_audit`. It refuses
   formally promoted caches, binds its report to the sample and target hashes,
   and shows source-aligned target vertices, independent body/hand tracks, and
   a depth-colored side view. Audit these overlays before training. Require
   catastrophic failure below 10% overall and separately for body, left hand,
   and right hand. Record the sample seed, frozen-manifest hash, reviewer, and
   completion timestamp using
   `phase2_refiner/configs/signavatars_target_audit.example.json`; formal intake
   rejects missing strata or aggregate-only audits.
5. Rematerialize the target cache without `--audit-candidate` and with the
   passing `--target-audit`. This creates a new append-only formal cache; the
   audit-candidate cache is never promoted in place.
6. Run A1R over train/validation/calibration; do not reuse H32 residuals.
7. Bind A1R with `materialize_a1r_cache`, recompute reprojection residuals from
   A1R, and run `audit_formal_phase2r` over every clip.
8. Require G2 volume, ≥80% clips of length ≥16, ≥70% complete body/two-hand
   fields, deterministic rebuild hashes, and no source-group or signer overlap.

### Stage C — mechanism development, without Lane-L tuning

Train on development splits only:

| Run | Temporal attention | Mesh checkpoint | Benefit head | U1 feedback | Purpose |
|---|:---:|:---:|:---:|:---:|---|
| F0 | No | Yes | Yes | U0 | Matched-capacity framewise control |
| T0 | Yes | No | angular | U0 | Reproduce objective-mismatch failure |
| T1 | Yes | Yes | No | U0 | Isolate mesh objective |
| T2 | Yes | Yes | mesh | U0 | Test selective regional safety |
| T3 | Yes | Yes | mesh | U1 | Final candidate; causal U1 intervention |

Use identical data, seeds, schedules, parameter capacity where possible, and
corruption streams. Temporal causality requires T2/T3 to outperform F0 on
4/8/16-frame bursts and hard real residuals, not only average smoothness.

### Stage D — development gates

- **G3:** ≥30% centered regional mesh recovery at 4/8/16 frames and clean-to-
  injected ratio <2% for all regions after final selection.
- **G4:** ≥3% equal-region improvement, no region worse than 1%, and ≥8% hard-
  subset gain on source-disjoint external validation with a passing formal audit.
- **G5:** all regional Spearman/AUC/risk/NLL checks plus same-checkpoint causal
  feedback gain and ≤1% clean regression.

Only a passing development candidate is frozen.

### Stage E — untouched confirmatory G6

Run seeds 42/123/456 once on the locked 57-sign/1,493-frame Lane-L manifest.
Require identical coverage, no region >0.20 mm worse, at least two regions with
paired sign-clustered 95% CI excluding zero, ≥3% equal-region gain for every
seed, ≥8% frozen hard-subset gain, <1% clean regressions, <1% group/frame
fallback, and <0.20 mm regional seed SD. If G6 fails, report the negative result;
do not tune on Lane-L.

## 7. Active execution and blockers

The ordered GPU queue is launched in tmux session `dexavatar_phase2_ordered`.
It first runs the repaired exact-A1R fitter over the retained 32-frame How2Sign
preflight, requiring 32/32 result PKLs. Only after that succeeds does it run the
mesh-aligned proxy experiment into append-only
`outputs/phase2r/mesh_aligned_proxy_v2_seed42`. The sequential A1R preflight
waits for **24,000 MiB** free; mesh training independently retains the safer
**40,000 MiB** threshold. At this report's latest check the shared RTX 5880 Ada had
about 8.3 GB free and remained occupied by unrelated jobs, so the queue does
not pre-empt them. The separate CPU evaluation of the old checkpoint under the
newly aligned centered-region metric is complete and reported in Section 5.1.

The local target inventory was re-audited rather than inferred from directory
names:

| Local source | Actual usable content | Formal Phase 2 role |
|---|---|---|
| `data/smplx_gt` | 4,152 SMPL-X meshes over 57 SGNify signs (2.6 GB) | Locked G6/G7 benchmark only; forbidden in training and development validation |
| How2Sign | 37,125 front-view videos and 133-point 2D tracks | Sign observations; no independent 3D targets or local multiview triangulation path |
| Motion-X | Incomplete local archive whose available entries are generic performance/music SMPL-X motion | Generic pretraining only; not paired sign-video residual supervision |
| WHIM | Web-image annotation object arrays; local train multipart download incomplete | Not video-aligned SMPL-X sign supervision |
| ARCTIC / InterHand | Generic complete body-hand / hand-only supervision | G3 corruption recovery and pretraining, not sign-domain G4 evidence |

Two external dependencies prevent an honest full-GO claim today:

1. **Owner/legal action:** an eligible nonbenchmark sign-domain 3D corpus is
   absent locally. SignAvatars requires a Google access form containing
   email, name, institute, task selection, and acceptance of non-commercial
   terms. Codex cannot supply or accept those on the owner's behalf.
2. **Shared compute state:** A1R preflight and the 5,000-step mesh experiment
   require substantially more free GPU memory than is currently available.

The first dependency is the critical path. The official dataset has 70,000
sequences, 8.34 million frames, 153 signers, and released SMPL-X annotations,
which is sufficient in scale for G2, but access is gated. Until the owner grants
that input, full G4/G5 evidence cannot be created, and therefore G6 must not be
reinterpreted as a completed Phase 2 acceptance.

## 8. Primary external references

- SignAvatars project and access: https://signavatars.github.io/
- SignAvatars official repository/data format: https://github.com/ZhengdiYu/SignAvatars
- Hand4Whole++ official repository: https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE
- Hand4Whole++ paper: https://arxiv.org/abs/2603.14726
- DanceHMR temporal body/hand fusion: https://arxiv.org/abs/2605.18102
- MaskHand confidence-guided generative hand reconstruction: https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html
- Tamaththul3D sign-specific SMPL-X reconstruction: https://arxiv.org/abs/2605.05367
