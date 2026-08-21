# SIGNAL-4D end-to-end completion audit

Date: 2026-08-20 (Asia/Ho_Chi_Minh)

## User constraints

| Requirement | Evidence | Status |
|---|---|---|
| Work directly on `main` | `git branch --show-current` → `main` | PASS |
| Isolate new implementation | all implementation/runtime artifacts are under `signal4d/` | PASS |
| Do not modify legacy method | only tracked change outside `signal4d/` is the requested specification Markdown; legacy source/output trees are read-only inputs | PASS |
| No PR/merge workflow | both requested documents contain no PR/merge wording; spec section 30 uses direct work packages | PASS |
| Run long experiments in tmux and save outputs | Ensemble, fallback, candidate, and confirmatory tmux logs/artifacts retained under `signal4d/logs`, `artifacts`, and `runs` | PASS |
| Reach the registered target honestly | prospective assessor decision `PASS`; all six preregistered gates pass | PASS |

Unrelated untracked `HandFlow/`, `docs/proposal3/SIGNAL-4D_paper_to_implementation_map_2026-08-16.md`, and `docs/proposal4/` are user-owned and were not modified or removed.

## Milestones and work packages

| Scope | Authoritative evidence | Status |
|---|---|---|
| M0 / WP-001–007 | package/config/logging; manifest guards; SO(3), projection, alignment and handedness core; evaluator; adapters/cache; SMPL-X state/factors/window solver; synthetic end-to-end and frame-count tests | PASS |
| M1 / WP-008–009 | split-conformal calibration artifact/report; uncertainty/risk; change-point adaptive temporal factors; M1 dev/test runs and ablations | PASS |
| M2 / WP-010 | contact registry/proposer/switch/persistence/collision/evaluator, synthetic contact tests and warm-started dev run | IMPLEMENTED; G4/G5 FAIL for real claim, correctly excluded |
| WP-011 runner/statistics/reporting | one-command scripts, fail-closed SGNify evaluator, paired 10,000-replicate clip bootstrap, deterministic report CLI and result pack | PASS |
| G0/G1/G2/G3/G6 | recorded decisions under `reports/gates/`; v5 G6 extension verifies 5,550 release files | PASS |
| G4 contact labels | no reliable independent labels | FAIL BY EVIDENCE; contact claim removed |
| G5 M2 incremental value | no real contact activation/value and target regression | FAIL BY EVIDENCE; M1 path retained |

Gate failures above are expected decision branches in the specification, not
missing implementation. They narrow claims rather than permit unsupported
contact/semantic assertions.

## Prospective confirmation evidence

- Population: 56 clips/769 frames, manifest SHA-256
  `33825a3f1ac8aa6d063f90bc12c8061ed60680267615b6d76cbe1e8cee625b32`.
- Availability hierarchy before GT: 607 Ensemble A1, 145 HaMeR A0, 17 raw
  SMPLer-X terminal; zero missing.
- Gate selection without GT: A1 127, M1×1.0 442, M1×1.5 7, M1×3.0 193; zero
  switches.
- Release freeze SHA-256:
  `0c5808308b6de3f965fa50f0b05bad21cfe04e9e9161e211f2b03740f825975d`.
  Release was written at 11:43:31 +07; GT cache at 11:43:37 +07.
- Freeze verification: 5,550 files rehashed, zero mismatch.
- Gate reproducibility: 112/112 files byte-identical.
- Evaluation completeness: both methods 56 clips/769 frames, coverage 1.0.
- Left-hand superiority: -2.1411 mm, paired 95% CI [-2.9547, -1.4191].
- Upper-body: -1.0745 mm; right-hand: +0.0020 mm, both pass registered margins.
- Velocity -0.0143, acceleration -0.8929, jerk -21.5851; all dynamics gates pass.
- Assessor: `reports/confirmatory_extended_post_v5.json` → `decision: PASS`.

## Final quality and claim boundary

- Ruff: clean.
- Pytest: 38/38 pass.
- Shell syntax: all `signal4d/scripts/*.sh` pass `bash -n`.
- Release integrity: 5,550/5,550 frozen files match.
- Reports: `reports/confirmatory_extended_post_v5.md` and
  `reports/final_extended_post_v5/` are generated from raw evaluator/bootstrap
  outputs.

Permitted claim: new best result on the prospective SIGNAL-4D extended-post
SGNify endpoint versus the strongest pre-frozen, recomputed same-protocol A1
baseline. Not permitted: published external leaderboard SOTA, unseen-signer or
broad-language generalization, contact correctness, semantic fidelity,
biomechanical accuracy, or real-time deployment.
