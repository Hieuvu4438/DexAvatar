# SignPK-BA implementation audit

Audit date: 2026-08-30. Source specification:
`../docs/proposal11/SignPK-BA_End-to-End_Method_and_Implementation.md`.

## Outcome

The implementation path is complete for manifest construction, frozen observer
interfaces, PKC inputs/model/heads, leakage-safe training-window preparation and
three-stage training curriculum, all three inference modes, staged clip BA,
standard SMPL-X export, strict evaluation, subgroups, temporal diagnostics, and
front/side inspection.

Two artifacts are intentionally not produced yet:

1. a real SGNify OmniHands cache, because another extraction workload currently
   owns the GPU and concurrent inference could interfere or OOM;
2. a learned PKC checkpoint/final three-seed results, because SignAvatars
   extraction is still running and the request explicitly excludes training for
   now.

The learned interfaces were nevertheless exercised end to end on CPU with a
synthetic, schema-valid OmniHands cache and an identity-initialized PKC state.
Those smoke metrics are interface checks, not learned-method results.

## Requirement traceability

| Proposal area | Implementation/evidence | Status |
|---|---|---|
| Exact `start:end`, x2 identity, no ordinal pairing | `frame_manifest.py`, 57 checked manifests, strict evaluator ID equality | Complete |
| Canonical axes, units, rotations, left convention | `coordinates.py`, `rotations.py`, `palm_frame.py`, left-mirror/SO(3) tests | Complete |
| MANO to SMPL-X and fixed topology | `mano_smplx.py`, `topology.py`, differentiable `SMPLXLayer`, 10,475-vertex/face assertions | Complete |
| H4W++ frozen interface | `h4w_wrapper.py`; 1,493 cached frames validated at pinned commit | Complete |
| Optional H4W++ feature export | `h4w_feature_hook.py`; tested detached FP16 hooks and exact-ID cache | Complete in code; feature cache optional |
| OmniHands T=9 video exporter | `cache_omnihands.py`, raw pre-render cache, explicit gap/reflection/padding/provenance | Complete in code; real cache pending runtime |
| Explicit palm-kinematic tokens | 12-D body, 54-D hand, 20-D relation; body-relative location, shape, tips, flexion/splay, motion, gates, disagreement | Complete |
| Frozen expert features enter PKC | Omni 1024-D projections plus optional H4W body/WiLoR projections | Complete |
| PKC architecture/heads | intra-hand transformers, bimanual gated attention, biased hand-to-body coupling, temporal encoding, SO(3) residual composition, uncertainty/phase/interaction/motion heads | Complete |
| One-hand routing | class prior token and weak 0.25 non-dominant BA routing; no hard arm disable | Complete |
| Leakage-safe training | `DualObserverBundle`, `prepare_training_windows.py`, signer-disjoint dataset, quality weights, SGNify path/source rejection | Complete in code |
| Training objective | centered/un-centered geometry, rotations, FK, palm/relative palm, learned velocities, robust 2D, NLL, penetration, residual, weak latent priors and gate BCE | Complete in code |
| Curriculum/optimizer | Stage A/B/C configs, AdamW/cosine/warmup, separate uncertainty LR, Stage C logvar-only calibration | Complete in code |
| Clip BA | shared shape, residual SO(3) state, robust 2D/H4/Omni/palm/PKC/motion/relation/penetration/prior factors, four stages, early stop and fallbacks | Complete |
| Standard export | official SMPL-X forward, PCA off, fixed topology, parameter PKLs and OBJ IDs from manifest | Complete |
| Audited evaluation | official-compatible regional formula/class-0 rule, strict IDs/topology/finiteness, per-sign/percentile/subgroup/temporal reports | Complete |
| Qualitative inspection | y-up-aware front and side renderer; corrected image inspected on `Ablehnen` | Complete |
| Three final seeds and paper results | Requires completed extraction and actual training | Intentionally pending |

## Verification record

- `pytest -q`: 29 passed.
- `ruff check signpk scripts tests`: passed.
- `python -m compileall -q signpk scripts tests`: passed.
- Manifests: 57 signs, 1,493 central frames.
- H4W++ observer validation: 57 signs, 1,493 frames.
- Real 9-frame H4W++→explicit-token→PKC smoke: body `[1,9,14,12]`,
  hands `[1,9,15,54]`, Omni feature `[1,9,1024]`, finite output.
- Standard SMPL-X smoke: vertices `[14,10475,3]`, fixed faces, finite,
  differentiable state.
- Learned-mode interface smoke: all 14 windows, feed-forward export,
  per-frame diagnostics, and four BA stages completed on CPU.
- Training bridge smoke: three canonical windows passed DataLoader→PKC; one
  differentiable optimizer step through standard SMPL-X completed on CPU.
- BA smoke log contained all implemented factors: H4, Omni hand, palm, PKC,
  robust 2D, learned motion, relation, shape, penetration, residual and
  SignB/H prior.

## Integration baseline (not learned performance)

`Ablehnen`, 14 frames, H4W++/identity-PKC interface smoke under the audited
strict protocol:

| Path | UBody(-F) mm | LHand mm | RHand mm |
|---|---:|---:|---:|
| Direct H4W++ export | 39.8154 | 16.8775 | 15.7350 |
| Standard SMPL-X decode / identity PKC | 39.3578 | 15.3019 | 13.5465 |

The comparison established the required benchmark-only export rotation
`diag(1,-1,-1)`. It must not be applied inside canonical optimization.

## Runtime completion commands

After the current GPU extraction is clear:

```bash
python scripts/cache_omnihands.py --config configs/data/sgnify.yaml
python scripts/validate_observers.py --config configs/data/sgnify.yaml --require-omni
```

After SignAvatars extraction completes, prepare training windows, run Stage
A/B/C with at least seeds 42, 123, and 2026, then use the resulting Stage C
checkpoint for `pkc_feedforward` and `signpk_ba`. Do not use SGNify GT for
training, validation, hyperparameter search, early stopping, or calibration.
Use `--seed` and a distinct `--output-dir` for every run.

## Third-party boundary

- Hand4Whole++: pinned commit above; local license is MIT.
- DexAvatar: parent repository license is MIT.
- OmniHands: pinned commit above; no license file was found in the checked
  clone, so usage/redistribution terms require confirmation.
- SMPL-X/MANO: separately licensed assets, referenced but not redistributed.
