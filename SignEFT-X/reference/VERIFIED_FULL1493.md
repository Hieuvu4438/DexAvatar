# Verified clean inference: 57 signs / 1,493 frames

The active `configs/inference.yaml` pipeline was executed from the direct
frozen initializer, without the historical Transformer cache. Inference
completed with `status=ok`, 57 signs, and 1,493 frames. The subsequent
fail-closed release audit reported:

- 57 canonical sequences;
- 1,493 final SMPL-X states;
- 1,493 final meshes;
- 1,493 per-frame decisions with valid output hashes;
- 2,596 refined hand sides;
- 27 exact full-frame fallbacks when neither side was available;
- zero evaluation/ground-truth paths in the inference config.

## Post-hoc metrics

| Metric | Clean direct-initializer run | Former run | Delta |
|---|---:|---:|---:|
| Official TR All | 42.0535 | 42.0501 | +0.0034 |
| Official TR upper body | 25.7755 | 25.7788 | −0.0033 |
| Official TR upper body minus face | 29.0791 | 29.0829 | −0.0038 |
| Official TR left hand | 12.2806 | 12.2807 | −0.0001 |
| Official TR right hand | 11.4150 | 11.4156 | −0.0006 |
| PA-MPVPE upper body | 26.4008 | 26.4034 | −0.0025 |
| PA-MPVPE upper body minus face | 30.1391 | 30.1418 | −0.0027 |
| PA-MPVPE left hand | 8.1493 | 8.1493 | +0.0000 |
| PA-MPVPE right hand | 8.7999 | 8.7987 | +0.0011 |

Positive delta means the clean run has higher error. All differences are below
0.004 mm. This rerun shows that removing the learned sequence model preserves
the reported result; the effective method is the signer-consistent initializer
and bounded palm-canonical finger refinement.

Authoritative artifacts:

- `outputs/full1493/inference_summary.json`
- `outputs/full1493/release_audit.json`
- `outputs/full1493/evaluation/official/official_result.json`
- `outputs/full1493/evaluation/pa_mpvpe.json`
