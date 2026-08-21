## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + reproducibility validation
- Origin Date: 2026-08-20
- Verification Status: VERIFIED
- Version Label: author_sgnify_eval_v1
- Author evaluator SHA-256: `400bfbd736fc59fcc1867af7650188b61772136982f64b623df31494e6116877`
- Author SMPL-X SHA-256: `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`
- Prospective manifest SHA-256: `33825a3f1ac8aa6d063f90bc12c8061ed60680267615b6d76cbe1e8cee625b32`

# Evaluation with the SGNify author code

The original file `data/evaluation_from_author/evaluate_new_fitting.py` was not
modified. The portable SIGNAL-4D adapter imports its
`read_verts_and_faces`, `transl_point_error`, and
`point_error_common_center` functions directly, uses the author's region files,
MANO vertex IDs, joint regressor, sign classes, and class-0 exclusion rule, and
fails closed on any clip, frame, model-hash, coordinate, vertex-count, or face-topology
mismatch.

All values below are the author's vertex-micro mean in millimeters; lower is
better. Parentheses show method minus A1. Coverage is 100% for every row.

## Prospective extended-post comparison

This is the main clean comparison: 56 clips and 769 frames. The author metric
kernel is applied to the preregistered prospective manifest. The original author
script itself only enumerates central frames, so this is correctly described as
"author metric kernel on prospective frames," not an official published author
benchmark table.

| Method | TR all | TR upper body | TR left hand | TR right hand |
|---|---:|---:|---:|---:|
| A1 baseline | 43.0174 | 29.2938 | 14.4918 | 12.8413 |
| M1 alpha 1.0 | 42.9994 (-0.0180) | 29.2720 (-0.0218) | **14.0304 (-0.4614)** | 12.8420 (+0.0006) |
| M1 alpha 1.5 | 43.0613 (+0.0439) | 29.3597 (+0.0659) | 14.2332 (-0.2587) | 12.8430 (+0.0016) |
| M1 alpha 3.0 | 43.5322 (+0.5148) | 29.9908 (+0.6970) | 16.4099 (+1.9181) | 12.8463 (+0.0050) |
| **SIGNAL-4D v5 multigate** | **43.0072 (-0.0102)** | **29.2774 (-0.0164)** | **14.0354 (-0.4564)** | **12.8425 (+0.0012)** |

SIGNAL-4D v5 improves the author's left-hand endpoint by 0.4564 mm, or 3.15%,
relative to A1. It also slightly improves TR-all and upper body. Right-hand error
changes by +0.0012 mm. Among the frozen component hypotheses, alpha 1.0 is 0.0049
mm lower than the final multigate on this particular left-hand aggregation; the
multigate was selected for the registered SIGNAL-4D clip-macro endpoint and
dynamics, not for this post hoc author-micro endpoint.

Secondary author regions:

| Method | Upper body minus head | Upper body minus face |
|---|---:|---:|
| A1 baseline | 45.5681 | 33.2432 |
| SIGNAL-4D v5 | **45.4136 (-0.1546)** | **33.2042 (-0.0390)** |

Under the author's one-hand policy, left-hand error is evaluated on 41 eligible
clips; 15 class-0 one-handed clips are excluded. SIGNAL-4D v5 is better than A1
on 24 eligible clips, worse on 9, and exactly equal on 8.

## Frozen central-test comparison

This table uses the exact author central-frame selection on the frozen SIGNAL-4D
test partition: 24 clips and 655 frames. It is retained as a protocol audit. The
M1-v5 gate had already seen the revealed central partitions through grouped OOF
training, so its row is descriptive and not confirmatory.

| Method | TR all | TR upper body | TR left hand | TR right hand |
|---|---:|---:|---:|---:|
| A1 baseline | 42.1631 | 26.5916 | 12.5624 | 12.0759 |
| SIGNAL-4D M0 | **37.5743 (-4.5889)** | **25.5698 (-1.0217)** | 20.3612 (+7.7988) | 20.5029 (+8.4269) |
| SIGNAL-4D M1 corrected | 42.3179 (+0.1548) | 26.8280 (+0.2364) | 13.0771 (+0.5148) | 12.3445 (+0.2686) |
| SIGNAL-4D M1-v5 multigate, revealed | 42.1652 (+0.0021) | 26.6123 (+0.0208) | **11.9978 (-0.5645)** | 12.0776 (+0.0017) |

M0's low whole-body/upper-body values do not constitute a better balanced method:
both hands regress sharply. M1 corrected also does not beat A1 on this author
central-test aggregation. M2 is omitted because no corrected frozen test artifact
exists after the left-hand convention audit, and its contact/incremental-value
gates failed.

## Protocol differences from the SIGNAL-4D confirmatory report

The earlier 2.1411 mm prospective improvement and the 0.4564 mm result above are
both reproducible but answer different estimands:

| Property | Author code | SIGNAL-4D registered evaluator |
|---|---|---|
| Aggregation | vertex-micro across eligible samples | equal-weight clip macro |
| Left hand for class-0 signs | excluded entirely | evaluated for every manifest clip |
| Upper-body vertices | author's 8,888-vertex region with class-0 left-hand removal | frozen SIGNAL-4D upper-body region |
| Reported geometry here | TR all, hands, upper variants | preregistered upper/left/right regions |
| Dynamics/contact/uncertainty | not evaluated | evaluated separately |

Therefore, 2.1411 mm must not be quoted as an author-protocol result. Under the
author protocol the prospective left-hand improvement is 0.4564 mm. Conversely,
the author script cannot test SIGNAL-4D's dynamics, uncertainty, contact, or
coverage gates.

## Portability and source audit

The supplied author script cannot run unmodified on this machine because it
contains absolute `/home/kaustubh/...` asset paths. Its `--central` flag is parsed
but not used, and its original directory traversal pairs items positionally.
The adapter changes only transport and validation:

- author source and assets remain read-only;
- metric functions are imported from the supplied source rather than rewritten;
- absolute paths become explicit CLI arguments;
- prediction tensors are read directly instead of exporting lossy/intermediate OBJ copies;
- clip and frame IDs are matched explicitly;
- GT and model faces must be identical;
- structured per-frame, per-clip, summary, and comparison files are emitted.

The complete prospective output tree was rerun independently and compared with
`diff -qr`; all 18 files were byte-identical. Both comparison JSON files have
SHA-256 `f9034eff1a938baf368abd2138c9d4456c8d3cf423e427676c47795525ddc2d0`.

## Artifacts

- `central_test/comparison.csv` and `central_test/comparison.json`
- `prospective_extended_post/comparison.csv` and `comparison.json`
- `prospective_extended_post_repro/` — byte-identical deterministic rerun
- `methods/*/per_frame.csv`, `per_clip.csv`, and `summary.json` under each run
- Portable command: `signal4d evaluate-author-sgnify --help`

