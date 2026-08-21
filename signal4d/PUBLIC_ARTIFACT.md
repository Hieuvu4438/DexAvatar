# Anonymous/public artifact boundary

The redistributable SIGNAL-4D artifact consists of source, schemas, configs,
tests, environment locks, audit/report Markdown, evaluator summaries,
bootstrap JSON, CSV tables, and SVG figures under `signal4d/`. These files can
be published with repository-relative paths and without SGNify images/OBJ,
SMPL-X model bytes, estimator checkpoints, or trusted legacy pickle files.

Required licensed materials and placement are documented in `ASSETS.md`.
Their authoritative licenses remain with their owners. The local prospective
release record is identified publicly by SHA-256
`0c5808308b6de3f965fa50f0b05bad21cfe04e9e9161e211f2b03740f825975d`;
it contains hashes/provenance for locally licensed inputs but those input bytes
must not be redistributed.

Public result entry points:

- `reports/confirmatory_extended_post_v5.md`
- `reports/confirmatory_extended_post_v5.json`
- `reports/final_extended_post_v5/results.md`
- `reports/final_extended_post_v5/primary_metrics.csv`
- `reports/final_extended_post_v5/primary_geometry.svg`
- `reports/final_extended_post_v5/comparisons.json`
- `reports/completion_audit_20260820.md`

The scoped claim and limitations must be retained verbatim: this is a new best
result on the prospective SIGNAL-4D extended-post SGNify endpoint versus the
pre-frozen recomputed same-protocol A1 comparator, not an external leaderboard,
unseen-signer, contact-correctness, or semantic-fidelity claim.
