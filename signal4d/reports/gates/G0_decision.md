# G0 protocol decision

Decision: **pass** for the frozen clean protocol.

- Exact explicit-frame endpoint: 57 clips / 1,493 frames at 15 fps.
- Frozen v2 roles: calibration 12/260, development 21/578, untouched test 24/655.
- The evaluator rejects missing, duplicated, reordered, or extra frame IDs and never
  imports the fitting/optimization package.
- Coordinate, SO(3), projection, alignment, manifest, cache, factor, contact,
  synthetic end-to-end, and frame-count regression tests pass.
- `Ablehnen`, viewed during evaluator debugging, is quarantined to development and
  is forbidden for calibration/final reporting.

The legacy 2,872-frame arithmetic and the explicit 1,493-frame endpoint are not
silently mixed. Results may only be compared when manifest, regions, alignment,
units, and aggregation are identical.
