# Existing DexAvatar PKL/OBJ replay finding

The existing fitted PKLs and saved OBJs are not byte/geometry-equivalent when
re-forwarded through the pinned SMPL-X model. Two audited examples show:

- `Ablehnen/low_149`: 0.4689 mm mean, 15.8891 mm maximum vertex difference;
- `Blitz/low_143`: 0.2336 mm mean, 9.3414 mm maximum, localized to the active
  right hand (3.0333 mm mean there, while the unused left hand is ~0.0021 mm).

The DCG 6D→matrix→axis-angle round trip is within 1.2e-7 in the implicated hand,
so it does not explain the mismatch. The legacy fitting source selects one
result for the PKL but its mesh branch can render state held elsewhere in the
loop, and one-handed branches can omit a stored hand pose. Therefore old OBJ is
retained as historical evaluator output, while DCG initialization uses the PKL
parameters and regenerates a pinned SMPL-X forward artifact. The regenerated
artifact, not the inconsistent old OBJ, is subject to the exact replay gate.

