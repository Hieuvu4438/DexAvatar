#!/usr/bin/env bash
# Fails if anything outside probes/ changes relative to the approved dirty baseline.
set -euo pipefail

EXPECTED_SUPER_SHA="2b2a3d403075f196cd8970c0267efeb1b9b903db89771be013346ce62786ca61"
EXPECTED_SAPIENS_SHA="03e3aeba37f25b1a6b16f5d9d48747eaf990a1cf1ab911c6e9adba3d205077c0"

# Probe implementation belongs under probes/. The user also explicitly requires
# one phase report per E0/E1/E2/... under docs/dexavatar_diagnosis/.
CURRENT_STATUS=$(git status --porcelain \
  | grep -v '^.. probes/' \
  | grep -vE '^\?\? docs/dexavatar_diagnosis/E[0-9]+(_[A-Z0-9]+)*_PHASE_REPORT\.md$' \
  || true)
CURRENT_SUPER_SHA=$(printf '%s\n' "$CURRENT_STATUS" | sha256sum | cut -d' ' -f1)
CURRENT_SAPIENS_SHA=$(git -C sapiens status --porcelain -uall | sha256sum | cut -d' ' -f1)

if [ "$CURRENT_SUPER_SHA" != "$EXPECTED_SUPER_SHA" ] || [ "$CURRENT_SAPIENS_SHA" != "$EXPECTED_SAPIENS_SHA" ]; then
  echo "UPSTREAM CONTAMINATION DETECTED RELATIVE TO APPROVED BASELINE:"
  printf '%s\n' "$CURRENT_STATUS"
  echo "superproject baseline: $EXPECTED_SUPER_SHA"
  echo "superproject current:  $CURRENT_SUPER_SHA"
  echo "sapiens baseline:      $EXPECTED_SAPIENS_SHA"
  echo "sapiens current:       $CURRENT_SAPIENS_SHA"
  exit 1
fi

echo "OK: no new changes outside probes/ relative to approved baseline"
