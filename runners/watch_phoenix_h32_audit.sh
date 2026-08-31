#!/usr/bin/env bash
set -euo pipefail

project=/home/haipd/DexAvatar
selection_root="${project}/outputs/phoenix_soke_full_v1/selections"
h32_root="${project}/data/SignAvatars/datasets/language2motion/annotations/SMPL-X_phoenix"
output="${project}/outputs/phoenix_soke_full_v1/h32_train_dev_frontend_audit.json"

cd "${project}"
while true; do
    if [[ -f "${output}" ]]; then
        echo "H32 audit already complete: ${output}"
        exit 0
    fi
    missing=$(python - "${selection_root}" "${h32_root}" <<'PY'
import json, sys
from pathlib import Path
selection_root, h32_root = map(Path, sys.argv[1:])
names = []
for split in ("train", "dev"):
    payload = json.loads((selection_root / split / "selection.json").read_text())
    names.extend(str(item["source_clip"]) for item in payload["clips"])
print(sum(not (h32_root / f"{name}.pkl").is_file() for name in names))
PY
)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] H32 audit watcher missing=${missing}"
    if [[ "${missing}" == 0 ]]; then
        # Existing workers were launched before atomic publication was added.
        # Let the last direct pickle write settle before opening every payload.
        sleep 10
        python -m phase2_refiner.data.audit_phoenix_h32_frontend \
            --selection-root "${selection_root}" \
            --h32-root "${h32_root}" \
            --splits train dev \
            --minimum-age-seconds 5 \
            --output "${output}"
        exit 0
    fi
    sleep 60
done
