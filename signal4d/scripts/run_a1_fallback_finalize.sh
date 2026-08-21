#!/usr/bin/env bash
set -euo pipefail

project="/home/haipd/DexAvatar"
manifest="${project}/signal4d/artifacts/manifests/sgnify_extended_post_test_v1.jsonl"
primary="${project}/signal4d/artifacts/legacy_a1_extended_post_v1"
fallback="${project}/signal4d/artifacts/legacy_a1_hamer_extended_post_v1"
terminal="${project}/outputs/output_baseline"
required="${project}/signal4d/artifacts/legacy_a1_hamer_extended_post_v1/fallback_required.txt"
segments="${project}/signal4d/artifacts/legacy_a1_hamer_extended_post_v1/fallback_segments.json"
retry_required="${project}/signal4d/artifacts/legacy_a1_hamer_extended_post_v1/ensemble_retry_required.txt"
retry_segments="${project}/signal4d/artifacts/legacy_a1_hamer_extended_post_v1/ensemble_retry_segments.json"
report="${project}/signal4d/artifacts/legacy_a1_hamer_extended_post_v1/fallback_finalize.json"

cd "${project}"

# The active primary work queue has PID 1537849. Its jobs may briefly hand off
# between clips, so wait for both the queue and all primary fit children.
while kill -0 1537849 2>/dev/null; do
    sleep 30
done
while pgrep -f 'smplifyx/main.py.*legacy_a1_extended_post_v1' >/dev/null; do
    sleep 30
done

# Retry only the missing Ensemble ranges at conservative concurrency. This
# recovers transient OOM/allocator failures without using labels and preserves
# the strongest preregistered A1 source before falling back to HaMeR/A0.
python - "${manifest}" "${primary}" "${retry_required}" "${retry_segments}" <<'PY'
import json
import sys
from pathlib import Path

manifest, primary_root, output, segment_output = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
required = []
segments = {}
for row in rows:
    missing = [
        frame
        for frame in row["frame_ids"]
        if not (
            primary_root / row["clip_id"] / "smplifyx/results" / f"low_{frame:03d}.pkl"
        ).is_file()
    ]
    if missing:
        required.append(row["clip_id"])
        segments[row["clip_id"]] = [min(missing), max(missing)]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(required) + ("\n" if required else ""), encoding="utf-8")
segment_output.write_text(json.dumps(segments, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ -s "${retry_required}" ]]; then
    set +e
    xargs -r -P2 -I{} conda run -n dexavatar \
        bash signal4d/scripts/run_a1_extended_post_retry_one.sh "{}" "${retry_segments}" \
        <"${retry_required}"
    primary_retry_status=$?
    set -e
else
    primary_retry_status=0
fi

python - "${manifest}" "${primary}" "${fallback}" "${required}" "${segments}" <<'PY'
import json
import sys
from pathlib import Path

manifest, primary_root, fallback_root, output, segment_output = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
required = []
segments = {}
for row in rows:
    missing = [
        frame
        for frame in row["frame_ids"]
        if not (
            primary_root / row["clip_id"] / "smplifyx/results" / f"low_{frame:03d}.pkl"
        ).is_file()
        and not (
            fallback_root / row["clip_id"] / "smplifyx/results" / f"low_{frame:03d}.pkl"
        ).is_file()
    ]
    if missing:
        required.append(row["clip_id"])
        segments[row["clip_id"]] = [min(missing), max(missing)]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(required) + ("\n" if required else ""), encoding="utf-8")
segment_output.write_text(json.dumps(segments, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ -s "${required}" ]]; then
    set +e
    xargs -r -P2 -I{} conda run -n dexavatar \
        bash signal4d/scripts/run_a1_hamer_fallback_one.sh "{}" "${segments}" <"${required}"
    fallback_fit_status=$?
    set -e
else
    fallback_fit_status=0
fi

python - "${manifest}" "${primary}" "${fallback}" "${terminal}" "${required}" "${retry_required}" "${report}" "${primary_retry_status}" "${fallback_fit_status}" <<'PY'
import json
import sys
from pathlib import Path

manifest, primary_root, fallback_root, terminal_root, required_path, retry_path, output = map(
    Path, sys.argv[1:8]
)
primary_retry_status = int(sys.argv[8])
fallback_fit_status = int(sys.argv[9])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
required = [line for line in required_path.read_text().splitlines() if line]
retry_required = [line for line in retry_path.read_text().splitlines() if line]
missing = []
source_counts = {"ensemble_primary": 0, "hamer_a0": 0, "smplerx_a0_terminal": 0}
for row in rows:
    for frame in row["frame_ids"]:
        primary = primary_root / row["clip_id"] / "smplifyx/results" / f"low_{frame:03d}.pkl"
        fallback = fallback_root / row["clip_id"] / "smplifyx/results" / f"low_{frame:03d}.pkl"
        terminal = terminal_root / row["clip_id"] / "smplerx/smplx" / f"low_{frame:03d}.pkl"
        if primary.is_file():
            source_counts["ensemble_primary"] += 1
        elif fallback.is_file():
            source_counts["hamer_a0"] += 1
        elif terminal.is_file():
            source_counts["smplerx_a0_terminal"] += 1
        else:
            missing.append({"clip_id": row["clip_id"], "frame_id": frame})
report = {
    "schema_version": "1.0",
    "required_ensemble_retry_clips": retry_required,
    "required_fallback_clips": required,
    "source_counts": source_counts,
    "ensemble_retry_status": primary_retry_status,
    "fallback_fit_status": fallback_fit_status,
    "missing": missing,
    "passed": not missing,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if missing:
    raise SystemExit(f"fallback finalization left {len(missing)} missing frames")
PY
