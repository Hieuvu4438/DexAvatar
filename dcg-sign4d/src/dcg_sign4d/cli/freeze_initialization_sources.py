from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcg_sign4d.utils.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze hashes of existing DexAvatar PKLs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dexavatar-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable source registry exists: {output}")
    manifest_path = Path(args.manifest)
    items = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    root = Path(args.dexavatar_root)
    for item in items:
        clip_id = item["clip_id"]
        files = []
        for frame_id in item["frame_ids"]:
            path = root / clip_id / "smplifyx" / "results" / f"low_{frame_id}.pkl"
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(
                {
                    "frame_id": frame_id,
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
            )
        rows.append({"clip_id": clip_id, "files": files})
    payload = {
        "schema_version": "1.0",
        "status": "frozen_before_pickle_conversion",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "dexavatar_root": str(root.resolve()),
        "clips": rows,
        "clip_count": len(rows),
        "frame_count": sum(len(row["files"]) for row in rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "clips": len(rows), "frames": payload["frame_count"]}))


if __name__ == "__main__":
    main()
