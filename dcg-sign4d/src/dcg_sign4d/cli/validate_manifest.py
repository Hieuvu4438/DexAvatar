from __future__ import annotations

import argparse

from dcg_sign4d.data.manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DCG-Sign4D JSONL manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-existing-video", action="store_true")
    args = parser.parse_args()
    items = load_manifest(args.manifest, require_existing_video=args.require_existing_video)
    print(f"valid clips={len(items)} frames={sum(x.effective_frame_count for x in items)}")


if __name__ == "__main__":
    main()
