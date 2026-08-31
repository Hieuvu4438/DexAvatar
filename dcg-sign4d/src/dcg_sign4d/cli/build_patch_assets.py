from __future__ import annotations

import argparse
import json

from dcg_sign4d.geometry.patch_builder import (
    build_provisional_smplx_patch_map,
    write_patch_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic development-only SMPL-X patch map"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--vertex-ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_provisional_smplx_patch_map(
        model_path=args.model,
        segmentation_path=args.segmentation,
        vertex_ids_path=args.vertex_ids,
    )
    output = write_patch_map(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": payload["sha256"],
                "patch_count": len(payload["patches"]),
                "edge_count": len(payload["admissible_edges"]),
                "development_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
