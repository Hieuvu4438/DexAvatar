from __future__ import annotations

import argparse
import json

from dcg_sign4d.evaluation.sgnify import evaluate_sgnify_obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strict SGNify OBJ predictions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--author-asset-root", required=True)
    parser.add_argument("--author-sign-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trusted-author-assets", action="store_true")
    args = parser.parse_args()
    summary = evaluate_sgnify_obj(
        manifest_path=args.manifest,
        prediction_root=args.predictions,
        gt_root=args.gt_root,
        author_asset_root=args.author_asset_root,
        author_sign_file=args.author_sign_file,
        output_root=args.output,
        trusted_author_assets=args.trusted_author_assets,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
