from __future__ import annotations

import argparse
import json

from dcg_sign4d.inference.artifacts import validate_prediction_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an immutable DCG prediction artifact")
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    report = validate_prediction_artifact(args.artifact)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
