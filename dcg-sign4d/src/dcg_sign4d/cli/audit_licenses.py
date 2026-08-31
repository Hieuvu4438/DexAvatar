from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcg_sign4d.third_party import audit_dposer_runtime, audit_third_party


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit source/license/checkpoint provenance")
    parser.add_argument("--third-party", default="third_party")
    parser.add_argument("--manifest", default="third_party/manifest.yaml")
    parser.add_argument("--dposer-runtime-root")
    parser.add_argument("--dposer-registry", default="configs/diffusion/dposer_x_registry.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"third_party": audit_third_party(args.third_party, args.manifest)}
    if args.dposer_runtime_root:
        report["dposer_runtime"] = audit_dposer_runtime(
            args.dposer_runtime_root, args.dposer_registry
        )
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(f"immutable audit exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
