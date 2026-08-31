"""Train SIGNAL4D's isolated external-only residual architecture.

This entry point intentionally reuses the mature Phase-2 optimizer, losses,
checkpointing, and validation loop without modifying that frozen code.  It
replaces only the model factory after a fail-closed lineage audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from phase2_refiner import train as phase2_train

from .leakage import audit_protocol
from .model import model_from_config


def _preflight(argv: list[str]) -> tuple[dict, dict]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    with known.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    data = config.get("data", {})
    audit = audit_protocol(
        data["train_glob"],
        data["val_glob"],
        data.get("calibration_glob"),
        scan_clips=bool(data.get("audit_all_clips", True)),
    )
    print("external_only_lineage=" + json.dumps(audit, sort_keys=True))
    return config, audit


def main() -> None:
    _preflight(sys.argv[1:])

    def make_model(config: dict):
        return model_from_config(config, initialize=True)

    phase2_train.make_model = make_model
    phase2_train.main()


if __name__ == "__main__":
    main()
