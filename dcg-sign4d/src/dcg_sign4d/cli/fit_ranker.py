"""Fit and freeze hypothesis-ranking weights on validation clips only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dcg_sign4d.inference.ranker_fit import fit_ranker
from dcg_sign4d.utils.hashing import file_sha256


class RankerFitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    development_only: bool = False
    input_csv: Path
    output: Path
    steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    l2_weight: float = Field(ge=0)
    minimum_pair_accuracy: float = Field(ge=0, le=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = RankerFitConfig.model_validate(yaml.safe_load(config_path.read_text("utf-8")))
    report = fit_ranker(
        config.input_csv,
        config.output,
        steps=config.steps,
        learning_rate=config.learning_rate,
        l2_weight=config.l2_weight,
        minimum_pair_accuracy=config.minimum_pair_accuracy,
        seed=config.seed,
        development_only=config.development_only,
        config_sha256=file_sha256(config_path),
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
