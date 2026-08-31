"""Config-driven strict SGNify evaluator required by the proposal CLI contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dcg_sign4d.evaluation.sgnify import evaluate_sgnify_obj
from dcg_sign4d.utils.hashing import file_sha256


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development_only: bool = False
    manifest: Path
    predictions: Path
    gt_root: Path
    author_asset_root: Path
    author_sign_file: Path
    trusted_author_assets: bool
    output: Path
    primary_endpoint: str = Field(pattern="^root_aligned_hand_pve$")
    bootstrap_unit: str = Field(pattern="^signer$")


def _manifest_rows(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("empty evaluation manifest")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = EvaluationConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    if not config.trusted_author_assets:
        raise PermissionError("strict evaluation requires explicit trusted author assets")
    rows = _manifest_rows(config.manifest)
    signer_ids = [str(row.get("signer_id", "unknown")) for row in rows]
    if not config.development_only:
        if any(value.lower() in {"", "unknown"} for value in signer_ids):
            raise ValueError("production evaluation requires signer IDs")
        if len(set(signer_ids)) < 2:
            raise ValueError("production evaluation requires at least two signer clusters")
        if any(row.get("split") != "test" for row in rows):
            raise ValueError("production evaluation manifest must contain only test clips")
        if any(row.get("allowed_for_final_reporting") is not True for row in rows):
            raise ValueError("production clips must be explicitly allowed for final reporting")
    summary = evaluate_sgnify_obj(
        manifest_path=config.manifest,
        prediction_root=config.predictions,
        gt_root=config.gt_root,
        author_asset_root=config.author_asset_root,
        author_sign_file=config.author_sign_file,
        output_root=config.output,
        trusted_author_assets=True,
    )
    # The lower-level evaluator marks its standalone artifact complete.  This
    # orchestration layer adds frozen config identity, so remove that marker
    # until every additional file has been committed successfully.
    (config.output / "EVALUATION_COMPLETE").unlink()
    summary.update(
        {
            "evaluation_config_sha256": file_sha256(config_path),
            "development_only": config.development_only,
            "scientific_status": (
                "DEVELOPMENT_ONLY" if config.development_only else "FROZEN_TEST_EVALUATION"
            ),
        }
    )
    (config.output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (config.output / "evaluation_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (config.output / "EVALUATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
