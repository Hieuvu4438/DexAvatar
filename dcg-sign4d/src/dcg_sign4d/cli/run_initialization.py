"""Proposal-name initializer command with immutable DexAvatar artifact reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dcg_sign4d.data.reuse_index import build_reuse_index


class InitializationReuseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = Field(pattern="^artifact_replay$")
    development_only: bool
    source_root: Path
    source_report_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    config = InitializationReuseConfig.model_validate(
        yaml.safe_load(Path(args.config).read_text("utf-8"))
    )
    report = build_reuse_index(
        artifact_type="dexavatar_initialization",
        source_root=config.source_root,
        source_report_name="conversion_report.json",
        expected_source_report_sha256=config.source_report_sha256,
        source_marker="CONVERSION_COMPLETE",
        per_clip_artifact_name="trajectory.npz",
        per_clip_hash_field="trajectory_sha256",
        manifest_path=args.manifest,
        output=config.output,
        development_only=config.development_only,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
