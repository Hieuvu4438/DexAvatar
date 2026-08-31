"""Proposal-name contact command with verified reuse of existing pseudo labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dcg_sign4d.data.reuse_index import build_reuse_index


class ContactReuseConfig(BaseModel):
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
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    config = ContactReuseConfig.model_validate(yaml.safe_load(Path(args.config).read_text("utf-8")))
    if not config.development_only:
        raise PermissionError("provisional proximity labels cannot enter production training")
    report = build_reuse_index(
        artifact_type="provisional_contact_geometry",
        source_root=config.source_root,
        source_report_name="compilation_report.json",
        expected_source_report_sha256=config.source_report_sha256,
        source_marker="COMPILATION_COMPLETE",
        per_clip_artifact_name="contact_geometry.npz",
        per_clip_hash_field="contact_geometry_sha256",
        manifest_path=args.manifest,
        output=config.output,
        development_only=True,
        required_split=args.split,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
