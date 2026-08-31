"""Proposal-name observation command that reuses frozen detector outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dcg_sign4d.data.reuse_index import build_reuse_index
from dcg_sign4d.observations.compiler import compile_calibrated_keypoint_caches


class ObservationReuseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = Field(pattern="^(artifact_replay|calibrate_reused_raw)$")
    development_only: bool
    source_root: Path
    source_report_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    output: Path
    calibrator: Path | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    extractor_checkpoint_sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    preprocessing: dict[str, object] | None = None
    allow_incomplete_extractor_provenance: bool = False
    use_raw_frame_source_identity: bool = False

    @model_validator(mode="after")
    def calibrated_fields(self) -> ObservationReuseConfig:
        if self.backend == "calibrate_reused_raw" and any(
            value is None
            for value in (
                self.calibrator,
                self.extractor_name,
                self.extractor_version,
                self.extractor_checkpoint_sha256,
                self.preprocessing,
            )
        ):
            raise ValueError("calibrated compilation requires calibrator/extractor/preprocessing")
        if self.allow_incomplete_extractor_provenance and not self.development_only:
            raise ValueError("incomplete extractor provenance is development-only")
        if self.use_raw_frame_source_identity and not self.development_only:
            raise ValueError("raw frame source identity is development-only")
        return self


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    config = ObservationReuseConfig.model_validate(
        yaml.safe_load(Path(args.config).read_text("utf-8"))
    )
    if config.backend == "calibrate_reused_raw":
        report = compile_calibrated_keypoint_caches(
            raw_root=config.source_root,
            source_report_sha256=config.source_report_sha256,
            manifest_path=args.manifest,
            calibrator_path=config.calibrator,
            extractor={
                "name": config.extractor_name,
                "version": config.extractor_version,
                "checkpoint_sha256": config.extractor_checkpoint_sha256,
            },
            preprocessing=config.preprocessing,
            output=config.output,
            development_only=config.development_only,
            allow_incomplete_extractor_provenance=(config.allow_incomplete_extractor_provenance),
            use_raw_frame_source_identity=config.use_raw_frame_source_identity,
        )
        print(
            json.dumps({key: value for key, value in report.items() if key != "per_clip"}, indent=2)
        )
        return
    if not config.development_only:
        raise PermissionError("raw reused Sapiens scores are not calibrated observations")
    report = build_reuse_index(
        artifact_type="raw_sapiens_observations",
        source_root=config.source_root,
        source_report_name="compilation_report.json",
        expected_source_report_sha256=config.source_report_sha256,
        source_marker="COMPILATION_COMPLETE",
        per_clip_artifact_name="raw_keypoints.npz",
        per_clip_hash_field="artifact_sha256",
        manifest_path=args.manifest,
        output=config.output,
        development_only=True,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
