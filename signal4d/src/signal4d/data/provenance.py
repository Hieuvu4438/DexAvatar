from __future__ import annotations

import json
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


@dataclass
class RunProvenance:
    run_id: str
    git_commit: str
    dirty_worktree: bool
    manifest_sha256: str
    config_sha256: str
    seed: int
    environment: dict[str, str]
    started_at_utc: str
    completed_at_utc: str | None = None

    @classmethod
    def start(cls, manifest_sha256: str, config_sha256: str, seed: int) -> RunProvenance:
        return cls(
            run_id=str(uuid.uuid4()),
            git_commit=_git(["rev-parse", "HEAD"]),
            dirty_worktree=bool(_git(["status", "--porcelain"])),
            manifest_sha256=manifest_sha256,
            config_sha256=config_sha256,
            seed=seed,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": str(torch.version.cuda),
                "gpu_class": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def finish(self) -> None:
        self.completed_at_utc = datetime.now(timezone.utc).isoformat()

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
