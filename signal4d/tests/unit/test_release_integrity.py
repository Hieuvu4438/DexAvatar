import pytest

from signal4d.cli.confirmatory import run as confirmatory_run
from signal4d.cli.freeze_release import run as freeze_run


def test_confirmatory_rejects_changed_frozen_artifact(tmp_path) -> None:
    config = tmp_path / "method.yaml"
    manifest = tmp_path / "manifest.jsonl"
    artifact = tmp_path / "artifact.txt"
    config.write_text("method: frozen\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    artifact.write_text("before\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze_run(str(freeze), [str(config)], [str(manifest)], [str(artifact)])
    artifact.write_text("after\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen artifact tree changed"):
        confirmatory_run(
            str(freeze),
            str(config),
            str(manifest),
            "unused-cache",
            str(tmp_path / "output"),
            "unused-gt",
            "unused-gt-cache",
            "unused-model",
            "unused-upper",
            "unused-left",
            "unused-right",
            "cpu",
        )
