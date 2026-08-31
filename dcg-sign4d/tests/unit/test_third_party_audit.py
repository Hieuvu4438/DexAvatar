import json
import subprocess

import yaml

from dcg_sign4d.third_party import audit_dposer_runtime, audit_third_party


def init_repo(root, license_text="license"):
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "fixture@example.test"], check=True
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Fixture"], check=True)
    (root / "LICENSE").write_text(license_text)
    subprocess.run(["git", "-C", str(root), "add", "LICENSE"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "remote", "add", "origin", "https://example.test/x.git"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_source_and_license_audit(tmp_path):
    source = tmp_path / "third" / "fixture"
    source.parent.mkdir()
    commit = init_repo(source)
    import hashlib

    license_hash = hashlib.sha256((source / "LICENSE").read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "status": "candidate",
                "repositories": [
                    {
                        "name": "fixture",
                        "url": "https://example.test/x.git",
                        "commit": commit,
                        "license_file": "LICENSE",
                        "license_sha256": license_hash,
                    }
                ],
            }
        )
    )
    report = audit_third_party(source.parent, manifest)
    assert report["engineering_pass"] is True
    assert report["scientifically_frozen"] is False


def test_dposer_registry_hashes(tmp_path):
    root = tmp_path / "dposer"
    commit = init_repo(root)
    import hashlib

    digest = hashlib.sha256((root / "LICENSE").read_bytes()).hexdigest()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "status": "candidate",
                "source_commit": commit,
                "files": [{"path": "LICENSE", "sha256": digest}],
            }
        )
    )
    report = audit_dposer_runtime(root, registry)
    assert report["source_commit_pass"]
    assert report["checkpoint_hashes_pass"]
