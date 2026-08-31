import json
from pathlib import Path

import importlib.util


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "report_phoenix_soke_full_progress.py"
)
SPEC = importlib.util.spec_from_file_location("phoenix_progress_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_h32_worker_progress_uses_exact_sorted_modulo_and_critical_eta(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "videos"
    h32_root = tmp_path / "h32"
    (video_root / "train").mkdir(parents=True)
    h32_root.mkdir()
    for name in ("a", "b", "c", "d", "e"):
        (video_root / "train" / f"{name}.mp4").touch()

    now = 10_000.0
    for name, age in (("a", 10.0), ("c", 20.0), ("b", 30.0)):
        path = h32_root / f"{name}.pkl"
        path.touch()
        path.touch()
        import os

        os.utime(path, (now - age, now - age))

    result = MODULE._h32_worker_progress(
        video_root=video_root,
        h32_root=h32_root,
        required_names=["a", "b", "c", "d", "e"],
        required_frame_counts={"a": 10, "b": 20, "c": 30, "d": 40, "e": 50},
        num_workers=2,
        recent_window_seconds=600.0,
        now=now,
    )

    worker0, worker1 = result["workers"]
    assert worker0["declared_clips"] == 3
    assert worker0["complete_clips"] == 2
    assert worker0["missing_clips"] == 1
    assert worker0["recent_complete_clips"] == 2
    assert worker0["clip_rate_estimated_remaining_hours"] == 1 / 12
    assert worker0["declared_source_frames"] == 90
    assert worker0["complete_source_frames"] == 40
    assert worker0["estimated_remaining_hours"] == 50 / 240
    assert worker0["eta_basis"] == "source_video_frames"
    assert worker1["declared_clips"] == 2
    assert worker1["complete_clips"] == 1
    assert worker1["missing_clips"] == 1
    assert worker1["recent_complete_clips"] == 1
    assert worker1["clip_rate_estimated_remaining_hours"] == 1 / 6
    assert worker1["declared_source_frames"] == 60
    assert worker1["complete_source_frames"] == 20
    assert worker1["estimated_remaining_hours"] == 40 / 120
    assert result["critical_estimated_remaining_hours"] == 40 / 120
    assert result["unknown_required_clips"] == []
    assert result["unknown_required_frame_counts"] == []


def test_h32_worker_progress_refuses_finite_critical_eta_for_unknown_clip(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "videos"
    h32_root = tmp_path / "h32"
    (video_root / "train").mkdir(parents=True)
    h32_root.mkdir()
    (video_root / "train" / "known.mp4").touch()

    result = MODULE._h32_worker_progress(
        video_root=video_root,
        h32_root=h32_root,
        required_names=["known", "unknown"],
        num_workers=2,
        recent_window_seconds=600.0,
        now=10_000.0,
    )

    assert result["unknown_required_clips"] == ["unknown"]


def test_h32_segment_progress_tracks_disjoint_bounded_slices(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "videos"
    h32_root = tmp_path / "h32"
    (video_root / "train").mkdir(parents=True)
    h32_root.mkdir()
    for name in ("a", "b", "c", "d", "e"):
        (video_root / "train" / f"{name}.mp4").touch()

    now = 10_000.0
    import os

    for name in ("a", "b", "c"):
        path = h32_root / f"{name}.pkl"
        path.touch()
        os.utime(path, (now - 10.0, now - 10.0))

    result = MODULE._h32_segment_progress(
        video_root=video_root,
        h32_root=h32_root,
        required_names=["a", "b", "c", "d", "e"],
        required_frame_counts={name: 10 for name in "abcde"},
        segments=[
            {"segment": "worker0", "worker": 0, "num_workers": 2},
            {
                "segment": "worker1_lower",
                "worker": 1,
                "num_workers": 2,
                "assigned_stop": 1,
            },
            {
                "segment": "worker1_tail",
                "worker": 1,
                "num_workers": 2,
                "assigned_start": 1,
            },
        ],
        recent_window_seconds=600.0,
        now=now,
    )

    assert result["covered_required_clips"] == 5
    assert result["missing_required_clips"] == []
    assert result["overlap_required_clips"] == []
    worker0, lower, tail = result["segments"]
    assert worker0["declared_required_clips"] == 3
    assert worker0["complete_required_clips"] == 2
    assert lower["declared_required_clips"] == 1
    assert lower["complete_required_clips"] == 1
    assert tail["declared_required_clips"] == 1
    assert tail["complete_required_clips"] == 0


def test_wilor_audit_status_summarizes_ledger_without_payloads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "schema": "audit-v1",
                "timestamp_utc": "2026-08-26T00:00:00+00:00",
                "all_verified": False,
                "splits": {"train": {"verified_shards": 2}},
                "verified": {
                    "train/shard_0000": {"records": 10, "hamer_dropouts": 1},
                    "train/shard_0001": {"records": 12, "hamer_dropouts": 2},
                },
            }
        )
    )
    status = MODULE._wilor_audit_status(path, path.stat().st_mtime + 5.0)
    assert status["exists"]
    assert status["verified_shards"] == 2
    assert status["verified_records"] == 22
    assert status["hamer_dropouts"] == 3
    assert status["age_seconds"] == 5.0


def test_h32_incremental_audit_status_omits_verified_payload_ledger(
    tmp_path: Path,
) -> None:
    path = tmp_path / "h32.json"
    path.write_text(
        json.dumps(
            {
                "schema": "h32-audit-v1",
                "timestamp_utc": "2026-08-26T00:00:00+00:00",
                "all_verified": False,
                "declared_clips": 10,
                "verified_clips": 4,
                "pending_clips": 6,
                "unstable_clips": ["writing"],
                "newly_validated": 2,
                "reused": 2,
                "h32_verified_content_set_sha256": "abc",
                "splits": {"train": {"verified_clips": 4}},
                "verified": {"large-payload": {"sha256": "do-not-copy"}},
            }
        )
    )
    status = MODULE._h32_incremental_audit_status(
        path, path.stat().st_mtime + 5.0
    )
    assert status["exists"]
    assert status["verified_clips"] == 4
    assert status["unstable_clips"] == 1
    assert "verified" not in status
