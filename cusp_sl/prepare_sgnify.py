"""Build the locked 1,493-frame input cache without reading SGNify pose labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from pathlib import Path

from cusp_sl.config import load_config
from phase2_refiner.data.build_observation_cache import build_clip
from phase2_refiner.data.cache_schema import SCHEMA_VERSION, save_cache_clip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"No frame number in {path}")
    return int(match.group(1))


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def protocol_rows(config) -> list[dict[str, str]]:
    baseline = Path(config.protocol.baseline_root)
    gt_root = Path(config.protocol.gt_root)
    segments = json.loads(Path(config.protocol.segments_file).read_text(encoding="utf-8"))
    signs = [line.split()[0] for line in Path(config.protocol.signs_file).read_text().splitlines() if line.strip()]
    rows = []
    for sign in sorted(signs):
        start, end = segments[sign]
        gt_candidates = sorted((gt_root / sign).glob("*.obj"), key=number)
        gt_selected = [path for path in gt_candidates if start * 2 <= number(path) <= end * 2]
        result_dir = baseline / sign / "smplifyx" / "results"
        mesh_dir = baseline / sign / "smplifyx" / "meshes"
        predictions = sorted(result_dir.glob("*.pkl"), key=number)
        if len(predictions) != len(gt_selected):
            raise ValueError(f"{sign}: {len(predictions)} predictions != {len(gt_selected)} selected GT meshes")
        for position, (prediction, gt) in enumerate(zip(predictions, gt_selected)):
            mesh = mesh_dir / f"{prediction.stem}.obj"
            image = Path(config.protocol.frames_root) / sign / f"{prediction.stem}.png"
            if not mesh.is_file() or not image.is_file():
                raise FileNotFoundError(mesh if not mesh.is_file() else image)
            rows.append({
                "sign": sign,
                "position": str(position),
                "frame_name": prediction.stem,
                "frame_number": str(number(prediction)),
                "prediction_path": str(prediction.resolve()),
                "baseline_mesh": str(mesh.resolve()),
                "image_path": str(image.resolve()),
                "gt_path_author_pairing": str(gt.resolve()),
            })
    if len(rows) != config.protocol.expected_frames:
        raise ValueError(f"Expected {config.protocol.expected_frames} rows, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new empty versioned output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = protocol_rows(config)
    csv_path = args.output / "author_1493_frame_manifest.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = Path(config.protocol.baseline_root)
    cache_entries = []
    for sign in sorted({row["sign"] for row in rows}):
        sign_rows = [row for row in rows if row["sign"] == sign]
        results = [Path(row["prediction_path"]) for row in sign_rows]
        sapiens_path = baseline / sign / "sapiens.pkl"
        hamer_path = baseline / sign / "hamer" / "hamer.pkl"
        clip = build_clip(
            sign, results, Path(config.protocol.frames_root).resolve(),
            load_pickle(sapiens_path) if sapiens_path.exists() else {},
            load_pickle(hamer_path) if hamer_path.exists() else {},
            None, config.data.fps,
            {"dataset": "SGNify", "role": "locked_evaluation_input_only", "target_provider": "none"},
        )
        cache_path = args.output / "clips" / f"{sign}.npz"
        save_cache_clip(cache_path, clip)
        cache_entries.append(str(cache_path.relative_to(args.output)))
        print(f"[prepare] {sign}: {len(results)} frames")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "role": "evaluation_input_only_no_pose_targets",
        "expected_frames": config.protocol.expected_frames,
        "observed_frames": len(rows),
        "frame_manifest": csv_path.name,
        "frame_manifest_sha256": sha256(csv_path),
        "signs_sha256": sha256(Path(config.protocol.signs_file)),
        "segments_sha256": sha256(Path(config.protocol.segments_file)),
        "clips": cache_entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

