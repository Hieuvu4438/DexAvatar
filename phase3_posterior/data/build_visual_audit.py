"""Build deterministic relation-geometry evidence sheets for the P3-G0 audit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from phase3_posterior.data.cache_schema import load_relation_sidecar
from phase3_posterior.provenance import atomic_json, sha256_file


def _panel(nodes, valid, edge_index, contact, contact_valid, label: str) -> np.ndarray:
    canvas = np.full((240, 320, 3), 248, dtype=np.uint8)
    usable = valid & np.isfinite(nodes).all(axis=-1)
    span = float(np.ptp(nodes[usable], axis=0).max()) if usable.any() else 0.0

    def draw_projection(
        dimensions: tuple[int, int], x_center: float, title: str
    ) -> None:
        points = nodes[:, list(dimensions)].copy()
        if not usable.any():
            return
        low = np.quantile(points[usable], 0.02, axis=0)
        high = np.quantile(points[usable], 0.98, axis=0)
        scale = 90.0 / max(float((high - low).max()), 1e-5)
        center = (low + high) * 0.5
        pixels = (points - center) * scale + np.array([x_center, 125.0])
        pixels[:, 1] = 250.0 - pixels[:, 1]
        for edge, (start, end) in enumerate(edge_index.T):
            if not (usable[start] and usable[end]):
                continue
            color = (
                (20, 20, 230)
                if contact_valid[edge] and contact[edge]
                else (175, 175, 175)
            )
            cv2.line(
                canvas,
                tuple(np.rint(pixels[start]).astype(int)),
                tuple(np.rint(pixels[end]).astype(int)),
                color,
                1,
                cv2.LINE_AA,
            )
        for index in np.flatnonzero(usable):
            color = (255, 100, 20) if index >= 10 else (40, 155, 30)
            cv2.circle(canvas, tuple(np.rint(pixels[index]).astype(int)), 3, color, -1)
        cv2.putText(
            canvas,
            title,
            (int(x_center - 70), 232),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (60, 60, 60),
            1,
        )

    draw_projection((0, 1), 80.0, "XY")
    draw_projection((0, 2), 240.0, "XZ")
    cv2.putText(
        canvas,
        f"{label} span={span:.3f}m",
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
    )
    return canvas


def build(
    relation_roots: list[Path], output: Path, samples_per_source: int, seed: int
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    clips_dir = output / "clips"
    sheets_dir = output / "sheets"
    clips_dir.mkdir()
    sheets_dir.mkdir()
    rng = random.Random(seed)
    records = []
    rendered = []
    for root in relation_roots:
        candidates = sorted(root.glob("*.npz"))
        if len(candidates) < samples_per_source:
            raise ValueError(
                f"{root} has {len(candidates)} sidecars, need {samples_per_source}"
            )
        selected = rng.sample(candidates, samples_per_source)
        for path in selected:
            relation = load_relation_sidecar(path)
            positions = np.linspace(0, len(relation.node_positions) - 1, 4, dtype=int)
            panels = [
                _panel(
                    relation.node_positions[position],
                    relation.node_valid[position],
                    relation.edge_index,
                    relation.contact_target[position],
                    relation.contact_valid[position],
                    f"{root.name} {relation.clip_id} f={position}",
                )
                for position in positions
            ]
            image = np.concatenate(panels, axis=1)
            target = clips_dir / f"{root.name}_{relation.clip_id}.jpg"
            cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            rendered.append(target)
            records.append(
                {
                    "clip_id": relation.clip_id,
                    "source": root.name,
                    "relation_path": str(path.resolve()),
                    "relation_sha256": sha256_file(path),
                    "evidence_image": str(target.resolve()),
                    "frames_inspected": positions.tolist(),
                    "decision": "PENDING_VISUAL_REVIEW",
                }
            )
    for start in range(0, len(rendered), 10):
        images = [cv2.imread(str(path)) for path in rendered[start : start + 10]]
        sheet = np.concatenate(images, axis=0)
        cv2.imwrite(
            str(sheets_dir / f"sheet_{start // 10 + 1:02d}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
    report = {
        "schema": "phase3-relation-visual-audit-v1",
        "seed": seed,
        "samples_per_source": samples_per_source,
        "reviewed_clips": 0,
        "catastrophic_failures": 0,
        "records": records,
    }
    atomic_json(output / "audit_pending.json", report)
    return {"clips": len(records), "sheets": len(list(sheets_dir.glob("*.jpg")))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.relation_root, args.output, args.samples_per_source, args.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
