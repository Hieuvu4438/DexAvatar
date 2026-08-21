from __future__ import annotations

import csv
import html
import json
from pathlib import Path

PRIMARY_METRICS = (
    "tr_v2v_upper_body_mm",
    "tr_v2v_left_hand_mm",
    "tr_v2v_right_hand_mm",
    "velocity_error",
    "acceleration_error",
    "jerk_error",
    "coverage",
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("inputs must use NAME=PATH")
    return name, Path(path)


def _geometry_svg(rows: list[dict[str, object]]) -> str:
    metrics = PRIMARY_METRICS[:3]
    values = [float(row[metric]) for row in rows for metric in metrics]
    maximum = max(values, default=1.0)
    width = 900
    height = 100 + 70 * len(rows)
    colors = ("#2457C5", "#11A579", "#F2B701", "#E73F74")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18">'
        "Frozen-test TR-V2V (lower is better)</text>",
    ]
    for row_index, row in enumerate(rows):
        y = 55 + row_index * 70
        parts.append(
            f'<text x="20" y="{y + 15}" font-family="sans-serif" font-size="13">'
            f"{html.escape(str(row['method']))}</text>"
        )
        for metric_index, metric in enumerate(metrics):
            value = float(row[metric])
            bar_width = 540 * value / maximum if maximum else 0
            bar_y = y + metric_index * 16
            parts.append(
                f'<rect x="220" y="{bar_y}" width="{bar_width:.2f}" height="12" '
                f'fill="{colors[metric_index]}"/>'
            )
            parts.append(
                f'<text x="{230 + bar_width:.2f}" y="{bar_y + 11}" '
                f'font-family="sans-serif" font-size="11">{metric.split("_")[2]}: '
                f"{value:.3f}</text>"
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def run(
    methods: list[str],
    comparisons: list[str],
    output_root: str,
    stress_slices: str | None = None,
) -> dict[str, object]:
    if not methods:
        raise ValueError("at least one method evaluation is required")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    source_files: dict[str, str] = {}
    for value in methods:
        name, evaluation = _named_path(value)
        summary_path = evaluation / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row: dict[str, object] = {"method": name}
        row.update({metric: summary.get(metric) for metric in PRIMARY_METRICS})
        rows.append(row)
        source_files[name] = str(summary_path.resolve())
    comparison_payload: dict[str, object] = {}
    for value in comparisons:
        name, path = _named_path(value)
        comparison_payload[name] = json.loads(path.read_text(encoding="utf-8"))
        source_files[f"comparison:{name}"] = str(path.resolve())

    slice_rows: list[dict[str, object]] = []
    if stress_slices is not None:
        slice_path = Path(stress_slices)
        slice_config = json.loads(slice_path.read_text(encoding="utf-8"))
        source_files["stress_slices"] = str(slice_path.resolve())
        for value in methods:
            name, evaluation = _named_path(value)
            with (evaluation / "per_clip.csv").open(newline="", encoding="utf-8") as handle:
                clips = list(csv.DictReader(handle))
            for slice_name, specification in slice_config["slices"].items():
                threshold = float(specification["value"])
                operator = specification["operator"]
                selected = [
                    clip
                    for clip in clips
                    if (
                        float(clip["frames"]) <= threshold
                        if operator == "<="
                        else float(clip["frames"]) > threshold
                    )
                ]
                if not selected:
                    continue
                slice_row: dict[str, object] = {
                    "method": name,
                    "slice": slice_name,
                    "clips": len(selected),
                }
                for metric in PRIMARY_METRICS[:6]:
                    total = sum(float(clip[metric]) for clip in selected)
                    slice_row[metric] = total / len(selected)
                slice_rows.append(slice_row)

    with (output / "primary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *PRIMARY_METRICS])
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparisons.json").write_text(
        json.dumps(comparison_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "primary_geometry.svg").write_text(_geometry_svg(rows), encoding="utf-8")
    if slice_rows:
        slice_columns = ["method", "slice", "clips", *PRIMARY_METRICS[:6]]
        with (output / "stress_slices.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=slice_columns)
            writer.writeheader()
            writer.writerows(slice_rows)

    header = "| Method | Body (mm) | Left (mm) | Right (mm) | Velocity | Coverage |"
    markdown = [
        "# Frozen result table",
        "",
        "Generated directly from immutable evaluator summaries.",
        "",
        header,
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {float(row['tr_v2v_upper_body_mm']):.4f} | "
            f"{float(row['tr_v2v_left_hand_mm']):.4f} | "
            f"{float(row['tr_v2v_right_hand_mm']):.4f} | "
            f"{float(row['velocity_error']):.4f} | {float(row['coverage']):.4f} |"
        )
    (output / "results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "generated_from_raw_evaluator_outputs",
        "sources": source_files,
        "methods": [row["method"] for row in rows],
        "comparisons": sorted(comparison_payload),
        "stress_slices": stress_slices,
    }
    (output / "report_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
