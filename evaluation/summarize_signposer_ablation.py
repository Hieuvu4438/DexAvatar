#!/usr/bin/env python3
"""Create the 57-sign comparison table for the SignPoser ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = {
    "UBody(-F)": "tr_upper_body_minus_face_mm",
    "LHand": "tr_left_hand_mm",
    "RHand": "tr_right_hand_mm",
}


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "", "None") else float(value)


def _bootstrap_ci(values: np.ndarray, seed: int = 42, samples: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/signposer_ablation_evaluation"),
    )
    parser.add_argument("--with-label", default="DexAvatar_with_SignPosers")
    parser.add_argument("--without-label", default="ExpertInit_no_SignPosers")
    args = parser.parse_args()

    root = args.evaluation_root
    with_root = root / "methods" / args.with_label
    without_root = root / "methods" / args.without_label
    with_rows = _read_rows(with_root / "per_clip.csv")
    without_rows = _read_rows(without_root / "per_clip.csv")
    if with_rows.keys() != without_rows.keys() or len(with_rows) != 57:
        raise ValueError("Expected identical 57-sign populations")

    fieldnames = ["sign", "frames", "one_handed_class_0"]
    for display_name in METRICS:
        prefix = display_name.lower().replace("(-f)", "_minus_face")
        fieldnames.extend(
            [
                f"{prefix}_with_signposers_mm",
                f"{prefix}_expert_init_mm",
                f"{prefix}_effect_mm",
                f"{prefix}_winner",
            ]
        )

    comparison_rows: list[dict[str, object]] = []
    effects: dict[str, list[float]] = {name: [] for name in METRICS}
    for sign in sorted(with_rows):
        current_with = with_rows[sign]
        current_without = without_rows[sign]
        row: dict[str, object] = {
            "sign": sign,
            "frames": int(current_with["frames"]),
            "one_handed_class_0": current_with["one_handed_class_0"],
        }
        for display_name, metric in METRICS.items():
            prefix = display_name.lower().replace("(-f)", "_minus_face")
            with_value = _optional_float(current_with[metric])
            without_value = _optional_float(current_without[metric])
            if with_value is None or without_value is None:
                effect = None
                winner = "not_evaluated"
            else:
                # Positive effect means applying the SignPosers lowers error.
                effect = without_value - with_value
                effects[display_name].append(effect)
                winner = (
                    "with_signposers"
                    if effect > 1e-9
                    else "expert_init"
                    if effect < -1e-9
                    else "tie"
                )
            row[f"{prefix}_with_signposers_mm"] = with_value
            row[f"{prefix}_expert_init_mm"] = without_value
            row[f"{prefix}_effect_mm"] = effect
            row[f"{prefix}_winner"] = winner
        comparison_rows.append(row)

    csv_path = root / "per_sign_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    per_sign_markdown = [
        "# Per-sign SignPoser ablation",
        "",
        "Each metric cell is `with SignPosers / expert init / effect` in mm. Effect is expert init minus with-SignPosers, so positive means the SignPoser pipeline is better.",
        "",
        "| Sign | Frames | UBody(-F) | LHand | RHand |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        cells: list[str] = []
        for display_name in METRICS:
            prefix = display_name.lower().replace("(-f)", "_minus_face")
            with_value = row[f"{prefix}_with_signposers_mm"]
            without_value = row[f"{prefix}_expert_init_mm"]
            effect = row[f"{prefix}_effect_mm"]
            cells.append(
                "—"
                if effect is None
                else f"{float(with_value):.3f} / {float(without_value):.3f} / {float(effect):+.3f}"
            )
        per_sign_markdown.append(
            f"| {row['sign']} | {row['frames']} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    per_sign_markdown.append("")
    (root / "PER_SIGN.md").write_text("\n".join(per_sign_markdown), encoding="utf-8")

    with_summary = json.loads((with_root / "summary.json").read_text(encoding="utf-8"))
    without_summary = json.loads((without_root / "summary.json").read_text(encoding="utf-8"))
    statistics: dict[str, object] = {
        "effect_definition": "expert_init_error - with_signposers_error; positive means SignPosers improve",
        "clips": 57,
        "frames": 1493,
        "metrics": {},
    }
    markdown = [
        "# SignBPoser + SignHPoser combined ablation",
        "",
        "Effect = expert-initialization error minus fitted-with-SignPosers error; positive is better for SignPosers.",
        "",
        "| Metric | With SignPosers (micro) | Expert init (micro) | Effect | Relative | Sign wins | Expert wins | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for display_name, metric in METRICS.items():
        values = np.asarray(effects[display_name], dtype=np.float64)
        with_micro = float(with_summary[metric])
        without_micro = float(without_summary[metric])
        micro_effect = without_micro - with_micro
        relative = micro_effect / without_micro * 100.0
        wins = int((values > 1e-9).sum())
        losses = int((values < -1e-9).sum())
        ties = int(len(values) - wins - losses)
        test = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
        metric_stats = {
            "evaluated_signs": int(len(values)),
            "with_signposers_vertex_micro_mm": with_micro,
            "expert_init_vertex_micro_mm": without_micro,
            "vertex_micro_effect_mm": micro_effect,
            "vertex_micro_relative_effect_percent": relative,
            "sign_macro_effect_mean_mm": float(values.mean()),
            "sign_macro_effect_median_mm": float(np.median(values)),
            "sign_macro_effect_bootstrap_95ci_mm": _bootstrap_ci(values),
            "signposers_wins": wins,
            "expert_init_wins": losses,
            "ties": ties,
            "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_two_sided_p": float(test.pvalue),
        }
        statistics["metrics"][display_name] = metric_stats
        markdown.append(
            f"| {display_name} | {with_micro:.4f} | {without_micro:.4f} | "
            f"{micro_effect:+.4f} | {relative:+.2f}% | {wins}/{len(values)} | "
            f"{losses}/{len(values)} | {test.pvalue:.3g} |"
        )

    markdown.extend(
        [
            "",
            "This is a combined end-to-end ablation. It compares the released poser-constrained fitting output against the frozen expert initialization with zero optimization steps; it does not separate SignBPoser from SignHPoser or isolate the optimizer from the priors.",
            "",
        ]
    )
    (root / "ablation_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "REPORT.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
