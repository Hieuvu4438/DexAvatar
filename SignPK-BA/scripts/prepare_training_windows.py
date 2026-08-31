#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.data.cache_schema import DualObserverBundle, load_dual_observer_bundle
from signpk.data.signavatars_dataset import FORBIDDEN_BENCHMARK_MARKERS, WINDOW_SCHEMA
from signpk.data.window_sampler import all_windows
from signpk.export.diagnostics import write_jsonl
from signpk.models.explicit_tokens import ExplicitTokenBuilder


TARGET_SCHEMA = "signpk-pseudogt-sequence-v1"
REQUIRED_INDEX_FIELDS = {
    "observer_cache",
    "target_cache",
    "signer_id",
    "sequence_id",
    "split",
    "quality_weight",
    "source_dataset",
}
REQUIRED_TARGET_FIELDS = {
    "target_upper_rotmat",
    "target_left_hand_rotmat",
    "target_right_hand_rotmat",
    "base_body_rotmat",
    "betas",
    "translation",
}


def _resolve(index_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (index_path.parent / path).resolve()


def _slice_dataclass(value, indices: tuple[int, ...]):
    index = torch.as_tensor(indices, dtype=torch.long)
    return type(value)(
        **{
            field.name: (
                None
                if getattr(value, field.name) is None
                else getattr(value, field.name).index_select(0, index)
            )
            for field in fields(value)
        }
    )


def _validate_source(row: dict[str, Any], index_path: Path) -> tuple[Path, Path]:
    missing = REQUIRED_INDEX_FIELDS - set(row)
    if missing:
        raise KeyError(f"sequence index row missing {sorted(missing)}")
    observer_path = _resolve(index_path, row["observer_cache"])
    target_path = _resolve(index_path, row["target_cache"])
    joined = f"{observer_path} {target_path} {row['source_dataset']}".lower()
    if any(marker in joined for marker in FORBIDDEN_BENCHMARK_MARKERS):
        raise ValueError(f"SGNify benchmark leakage in training sequence {row['sequence_id']}")
    if not 0 < float(row["quality_weight"]) <= 1:
        raise ValueError("quality_weight must be in (0,1]")
    return observer_path, target_path


def _load_targets(path: Path, length: int) -> dict[str, Any]:
    target = torch.load(path, map_location="cpu", weights_only=True)
    if target.get("schema_version") != TARGET_SCHEMA:
        raise ValueError(f"unsupported pseudo-GT schema at {path}")
    missing = REQUIRED_TARGET_FIELDS - set(target)
    if missing:
        raise KeyError(f"pseudo-GT sequence missing {sorted(missing)}")
    for name, value in target.items():
        if name == "schema_version" or not isinstance(value, torch.Tensor):
            continue
        if value.shape[0] != length:
            raise ValueError(f"{name} length {value.shape[0]} != observer length {length}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf")
    return target


def _window_payload(
    bundle: DualObserverBundle,
    target: dict[str, Any],
    indices: tuple[int, ...],
    center_index: int,
    handedness_class: float,
) -> dict[str, Any]:
    tokens = ExplicitTokenBuilder().build(
        _slice_dataclass(bundle.body, indices),
        _slice_dataclass(bundle.h4w_left, indices),
        _slice_dataclass(bundle.h4w_right, indices),
        _slice_dataclass(bundle.omni_left, indices),
        _slice_dataclass(bundle.omni_right, indices),
        bundle.root_rel[list(indices)],
        bundle.timestamps[list(indices)],
        torch.tensor([handedness_class]),
    )
    payload: dict[str, Any] = {"schema_version": WINDOW_SCHEMA}
    for field in fields(tokens):
        value = getattr(tokens, field.name)
        if value is not None:
            payload[field.name] = value.squeeze(0).detach().cpu()
    for name, value in target.items():
        if name == "schema_version" or not isinstance(value, torch.Tensor):
            continue
        payload[name] = value[center_index].detach().cpu()
    payload["center_frame_id"] = bundle.frame_ids[center_index].detach().cpu()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert frozen dual-observer SignAvatars sequences into PKC windows"
    )
    parser.add_argument("--sequence-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--padding", choices=["reflect", "replicate"], default="reflect")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    index_path = args.sequence_index.resolve()
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_root = args.output_root.resolve()
    output_rows = []
    for row in rows:
        observer_path, target_path = _validate_source(row, index_path)
        bundle = load_dual_observer_bundle(observer_path)
        target = _load_targets(target_path, len(bundle.timestamps))
        class_value = float(row.get("handedness_class", 0.0))
        for gap in args.gaps:
            specs = all_windows(len(bundle.timestamps), args.window_size, gap, args.padding)
            for center_index, spec in enumerate(specs):
                relative = (
                    Path(str(row["split"]))
                    / str(row["sequence_id"])
                    / f"frame_{int(bundle.frame_ids[center_index]):08d}_gap{gap}.pt"
                )
                output_path = output_root / relative
                if args.overwrite or not output_path.is_file():
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        _window_payload(
                            bundle,
                            target,
                            spec.indices,
                            center_index,
                            class_value,
                        ),
                        output_path,
                    )
                output_rows.append(
                    {
                        "cache_path": str(output_path),
                        "signer_id": str(row["signer_id"]),
                        "sequence_id": str(row["sequence_id"]),
                        "split": str(row["split"]),
                        "quality_weight": float(row["quality_weight"]),
                        "source_dataset": str(row["source_dataset"]),
                        "frame_id": int(bundle.frame_ids[center_index]),
                        "temporal_gap": gap,
                    }
                )
    write_jsonl(args.output_index.resolve(), output_rows)
    print(f"wrote {len(output_rows)} windows to {output_root}")


if __name__ == "__main__":
    main()
