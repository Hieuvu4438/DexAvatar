#!/usr/bin/env python3
"""
Build SMPL-X vertex region indices for TR-V2V evaluation.

Outputs:
- smplx_upper_body_minus_face_vidx.npy
- smplx_left_hand_vidx.npy
- smplx_right_hand_vidx.npy
- smplx_region_manifest.json (metadata for reproducibility)
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _b2s(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    # numpy bytes_
    if isinstance(x, np.bytes_):
        return bytes(x).decode("utf-8", errors="ignore")
    return x


def _as_str(x) -> str:
    x = _b2s(x)
    return str(x)


def _find_label_map(obj: dict) -> dict:
    """Return {label_id:int -> label_name:str} from flexible pickle schemas."""
    # Normalize possible bytes keys to str for lookup.
    # We'll access values by scanning rather than relying on direct key lookup.

    # name -> id
    for k in ["part2num", "part_to_label", "name2id", "part_names_to_id", "segm_names"]:
        v = obj.get(k) or obj.get(_b2s(k))
        if isinstance(v, dict):
            keys = list(v.keys())
            vals = list(v.values())
            if keys and all(isinstance(_b2s(x), str) for x in keys) and all(isinstance(x, (int, np.integer)) for x in vals):
                return {int(i): _as_str(n) for n, i in v.items()}

    # id -> name
    for k in ["num2part", "label_to_part", "id2name", "segm_id_to_name"]:
        v = obj.get(k) or obj.get(_b2s(k))
        if isinstance(v, dict):
            keys = list(v.keys())
            vals = list(v.values())
            if keys and all(isinstance(x, (int, np.integer)) for x in keys) and all(isinstance(_b2s(x), str) for x in vals):
                return {int(i): _as_str(n) for i, n in v.items()}

    # list/tuple/np array: index = id
    for k in ["part_names", "labels", "names", "segm_names"]:
        v = obj.get(k) or obj.get(_b2s(k))
        if isinstance(v, (list, tuple, np.ndarray)):
            vv = list(v)
            if vv and all(isinstance(_b2s(x), str) for x in vv):
                return {i: _as_str(name) for i, name in enumerate(vv)}

    # generic scan of dict fields (also handle bytes keys)
    for v in obj.values():
        if isinstance(v, dict):
            keys = list(v.keys())
            vals = list(v.values())
            # name -> id
            if keys and all(isinstance(_b2s(x), str) for x in keys) and all(isinstance(x, (int, np.integer)) for x in vals):
                return {int(i): _as_str(n) for n, i in v.items()}
            # id -> name
            if keys and all(isinstance(x, (int, np.integer)) for x in keys) and all(isinstance(_b2s(x), str) for x in vals):
                return {int(i): _as_str(n) for i, n in v.items()}

    raise RuntimeError("Could not infer label-name mapping from smplx_parts_segm.pkl")


def _build_from_smplx_weights(smplx_model_path: Path, out_dir: Path, hand_thr=0.25, ubody_thr=0.20):
    data = np.load(smplx_model_path, allow_pickle=True)
    weights = np.asarray(data["weights"], dtype=np.float32)  # (V, J)

    # SMPL-X joint ids (common ordering)
    L_HAND = [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    R_HAND = [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

    # upper-body proxies
    UBODY = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # spine/neck/shoulder/arms/wrists/head
    FACE_EXCL = [22, 23, 24]  # jaw + eyes

    l_score = weights[:, L_HAND].sum(axis=1)
    r_score = weights[:, R_HAND].sum(axis=1)
    u_score = weights[:, UBODY].sum(axis=1)
    f_score = weights[:, FACE_EXCL].sum(axis=1)

    l_idx = np.where(l_score >= hand_thr)[0].astype(np.int64)
    r_idx = np.where(r_score >= hand_thr)[0].astype(np.int64)

    u_idx = np.where((u_score >= ubody_thr) & (f_score < 0.10))[0].astype(np.int64)
    # loại hand khỏi upper-body
    u_set = set(u_idx.tolist()) - set(l_idx.tolist()) - set(r_idx.tolist())
    u_idx = np.asarray(sorted(u_set), dtype=np.int64)

    np.save(out_dir / "smplx_left_hand_vidx.npy", l_idx)
    np.save(out_dir / "smplx_right_hand_vidx.npy", r_idx)
    np.save(out_dir / "smplx_upper_body_minus_face_vidx.npy", u_idx)

    manifest = {
        "method": "smplx_weights_fallback",
        "smplx_model_path": str(smplx_model_path),
        "thresholds": {"hand_thr": hand_thr, "ubody_thr": ubody_thr, "face_excl_thr": 0.10},
        "output_counts": {
            "left_hand_vertices": int(l_idx.size),
            "right_hand_vertices": int(r_idx.size),
            "upper_body_minus_face_vertices": int(u_idx.size),
        },
    }
    with (out_dir / "smplx_region_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Saved region index files (weights fallback):")
    print(out_dir / "smplx_left_hand_vidx.npy")
    print(out_dir / "smplx_right_hand_vidx.npy")
    print(out_dir / "smplx_upper_body_minus_face_vidx.npy")
    print(out_dir / "smplx_region_manifest.json")
    print("Counts:", manifest["output_counts"])


def _labels_matching(label_to_name, include_any=(), exclude_any=()):
    out = []
    for lid, name in label_to_name.items():
        n = _norm(name)
        if include_any and not any(tok in n for tok in include_any):
            continue
        if exclude_any and any(tok in n for tok in exclude_any):
            continue
        out.append(int(lid))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segm_pkl", default="dexavatar_fitting/assets/smplx_parts_segm.pkl")
    ap.add_argument("--out_dir", default="dexavatar_fitting/assets")
    ap.add_argument("--smplx_model", default="SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz")
    ap.add_argument("--hand_thr", type=float, default=0.25)
    ap.add_argument("--ubody_thr", type=float, default=0.20)
    args = ap.parse_args()

    segm_pkl = Path(args.segm_pkl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with segm_pkl.open("rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    if not isinstance(obj, dict) or "segm" not in obj:
        raise RuntimeError(f"Unexpected segmentation pickle schema in {segm_pkl}")

    segm = np.asarray(obj["segm"], dtype=np.int64).reshape(-1)
    try:
        label_to_name = _find_label_map(obj)
    except RuntimeError:
        print("Could not infer label-name mapping from pickle. Falling back to SMPL-X skinning weights.")
        _build_from_smplx_weights(Path(args.smplx_model), out_dir, args.hand_thr, args.ubody_thr)
        return

    # hand labels
    lhand_labels = _labels_matching(
        label_to_name,
        include_any=("lefthand", "leftpalm", "leftthumb", "leftindex", "leftmiddle", "leftring", "leftpinky", "leftlittle"),
    )
    rhand_labels = _labels_matching(
        label_to_name,
        include_any=("righthand", "rightpalm", "rightthumb", "rightindex", "rightmiddle", "rightring", "rightpinky", "rightlittle"),
    )

    # upper-body (exclude face/head and hands)
    ubody_labels = _labels_matching(
        label_to_name,
        include_any=("torso", "chest", "spine", "neck", "shoulder", "clavicle", "collar", "upperarm", "forearm", "arm", "elbow", "wrist", "back", "body", "abdomen"),
        exclude_any=("face", "head", "jaw", "nose", "eye", "ear", "brow", "lip", "mouth", "cheek", "contour", "scalp", "hair"),
    )
    ubody_labels = sorted(set(ubody_labels) - set(lhand_labels) - set(rhand_labels))

    lhand_idx = np.where(np.isin(segm, lhand_labels))[0].astype(np.int64)
    rhand_idx = np.where(np.isin(segm, rhand_labels))[0].astype(np.int64)
    ubody_idx = np.where(np.isin(segm, ubody_labels))[0].astype(np.int64)

    np.save(out_dir / "smplx_left_hand_vidx.npy", lhand_idx)
    np.save(out_dir / "smplx_right_hand_vidx.npy", rhand_idx)
    np.save(out_dir / "smplx_upper_body_minus_face_vidx.npy", ubody_idx)

    manifest = {
        "segm_pkl": str(segm_pkl),
        "num_vertices": int(segm.shape[0]),
        "num_labels": int(len(label_to_name)),
        "labels": {str(k): str(v) for k, v in sorted(label_to_name.items(), key=lambda kv: kv[0])},
        "selected_labels": {
            "left_hand": {str(k): label_to_name[k] for k in lhand_labels},
            "right_hand": {str(k): label_to_name[k] for k in rhand_labels},
            "upper_body_minus_face": {str(k): label_to_name[k] for k in ubody_labels},
        },
        "output_counts": {
            "left_hand_vertices": int(lhand_idx.size),
            "right_hand_vertices": int(rhand_idx.size),
            "upper_body_minus_face_vertices": int(ubody_idx.size),
        },
    }
    with (out_dir / "smplx_region_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Saved region index files:")
    print(out_dir / "smplx_left_hand_vidx.npy")
    print(out_dir / "smplx_right_hand_vidx.npy")
    print(out_dir / "smplx_upper_body_minus_face_vidx.npy")
    print(out_dir / "smplx_region_manifest.json")
    print("Counts:", manifest["output_counts"])


if __name__ == "__main__":
    main()
