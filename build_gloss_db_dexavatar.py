"""
KQ-05 — gộp tham số SMPL-X do DexAvatar (SMPLify-X + WiLoR) sinh ra vào một bản
sao của gloss_db.pkl, để render lại các câu định tính với chất lượng bàn tay tốt hơn.

File ĐỘC LẬP: chỉ cần numpy, không cần text2sign_pipeline. Chạy được ngay trên
server của haipd, nơi chỉ có DexAvatar và một bản copy của gloss_db.pkl.

Bố trí thư mục mặc định (giống text2sign_pipeline):

    <thư mục làm việc>/
        data/gloss_db.pkl          <- copy từ text2sign_pipeline/data/
        outputs/vsl/<GLOSS>/smplifyx/results/*.pkl
        build_gloss_db_dexavatar.py

    python build_gloss_db_dexavatar.py

Đường dẫn khác thì chỉ ra bằng cờ:

    python build_gloss_db_dexavatar.py \
        --gloss-db /home/haipd/data/gloss_db.pkl \
        --dex-root /home/haipd/DexAvatar/outputs/vsl \
        --out      /home/haipd/data/gloss_db_dexavatar.pkl

KHÔNG ghi đè gloss_db.pkl gốc. Chỉ thay trường 'smplx' của những gloss chạy được;
trường 'embedding' giữ nguyên, mọi gloss khác giữ nguyên.

Khung frame đã được cắt sẵn ở thượng nguồn: segment_vsl.json chọn đúng một lượt
diễn (theo 'suggest' trong repetitions.json), và data_parser của DexAvatar chỉ nạp
các frame trong khoảng đó. Nên ở đây chỉ cần đọc theo đúng thứ tự frame.
"""
import argparse
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

# Đúng 11 khoá của một frame trong gloss_db.pkl gốc, kèm shape mong đợi.
FRAME_KEYS = {
    "global_orient": (1, 3), "body_pose": (1, 63),
    "left_hand_pose": (1, 45), "right_hand_pose": (1, 45),
    "jaw_pose": (1, 3), "betas": (1, 10), "expression": (1, 10),
    "leye_pose": (1, 3), "reye_pose": (1, 3),
    "camera_rotation": (1, 3, 3), "camera_translation": (1, 3),
}


def nfc(s):
    """Tên thư mục từ macOS ở dạng NFD, từ Linux ở dạng NFC — quy về một mối."""
    return unicodedata.normalize("NFC", s)


def frame_no(path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def load_frame(path):
    """SMPLify-X tuỳ bản lưu thẳng dict hoặc bọc trong {'result': ...}."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict) and "result" in d and isinstance(d["result"], dict):
        d = d["result"]
    if isinstance(d, (list, tuple)) and len(d) and isinstance(d[0], dict):
        d = d[0]                      # một số bản lưu list theo person_id
    if not isinstance(d, dict):
        raise ValueError(f"{path.name}: không phải dict mà là {type(d).__name__}")
    return d


def normalise(raw, fallback, path):
    """Ép về đúng 11 khoá, đúng shape, float32. Khoá thiếu lấy từ frame gốc."""
    out, missing = {}, []
    for k, shape in FRAME_KEYS.items():
        v = raw.get(k)
        if v is None:
            out[k] = np.array(fallback[k], dtype=np.float32).reshape(shape)
            missing.append(k)
            continue
        v = np.asarray(v, dtype=np.float32)
        want = int(np.prod(shape))
        if v.size != want:
            raise ValueError(
                f"{path.name}: khoá '{k}' có {v.size} phần tử (shape {v.shape}), "
                f"cần {want} (shape {shape}). Các khoá có trong file: "
                f"{sorted(raw.keys())}")
        out[k] = v.reshape(shape)
    return out, missing


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Gộp SMPL-X của DexAvatar vào bản sao gloss_db.pkl")
    ap.add_argument("--gloss-db", default="data/gloss_db.pkl",
                    help="gloss_db.pkl gốc (chỉ ĐỌC, không bị ghi đè)")
    ap.add_argument("--dex-root", default="outputs/vsl",
                    help="thư mục chứa <GLOSS>/smplifyx/results của DexAvatar")
    ap.add_argument("--out", default="data/gloss_db_dexavatar.pkl",
                    help="file kết quả")
    ap.add_argument("--min-frames", type=int, default=10,
                    help="ít frame hơn mức này thì giữ bản gốc — quá ngắn để render")
    ap.add_argument("--max-bad-frac", type=float, default=0.1,
                    help="tỷ lệ file pkl hỏng tối đa cho phép; vượt thì giữ bản gốc")
    ap.add_argument("--dry-run", action="store_true",
                    help="chỉ in bảng đối chiếu, không ghi file")
    args = ap.parse_args()

    db_path, root, out = Path(args.gloss_db), Path(args.dex_root), Path(args.out)
    for pth, what in ((db_path, "gloss_db.pkl"), (root, "thư mục outputs/vsl")):
        if not pth.exists():
            sys.exit(f"LỖI: không thấy {what} tại {pth.resolve()}\n"
                     f"      chỉ đường dẫn đúng bằng --gloss-db / --dex-root")
    if out.resolve() == db_path.resolve():
        sys.exit("LỖI: --out trùng --gloss-db, sẽ ghi đè bản gốc. Đổi tên file ra.")

    with open(db_path, "rb") as f:
        db = pickle.load(f)
    print(f"gloss_db gốc : {db_path.resolve()}  ({len(db)} mục)")
    print(f"DexAvatar    : {root.resolve()}")

    key_by_nfc = {nfc(k): k for k in db}
    rows, replaced, skipped, all_bad = [], 0, [], []

    for gdir in sorted(root.iterdir()):
        if not gdir.is_dir():
            continue
        name = nfc(gdir.name)
        res = gdir / "smplifyx" / "results"
        pkls = sorted(res.glob("*.pkl"), key=frame_no) if res.is_dir() else []

        if name not in key_by_nfc:
            skipped.append((name, len(pkls), "không có trong gloss_db"))
            continue
        key = key_by_nfc[name]
        n_old = len(db[key]["smplx"])
        if len(pkls) < args.min_frames:
            skipped.append((name, len(pkls), f"< {args.min_frames} frame, giữ bản gốc"))
            rows.append((name, n_old, len(pkls), "GIỮ GỐC"))
            continue

        fallback = db[key]["smplx"][0]
        frames, miss_all, bad = [], set(), []
        for p in pkls:
            # Tiến trình bị kill giữa lúc ghi để lại pkl rỗng/cụt. Bỏ frame đó và
            # đếm lại, đừng để một file hỏng làm sập cả lượt gộp.
            try:
                raw = load_frame(p)
            except (EOFError, pickle.UnpicklingError, ValueError) as e:
                bad.append((p.name, e.__class__.__name__))
                continue
            fr, miss = normalise(raw, fallback, p)
            frames.append(fr)
            miss_all |= set(miss)
        if bad:
            all_bad.extend((name, n, w) for n, w in bad)
        if len(frames) < args.min_frames:
            skipped.append((name, len(frames),
                            f"chỉ còn {len(frames)} frame đọc được / {len(pkls)} file, giữ bản gốc"))
            rows.append((name, n_old, len(frames), "GIỮ GỐC"))
            continue
        if bad and len(bad) > args.max_bad_frac * len(pkls):
            skipped.append((name, len(frames),
                            f"{len(bad)}/{len(pkls)} file hỏng (> {args.max_bad_frac:.0%}), giữ bản gốc"))
            rows.append((name, n_old, len(frames), "GIỮ GỐC"))
            continue
        if not args.dry_run:
            db[key] = {"smplx": frames, "embedding": db[key]["embedding"]}
        replaced += 1
        note = "thay mới"
        if bad:
            note += f" (bỏ {len(bad)} frame hỏng)"
        if miss_all:
            note += f" (thiếu {','.join(sorted(miss_all))} → lấy từ bản gốc)"
        rows.append((name, n_old, len(frames), note))

    if not rows:
        sys.exit(f"LỖI: không tìm thấy gloss nào dưới {root.resolve()}\n"
                 f"      cấu trúc cần là <GLOSS>/smplifyx/results/*.pkl")

    w = max(max(len(r[0]) for r in rows), 8)
    print(f"\n{'gloss':<{w}} {'gốc':>5} {'mới':>5} {'chênh':>6}  ghi chú")
    print("-" * (w + 50))
    for name, a, b, note in sorted(rows):
        print(f"{name:<{w}} {a:>5} {b:>5} {b - a:>+6}  {note}")
    if skipped:
        print("\nBỎ QUA:")
        for name, n, why in skipped:
            print(f"  {name}  ({n} frame) — {why}")

    if all_bad:
        print(f"\nFILE PKL HỎNG ({len(all_bad)}) — đã bỏ qua:")
        for g, fn, why in all_bad[:30]:
            print(f"  {g}/{fn}  ({why})")
        if len(all_bad) > 30:
            print(f"  ... và {len(all_bad) - 30} file nữa")
        print("  Xoá chúng rồi chạy lại tầng 4 nếu muốn đủ frame:")
        print("  find outputs/vsl -path '*/smplifyx/results/*.pkl' -size -2c -delete")

    if args.dry_run:
        print(f"\n[dry-run] Sẽ thay {replaced}/{len(rows)} gloss. Chưa ghi file nào.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(db, f)
    mb = out.stat().st_size / 2**20
    print(f"\nĐã thay {replaced}/{len(rows)} gloss. Ghi: {out.resolve()} ({mb:.0f} MB)")
    print("gloss_db.pkl gốc KHÔNG bị đụng tới.")


if __name__ == "__main__":
    main()
