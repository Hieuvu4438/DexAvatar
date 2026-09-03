"""
Chuẩn bị MỘT gloss mới cho DexAvatar: cắt đầu-cuối, tách frame, chọn lượt diễn,
sinh dòng metadata cần thêm vào signs_vsl.txt và segment_vsl.json.

Dùng khi bổ sung một gloss lẻ (ví dụ VÀO cho câu 2) mà không muốn chạy lại cả 40.

Chỉ cắt PHẦN ĐẦU và PHẦN CUỐI ít chuyển động — KHÔNG cắt phần giữa, để không
ảnh hưởng chất lượng ký hiệu.

    python atc_rebuttal/prep_one_gloss.py --video VAO.mp4 --gloss VÀO \
        --out-root /home/haipd/DexAvatar/data/vsl_glosses

Sau khi chạy, làm theo đúng các bước script in ra ở cuối.
"""
import argparse
import json
import shutil
import sys
import unicodedata as u
from pathlib import Path

import numpy as np


def nfc(s):
    return u.normalize("NFC", str(s))


def read_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Không mở được video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        sys.exit(f"Video không có frame nào: {path}")
    return frames, fps


def motion_energy(frames):
    """Năng lượng chuyển động từng frame: sai khác tuyệt đối trung bình với frame trước."""
    import cv2
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    e = np.zeros(len(g))
    for i in range(1, len(g)):
        e[i] = np.abs(g[i] - g[i - 1]).mean()
    if len(e) > 1:
        e[0] = e[1]
    return e


def smooth(x, r=2):
    k = np.ones(2 * r + 1) / (2 * r + 1)
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def trim_ends(e, thr_frac=0.25):
    """Chỉ bỏ phần đầu và phần cuối dưới ngưỡng. Không đụng phần giữa."""
    thr = e.min() + thr_frac * (e.max() - e.min())
    active = np.where(e >= thr)[0]
    if len(active) == 0:
        return 0, len(e) - 1
    return int(active[0]), int(active[-1])


def find_reps(e, thr_frac=0.45, min_len=8, gap=4):
    """Các đoạn liên tục có năng lượng cao = các lượt diễn."""
    thr = e.min() + thr_frac * (e.max() - e.min())
    on = e >= thr
    spans, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        hole = 0
        while j + 1 < len(on) and (on[j + 1] or hole < gap):
            hole = 0 if on[j + 1] else hole + 1
            j += 1
        j -= hole
        if j - i + 1 >= min_len:
            spans.append((i, j))
        i = j + 1
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video QIPEDC đã tải về")
    ap.add_argument("--gloss", required=True, help="tên gloss, ví dụ VÀO")
    ap.add_argument("--out-root", required=True,
                    help="thư mục data/vsl_glosses của DexAvatar")
    ap.add_argument("--margin", type=int, default=6,
                    help="nới đoạn ký hiệu mỗi đầu, giữ pha đưa tay vào/rút ra")
    ap.add_argument("--min-len", type=int, default=29, help="độ dài đoạn tối thiểu")
    ap.add_argument("--rep", type=int, default=None,
                    help="chọn lượt diễn thứ mấy (0-based); mặc định lượt nhiều năng lượng nhất")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    g = nfc(args.gloss)
    frames, fps = read_frames(args.video)
    e_raw = motion_energy(frames)
    e = smooth(e_raw)
    a, b = trim_ends(e)
    print(f"Video      : {args.video}  {len(frames)} frame @ {fps:.0f} fps")
    print(f"Cắt đầu-cuối: giữ frame {a}..{b}  ({b - a + 1} frame, bỏ {a} đầu + "
          f"{len(frames) - 1 - b} cuối)")

    kept = frames[a:b + 1]
    ek = e[a:b + 1]
    spans = find_reps(ek)
    print(f"\nCác lượt diễn phát hiện được trong video đã cắt:")
    if not spans:
        spans = [(0, len(kept) - 1)]
        print("  (không tách được lượt nào — dùng cả đoạn)")
    scored = []
    for i, (s0, s1) in enumerate(spans):
        en = float(ek[s0:s1 + 1].sum())
        scored.append((en, i, s0, s1))
        print(f"  #{i}  [{s0:3d},{s1:3d}]  len={s1 - s0 + 1:3d}  energy={en:8.2f}")
    pick = args.rep if args.rep is not None else max(scored)[1]
    s0, s1 = spans[pick]
    lo = max(0, s0 - args.margin)
    hi = min(len(kept) - 1, s1 + args.margin)
    while (hi - lo + 1) < args.min_len and (lo > 0 or hi < len(kept) - 1):
        if lo > 0:
            lo -= 1
        if (hi - lo + 1) < args.min_len and hi < len(kept) - 1:
            hi += 1
    seg = [lo + 1, hi + 1]                     # segment_vsl.json dùng chỉ số 1-based
    print(f"\nChọn lượt #{pick} -> segment 1-based {seg}  ({seg[1] - seg[0] + 1} frame "
          f"≈ {(seg[1] - seg[0] + 1) / fps:.1f}s)")

    if args.dry_run:
        print("\n[dry-run] Chưa ghi frame nào.")
        return

    import cv2
    out = Path(args.out_root) / g
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for i, fr in enumerate(kept, start=1):
        cv2.imwrite(str(out / f"{i:05d}.png"), fr)
    print(f"\nĐã ghi {len(kept)} frame vào {out}")

    print("\n" + "=" * 70)
    print("CÁC BƯỚC TIẾP THEO trên máy haipd")
    print("=" * 70)
    print(f"""
1. Thêm gloss vào metadata:

   cd /home/haipd/DexAvatar/vsl_meta
   cp signs_vsl.txt signs_vsl.txt.bak && cp segment_vsl.json segment_vsl.json.bak
   echo '{g} 0' >> signs_vsl.txt
   python -c "
import json,io
p='segment_vsl.json'; d=json.load(open(p,encoding='utf-8'))
d['{g}']={seg}
json.dump(d,open(p,'w',encoding='utf-8'),ensure_ascii=False,indent=1)
print('segment_vsl.json:',len(d),'gloss')"

2. Chạy DexAvatar tầng 1-4 cho riêng gloss này:

   cd /home/haipd/DexAvatar
   export ROOT_PATH="/home/haipd/DexAvatar/data/vsl_glosses/{g}"
   export OUTPUT_PATH="/home/haipd/DexAvatar/outputs/vsl/{g}"
   export SIGN_NAME="{g}"
   bash scripts/M3.5_wilor_extract.sh
   (cd dexavatar_fitting && PYTHONPATH=$PYTHONPATH:$(pwd)/smplifyx:$(pwd) \\
     python script.py --path "$ROOT_PATH" --out_path "$OUTPUT_PATH" \\
       --gpu_id 0 --split_num 1 --config /home/haipd/DexAvatar/vsl_meta/fit_vsl.yaml)

3. Kiểm tra (phải ra khoảng {seg[1] - seg[0] + 1}):

   ls "$OUTPUT_PATH/smplifyx/results" | wc -l

4. Gộp lại vào gloss_db:

   python build_gloss_db_dexavatar.py

5. Copy data/gloss_db_dexavatar.pkl sang máy render rồi render lại câu 2:

   python atc_rebuttal/render_sentences.py --db data/gloss_db_dexavatar.pkl \\
     --sentences atc_rebuttal/e2e/demo_sentences.json \\
     --out output_kq05_dex_v2 --smooth-r 2 --bg white --only 2
""")


if __name__ == "__main__":
    main()
