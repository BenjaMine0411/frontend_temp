#!/usr/bin/env python3
"""Build a SAMPLE video library in the real library format, so the website can be tried
before the real videos exist.

  python make_sample_library.py            # creates ./video_library/ with FULL coverage (every option works)
  python make_sample_library.py --sparse   # incomplete coverage, to try the greyed-out options

It writes exactly what the real pipeline writes (same folder names, same per-video sidecar json,
and manifest.json produced by video_generation/build_index.py), using the 5 sample videos in
sample_assets/ as stand-ins for every video:

    video_library/<下身>+<上身>/<gender>/h<height>_w<weight:03d>/<motion>.mp4 (+ .json)
    video_library/manifest.json
    video_library/catalog.json
    video_library/thumbnails/<garment>.png

When the real videos arrive, replace the contents of video_library/ with the real library
(merge_library.py + build_index.py output, plus catalog.json and thumbnails/) -- the website
needs no change. By default every option has a video; --sparse leaves some genders, body sizes
and garment pairs without one so the greyed-out options can be tried out.
"""

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "video_generation"))
from toolkit_common import Job, atomic_write_json, make_sidecar  # noqa: E402
from build_index import build  # noqa: E402

ASSETS = HERE / "sample_assets"
OUT = HERE / "video_library"

# ---- 預設：完整覆蓋（每個選項都有影片、都能點）----
# 用 5 支示範影片輪流充當每一支影片。
_A = {m: f"00170_{m}" for m in ("05_08", "32_01", "76_10")}          # 百褶短裙 + 格紋背心 的示範影片
_B = {"05_08": "00176_05_08", "76_10": "00176_76_10", "32_01": "00170_32_01"}   # 牛仔褲 + 襯衫（沒有第三支，借用 A 的）
FULL_PAIRS = {
    ("00170_lower", "00170_upper"): _A,
    ("00176_lower", "00176_upper"): _B,
    ("00176_lower", "00170_upper"): _A,
    ("00170_lower", "00176_upper"): _B,
}
# 身高、體重用 catalog.json 裡所有區間的中間值（4 個身高 × 5 個體重），所以每個區間都選得到
FULL_BODIES = {g: [(h, w) for h in (155, 165, 175, 185) for w in (45, 55, 65, 75, 85)] for g in ("female", "male")}

# ---- --sparse：刻意做成覆蓋不完整，用來試「沒有影片的選項會變灰」----
SPARSE_PAIRS = {
    ("00170_lower", "00170_upper"): {"05_08": "00170_05_08", "32_01": "00170_32_01", "76_10": "00170_76_10"},
    ("00176_lower", "00176_upper"): {"05_08": "00176_05_08", "76_10": "00176_76_10"},
    ("00176_lower", "00170_upper"): {"05_08": "00170_05_08"},        # 混搭
}
SPARSE_BODIES = {
    "female": [(155, 45), (165, 55), (165, 65), (175, 65)],
    "male": [(175, 65), (175, 75), (185, 75)],
}
SPARSE_PAIR_BODIES = {   # 每套搭配有哪些身形；沒列出的搭配 = 全部身形
    ("00176_lower", "00176_upper"): {("female", 165, 55), ("female", 165, 65), ("female", 175, 65), ("male", 175, 65)},
    ("00176_lower", "00170_upper"): {("female", 165, 55)},
}
FRAMES, FPS = 101, 24      # 示範影片實際的規格（正式影片是 250 幀）


def link_or_copy(src: Path, dst: Path):
    """Every video is one of 5 sample files; hard-link instead of copying so the library takes
    almost no extra disk space (falls back to a real copy where hard links are not available)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    sparse = "--sparse" in sys.argv[1:]
    pairs, bodies_by_gender, pair_bodies = (SPARSE_PAIRS, SPARSE_BODIES, SPARSE_PAIR_BODIES) if sparse \
        else (FULL_PAIRS, FULL_BODIES, {})
    if OUT.exists():
        shutil.rmtree(OUT)
    n = 0
    for (lower, upper), motions in pairs.items():
        for gender, bodies in bodies_by_gender.items():
            for height, weight in bodies:
                allowed = pair_bodies.get((lower, upper))
                if allowed is not None and (gender, height, weight) not in allowed:
                    continue
                for motion, sample in motions.items():
                    job = Job((lower, upper), gender, height, weight, motion, FRAMES, FPS)
                    mp4 = job.video_path(OUT)
                    mp4.parent.mkdir(parents=True, exist_ok=True)
                    link_or_copy(ASSETS / "videos" / f"{sample}.mp4", mp4)
                    atomic_write_json(job.sidecar_path(OUT), make_sidecar(job, "done", elapsed_seconds=0, sample=True))
                    n += 1
    shutil.copytree(ASSETS / "thumbnails", OUT / "thumbnails")
    shutil.copyfile(ASSETS / "catalog.json", OUT / "catalog.json")
    manifest = build(OUT)
    atomic_write_json(OUT / "manifest.json", manifest)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6   # 硬連結以邏輯大小計
    print(f"示範資料庫完成（{'不完整覆蓋' if sparse else '完整覆蓋'}）：{n} 支影片（{size:.0f} MB），索引 {manifest['count']} 筆 → {OUT}")


if __name__ == "__main__":
    main()
