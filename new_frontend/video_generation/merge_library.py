#!/usr/bin/env python3
"""Merge another person's video library into yours.

  python merge_library.py /path/to/their/video_library --config paths.json [--overwrite]

Videos are matched by their path (outfit/gender/height_weight/motion), which is a pure function of
the parameters, so the same video from two people has the same path. For each finished video
(sidecar status=done and the mp4 present) in the source:
  - not in your library yet            -> copied
  - already in your library, same frames/fps -> skipped
  - already in your library, different frames/fps -> listed as a conflict, not touched
    (use --overwrite to replace yours with theirs)

Run build_index.py afterwards to refresh manifest.json.
"""

import argparse
import shutil
import sys
from pathlib import Path

from toolkit_common import ToolkitError, atomic_write_json, load_paths, print_err, read_sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="別人傳來的 video_library 資料夾")
    ap.add_argument("--config", required=True, type=Path, help="paths.json")
    ap.add_argument("--overwrite", action="store_true", help="衝突時用對方的影片取代自己的")
    args = ap.parse_args()
    try:
        paths = load_paths(args.config)
    except ToolkitError as e:
        print_err(f"錯誤：{e}")
        sys.exit(2)
    src_root, dst_root = args.source.resolve(), Path(paths["library_root"]).resolve()
    if not src_root.is_dir():
        print_err(f"錯誤：找不到來源資料夾 {src_root}")
        sys.exit(2)
    if src_root == dst_root:
        print_err("錯誤：來源和你的 library_root 是同一個資料夾")
        sys.exit(2)

    copied = skipped = 0
    conflicts = []
    for sc_path in sorted(src_root.rglob("*.json")):
        if sc_path.name == "manifest.json":
            continue
        sc = read_sidecar(sc_path)
        mp4 = sc_path.with_suffix(".mp4")
        if not sc or sc.get("status") != "done" or not mp4.is_file():
            continue
        rel = sc_path.relative_to(src_root)
        dst_sc, dst_mp4 = dst_root / rel, (dst_root / rel).with_suffix(".mp4")
        mine = read_sidecar(dst_sc)
        if mine and mine.get("status") == "done" and dst_mp4.is_file():
            if (mine.get("frames"), mine.get("fps")) == (sc.get("frames"), sc.get("fps")):
                skipped += 1
                continue
            if not args.overwrite:
                conflicts.append(f"{rel.with_suffix('')}：你的 {mine.get('frames')} 幀/{mine.get('fps')}fps，"
                                 f"對方 {sc.get('frames')} 幀/{sc.get('fps')}fps")
                continue
        dst_mp4.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_mp4.with_name(dst_mp4.name + ".tmp")
        shutil.copyfile(mp4, tmp)
        tmp.replace(dst_mp4)
        atomic_write_json(dst_sc, sc)            # sidecar last, so a half-copied video is never "done"
        copied += 1

    print(f"合併完成：新增 {copied} 支，已存在跳過 {skipped} 支，衝突 {len(conflicts)} 支。")
    if conflicts:
        print("衝突（未處理，要用對方的請加 --overwrite）：")
        for c in conflicts:
            print(f"  - {c}")
    print("接著執行：python build_index.py --config paths.json")


if __name__ == "__main__":
    main()
