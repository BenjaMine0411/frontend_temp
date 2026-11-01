#!/usr/bin/env python3
"""Batch video renderer: read a job file, expand it into individual videos, render each.

  python batch_render.py my_jobs.json --config paths.json --dry-run   # count + time estimate only
  python batch_render.py my_jobs.json --config paths.json             # render

Pause / resume: Ctrl+C at any time (or a crash / power loss), then run the SAME command again.
Videos already done are skipped; the video that was in progress starts over; untangle results
and fitted bodies cached under work_root are reused.

Failures: a failed video is retried (--retries, default 1 extra attempt); if it still fails it is
logged to <library_root>/failed.log and the batch moves on. Failed videos are tried again on the
next run.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from toolkit_common import (ToolkitError, calibration, estimate_seconds, expand_jobs, fmt_duration, is_done,
                            load_catalog, load_paths, motion_length_problems, print_err,
                            validate_jobs_against_catalog)
from render_one_video import RenderFailed, format_timings, log_path, render_video, steps_for_frames


def plan(jobs, library, work_root=None):
    todo = [j for j in jobs if not is_done(library, j)]
    calib = calibration(library)
    seen_outfits, total = set(), 0.0
    for j in todo:
        first = j.outfit not in seen_outfits
        seen_outfits.add(j.outfit)
        cached = bool(work_root) and (Path(work_root) / "_staging_garment_dicts" / j.outfit).exists()
        total += estimate_seconds(j, first, steps_for_frames(j.frames), calib, cached)
    return todo, total, calib


def append_failed_log(library: Path, job, error: str):
    library.mkdir(parents=True, exist_ok=True)
    with (library / "failed.log").open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {job.key} frames={job.frames}\n{error}\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", type=Path, help="工作單 (json)")
    ap.add_argument("--config", required=True, type=Path, help="paths.json")
    ap.add_argument("--dry-run", action="store_true", help="只列出數量與預估時間，不執行")
    ap.add_argument("--retries", type=int, default=1, help="失敗後額外重試幾次（預設 1）")
    args = ap.parse_args()

    try:
        paths = load_paths(args.config)
        catalog = load_catalog(paths["data_root"])
        jobs = expand_jobs(args.jobs)
        problems = validate_jobs_against_catalog(jobs, paths, catalog)
        if not problems:
            problems = motion_length_problems(jobs, paths, catalog)
    except ToolkitError as e:
        print_err(f"錯誤：{e}")
        sys.exit(2)
    if problems:
        print_err("工作單有問題：")
        for p in problems:
            print_err(f"  - {p}")
        sys.exit(2)

    library = Path(paths["library_root"])
    todo, total, calib = plan(jobs, library, paths["work_root"])
    print(f"工作單共 {len(jobs)} 支影片：已完成 {len(jobs) - len(todo)} 支，待產生 {len(todo)} 支。")
    if calib:
        print(f"預估還需要 約 {fmt_duration(total)}（依這台電腦已完成影片的實測速度："
              f"模擬 {calib['step']:.1f} 秒/步、算圖 {calib['frame']:.1f} 秒/幀）。")
    else:
        print(f"預估還需要 約 {fmt_duration(total)}（粗估，以 RTX 5060 筆電的速度為準；你的電腦可能差很多，"
              f"跑完第一支影片後會改用實測速度重新推算）。")
    if args.dry_run:
        return
    if not todo:
        print("沒有需要產生的影片。")
        return

    n_done = n_failed = 0
    failed = []
    t_start = time.perf_counter()
    try:
        for i, job in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {job.key}  ({job.frames} 幀)", flush=True)
            for attempt in range(1, args.retries + 2):
                try:
                    sc = render_video(paths, catalog, job, job.video_path(library), job.sidecar_path(library))
                    print(f"  完成，耗時 {sc['elapsed_seconds']} 秒", flush=True)
                    if sc.get("stage_seconds"):
                        print("  階段耗時：" + format_timings(sc["stage_seconds"], steps_for_frames(job.frames), job.frames), flush=True)
                    n_done += 1
                    break
                except RenderFailed as e:
                    print(f"  第 {attempt} 次失敗", flush=True)
                    if attempt > args.retries:
                        append_failed_log(library, job, str(e))
                        print(f"  放棄這支，已記錄到 {library / 'failed.log'}；log：{log_path(paths, job)}", flush=True)
                        n_failed += 1
                        failed.append(job.key)
    except KeyboardInterrupt:
        print("\n\n已暫停。要繼續時，重新執行同一個指令即可（已完成的會自動跳過）。")
        sys.exit(130)

    print(f"\n結束：新完成 {n_done} 支，失敗 {n_failed} 支，總耗時 {fmt_duration(time.perf_counter() - t_start)}。")
    if failed:
        print("失敗的影片（下次執行會再試一次）：")
        for k in failed:
            print(f"  - {k}")
        sys.exit(1)


if __name__ == "__main__":
    main()
