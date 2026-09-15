#!/usr/bin/env python3
"""Minimal local demo server for run_outfit_preview_pipeline.py.

Single file, no database, no job queue -- just enough to click a button in a
browser and watch the two-phase preview pipeline's output appear. Meant for
trying the idea out locally, not for deploying to real users.

NEW SCRIPT: doesn't modify run_outfit_preview_pipeline.py or anything else --
just shells out to it exactly the way you'd run it from the command line, and
serves whatever files it produces.

EDIT THESE before using: GARMENTS (your actual catalog, one entry per
individual garment) and MOTIONS (your actual short motion clips) are
hardcoded example lists below -- swap them for your real catalog. A visitor
picks one "upper" + one "lower" garment independently (see GARMENTS' own
comment for why any pairing is allowed, not just pre-cached ones).

Run:
    /path/to/conda/envs/ccraft/bin/python preview_server.py
    then open http://127.0.0.1:5000 in a browser.

How it works (matches run_outfit_preview_pipeline.py's own docstring):
  1. Browser POSTs the form to /generate. This launches
     run_outfit_preview_pipeline.py as a background subprocess (doesn't block
     the request) and immediately returns a run_id. A single overall
     progress bar (STEP_LABELS/PHASE_STEP in INDEX_HTML, 6 equal slices) is
     built right away and stays visible the whole run, with a text label
     above it naming the current step -- a coarser, more narrative view than
     showing status.json's raw `phase` string.
  2. Browser polls /status/<run_id> every ~1.5s -- this just reads the
     status.json that script already writes at each phase transition (plus
     rendered_frames, added by this route itself -- see status() below).
  3. As soon as status.json says preview_ready (with a turntable_frames
     count), the browser fetches all turntable/angle_NNN.png images and
     auto-cycles through them for ONE slow spin (back to the front view),
     then stops -- prev/next buttons stay live afterwards for anyone who
     wants to keep looking around, but nothing auto-rotates on repeat for
     the whole wait (that reads as gimmicky rather than useful).
  4. While the full animation simulates, then renders, the progress bar's
     label gains rendered_frames/total_frames and an estimated-time-
     remaining line, and the bar itself fills fractionally within its
     "渲染動畫影格" slice instead of just jumping to the next one -- no
     per-frame image swapping. A rotating "did you know" tip line keeps
     running the whole time.
  5. Once status.json says video_ready, the browser swaps to video.mp4
     (the progress bar above it stays, now full).
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "gg_outputs_fixedsize"
THUMBNAIL_DIR = SCRIPT_DIR / "catalog_thumbnails"

# --- EDIT THESE for your real catalog ---
# Item-by-item picker (one "upper" + one "lower" garment, chosen
# independently) rather than fixed pre-paired combos -- ANY upper+lower pair
# can be requested; run_outfit_preview_pipeline.py's untangle cache just
# takes a few extra minutes to build the first time a specific pair+gender
# hasn't been seen before (see outfit_name_for below), same as it always
# has for a brand new combo. Pre-warm the pairs you expect to be popular
# ahead of time (e.g. via this script's own /generate, or
# run_multigarment_pipeline_fixedsize.py --skip-render) so visitors don't
# pay that cost live.
#
# "thumbnail": a filename under catalog_thumbnails/ -- generate one by
# running generate_garment_thumbnail.py once for that garment (ghost-
# mannequin style, ~1-2 min, ONE garment only -- no untangle needed, ~the
# cheapest asset this catalog has). "source_gender": whichever gender this
# garment was actually recovered with (Stage 1-2) -- passed as
# --source-genders regardless of what a visitor picks in the gender
# dropdown, since that's a fixed, already-cached fact about the garment
# file, independent of the per-request TARGET --gender.
GARMENTS = [
    {"name": "00170_upper", "label": "格紋毛衣背心 (00170)", "slot": "upper",
     "source_gender": "female", "thumbnail": "00170_upper.png"},
    {"name": "00170_lower", "label": "百褶短裙 (00170)", "slot": "lower",
     "source_gender": "female", "thumbnail": "00170_lower.png"},
    {"name": "00176_upper", "label": "襯衫 (00176)", "slot": "upper",
     "source_gender": "female", "thumbnail": "00176_upper.png"},
    {"name": "00176_lower", "label": "牛仔褲 (00176)", "slot": "lower",
     "source_gender": "female", "thumbnail": "00176_lower.png"},
    # Real GG training-result import (import_official_gg_garment.py, official
    # ActorsHQ SMPL-X capture), not the generic-pretrained-checkpoint recovery
    # the four garments above use -- see recovery_metadata.source in each
    # garment_dict.pkl. Thumbnails made via generate_garment_thumbnail_from_
    # source.py's official_actorshq_smplx_import branch (reads relax_
    # trajectory.pkl, not a source_trajectory).
    {"name": "Actor05_upper_stage3_clean", "label": "素色T恤 (Actor05)", "slot": "upper",
     "source_gender": "male", "thumbnail": "Actor05_upper_stage3_clean.png"},
    {"name": "Actor05", "label": "工裝褲 (Actor05)", "slot": "lower",
     "source_gender": "male", "thumbnail": "Actor05.png"},
]
MOTIONS = [
    {"label": "走路 (01_01)", "path": "CMU/01/01_01_stageii.npz"},
    {"label": "走路-慢速 (108_01)", "path": "CMU/108/108_01_stageii.npz"},
    {"label": "走路-悠閒 (05_01)", "path": "CMU/05/05_01_stageii.npz"},
    {"label": "走路-正常 (07_01)", "path": "CMU/07/07_01_stageii.npz"},
    {"label": "走路-快速 (02_02)", "path": "CMU/02/02_02_stageii.npz"},
]
GENDERS = [
    {"value": "female", "label": "女性"},
    {"value": "male", "label": "男性"},
]
# -----------------------------------------

app = Flask(__name__)
RUNS = {}  # run_id -> {"dir": Path, "outfit_name": str}
GARMENTS_BY_NAME = {g["name"]: g for g in GARMENTS}

# Two-stage flow: a visitor fits their body ONCE (page 1), then tries however
# many outfits/motions against it (page 2) without re-entering height/weight/
# gender or re-paying fit_body_from_measurements_fast.py's ~20s optimization
# loop each time. In-memory only (resets on server restart), same style as
# RUNS above -- this is a single-visitor-at-a-time local demo, no session/
# cookie machinery needed. Deliberately NOT a cache keyed by (gender, height,
# weight) reusable across different visits -- two people matching on exact
# height+weight+gender is unlikely enough that it isn't worth building; this
# is just "the one body the current visitor is using right now".
CURRENT_BODY = None  # None, or {"betas_path": Path, "gender", "height", "weight"}
CURRENT_BODY_DIR = OUTPUT_ROOT / "web_current_body"


def outfit_name_for(garment_names, gender):
    # run_outfit_preview_pipeline.py's untangle cache (Stage 3,
    # _staging_garment_dicts/<outfit-name>/) is keyed on --outfit-name AND
    # bakes in whatever --gender the outfit was untangled for (confirmed the
    # hard way: requesting a different --gender against an existing
    # --outfit-name raises "exists but was built with different settings").
    # "female" keeps the SAME outfit-name this project has always used (so
    # the existing pre-warmed cache still hits, back from when this was a
    # fixed combo catalog); any other target gender gets its own namespace,
    # pre-warmed the same way -- see GARMENTS' "source_gender" for the
    # (unrelated, fixed) gender the garments themselves were recovered with.
    base = "+".join(garment_names)
    return base if gender == "female" else f"{base}_{gender}"


INDEX_HTML = """<!doctype html>
<title>Outfit Preview Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 16px; }
  label { display: block; margin-top: 12px; font-weight: 600; }
  select, input { width: 100%; padding: 8px; font-size: 16px; box-sizing: border-box; }
  button { margin-top: 20px; padding: 12px; width: 100%; font-size: 16px; cursor: pointer; }
  /* Left to right: generated result, garment pickers (left-aligned), params. */
  .layout { display: flex; gap: 32px; align-items: flex-start; margin-top: 16px; }
  .col-result {
    flex: 1 1 380px; min-width: 260px; min-height: 480px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }
  .col-result h2 { margin: 0 0 16px; text-align: center; }
  .col-garments { flex: 1 1 280px; min-width: 220px; text-align: left; }
  .col-params { flex: 1 1 220px; min-width: 200px; }
  #result { width: 100%; text-align: center; }
  img, video { max-width: 100%; border-radius: 8px; }
  .combo-grid { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
  .combo-card {
    flex: 1 1 45%; box-sizing: border-box; margin: 0; padding: 3px;
    border: 1px solid #ddd; border-radius: 6px; text-align: center; cursor: pointer;
  }
  .combo-card input { display: none; }
  .combo-card img { width: 100%; display: block; border-radius: 4px; }
  .combo-card.selected { border-color: #1a73e8; border-width: 2px; padding: 2px; background: #eef4fd; }
  #progress-label { font-size: 14px; font-weight: 600; margin-bottom: 6px; text-align: left; }
  #overall-bar { width: 100%; height: 12px; }
  #status-sub { font-size: 12px; color: #999; margin: 10px 0 4px; }
  #turntable-wrap { position: relative; display: inline-block; max-width: 100%; }
  #turntable-wrap button {
    position: absolute; top: 50%; transform: translateY(-50%);
    width: 36px; height: 36px; border-radius: 50%; border: none;
    background: rgba(0,0,0,0.35); color: #fff; font-size: 18px; cursor: pointer;
  }
  #tt-prev { left: 6px; } #tt-next { right: 6px; }
  #tip { font-size: 13px; color: #888; margin-top: 16px; min-height: 1.4em; font-style: italic; }
  #page-body { max-width: 420px; margin: 0 auto; text-align: center; }
  #page-body p.intro { color: #666; font-size: 14px; text-align: left; }
  #body-preview-area { margin-top: 20px; }
  #body-preview-img { max-width: 260px; border-radius: 8px; border: 1px solid #ddd; }
  #body-status-msg { margin-top: 12px; font-size: 14px; color: #666; }
  #change-body-btn { background: none; border: 1px solid #ccc; margin-top: 8px; }
</style>
<div id="page-body">
  <h2>穿搭預覽 Demo</h2>
  <p class="intro">先填一次身體資料，等一下不管試穿幾套衣服都會用同一個身體，不用每套都重填。</p>
  <form id="body-form">
    <label>性別</label>
    <select name="gender">
      {% for g in genders %}<option value="{{g.value}}">{{g.label}}</option>{% endfor %}
    </select>
    <label>身高 (m)</label>
    <input type="number" step="0.01" name="height" value="1.70" min="1.0" max="2.2" required>
    <label>體重 (kg)</label>
    <input type="number" step="0.1" name="weight" value="58" min="25" max="180" required>
    <button type="submit">建立我的身體模型</button>
  </form>
  <div id="body-status-msg"></div>
  <div id="body-preview-area" hidden>
    <img id="body-preview-img">
    <div><button type="button" id="body-next-btn">下一步：選衣服</button></div>
  </div>
</div>
<div id="page-outfit" hidden>
<form id="f" class="layout">
  <div class="col-garments">
    <label>上身</label>
    <div class="combo-grid">
      {% for i, g in enumerate(uppers) %}
      <label class="combo-card{{ ' selected' if i == 0 else '' }}">
        <input type="radio" name="upper" value="{{g.name}}"{{ ' checked' if i == 0 else '' }}>
        <img src="/garment_thumbnail/{{g.name}}" alt="{{g.label}}">
      </label>
      {% endfor %}
    </div>
    <label>下身</label>
    <div class="combo-grid">
      {% for i, g in enumerate(lowers) %}
      <label class="combo-card{{ ' selected' if i == 0 else '' }}">
        <input type="radio" name="lower" value="{{g.name}}"{{ ' checked' if i == 0 else '' }}>
        <img src="/garment_thumbnail/{{g.name}}" alt="{{g.label}}">
      </label>
      {% endfor %}
    </div>
  </div>
  <div class="col-params">
    <label>動作</label>
    <select name="motion">
      {% for i, m in enumerate(motions) %}<option value="{{i}}">{{m.label}}</option>{% endfor %}
    </select>
    <button type="submit">生成</button>
    <button type="button" id="change-body-btn">換人（重新填身體資料）</button>
  </div>
  <div class="col-result"><h2>穿搭預覽 Demo</h2><div id="result"></div></div>
</form>
</div>
<script>
const form = document.getElementById('f');
const result = document.getElementById('result');
let poll = null;

const pageBody = document.getElementById('page-body');
const pageOutfit = document.getElementById('page-outfit');

function showPage(name) {
  pageBody.hidden = name !== 'body';
  pageOutfit.hidden = name !== 'outfit';
}

// Polled once on page load (not just assumed client-side) so a mid-flow
// browser refresh lands on whichever page actually matches server state,
// e.g. after a server restart cleared CURRENT_BODY.
async function refreshBodyStatus() {
  const resp = await fetch('/body_status', { cache: 'no-store' });
  const s = await resp.json();
  showPage(s.has_body ? 'outfit' : 'body');
}
refreshBodyStatus();

const bodyForm = document.getElementById('body-form');
const bodyStatusMsg = document.getElementById('body-status-msg');
const bodyPreviewArea = document.getElementById('body-preview-area');
const bodyPreviewImg = document.getElementById('body-preview-img');

bodyForm.onsubmit = async (e) => {
  e.preventDefault();
  // Disable the button for the whole ~20s request -- without this a
  // double-click (or a submit fired while still mid-edit of a field) could
  // send two overlapping /fit_body calls; the server now rejects the second
  // one (409) either way, but this avoids the confusing 409 entirely and is
  // the more important half of the fix (a real 152cm/5kg body was produced
  // this way once, before the server-side guard existed).
  const submitBtn = bodyForm.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  bodyPreviewArea.hidden = true;
  bodyStatusMsg.textContent = '建立身體模型中，約需 20 秒...';
  try {
    const body = new URLSearchParams(new FormData(bodyForm));
    const resp = await fetch('/fit_body', { method: 'POST', body });
    if (!resp.ok) { bodyStatusMsg.textContent = '失敗：' + await resp.text(); return; }
    const { preview_image } = await resp.json();
    bodyStatusMsg.textContent = '';
    bodyPreviewImg.src = preview_image;
    bodyPreviewArea.hidden = false;
  } finally {
    submitBtn.disabled = false;
  }
};

document.getElementById('body-next-btn').onclick = () => showPage('outfit');

document.getElementById('change-body-btn').onclick = async () => {
  if (poll) clearInterval(poll);
  await fetch('/discard_body', { method: 'POST' });
  bodyPreviewArea.hidden = true;
  bodyStatusMsg.textContent = '';
  result.innerHTML = '';
  showPage('body');
};

// Image cards (upper/lower, each its own radio group): clicking anywhere in
// a <label class="combo-card"> already checks its radio natively (no JS
// needed for that) -- this just keeps the ".selected" highlight in sync,
// since that's simpler/more compatible than a CSS-only
// `:has(input:checked)` rule for a page meant to just work anywhere. Scoped
// per input `name` (upper vs. lower) so picking a lower doesn't clear the
// upper's highlight.
document.querySelectorAll('input[name="upper"], input[name="lower"]').forEach((el) => {
  el.addEventListener('change', () => {
    document.querySelectorAll(`input[name="${el.name}"]`).forEach((sibling) =>
      sibling.closest('.combo-card').classList.remove('selected'));
    el.closest('.combo-card').classList.add('selected');
  });
});

// Single overall progress bar: a coarser, more narrative view of
// status.json's own `phase` values than showing that raw phase text.
// Weighted by REAL measured wall-clock seconds (not equal slices) from a
// full default-parameter run through this exact server (the one /generate
// always launches: --preview-steps 5 --full-steps 60 --discard-frames 10
// --turntable-angles 12), read straight from that run's own
// "[status] <phase> (t=<elapsed>s)" log lines:
//   recovering_garments+untangling  0.0s  ->  0.1s   (usually cached, ~instant)
//   fitting_body+rebinding          0.1s  -> 21.6s
//   preview_simulating+rendering   21.6s  -> 73.9s
//   full_simulating                73.9s  -> 284.7s  (by far the biggest slice)
//   full_rendering+stitch         284.7s  -> 363.1s  (stitch itself is a few
//                                                      seconds of that, not
//                                                      separately logged)
// If you change what /generate passes to run_outfit_preview_pipeline.py
// (a different --full-steps etc.), these proportions will drift -- re-time
// a real run and update STEP_WEIGHTS to match.
const STEP_LABELS  = ['準備衣服素材', '量身訂做版型', '產生站姿預覽', '模擬完整動作', '渲染動畫影格', '合成影片'];
const STEP_WEIGHTS = [0.1,           21.5,          52.3,           210.8,         75.4,          3.0];
const STEP_TOTAL = STEP_WEIGHTS.reduce((a, b) => a + b, 0);
// STEP_OFFSETS[i] = cumulative seconds at which step i STARTS (e.g. [0, 0.1,
// 21.6, 73.9, 284.7, 360.1] for the weights above).
const STEP_OFFSETS = [];
for (let i = 0, sum = 0; i < STEP_WEIGHTS.length; i++) {
  STEP_OFFSETS.push(sum);
  sum += STEP_WEIGHTS[i];
}
const PHASE_STEP = {
  starting: 0, recovering_garments: 0, untangling: 0,
  fitting_body: 1, rebinding: 1,
  preview_simulating: 2, preview_rendering: 2, preview_ready: 2,
  full_simulating: 3,
  full_rendering: 4,
  done: STEP_LABELS.length,
};
// ETA: seconds-per-frame is measured from when full_rendering (with a known
// total_frames) is FIRST seen, using the server's own elapsed_seconds (not
// client Date.now()) so it's immune to clock skew -- both numbers come from
// the same status.json.
let renderBaseElapsed = null;

function updateProgress(phase, s) {
  const labelEl = document.getElementById('progress-label');
  const barEl = document.getElementById('overall-bar');
  if (!labelEl || !barEl) return;

  if (phase === 'done') {
    labelEl.textContent = '完成！';
    barEl.value = 100;
    return;
  }

  const idx = PHASE_STEP.hasOwnProperty(phase) ? PHASE_STEP[phase] : 0;
  const stepIdx = Math.min(idx, STEP_WEIGHTS.length - 1);
  const stepStart = STEP_OFFSETS[stepIdx];
  const stepWeight = STEP_WEIGHTS[stepIdx];
  let label = STEP_LABELS[stepIdx];
  let fraction;

  if (phase === 'full_rendering' && s.total_frames) {
    // Real sub-progress available here (actual frame count) -- more
    // accurate than the time-based estimate below, so prefer it.
    const rendered = s.rendered_frames || 0;
    fraction = s.total_frames > 0 ? rendered / s.total_frames : 0;
    label += '（' + rendered + '/' + s.total_frames + ' 影格';
    if (renderBaseElapsed === null) renderBaseElapsed = s.elapsed_seconds;
    const secsIntoRender = s.elapsed_seconds - renderBaseElapsed;
    if (rendered > 0 && secsIntoRender > 0) {
      const remaining = Math.max(0, Math.round((secsIntoRender / rendered) * (s.total_frames - rendered)));
      label += '，預估剩餘 ' + remaining + ' 秒';
    }
    label += '）';
  } else {
    // No direct sub-progress signal for this step -- estimate from how long
    // we've been in it vs. STEP_WEIGHTS' measured typical duration. Capped
    // below 1 so it doesn't visually finish the segment before the phase
    // actually changes if this run happens to be slower than the reference.
    const elapsedInStep = (s.elapsed_seconds || 0) - stepStart;
    fraction = stepWeight > 0 ? Math.max(0, Math.min(0.97, elapsedInStep / stepWeight)) : 0;
  }

  labelEl.textContent = label;
  const pct = ((stepStart + fraction * stepWeight) / STEP_TOTAL) * 100;
  barEl.value = Math.min(100, Math.round(pct));
}

// Turntable state -- lives across checkStatus() calls (its own slow
// setInterval, separate from the 1.5s status poll) so the spin doesn't
// restart/stutter every time a status poll comes back.
let turntableTimer = null;
let turntableUrls = [];
let turntableIdx = 0;
let turntablePauseUntil = 0;

// Rotating loading tips -- purely cosmetic, no backend cost. Own interval,
// independent of the 1.5s status poll, so it doesn't restart every poll.
const TIPS = [
  '小知識：轉盤圖其實是同一個畫面從不同角度拍的，幾乎不用額外算圖時間',
  '小知識：每一個衣服皺褶，都是物理模擬一步一步算出來的',
  '小知識：衣服的材質參數（密度、彈性）會影響它怎麼垂墜',
  '小知識：系統會依照你填的身高體重，重新調整每件衣服合身的樣子',
  '小知識：這台展示機只有一張顯卡，所以同時間只能生成一個結果',
];
let tipTimer = null;
let tipIdx = 0;

function startTips() {
  stopTips();
  tipIdx = 0;
  const el = document.getElementById('tip');
  if (el) el.textContent = TIPS[0];
  tipTimer = setInterval(() => {
    tipIdx = (tipIdx + 1) % TIPS.length;
    const el = document.getElementById('tip');
    if (el) el.textContent = TIPS[tipIdx];
  }, 4000);
}
function stopTips() {
  if (tipTimer) { clearInterval(tipTimer); tipTimer = null; }
}

form.onsubmit = async (e) => {
  e.preventDefault();
  if (poll) clearInterval(poll);
  stopTurntable();
  stopTips();
  renderBaseElapsed = null;
  // Everything lives inside #stage so the finished video (or an error) can
  // just replace the whole thing in one go later -- covering the progress
  // bar/label instead of leaving it sitting above the result.
  result.innerHTML =
    '<div id="stage">' +
      '<div id="progress-label"></div>' +
      '<progress id="overall-bar" max="100" value="0"></progress>' +
      '<div id="status-sub"></div>' +
      '<div id="preview-area"></div>' +
    '</div>';
  updateProgress('starting', {});
  const statusSub = document.getElementById('status-sub');
  if (statusSub) statusSub.textContent = '送出中...';
  const body = new URLSearchParams(new FormData(form));
  const resp = await fetch('/generate', { method: 'POST', body });
  if (!resp.ok) { if (statusSub) statusSub.textContent = '送出失敗：' + await resp.text(); return; }
  const { run_id } = await resp.json();
  poll = setInterval(() => checkStatus(run_id), 1500);
};

function stopTurntable() {
  if (turntableTimer) { clearInterval(turntableTimer); turntableTimer = null; }
  turntableUrls = [];
  turntableIdx = 0;
}

function turntableStep(delta, run_id) {
  if (!turntableUrls.length) return;
  turntableIdx = (turntableIdx + delta + turntableUrls.length) % turntableUrls.length;
  const img = document.getElementById('turntable-img');
  if (img) img.src = turntableUrls[turntableIdx];
  turntablePauseUntil = Date.now() + 3000;  // brief pause after manual nav
}

// Builds the turntable + tip layout ONCE per run (preview_ready's first
// poll), inside #preview-area (leaves #progress-label/#overall-bar/
// #status-sub alone -- those persist across the whole run, not just this
// phase; rendering progress folds into that same top bar, see updateProgress).
function setupTurntable(run_id, n) {
  turntableUrls = Array.from({length: n}, (_, i) =>
    '/file/' + run_id + '/turntable/angle_' + String(i).padStart(3, '0') + '.png');
  turntableIdx = 0;
  const area = document.getElementById('preview-area');
  if (!area) return;
  area.innerHTML =
    '<div id="turntable-wrap">' +
      '<img id="turntable-img" src="' + turntableUrls[0] + '">' +
      '<button type="button" id="tt-prev">‹</button>' +
      '<button type="button" id="tt-next">›</button>' +
    '</div>' +
    '<div id="tip"></div>';
  document.getElementById('tt-prev').onclick = () => turntableStep(-1, run_id);
  document.getElementById('tt-next').onclick = () => turntableStep(1, run_id);
  // Auto-rotate exactly ONE full loop (n steps -> back to angle_000, the
  // front view) then stop -- a slow spin is a nice "look, you can see it
  // from every side" intro, but left running the whole multi-minute wait it
  // just looks like a gimmick on repeat. Prev/next stay live afterwards for
  // anyone who wants to keep looking around.
  let stepsTaken = 0;
  turntableTimer = setInterval(() => {
    if (Date.now() < turntablePauseUntil) return;
    turntableIdx = (turntableIdx + 1) % turntableUrls.length;
    const img = document.getElementById('turntable-img');
    if (img) img.src = turntableUrls[turntableIdx];
    stepsTaken++;
    if (stepsTaken >= turntableUrls.length) {
      clearInterval(turntableTimer);
      turntableTimer = null;
    }
  }, 500);  // ~6s for one full loop at 12 angles -- a calmer product-viewer pace
  startTips();
}

async function checkStatus(run_id) {
  const resp = await fetch('/status/' + run_id, { cache: 'no-store' });
  if (!resp.ok) {
    console.warn('status poll failed', resp.status);  // should never be 304 now (server sends no-store)
    return;
  }
  const s = await resp.json();
  updateProgress(s.phase, s);
  const statusSub = document.getElementById('status-sub');
  if (statusSub) statusSub.textContent = s.elapsed_seconds != null ? '已進行 ' + s.elapsed_seconds + ' 秒' : '';

  if (s.phase === 'error') {
    stopTurntable();
    stopTips();
    // Replaces the whole #stage (progress bar/label included), not just
    // #preview-area -- the error covers everything above it too.
    const stage = document.getElementById('stage');
    if (stage) stage.innerHTML = '<div style="color:red">' + s.message + '</div>';
    clearInterval(poll);
    return;
  }

  if (s.video_ready) {
    stopTurntable();
    stopTips();
    // Same reasoning -- the finished video covers the progress bar/label
    // instead of appearing below it.
    const stage = document.getElementById('stage');
    if (stage) stage.innerHTML =
      '<video src="/file/' + run_id + '/video.mp4?t=' + Date.now() + '" controls autoplay loop></video>';
    clearInterval(poll);
    return;
  }

  if (s.preview_ready && !turntableUrls.length && s.turntable_frames) {
    setupTurntable(run_id, s.turntable_frames);
  }
}
</script>
"""


@app.route("/")
def index():
    from flask import render_template_string
    uppers = [g for g in GARMENTS if g["slot"] == "upper"]
    lowers = [g for g in GARMENTS if g["slot"] == "lower"]
    return render_template_string(INDEX_HTML, uppers=uppers, lowers=lowers,
                                   motions=MOTIONS, genders=GENDERS, enumerate=enumerate)


@app.route("/garment_thumbnail/<name>")
def garment_thumbnail(name):
    # Static, not tied to any run_id -- these are pre-generated once per
    # garment (see GARMENTS' own comment, generate_garment_thumbnail.py) and
    # just live in catalog_thumbnails/ next to this script, unlike
    # /file/<run_id>/... .
    garment = GARMENTS_BY_NAME.get(name)
    if garment is None:
        abort(404)
    filename = garment.get("thumbnail")
    if not filename:
        abort(404)
    return send_from_directory(THUMBNAIL_DIR, filename)


def is_run_active(info) -> bool:
    """True if this run hasn't reached a terminal phase yet. There's only one
    GPU -- letting a second /generate start while one is still running means
    both silently fight over it and crawl (this actually happened: a leftover
    test run turned an expected ~4min request into 927s with no error or
    indication of why)."""
    status_path = info["dir"] / "status.json"
    if not status_path.exists():
        return True  # just launched, hasn't written its first status yet
    try:
        phase = json.loads(status_path.read_text()).get("phase")
    except (json.JSONDecodeError, OSError):
        return True  # mid-write race -- treat as active, safer than a false "it's free"
    return phase not in ("done", "error")


FIT_BODY_IN_PROGRESS = False  # see fit_body()'s own comment -- guards against a double-submit race


def any_job_active() -> bool:
    """True if a /generate run OR a /fit_body call is still mid-flight -- both
    use the GPU (fit_body_from_measurements_fast.py runs on it too), same
    one-GPU-at-a-time discipline as is_run_active's own docstring, so every
    GPU-touching endpoint shares this single check."""
    return FIT_BODY_IN_PROGRESS or any(is_run_active(info) for info in RUNS.values())


# Reasonable adult human bounds -- catches typos (a missing digit, e.g. "5"
# instead of "50") before they reach fit_betas(). Without this, an
# implausible height/weight combination (e.g. 152cm/5kg, hit for real once)
# still "succeeds" numerically -- the optimizer just pushes betas far outside
# the range real bodies occupy to satisfy both loss terms, producing a
# grotesquely gaunt shape, not an error. Matches the <input min/max> in
# INDEX_HTML; enforced here too since the HTML constraint alone doesn't
# protect a direct API call.
MIN_HEIGHT_M, MAX_HEIGHT_M = 1.0, 2.2
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 25.0, 180.0


@app.route("/fit_body", methods=["POST"])
def fit_body():
    global CURRENT_BODY, FIT_BODY_IN_PROGRESS
    if any_job_active():
        return ("目前已經有一個生成正在進行中，請等它完成後再試一次。"
                "這台機器只有一顆 GPU，同時跑兩個會讓兩邊都變超慢。", 409)

    gender = request.form.get("gender", "female")
    if gender not in {g["value"] for g in GENDERS}:
        return (f"不支援的性別選項：{gender}", 400)
    try:
        height = float(request.form["height"])
        weight = float(request.form["weight"])
    except (KeyError, ValueError):
        return ("身高/體重格式錯誤", 400)
    if not (MIN_HEIGHT_M <= height <= MAX_HEIGHT_M):
        return (f"身高需介於 {MIN_HEIGHT_M}m 到 {MAX_HEIGHT_M}m 之間", 400)
    if not (MIN_WEIGHT_KG <= weight <= MAX_WEIGHT_KG):
        return (f"體重需介於 {MIN_WEIGHT_KG}kg 到 {MAX_WEIGHT_KG}kg 之間", 400)

    # Marks this GPU-touching step busy for any_job_active() -- without this,
    # two /fit_body submits racing (e.g. a double-click, or a submit that
    # fires while the user is still mid-edit of the weight field) could run
    # concurrently: both would fit_betas() on the GPU at once (violates the
    # one-job-at-a-time rule everywhere else in this project), and both would
    # rmtree/rewrite CURRENT_BODY_DIR at the same time, so whichever finishes
    # last silently wins with no error -- this is almost certainly what
    # produced one real observed case (a 152cm/5kg body, i.e. an impossible
    # weight nobody would deliberately submit).
    # Always start from a clean slate rather than requiring the frontend to
    # call /discard_body first -- fit_body_from_measurements_fast.py refuses
    # to overwrite an existing --output-betas path anyway, and "start fresh"
    # is exactly what submitting this form again should mean.
    shutil.rmtree(CURRENT_BODY_DIR, ignore_errors=True)
    CURRENT_BODY_DIR.mkdir(parents=True, exist_ok=True)
    betas_path = CURRENT_BODY_DIR / "betas.pkl"
    preview_path = CURRENT_BODY_DIR / "body_preview.png"

    # FIT_BODY_IN_PROGRESS guards against a double-submit race: two /fit_body
    # calls overlapping (a double-click, or submitting while still mid-edit
    # of a field) would otherwise both fit_betas() on the GPU at once
    # (violates the one-job-at-a-time rule everywhere else in this project)
    # and both rmtree/rewrite CURRENT_BODY_DIR at the same time, so whichever
    # finishes last silently wins with no error -- this is almost certainly
    # what produced one real observed case (a 152cm/5kg body, i.e. an
    # impossible weight nobody would deliberately submit).
    FIT_BODY_IN_PROGRESS = True
    try:
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "fit_body_from_measurements_fast.py"),
             "--body-model-root", str(SCRIPT_DIR / "ccraft_data" / "aux_data" / "body_models"),
             "--gender", gender, "--height", str(height), "--weight", str(weight),
             "--output-betas", str(betas_path), "--patience", "50"],
            cwd=str(SCRIPT_DIR), check=True, capture_output=True, text=True)
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "render_body_preview.py"),
             "--betas-path", str(betas_path), "--output-image", str(preview_path)],
            cwd=str(SCRIPT_DIR), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(CURRENT_BODY_DIR, ignore_errors=True)
        return (f"身體資料處理失敗：{exc.stderr[-2000:] if exc.stderr else exc}", 500)
    finally:
        FIT_BODY_IN_PROGRESS = False

    # Synchronous on purpose -- fit_betas' optimization loop is short (~20s)
    # and this is a one-shot step, not worth building a status.json/polling
    # setup like /generate's for. The browser just shows a spinner for the
    # duration of this one request.
    CURRENT_BODY = {"betas_path": betas_path, "gender": gender, "height": height, "weight": weight}
    return jsonify({"preview_image": f"/body_preview.png?t={int(time.time() * 1000)}"})


@app.route("/discard_body", methods=["POST"])
def discard_body():
    """"換人" -- discard the current body so /body_status sends the visitor
    back to the body-data page. Actually deletes betas.pkl/body_preview.png
    from disk immediately (not just clearing the in-memory pointer): matches
    what pressing this button should mean, and /fit_body would need to do
    this anyway on the next submit (see its own comment)."""
    global CURRENT_BODY
    if any_job_active():
        return ("目前已經有一個生成正在進行中，請等它完成後再試一次。", 409)
    CURRENT_BODY = None
    shutil.rmtree(CURRENT_BODY_DIR, ignore_errors=True)
    return jsonify({"ok": True})


@app.route("/body_status")
def body_status():
    # Polled once on page load so a mid-flow browser refresh shows the right
    # page (body form vs. outfit picker) instead of assuming state client-side.
    if CURRENT_BODY is None:
        return jsonify({"has_body": False})
    return jsonify({"has_body": True, "preview_image": "/body_preview.png",
                     "gender": CURRENT_BODY["gender"], "height": CURRENT_BODY["height"],
                     "weight": CURRENT_BODY["weight"]})


@app.route("/body_preview.png")
def body_preview_image():
    if CURRENT_BODY is None:
        abort(404)
    return send_from_directory(CURRENT_BODY_DIR, "body_preview.png")


def cleanup_old_web_runs(keep_run_dir):
    """Delete every previous web-generated run folder except the one just
    started. This demo has no history/gallery feature -- once a new /generate
    request is accepted, nobody will look at an old run's files again, and
    they otherwise just accumulate disk space (this project hit ~9.8GB once
    before a manual cleanup).

    Scoped tightly to run-name prefix "web_" -- the prefix this route itself
    always uses (see run_name below) -- so this only ever touches runs this
    demo generated for a visitor. It never touches: CLI-invoked runs_preview
    folders (this project's own manual testing/comparison runs use other
    run-names, e.g. "quality_check_*"), the separate runs/ directory tree
    from run_multigarment_pipeline_fixedsize.py, or any cache directory
    (_staging_garment_dicts/, per-garment dicts) -- none of those live under
    runs_preview/web_*. Safe to call unconditionally: by the time /generate
    reaches this point, is_run_active()'s check above has already confirmed
    no run is mid-flight, so nothing still being polled/served gets deleted
    out from under a browser tab.

    ALSO drops the matching RUNS entry, not just the directory -- found the
    hard way (a live 409 that never cleared): is_run_active() treats a
    missing status.json as "just launched, not written its first status yet"
    (true for a genuinely brand-new run), but a deleted OLD run's directory
    also has no status.json, and RUNS is never otherwise pruned -- so any
    run whose folder got cleaned up here would look permanently "active"
    forever after, wrongly 409-blocking every future /generate and
    /fit_body call for the rest of this server process's life. Deleting the
    RUNS entry alongside the directory keeps that "missing = active" rule
    true only for actually-fresh runs.
    """
    for run_dir in OUTPUT_ROOT.glob("outfits/*/runs_preview/web_*"):
        if run_dir.is_dir() and run_dir != keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)
            log_path = run_dir.parent / f"{run_dir.name}.log"
            log_path.unlink(missing_ok=True)
            RUNS.pop(run_dir.name, None)


@app.route("/generate", methods=["POST"])
def generate():
    if CURRENT_BODY is None:
        return ("請先回第一頁填身體資料", 400)
    active_runs = [rid for rid, info in RUNS.items() if is_run_active(info)]
    if active_runs:
        return (f"目前已經有一個生成正在進行中（{active_runs[0]}），請等它完成後再試一次。"
                f"這台機器只有一顆 GPU，同時跑兩個會讓兩邊都變超慢。", 409)

    upper = GARMENTS_BY_NAME.get(request.form.get("upper", ""))
    lower = GARMENTS_BY_NAME.get(request.form.get("lower", ""))
    if upper is None or lower is None:
        return ("請選擇上身跟下身各一件", 400)
    motion = MOTIONS[int(request.form["motion"])]
    # Body no longer comes from this form -- height/weight/gender were fixed
    # back on page 1 when /fit_body ran (see CURRENT_BODY's own comment).
    gender = CURRENT_BODY["gender"]
    height = CURRENT_BODY["height"]
    weight = CURRENT_BODY["weight"]

    # Innermost-to-outermost order, same convention this project's
    # multi-garment scripts have always used (lower first, upper second) --
    # matters for combine_garments_untangle.py and for reproducing the exact
    # outfit_name string any already pre-warmed cache was built under.
    garment_names = [lower["name"], upper["name"]]
    source_genders = [lower["source_gender"], upper["source_gender"]]

    outfit_name = outfit_name_for(garment_names, gender)
    run_name = f"web_{int(time.time() * 1000)}"
    run_dir = OUTPUT_ROOT / "outfits" / outfit_name / "runs_preview" / run_name

    # No history/gallery feature -- a web-generated run is never looked at
    # again once a new one starts, so clear all of them out before launching
    # this one (see cleanup_old_web_runs' own docstring for exactly what this
    # does and doesn't touch).
    cleanup_old_web_runs(run_dir)

    cmd = [sys.executable, str(SCRIPT_DIR / "run_outfit_preview_pipeline.py"),
           "--garment-names", *garment_names,
           "--gender", gender,
           "--source-genders", *source_genders,
           "--outfit-name", outfit_name,
           "--height", str(height), "--weight", str(weight),
           "--precomputed-betas", str(CURRENT_BODY["betas_path"]),
           "--motion", motion["path"],
           "--run-name", run_name,
           # Whichever garment is in the "lower" slot gets the waistband pin
           # -- every lower-body item in this catalog is pants/a skirt, and
           # this project's own convention is pinning that garment before
           # relax + a fit-aware pin check at rebind (see generate_garment_
           # thumbnail.py's --is-lower-body for the same assumption).
           "--pin-waistband-garments", lower["name"],
           "--pin-waistband-if-snug-garments", lower["name"]]

    log_path = run_dir.parent / f"{run_name}.log"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), stdout=log_file, stderr=subprocess.STDOUT)

    run_id = run_name
    RUNS[run_id] = {"dir": run_dir}
    return jsonify({"run_id": run_id})


@app.route("/status/<run_id>")
def status(run_id):
    # Reads and returns the JSON directly (not send_from_directory) and
    # stamps no-store on the response -- this is a polling endpoint, it must
    # never come back as a cached/304 response or the frontend silently stops
    # updating (fetch()'s resp.ok is false for 304, so a naive `if (!resp.ok)
    # return` -- what this page's JS does -- looks like every poll failed).
    info = RUNS.get(run_id)
    if info is None:
        abort(404)
    status_path = info["dir"] / "status.json"
    if not status_path.exists():
        payload = {"phase": "starting", "message": "啟動中..."}
    else:
        payload = json.loads(status_path.read_text())
    # write_status() (run_outfit_preview_pipeline.py) overwrites the whole
    # file on every call -- it doesn't merge with what was there before. That
    # means a field written at one phase transition (turntable_frames, at the
    # "preview_ready" write) is gone from the file the instant the NEXT
    # write_status() call happens, which here follows almost immediately
    # (straight on to "full_simulating", no work in between). A 1.5s poll
    # interval can easily never land on that one-call-wide window, so relying
    # on the JSON field directly silently loses the turntable forever (this
    # actually happened). Counting turntable/*.png on disk instead is stable
    # on every poll from preview_ready onward, same reasoning as
    # rendered_frames below for frames/renders/.
    if payload.get("preview_ready"):
        turntable_dir = info["dir"] / "turntable"
        if turntable_dir.is_dir():
            count = len(list(turntable_dir.glob("angle_*.png")))
            if count:
                payload["turntable_frames"] = count
    # run_outfit_preview_pipeline.py writes total_frames once it knows it
    # (full_rendering's status write), but has no way to report progress
    # *during* that subprocess run (render() blocks until render_lean.py
    # exits) without new inter-process plumbing. Counting frames/renders/
    # here instead needs no changes to that script at all -- the files
    # showing up on disk already are the progress signal.
    if payload.get("phase") == "full_rendering" and payload.get("total_frames"):
        renders_dir = info["dir"] / "frames" / "renders"
        payload["rendered_frames"] = len(list(renders_dir.glob("*.png"))) if renders_dir.is_dir() else 0
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/latest_frame/<run_id>")
def latest_frame(run_id):
    info = RUNS.get(run_id)
    if info is None:
        abort(404)
    renders_dir = info["dir"] / "frames" / "renders"
    if not renders_dir.is_dir():
        frame = None
    else:
        frames = sorted(renders_dir.glob("*.png"))
        frame = frames[-1].name if frames else None
    resp = jsonify({"frame": frame})
    resp.headers["Cache-Control"] = "no-store"  # same reasoning as /status
    return resp


@app.route("/file/<run_id>/<path:relpath>")
def file(run_id, relpath):
    info = RUNS.get(run_id)
    if info is None:
        abort(404)
    full_path = (info["dir"] / relpath).resolve()
    if info["dir"].resolve() not in full_path.parents and full_path != info["dir"].resolve():
        abort(403)  # basic guard against path traversal outside this run's own folder
    return send_from_directory(full_path.parent, full_path.name)


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Serving on http://127.0.0.1:5000  (output root: {OUTPUT_ROOT})")
    app.run(host="127.0.0.1", port=5000, debug=False)
