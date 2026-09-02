# API / 資料格式說明

## GG 原始輸出格式

兩種來源，兩條匯入路徑（細節見 `ContourCraft/import_official_gg_garment.py` 和
`recover_resize_garment_bootstrap.py` 檔案內的說明）：

1. **已模擬好的服裝軌跡**（`source_trajectory` + `source_frame`）：一份 `.pkl`，內含
   `pred`（服裝逐幀頂點座標）、`obstacle`（身體逐幀頂點座標）、`cloth_faces`、`obstacle_faces`。
   `recover_resize_garment_bootstrap.py` 從中切出一幀，還原成 `garment_dict.pkl`。
2. **官方 SMPL-X 真實捕捉**（`smplx_source` + `template_frame`）：一份 `.npz`，內含逐幀的
   `betas`/`global_orient`/`transl`/`body_pose`/`jaw_pose`/`expression`/`left_hand_pose`/
   `right_hand_pose`。`import_official_gg_garment.py` 直接用這些參數 + 官方註冊好的服裝樣板
   （`stage1/template_uv.obj`）匯入，同時寫出 `relax_trajectory.pkl`（跟上面同樣的
   `pred`/`obstacle`/`cloth_faces`/`obstacle_faces` schema，是服裝settle到身體上的過程逐步記錄）。

兩者匯入後都會產生同一種格式的 `garment_dict.pkl`（頂層鍵值：`rest_pos`、`faces`、`node_type`、
`garment_id`、`lbs`、`center`、`coarse_edges`、`gender`、`model_type`、`recovery_metadata`），
`recovery_metadata` 裡的 `source` 欄位（`source_trajectory` 存在 vs.
`source == "official_actorshq_smplx_import"`）用來分辨是哪條路徑進來的，多處程式（例如縮圖產生
工具）會依此分流。

## 額外處理程式與指令

```
GG output（garment_dict.pkl 或 SMPL-X npz）
  → recover_resize_garment_bootstrap.py（路徑1）或 import_official_gg_garment.py（路徑2）
  → combine_garments_untangle.py         # 上下身服裝合併去穿插
  → contourcraft_fixed_size_upper.py     # 依身高體重/動作綁定
  → simulate_recovered_outfit.py         # 物理模擬，輸出 trajectory.pkl
  → render_turntable.py / render_lean.py # 算圖，輸出 PNG
  → (moviepy ImageSequenceClip)          # PNG 序列組成 mp4，preview_server.py 內的 subprocess 呼叫
```

指令範例（實際參數見 `run_outfit_preview_pipeline.py`，這裡只列關鍵幾步）：

```bash
python combine_garments_untangle.py --garment-names <lower> <upper> \
    --outfit-name <lower>+<upper> --gender <target-gender> ...

python simulate_recovered_outfit.py --outfit-name <lower>+<upper> \
    --sequence <motion.npz> --output-trajectory trajectory.pkl ...

python render_lean.py --traj_path trajectory.pkl --output_path frames/ --bake-samples 4
```

## 前端最終需要的格式

- 靜態轉台：`turntable/angle_000.png` ~ `angle_011.png`（12 張 PNG，數量由 `--turntable-angles`
  決定，預設 12）
- 完整動畫：`video.mp4`（H.264，`moviepy` 用 `libx264` codec 編碼）
- 進度輪詢：`status.json`

**瀏覽器可以直接顯示以上所有格式**（`<img>`/`<video>` 標籤），不需要任何外掛或轉檔——
`preview_server.py` 用 Flask 的 `send_from_directory`/`send_file` 原封不動把檔案吐出去。

**目前實際使用的是影片（mp4）+ PNG，沒有 PLY/SPLAT/mesh 進到前端這一層**——算圖過程會在
`turntable/`、`frames/` 等中間目錄產生 `.ply`/`.obj`/貼圖檔案（算圖 pipeline 自己的工作檔案），
但 `preview_server.py` 從來不會讀取或提供這些檔案，瀏覽器也拿不到，純粹是後端內部的中間產物。

## 轉換前後的目錄範例

三個階段，對應上方「額外處理程式與指令」的流程圖：

```
# 1. Pipeline 輸入：GG 原始輸出（路徑1範例，服裝軌跡）
gaugar_trajectories/individual-garments/00170_upper_Take1.pkl

# 1. Pipeline 輸入：GG 原始輸出（路徑2範例，官方 SMPL-X 捕捉）
actorshq_official_smplx/Actor05/Sequence1/smpl_params.npz

# 2. Pipeline 中繼：還原/合併後的服裝快取（供後續模擬讀取，非前端直接存取）
gg_outputs_fixedsize/
├── <garment-name>/garment_dict.pkl          # 單件服裝，還原後
└── _staging_garment_dicts/<outfit-name>/    # 上下身合併去穿插後
    ├── <lower-garment-name>.pkl
    ├── <upper-garment-name>.pkl
    └── manifest.json

# 3. 前端讀取：Pipeline 最終輸出（單次 /generate 產生，見 sample_assets/ 的真實範例）
gg_outputs_fixedsize/outfits/<outfit-name>/runs_preview/<run-id>/
├── status.json
├── preview.png
├── turntable/
│   ├── angle_000.png
│   ├── ...
│   └── angle_011.png
└── video.mp4
```

## `status.json` 欄位

真實範例見 `sample_assets/status.json`：

```json
{
  "phase": "done",
  "updated_at": 1788357302.23,
  "elapsed_seconds": 383.1,
  "message": "Done",
  "preview_ready": true,
  "video_ready": true
}
```

`phase` 依序經過：`recovering_garments` → `untangling` → `fitting_body` → `rebinding` →
`preview_simulating` → `preview_rendering` → `preview_ready`（此時 `turntable_frames` 欄位出現）
→ `full_simulating` → `full_rendering`（此時 `rendered_frames`/`total_frames` 欄位出現）→ `done`。
前端每約 1.5 秒輪詢一次 `/status/<run_id>`。

## outfit 命名與快取

`<下身服裝名>+<上身服裝名>`，目標性別非 `female` 時加 `_<gender>` 後綴——這個字串同時是
`_staging_garment_dicts/<outfit-name>/` untangle 快取的 key。細節與「怎麼幫既有的 CLI 建好的
outfit 建別名讓網頁重用」的技巧，見 [HANDOFF.md 第8節](HANDOFF.md#8-api-或格式相依關係)。
