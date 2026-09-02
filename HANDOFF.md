# HANDOFF：穿搭預覽 Demo 前端交接

## 路徑設定

程式碼中的路徑使用通用佔位符（`/path/to/GG_projects/...`、`/path/to/conda/envs/...`）。
部署前請搜尋 `/path/to/` 並替換為實際路徑。受影響檔案：

| 檔案 | 位置 |
|---|---|
| `ContourCraft/preview_server.py` | 1 處，檔案開頭啟動指令範例 |
| `ContourCraft/run_outfit_preview_pipeline.py` | 4 處，`--contourcraft-root`/`--gaugar-root`/`--ccraft-python`/`--gaugar-python` 預設值 |
| `ContourCraft/generate_garment_thumbnail_from_source.py` | 3 處，同類路徑預設值 |
| `ContourCraft/import_official_gg_garment.py` | 3 處，檔案開頭指令範例 |
| `ContourCraft/convert_actorshq_smplx_to_gaugar.py` | 2 處，檔案開頭指令範例 |

以上均為 argparse 預設值或說明文件中的範例指令，不影響程式邏輯。也可以不修改原始碼，執行時
用對應的 CLI 參數覆蓋（例如 `--contourcraft-root /你的/路徑`）。

`utils/defaults.py` 未包含在此交接範圍內，請依官方 GG/ContourCraft 安裝說明自行設定。

---

## 0. 概述

本機、非正式環境的服裝模擬預覽 Demo：使用者於網頁選擇上身服裝、下身服裝（各自獨立選擇）、
性別、身高體重、動作，系統透過 Gaussian-Garments + ContourCraft 的模擬/算圖 pipeline 產生
一段站立轉台預覽（約1分鐘），接著產生完整動畫（約4-6分鐘）。

前端本體為 **`ContourCraft/preview_server.py`**，單一 Flask 檔案，頁面 HTML/JS 內嵌於檔案內的
`INDEX_HTML` 字串，無 npm、無前端 build 工具鏈。

此交接資料夾的前提：接收方已具備官方 GG + ContourCraft 程式碼與可執行的 conda 環境。此資料夾
僅包含在官方程式碼之上新增/所需的部分，供疊加至既有 checkout 使用，非獨立安裝套件。

---

## 1. 目前功能（四分類）

前端共 6 個 Flask 路由：

| 路由 / 功能 | 分類 | 說明 |
|---|---|---|
| `/`（首頁：服裝選單、表單） | 已完整可用功能 | 純靜態頁面，不需後端即可開啟 |
| `/garment_thumbnail/<name>`（服裝縮圖） | 已完整可用功能 | 讀取 `catalog_thumbnails/` 內預先產生的縮圖檔 |
| `/generate`（POST，生成請求） | 必須依賴後端 | `subprocess` 呼叫 `run_outfit_preview_pipeline.py`，需 GPU、conda 環境、模型 checkpoint、快取資料 |
| `/status/<run_id>`（進度輪詢） | 必須依賴後端 | 讀取後端 pipeline 寫出的 `status.json` |
| `/latest_frame/<run_id>`（執行中即時畫面） | 必須依賴後端 | 需有執行中的模擬/算圖工作 |
| `/file/<run_id>/<path>`（最終影片/轉台圖） | 必須依賴後端 | 讀取後端 pipeline 產出的檔案 |

無「UI 未串接」與「mock data」兩類。

此處「後端」定義為 GG + ContourCraft 的 GPU 模擬/算圖 pipeline（`run_outfit_preview_pipeline.py`
以 `subprocess` 執行），非獨立 API 服務。

**服裝資料來源，兩條路徑並存**：
- **通用預訓練 checkpoint 路徑**（`00170_*`/`00176_*`）：`recover_resize_garment_bootstrap.py`
  搭配通用預訓練 ContourCraft checkpoint 還原服裝。
- **GG 訓練結果路徑**（`Actor05`/`Actor05_upper_stage3_clean`）：`import_official_gg_garment.py`
  直接匯入官方 ActorsHQ SMPL-X 捕捉資料。已通過端到端測試，含與通用 checkpoint 路徑服裝混搭
  的情境（見第10節）。

---

## 2. 需要的輸入格式

- 上身服裝、下身服裝：各自從 `GARMENTS` 目錄選擇一件（任意組合，命名/快取機制見第7-8節）
- 性別：`female`/`male`（`GENDERS` 目錄）
- 身高（公尺）、體重（公斤）
- 動作：從 `MOTIONS` 目錄選擇一段 CMU 動作片段（`.npz`）

---

## 3. 輸出/畫面結果

- 快速預覽：12 個角度的靜態轉台 PNG（`turntable/angle_000.png` ~ `angle_011.png`），約 1 分鐘
- 完整動畫：mp4 影片（`video.mp4`），約 4-6 分鐘

瀏覽器可原生顯示以上格式。範例輸出見 `sample_assets/`。

---

## 4. 安裝、build、啟動與測試指令

無 build 步驟（無 npm/bundler）。

```bash
# 安裝唯一外部相依（其餘為 Python 標準函式庫）——requirements.txt 已鎖定確切版本
# 於本交接資料夾根目錄執行，安裝到之後啟動 preview_server.py 會用到的同一個 Python 環境
pip install -r requirements.txt

# 啟動（需使用具備 GG/ContourCraft 相依的 conda 環境；於部署後的 ContourCraft/ 目錄執行）
/path/to/conda/envs/ccraft/bin/python preview_server.py

# 開啟
http://127.0.0.1:5000
```

執行安裝與啟動、檢視頁面與縮圖，不需 GPU 或完整 GG/ContourCraft 環境——`preview_server.py`
僅 import `flask` 與 Python 標準庫。執行生成請求時才需要完整環境（見第6節）。

---

## 5. 已知限制

以下瑕疵源自 Stage1/Stage2 原始重建資料，非本 pipeline 造成，暫不修復：

1. **服裝下擺邊界摺疊**：部分服裝下擺邊界線向內摺疊、自我交叉，算圖呈現為一塊粗糙補丁
   （見 `Actor05_upper_stage3_clean` 縮圖與範例影片）。
2. **腳踝/袖口鋸齒狀邊緣**：邊界頂點靜止姿態有小幅鋸齒，源於 Stage1 對薄布料邊緣的重建限制
   （見 `Actor05` 縮圖與範例影片褲腳處）。

---

## 6. 設備與瀏覽器要求

**瀏覽頁面、縮圖、已生成範例**：現代瀏覽器即可。不需 GPU、不需 conda 環境、不需第7節的資料——
`preview_server.py` 僅相依 `flask`。

**執行生成請求**，需要以下全部項目：

- NVIDIA GPU
- 兩個 conda 環境：`ccraft`（執行 ContourCraft 部分，內含 `models/ccraft.py` 相依的 `cudf`）、
  `gaugar-sm120`（執行 Gaussian-Garments 算圖部分）——兩者不可混用
- ContourCraft 端資料：`ccraft_data/`（訓練模型 checkpoint、SMPL-X body model）、`CMU/`（動作
  捕捉資料）
- Gaussian-Garments 端資料：`output_root/`、`data_root/`、`aux_root/`（依官方
  `DataPreparation.md` 準備）
- 兩個 repo 的 `utils/defaults.py` 已依官方安裝說明填妥（未收錄於本交接資料夾，見上方「路徑設定」）

以上任一項缺漏，生成請求會直接失敗並回傳明確錯誤訊息（例如缺少資料目錄、或讀取到未填寫的路徑
佔位符），不會靜默失敗。

---

## 7. GG 到前端的完整資料流程

```
GG output（garment_dict.pkl / SMPL-X 訓練/捕捉資料）
   │  路徑1：recover_resize_garment_bootstrap.py（自已模擬服裝軌跡回推）
   │  路徑2：import_official_gg_garment.py（直接匯入官方 SMPL-X 捕捉資料）
   ▼
combine_garments_untangle.py（上下身服裝合併、去除穿插）
   ▼
contourcraft_fixed_size_upper.py + simulate_recovered_outfit.py（依身高體重/動作模擬）
   ▼
Gaussian-Garments/render_turntable.py + render_lean.py（算圖 → PNG）
   ▼
PNG 序列組成 mp4（見 preview_server.py 內的 subprocess 呼叫）
   ▼
preview_server.py 以 Flask send_from_directory 提供 PNG/mp4/JSON
```

算圖階段直接產生瀏覽器可顯示格式（PNG、mp4），前端不做額外轉檔。

**目錄範例**（單次 `/generate` 輸出）：
```
gg_outputs_fixedsize/outfits/<outfit-name>/runs_preview/<run-id>/
├── status.json
├── preview.png
├── turntable/
│   └── angle_000.png ~ angle_011.png
└── video.mp4
```

真實範例見 `sample_assets/`。

**格式**：影片（mp4）+ PNG。無 PLY/SPLAT/mesh 進入前端。算圖過程於 `turntable/`、`frames/`
等中間目錄產生 `.ply`/`.obj`/貼圖檔案，屬 pipeline 內部工作檔案，`preview_server.py` 不讀取
或提供這些檔案。

---

## 8. API 或格式相依關係

- **`GARMENTS` 目錄結構**（`preview_server.py`）：`{"name", "label", "slot", "source_gender", "thumbnail"}`。
  `slot` 為 `"upper"`/`"lower"`；`source_gender` 為該服裝於 GG 捕捉/還原時的性別，與使用者選擇的
  目標性別為獨立參數，分別傳入 pipeline。

- **outfit 命名規則**（`outfit_name_for()`）：`<下身服裝名>+<上身服裝名>`，目標性別非 `female`
  時加上 `_<gender>` 後綴。此名稱同時為 `_staging_garment_dicts/<outfit-name>/` untangle 快取
  目錄的鍵值。首次選擇未預先建置快取的組合，將額外花費約 4-5 分鐘進行 untangle。

- **快取重用**：若已透過 CLI 工具（如 `run_multigarment_pipeline_fixedsize.py`）建置某 outfit，
  可依以下步驟讓前端重用該快取，略過重新 untangle：
  1. 於 `_staging_garment_dicts/` 下建立資料夾，名稱與 `outfit_name_for()` 產生的結果一致
  2. 將既有的 `<garment>.pkl` 以 symlink 方式連入（不需複製，檔案通常為數十 MB）
  3. 建立 `manifest.json`，內容需與 `run_outfit_preview_pipeline.py` 的 `ensure_outfit()` 所
     計算的 `expected` dict 完全相等（`garment_names`、`outfit_name`、`gender`、`model_type`、
     `n_epochs`、`steps_per_stage`）。內容不符將直接拋出 `FileExistsError`，不會靜默重建。

---

## 9. 主要檔案及用途

**ContourCraft/**：

| 檔案 | 用途 |
|---|---|
| `preview_server.py` | 前端本體，Flask app。定義 `GARMENTS`/`MOTIONS`/`GENDERS` 目錄，處理第1節的 6 個路由，以 `subprocess` 依序呼叫下方各腳本組成完整 pipeline |
| `run_outfit_preview_pipeline.py` | 兩階段 pipeline 協調者，前端 `/generate` 直接呼叫此腳本。依序呼叫下方所有前處理/模擬/算圖腳本，過程中即時寫入 `status.json` 供前端輪詢，最終輸出到 `runs_preview/<run-id>/`（見第7節） |
| `recover_resize_garment_bootstrap.py` | 通用 checkpoint 路徑的服裝匯入：從 GG 已模擬的服裝軌跡 `.pkl` 切出一幀，還原成 `garment_dict.pkl`，寫入 `recovery_metadata`（`source_trajectory`/`source_frame`） |
| `pin_garment_waistband.py` | 標記下身服裝腰部邊界頂點，避免後續物理模擬時褲/裙從腰部滑落 |
| `relax_recovered_garment.py` | 用 ContourCraft 模型將剛還原的服裝從初始姿態鬆弛（relax）至接近真實靜止狀態，輸出 `relax_trajectory.pkl` |
| `combine_garments_untangle.py` | 合併上下身服裝並去除幾何穿插，輸出到 `_staging_garment_dicts/<outfit-name>/`（含每件服裝的 `.pkl` 與 `manifest.json`，見第8節） |
| `contourcraft_fixed_size_upper.py` | 將已合併的 outfit 依使用者指定的身高體重目標身體重新綁定（rebind），輸出到 `fixedsize_dicts/` |
| `simulate_recovered_outfit.py` | 讀取重新綁定後的 outfit + 動作序列 `.npz`，執行物理模擬，輸出 `trajectory.pkl`（快速預覽跑少量步數；完整動畫跑 60 步） |
| `fit_body_from_measurements_fast.py` | 將身高（公尺）、體重（公斤）轉換為 SMPL-X 身體參數，輸出 `betas.pkl` 與附加身體資訊的動作序列 `motion_with_body.npz` |
| `generate_garment_thumbnail_from_source.py` | 服裝縮圖產生工具（離線，非 `/generate` 流程呼叫）。支援兩種來源：通用 checkpoint 路徑讀 `source_trajectory`；GG 訓練結果路徑讀 `relax_trajectory.pkl`（見程式內註解），輸出到 `catalog_thumbnails/` |
| `import_official_gg_garment.py` | GG 訓練結果路徑的服裝匯入：讀取官方 ActorsHQ SMPL-X 捕捉 `.npz` 與 Stage1 服裝樣板，輸出 `garment_dict.pkl`（`recovery_metadata.source = "official_actorshq_smplx_import"`）與 `relax_trajectory.pkl` |
| `convert_actorshq_smplx_to_gaugar.py` | 將 ActorsHQ 官方 SMPL-X 資料轉換為 Gaussian-Garments 期望的輸入格式，供其 Stage1-3 pipeline 使用 |
| `catalog_thumbnails/` | 6 張服裝縮圖（4 張通用 checkpoint 路徑、2 張 GG 訓練結果路徑），`preview_server.py` 的 `/garment_thumbnail/<name>` 路由直接讀取 |

**Gaussian-Garments/**：

| 檔案 | 用途 |
|---|---|
| `render_lean.py` | 讀取 `trajectory.pkl`，逐幀算圖輸出 PNG 至 `frames/`，完整動畫每一幀使用，之後由 `preview_server.py` 呼叫的 subprocess 組成 `video.mp4`。`import inference as inf` 借用其 `Simulation` 類別與相機邏輯，但只算官方版四格比較圖中的 `final` 單一格，並將 `bake_texture` 換成自訂、取樣數可調的加速版（demo 用 4，官方原版寫死 16），非直接執行 `inference.py` |
| `render_turntable.py` | 讀取 `trajectory.pkl` 的單一幀，以多個相機角度算圖，輸出 `turntable/angle_000.png` ~ `angle_011.png`，快速預覽階段使用。同樣透過 `render_lean.py` 借用其加速烘焙邏輯 |
| `inference.py` | 官方檔案，修改一處：`Simulation.__init__` 新增 `garment_names` 為空時自檔名回推的 fallback。此為唯一被 `render_lean.py`/`render_turntable.py` 實際使用的修改——其餘改動（`--bake-samples` 參數、計時統計、`inference_mode()`）位於 `if __name__=="__main__"` 區塊，`render_lean.py`/`render_turntable.py` 以 import 方式使用該模組，不會執行到該區塊。本身從未被當作程式直接執行 |

---

## 10. 實際測試結果

**無 GPU**：首頁與 6 張服裝縮圖（含 Actor05 兩張）皆正常載入，僅需 `flask`，未使用 `ccraft`/
`gaugar-sm120` conda 環境。

**完整生成流程**：第6節所列需求齊備後，`POST /generate` 完整執行成功：

- 組合：「工裝褲 (Actor05)」+「素色T恤 (Actor05)」（GG 訓練結果路徑），目標性別男性
- 總耗時：435.8 秒，`fitting_body`/`rebinding` 階段快速完成（未觸發重新 untangle，第8節快取
  重用技巧生效）
- 結果：`preview_ready: true`、`video_ready: true`，產出有效 mp4 檔案

**混搭與併發**：另一次測試組合「工裝褲 (Actor05)」（GG 訓練結果路徑）+「00170 上衣」（通用
checkpoint 路徑）混搭、目標性別女性，總耗時 729 秒（含該組合首次觸發的全新 untangle，約
5.4 分鐘）；`preview_ready: true`、`video_ready: true`；執行期間第二次送出生成請求，收到
HTTP 409，未影響第一個工作。

**通用 checkpoint 路徑**：組合「牛仔褲 (00176)」+「襯衫 (00176)」，目標性別女性，總耗時
383.1 秒；`preview_ready: true`、`video_ready: true`。

`sample_assets/` 內容為此次測試的實際輸出。

---

## 11. 原始 branch、commit 與修改摘要

- **ContourCraft**：branch `main`，追蹤 `origin/main`，HEAD `d606432c2f7e96b05664e6a4ffb9d1cb98705baf`
- **Gaussian-Garments**：branch `main`，追蹤 `origin/main`，HEAD `ceb66fc36598da05b3b3391f4562cdc54d2a2ed2`

本交接資料夾以新增獨立腳本為主。對官方既有檔案僅一處修改：`Gaussian-Garments/inference.py`
（內容見第9節）。兩個 repo 的 `utils/defaults.py` 需依官方安裝說明自行設定，未收錄於本交接
範圍（見文件最上方「路徑設定」）。
