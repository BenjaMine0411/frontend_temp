# Fixed-Size

## 1. 概述

服裝套用到不同身高體重的目標身體時，服裝本身的尺寸不應該隨之改變——一件版型固定的上衣，換一個更高大的人穿，應該是「同一件衣服穿在不同身體上」，不是衣服本身被撐大。Fixed-size 協定解決的就是這個問題：讓服裝的尺寸只由它自己的來源資料決定，套用到任何目標身體時都維持原本大小，只有姿勢、貼合方式會依目標身體調整。

核心程式：`ContourCraft/contourcraft_fixed_size_upper.py`。

## 2. 核心機制：`shapedirs = 0`

服裝的頂點位置由兩部分決定：`shapedirs`（形狀變化方向，對應到身體的 betas 參數）跟 `posedirs`/`lbs_weights`（姿勢變化方式，對應到骨架動作）。Fixed-size 的做法分三步：

1. **保留服裝原始尺寸**：用服裝自己的來源 betas + 自己的 `shapedirs`，算出一份「已經套用過原始體型」的靜止頂點位置，直接寫死成新的 `v`（`contourcraft_fixed_size_upper.py` 第230行、251行）。
2. **`shapedirs` 歸零**：把輸出服裝字典的 `shapedirs` 設成全零陣列（第252行：`output_lbs["shapedirs"] = np.zeros_like(source_lbs["shapedirs"], dtype=np.float32)`）。之後不管套用到哪個目標身體、target betas 是什麼，服裝本身的靜止尺寸都不會再被 betas 影響。
3. **用目標身體重新綁定姿勢**：對服裝每個頂點，找出目標身體上最近的對應頂點，取用那個頂點的 `posedirs`/`lbs_weights`（第240-254行）——這一步只決定「服裝怎麼跟著目標身體的動作變形」，不影響尺寸。

這三步跑完後產生的新服裝字典，會一路保留到後續的模擬階段（`simulate_recovered_outfit.py` 只是把已經處理好的字典交給標準 ContourCraft dataloader，不會重新計算或覆蓋 `shapedirs`）。

## 3. 使用流程：`prepare` → `simulate`

### `prepare`：產生 fixed-size 服裝字典

不會跑模擬，純粹是資料轉換。

```bash
python contourcraft_fixed_size_upper.py prepare \
  --contourcraft-root /path/to/GG_projects/ContourCraft \
  --body-model-root /path/to/ccraft_data/aux_data/body_models/smplx \
  --source-garment-dict /path/to/source_garment.pkl \
  --source-betas /path/to/source_betas.pkl \
  --target-sequence /path/to/target_motion_smplx.npz \
  --output-garment-dict /path/to/output_fixedsize_garment.pkl \
  --model-type smplx \
  --source-gender female \
  --target-gender female
```

| 參數 | 意義 |
|---|---|
| `--source-garment-dict` | 原始服裝字典 |
| `--source-betas` | 服裝原本尺寸對應的身體 betas |
| `--target-sequence` | 要套用的目標身體/動作序列，內含 target betas |
| `--output-garment-dict` | 輸出路徑，檔案不可已存在 |

### `simulate`：跑 ContourCraft rollout

實際自動化流程（`/generate`）呼叫的是 `simulate_recovered_outfit.py`，不是這支腳本自己的 `simulate` 子指令——兩者最終呼叫的 ContourCraft rollout 邏輯相同，只是 `simulate_recovered_outfit.py` 多支援一次處理多件已合併的服裝（outfit）。

### 在自動化流程裡的位置

`run_outfit_preview_pipeline.py` 對每一件服裝，都會先呼叫 `prepare`（每個服裝一次，不論預覽或完整動畫都共用同一份結果），再呼叫 `simulate_recovered_outfit.py` 做實際模擬。

## 4. 合身自動判斷釘腰帶

`apply_fit_aware_pin()`（`contourcraft_fixed_size_upper.py` 第91-186行）在 `prepare` 階段，會比較服裝跟目標身體在腰帶位置的圍度（用凸包周長計算），判斷服裝是否夠貼身：

- 夠貼身：不需要額外釘住腰帶頂點，貼合本身就足以讓服裝不下滑
- 不夠貼身（寬鬆）：自動找出腰帶邊界頂點並釘住，避免物理模擬時因為沒有摩擦力模型而整件下滑（背景說明見 `pin_garment_waistband.py`）

對應參數 `--pin-waistband-if-snug`。**這個功能在網站實際運作時是自動套用的**：`preview_server.py` 每次生成請求都會把下身服裝的名字放進 `--pin-waistband-if-snug-garments`，所以每次選了下身服裝，都會自動做這個合身判斷，不需要使用者手動開啟。

## 5. 檢查工具：`inspect_garment_dict.py`

純讀取的診斷工具，印出一份服裝字典的內容摘要，不會修改任何檔案。

```bash
python inspect_garment_dict.py /path/to/garment_dict.pkl
```

輸出內容：
- `keys`：字典裡每個欄位的名稱與維度
- `vertex_count`／`face_count`：頂點數、面數
- `node_type_counts`：`node_type` 欄位裡每種數值各出現幾次
- `pinned_value_3_count`：其中數值 `3`（代表被釘住的頂點）的數量——用來確認腰帶/領口等錨點有沒有正確設定

建議使用時機：準備好一份 fixed-size 服裝字典之後、正式跑模擬之前，先執行一次確認拓樸/pin點是否符合預期，避免花時間跑完模擬才發現服裝資料本身有問題。

## 6. 相關檔案

服裝從原始資料到進入 fixed-size 流程、再到模擬的完整路徑：

```
recover_resize_garment_bootstrap.py（通用 checkpoint 路徑）
  或 import_official_gg_garment.py（GG 訓練結果路徑）
       ↓
pin_garment_waistband.py（視需要標記固定頂點）
       ↓
contourcraft_fixed_size_upper.py prepare（第2-4節）
       ↓
inspect_garment_dict.py（可選，健檢）
       ↓
simulate_recovered_outfit.py（實際模擬）
```

| 檔案 | 角色 | 輸入 → 輸出 |
|---|---|---|
| `recover_resize_garment_bootstrap.py` | 通用 checkpoint 路徑的服裝匯入 | GG 服裝軌跡 `.pkl` → `garment_dict.pkl` |
| `import_official_gg_garment.py` | GG 訓練結果路徑的服裝匯入 | 官方 SMPL-X 捕捉 `.npz` → `garment_dict.pkl` + `relax_trajectory.pkl` |
| `pin_garment_waistband.py` | 自動偵測並標記腰帶/領口等錨點頂點，避免無摩擦力模型下模擬時服裝滑落 | `garment_dict.pkl` → 原地加上 `node_type == 3` 標記 |
| `contourcraft_fixed_size_upper.py` | Fixed-size 核心邏輯，見第2-4節 | `garment_dict.pkl` + 目標身體/動作 → fixed-size `garment_dict.pkl` |
| `inspect_garment_dict.py` | 服裝字典檢查工具，見第5節 | `garment_dict.pkl` → 摘要報告（印在畫面上） |
| `simulate_recovered_outfit.py` | 實際自動化流程使用的模擬腳本 | fixed-size `garment_dict.pkl` + 動作 → `trajectory.pkl` |
