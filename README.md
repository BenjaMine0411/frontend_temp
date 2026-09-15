# 穿搭預覽 Demo — 前端交接

## 路徑設定

程式碼中的路徑使用通用佔位符（`/path/to/...`）。部署前請搜尋 `/path/to/` 並替換為實際路徑。
完整清單見 [HANDOFF.md](HANDOFF.md#路徑設定)。

## 概述

本機服裝模擬預覽網頁的前端部分。功能、架構、資料流程說明見 **[HANDOFF.md](HANDOFF.md)**；服裝尺寸固定機制（fixed-size）的完整說明見 **[FIXED_SIZE.md](FIXED_SIZE.md)**。
本文件僅說明安裝與啟動方式。

## 前提

需具備官方 [ContourCraft](https://github.com/Dolorousrtur/ContourCraft) +
[Gaussian-Garments](https://github.com/eth-ait/Gaussian-Garments) 程式碼與可執行的 conda 環境。
此資料夾非獨立安裝套件，內容為疊加至既有 checkout 使用。

## 安裝（不需 GPU）

於本資料夾（`frontend_handoff/`）根目錄執行，安裝 `preview_server.py` 唯一的外部相依：

```bash
pip install -r requirements.txt
```

## 部署

```bash
cp -r ContourCraft/*        /目標路徑/ContourCraft/
cp -r Gaussian-Garments/*   /目標路徑/Gaussian-Garments/
```

`ContourCraft/` 下的檔案均為新增的獨立腳本，不會覆蓋既有官方檔案。`Gaussian-Garments/inference.py`
為修改過的官方檔案，部署前請參閱 [HANDOFF.md 第9節](HANDOFF.md#9-主要檔案及用途)。

## 啟動（不需 GPU）

於複製後的 `ContourCraft/` 目錄下執行，沿用上一步安裝相依的 Python 環境：

```bash
python preview_server.py
```

開啟 http://127.0.0.1:5000 ，應可看到服裝選單與縮圖。

## 擬合身體（第一頁，需額外安裝）

網頁第一頁的「擬合身體」需要 `torch` + `open3d`。**注意**：這個功能是用啟動 `preview_server.py`
本身的那個 python 去執行的，不是另外指定的環境，所以不能用上面「安裝（不需GPU）」那個輕量環境
啟動——要用你本來就有的 `ccraft` conda 環境啟動，並額外安裝：

```bash
/目標路徑/conda/envs/ccraft/bin/pip install -r requirements-body-preview.txt
/目標路徑/conda/envs/ccraft/bin/python preview_server.py
```

（`torch`/`scipy`/`Pillow` 通常你的 `ccraft` 環境已經有，這份清單實際上只會新裝 `open3d`。）

## 執行生成請求（需 GPU + conda 環境）

啟動時需指定具備 ContourCraft/Gaussian-Garments 相依的 conda 環境：

```bash
/目標路徑/conda/envs/ccraft/bin/python preview_server.py
```

執行生成請求需 GPU、對應訓練模型，首次選擇新服裝組合需額外建置快取時間，詳見
[HANDOFF.md 第7-8節](HANDOFF.md)。

## 範例輸出

`sample_assets/` 提供一組實際測試輸出：轉台 PNG、預覽圖、完整影片、`status.json` 格式範例。
格式細節見 [API_FORMAT.md](API_FORMAT.md)。

## 已知限制

見 [HANDOFF.md 第5節](HANDOFF.md#5-已知限制)。
