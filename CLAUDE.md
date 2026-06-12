# CLAUDE.md — 駕駛疲勞/注意力偵測 (DMS) 邊緣 AI 專案

> 這份檔案會被 Claude Code 自動載入成上下文。更動結構或慣例時請同步更新這裡。

## 專案目標

一個「駕駛疲勞偵測 (Driver Monitoring System, DMS)」的邊緣 AI 作品集專案,
用於從車載軟體工程師轉職邊緣 AI。核心流程:

    攝影機影像 → MediaPipe 抓臉部特徵點 → 計算 EAR(眼睛張開比例)
    → 連續閉眼超過門檻 → 判定為疲勞 → 發出警示

## 產業背景(這是面試敘事的重點,別弄丟)

歐盟 GSR (General Safety Regulation) 自 2026/7 起,強制新車搭載
ADDW (Advanced Driver Distraction Warning) 駕駛監控系統。本專案直接對應此法規需求,
證明能把法規要求落地成可運行的邊緣 AI 系統。

## 分階段規劃

- **階段一(開發,目前進行中,零花費):** 筆電 + 內建鏡頭,
  Python + OpenCV + MediaPipe + NumPy,完成 EAR → 疲勞判定 → 警示的核心邏輯。
- **階段二(部署,之後):** 上樹莓派 5 / Jetson,做 INT8 量化,
  記錄 FPS 與延遲,產出 before/after 的效能數據。

## 環境

- Python **3.11 / 3.12**(MediaPipe 不支援 3.13,務必避開)。
- 開發/git/Claude Code/鏡頭測試全部在 **Windows 原生**
  (Day 0 已棄用 WSL2:預設抓不到內建鏡頭,環境摩擦太大)。
- 虛擬環境:`.venv`(Python 3.12.10,winget 安裝)。
- **mediapipe ≥0.10.3x 已移除舊版 `mp.solutions` API**,只能用 Tasks API
  (`mediapipe.tasks.python.vision.FaceLandmarker`,需要 `face_landmarker.task` 模型檔)。
  寫 MediaPipe 相關程式一律走 Tasks API,別再用 `mp.solutions.face_mesh`。

## 資料夾結構

```
src/dms/        可被 import 的核心套件
  face.py         MediaPipe 封裝:抓臉部特徵點
  ear.py          EAR 計算(純函式,易測試)
  drowsiness.py   連續閉眼 → 疲勞判定狀態機
  alert.py        警示(畫面/聲音)
scripts/        進入點(可執行腳本)
tests/          單元測試(不依賴鏡頭)
configs/        可調參數(EAR 門檻、連續閉眼幀數…)
data/samples/   測試影片(不進 git)
docs/           設計說明 + Mermaid 圖
```

## 程式慣例

- 核心演算法(尤其 EAR)寫成**純函式**,輸入座標、輸出數值,搭配單元測試。
- 可調參數放 `configs/default.yaml`,不要寫死在程式裡。
- 影像處理腳本一律支援 `--source`(攝影機 index 或影片檔路徑),
  讓同一份程式碼在 WSL2(影片)與 Windows(live)都能跑。
- 函式請寫型別註記與簡短 docstring。

## 給 Claude Code 的工作方式

- 改動前先說明計劃(要改哪些檔、怎麼改),確認後再實作。
- 寫演算法邏輯時,連同 `tests/` 的單元測試一起寫。
- 不確定 MediaPipe 特徵點 index 時,明確標註出處,別憑印象填數字。

## 現況 / 下一步

- [x] 冒煙測試 `scripts/test_camera_facemesh.py`(攝影機 + FaceLandmarker 能動)
- [x] `src/dms/ear.py`:用 MediaPipe 眼睛特徵點實作 EAR + 單元測試(18 tests 綠)
- [x] `scripts/test_camera_facemesh.py` 遷移到 Tasks API(`FaceLandmarker`,
      含即時 EAR 顯示;live 實測過:FPS 30~32,EAR 張眼 ~0.30 / 閉眼 <0.02)
- [x] `src/dms/drowsiness.py`:連續閉眼的疲勞判定狀態機
      (時間制,預設 ear_threshold=0.2、drowsy_seconds=1.0;20 tests 綠,累計 38)
- [ ] `src/dms/alert.py`:警示(畫面/聲音)
- [ ] 串起完整即時 pipeline 腳本(FaceLandmarker → ear → drowsiness → alert)
