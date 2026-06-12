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
- [x] `src/dms/alert.py`:警示(紅框 + sounddevice 嗶聲,冷卻 1.5s;12 tests)
- [x] `src/dms/face.py`:FaceLandmarker Tasks API 封裝(腳本共用)
- [x] 完整即時 pipeline `scripts/run_dms.py`(configs/default.yaml 可調參數)
- [x] git init + 首 commit(b063fca);測試累計 50 綠
- [x] live 驗收 run_dms.py(閉眼 1 秒 → 紅框 + 嗶聲)
- [x] GitHub repo + push(DAVIDOWHSU/DMS_AI,public + All Rights Reserved);
      docs/(README、ARCHITECTURE、DESIGN_DECISIONS、DEPLOYMENT_OPTIMIZATION)
- [x] 階段二前置:benchmark.py(分階段延遲/FPS/JSON)、inspect_model.py
      (.task = 3 顆 TFLite;blendshapes 佔 25% 但未用;EAR+狀態機僅 0.01ms/幀)
- [x] live 鏡頭 benchmark baseline(docs/benchmarks/laptop_camera.json:端到端 29.9 FPS,
      瓶頸是鏡頭讀幀 20.2ms,推理 13.0ms)
- [x] Q2 多人臉:src/dms/tracking.py 質心追蹤(穩定 id + 每人專屬狀態機,
      data_factory 注入、Q3 可複用;含 face_centroid/face_bbox)、face.py 加
      detect_faces()(detect() 保留兼容)、alert.py 加 draw_face_status()/
      draw_drowsy_banner()、run_dms.py 用 tracker 管多狀態機、configs 加
      face.num_faces=2 + tracking 參數。決策:駕駛=當幀最大臉,只有駕駛
      DROWSY 才全屏紅框+嗶聲。23 個新測試,累計 73 綠
- [ ] live 驗收 Q2(本人 + 手機照片各一張臉,確認 id 穩定、閉眼計時不互相干擾)
- [ ] Q3 分心偵測(手機/手離方向盤,MediaPipe ObjectDetector 或 YOLO,先技術探路)
- [ ] 階段二:部署 Pi 5 / Jetson(先 XNNPACK baseline,不夠快才量化)、FPS/延遲 before-after
