---
title: DMS 駕駛疲勞偵測 — 開發紀錄 (Dev Log)
tags: edge-ai, dms, computer-vision, portfolio
---

# DMS 駕駛疲勞偵測 — 開發紀錄

> 一個從車載軟體工程師轉職邊緣 AI 的作品集專案。
> 用一般攝影機即時偵測駕駛疲勞(連續閉眼)並發出警示,
> 對應歐盟 GSR 對車載駕駛監控(DMS)的法規趨勢。

[TOC]

---

## 專案背景 / 為什麼做這個

歐盟《一般安全法規》(GSR, Regulation (EU) 2019/2144)要求 M、N 類新車
搭載「進階駕駛分心警示」(ADDW)系統:**新車型自 2024/7/7、所有新車自 2026/7/7 強制**。
與之並行的還有針對「疲勞/注意力」的 DDAW(Driver Drowsiness and Attention Warning)。
兩者底層都是車載駕駛監控系統(DMS)。

本專案聚焦 **DDAW 面向**:用攝影機抓臉部特徵點、以 EAR(眼睛張開比例)判斷
連續閉眼 → 判定疲勞 → 警示。選這個切角的理由有三:

1. **真實產業相關性** —— 法規 2026 年全面上路,是我車載背景與邊緣 AI 之間最自然的橋。
2. **隱私/合規對齊** —— GSR 要求這類系統盡量不依賴臉部辨識等敏感生物特徵、只保留必要資料;
   本專案用的是「幾何特徵點」而非身分辨識,且全程在邊緣端本地運算,呼應這個方向(可連結我 ISO 21434 / 資安合規經驗)。
3. **能展示完整鏈路** —— 從演算法、單元測試、版本控制,到之後的裝置部署與 INT8 量化效能量測。

---

## 技術棧

| 層面 | 選用 |
|---|---|
| 開發階段 | Python 3.11/3.12、OpenCV、MediaPipe、NumPy(筆電 + 內建鏡頭,零花費) |
| 部署階段(規劃) | Raspberry Pi 5 / Jetson、INT8 量化、FPS / 延遲量測 |
| 核心方法 | MediaPipe 臉部特徵點 → EAR → 連續閉眼狀態機 → 警示 |
| 開發環境 | Windows 原生、VS Code、Claude Code |
| 版本控制 / 文件 | Git / GitHub、本檔(devlog)、HackMD(GitHub Sync 發布) |

---

## 目前進度

> 每完成一項就更新這裡,讓讀者(和未來的我)一眼看到現況。

- [x] 環境決策、專案骨架、攝影機 + MediaPipe 冒煙測試
- [x] `src/dms/ear.py`：EAR 計算(純函式)+ 單元測試
- [x] `src/dms/drowsiness.py`：連續閉眼的疲勞判定狀態機(時間制,非幀數)
- [x] `src/dms/alert.py`：警示(畫面紅框 + sounddevice 嗶聲,含冷卻)
- [x] 串起完整即時 pipeline 腳本(`scripts/run_dms.py`,live 驗收待跑)
- [ ] 部署到實體裝置、INT8 量化、記錄 before/after FPS 與延遲

---

## 開發紀錄

<!--
給 Claude Code 的維護規則:
1. 新的 entry 一律加在本區塊「最底下」(時間順序:舊在上、新在下)。
2. 新增 entry 後,順手更新上方「目前進度」的勾選狀態。
3. 每則 entry 保持精簡;高價值的是「問題怎麼解」與「為什麼這樣決策」,別省略這兩段。
4. 日期用 YYYY-MM-DD;Day N 從 0 起算。
-->

### Entry 模板(複製下面這段開新紀錄)

```markdown
### YYYY-MM-DD · Day N:<這次的主題>

**目標**
-

**做了什麼**
-

**遇到的問題 / 怎麼解**
-

**決策與取捨**(沒有就省略)
-

**學到的 / 下一步**
-
```

---

### 2026-06-11 · Day 0:環境決策與冒煙測試

**目標**
建立專案骨架、確認開發環境可行,並跑通第一支「攝影機 + MediaPipe」測試。

**做了什麼**
- 建立 `src/dms` 套件 + `scripts/` / `tests/` 的正規結構。
- 寫出 `scripts/test_camera_facemesh.py` 冒煙測試。
- 成功在畫面上看到臉部網格與即時 FPS,確認影像鏈路與 MediaPipe 可用。

**遇到的問題 / 怎麼解**
- 原本想在 WSL2 開發,但 WSL2 預設核心未編入 UVC/V4L2 驅動,抓不到筆電內建鏡頭;
  即使用 usbipd-win attach 進去,仍需自編核心、且常卡在串流錯誤(timeout / ECONNRESET)。
  → 改用 **Windows 原生環境**跑鏡頭,避開與專案無關的環境摩擦。
- MediaPipe 最新版只支援到 **Python 3.12**,誤用 3.13 會在安裝階段直接失敗。
  → 固定使用 Python 3.11/3.12。

**決策與取捨**
- 開發階段選 Windows 原生而非 WSL2:犧牲一點「Linux 開發經驗」的展示,
  換取零環境摩擦;反正部署階段的 Pi / Jetson 是真 Linux,鏡頭本就能直接用,Linux 經驗在那邊補回來。
- 把 EAR 規劃成純函式:為了能寫「不依賴鏡頭」的單元測試,對 CI 友善,
  也讓核心邏輯的正確性可被獨立驗證。

**學到的 / 下一步**
- 學到:WSL2 在硬體(USB/攝影機)穿透上的限制與成因。
- 下一步:實作 `src/dms/ear.py` —— 從 MediaPipe 468 點中挑出雙眼各 6 點套 EAR 公式,
  並用合成座標寫單元測試(不開鏡頭即可驗證)。

---

### 2026-06-12 · Day 1:環境重建、EAR 純函式 + 單元測試

**目標**
落地專案骨架、重建 Windows 開發環境,完成 `src/dms/ear.py` 與單元測試。

**做了什麼**
- 把 `files/` 裡的 CLAUDE.md / requirements / smoke test 搬回正規位置,
  建出 `src/dms`、`scripts`、`tests`、`configs`、`data/samples`、`docs` 骨架 + `.gitignore`。
- winget 安裝 Python 3.12.10、建 `.venv`、裝 opencv 4.13 / mediapipe 0.10.35 / numpy 2.4.6 / pytest。
- 實作 `src/dms/ear.py`:`compute_ear()`(EAR 純函式)、
  `eye_points_from_landmarks()`(正規化座標 → 像素座標)、`average_ear()`(單眼失效容錯)。
- `tests/test_ear.py` 18 個測試全綠:手算值、閉眼為 0、縮放/平移/旋轉不變性、
  退化保護、形狀檢查、index 健全性、座標換算、雙眼平均。
- 加 `pyproject.toml`(pytest `pythonpath = ["src"]`,src layout)。
- **冒煙測試遷移到 Tasks API**:`scripts/test_camera_facemesh.py` 改用
  `FaceLandmarker`(VIDEO 模式、同步 `detect_for_video()`),
  模型檔 `models/face_landmarker.task` 首次執行自動下載(3.8MB,gitignore)。
  順便整合 `dms.ear`:畫面即時顯示 EAR、12 個眼點標黃,等於用真實特徵點驗證 EAR 整合。
  無頭自檢(`scripts/_selfcheck_tasks_api.py`)通過:模型載入 + 黑畫面偵測 0 臉。

**遇到的問題 / 怎麼解**
- 這台機器的 `python` 其實是 Microsoft Store 空殼,環境根本沒裝過。
  → `winget install Python.Python.3.12` 重建,固定 3.12(MediaPipe 不支援 3.13)。
- **mediapipe 0.10.35 已整個移除舊版 `mp.solutions` API**(只剩 `mediapipe.tasks`)。
  影響:Day 0 的 `scripts/test_camera_facemesh.py` 在新環境跑不動。
  → 同日完成遷移(見上)。兩個遷移細節:
  ① VIDEO 模式要求 timestamp 嚴格遞增,影片檔讀太快會撞毫秒,用 last_ts+1 保證遞增;
  ② Pylance 把 `vision.FaceLandmarker` 當變數(`vision/__init__` 用賦值轉出),
     型別註記要從定義模組 `...vision.face_landmarker` 直接 import 類別。
- 眼睛特徵點 index 出處驗證:舊的 `face_mesh_connections` 模組已不存在,
  改用 `FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE/RIGHT_EYE` 驗證,
  確認 `ear.py` 的左右眼 6 點皆為官方眼部點集子集(`scripts/_verify_eye_idx.py`)。

**決策與取捨**
- EAR 退化(眼寬趨近 0)回 **NaN 而不是 0**:0 會被誤判成閉眼而誤觸警示,
  NaN 讓上層狀態機把該幀當「無量測」跳過。`average_ear()` 同理:單眼 NaN 用另一眼。
- `eye_points_from_landmarks()` 強制傳入影像寬高:MediaPipe 正規化座標在
  非正方形影像會扭曲長寬比,EAR 是「比例」,必須先還原像素尺度再算。
- 測試走純合成座標、不 import mediapipe:跑得快、CI 不需要相機或模型檔。

**學到的 / 下一步**
- 學到:mediapipe 0.10.3x 的破壞性變更(legacy solutions 移除);
  以及正規化座標對「比例類特徵」的長寬比陷阱。
- 下一步:在實機跑 live 冒煙測試確認鏡頭 + EAR 數值合理(張眼 ~0.25-0.35、閉眼趨近 0),
  然後實作 `src/dms/drowsiness.py` 疲勞判定狀態機。

---

### 2026-06-12 · Day 1(續):live 實測 EAR + 疲勞判定狀態機

**目標**
實機驗證 Tasks API 冒煙測試與 EAR 數值,完成 `src/dms/drowsiness.py`。

**做了什麼**
- **live 實測通過**(筆電內建鏡頭,CPU):FPS 30~32;
  EAR 實測 —— 雙眼張開 ≈ 0.30、單眼閉 ≈ 0.15、雙眼閉 < 0.02。
  張/閉間隔大,經典門檻 0.2 餘裕充足,直接採用。
- `src/dms/drowsiness.py`:`DrowsinessConfig`(dataclass,參數驗證)+
  `DrowsinessDetector` 狀態機,狀態 NO_MEASUREMENT / EYES_OPEN / EYES_CLOSING / DROWSY。
- `tests/test_drowsiness.py` 20 個測試全綠(累計 38):眨眼不誤觸、
  恰達門檻觸發、張眼歸零重算、NaN 斷檔保留累積、30FPS 完整情境模擬。

**決策與取捨**
- **時間制取代幀數制**:原規劃「連續閉眼幀數」,改成「連續閉眼秒數」。
  理由:FPS 隨裝置漂移(筆電 ~30,Pi 可能 10~15),幀數門檻在部署階段會失準;
  DDAW 法規的反應時間本來就以秒為單位。預設 `drowsy_seconds=1.0`(眨眼 0.1~0.4s)。
- **NaN 不累積、不重置**:臉短暫掉偵測(閉眼本來就較易掉)不清掉已累積的閉眼時間,
  但「沒看到」也不能當「閉眼」累積;臉回來仍閉眼則從原起點繼續算。
- 單眼閉(EAR ≈ 0.15)會低於 0.2 門檻 → 視為可接受:開車閉單眼超過 1 秒同樣值得警示。

**學到的 / 下一步**
- 學到:把「裝置相依的幀數」換成「物理量(秒)」是邊緣部署的關鍵習慣。
- 下一步:`src/dms/alert.py` 警示(畫面/聲音),然後串完整即時 pipeline
  (FaceLandmarker → ear → drowsiness → alert)。也該 `git init` 了。

---

### 2026-06-12 · Day 1(完):alert + 完整 pipeline + git init

**目標**
完成警示模組、串起完整即時 pipeline,專案進版控。

**做了什麼**
- `src/dms/face.py`:FaceLandmarker 的 Tasks API 封裝(`FaceLandmarkerVideo`,
  context manager;模型自動下載、BGR→mp.Image、timestamp 嚴格遞增都收進來),
  冒煙測試與 pipeline 共用,消除重複碼。
- `src/dms/alert.py`:`draw_alert()` 視覺警示(狀態列 + DROWSY 紅框大字)、
  `SoundAlert` 聲音警示(numpy 合成 880Hz 正弦波 + 5ms 淡入出防爆音,
  sounddevice 播放,內建 1.5s 冷卻防洗版)。
- `configs/default.yaml` + pyyaml:門檻、警示音、解析度全部外部化。
- `scripts/run_dms.py`:完整 pipeline(攝影機 → FaceLandmarker → EAR →
  狀態機 → 畫面/聲音警示),支援 `--source` 與 `--config`。
- 測試累計 **50 綠**(alert 12 個:波形/冷卻/紅框畫素驗證,全部不需喇叭與 GUI)。
- `git init` + 首 commit `b063fca`(18 檔、1482 行)。

**決策與取捨**
- 聲音用「numpy 合成 + sounddevice」而非 winsound:winsound 只有 Windows,
  sounddevice(mediapipe 已附帶)跨平台,部署到 Pi 不用改警示層。
- `SoundAlert` 的 player 設計成可注入的 callable:單元測試傳假 player,
  不需要喇叭就能驗證冷卻邏輯 —— 與 EAR 純函式同一個「可測性優先」原則。
- 視覺警示驗證直接斷言畫素(角落 BGR 值),不用 mock cv2。

**學到的 / 下一步**
- 下一步:live 跑 `scripts/run_dms.py` 驗收完整鏈路(閉眼 1 秒 → 紅框 + 嗶聲),
  然後建 GitHub repo 推上去;再來是 docs/ 設計說明 + Mermaid 圖,
  與階段二的裝置部署規劃。
