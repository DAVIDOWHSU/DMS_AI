# DMS — 駕駛疲勞偵測系統

> 一個邊緣 AI 的作品集專案。
> 用攝影機即時偵測駕駛疲勞(連續閉眼)並發出警示,對應歐盟 GSR 2026/7 的 DDAW 法規。

## 快速開始

### 需求

- Python 3.11 / 3.12(MediaPipe 不支援 3.13)
- 筆電 + 內建鏡頭(或任何 USB 攝影機)
- ~60 MB 磁碟(模型 3.8MB + 依賴)

### 安裝

```bash
# 建虛擬環境
python -m venv .venv
.venv\Scripts\activate

# 裝依賴
pip install -r requirements.txt
```

### 執行完整 DMS

```bash
# 即時偵測:對著鏡頭閉眼超過 1 秒 → 紅框 + 嗶聲
python scripts/run_dms.py

# 冒煙測試(只看 EAR 計算,不警示)
python scripts/test_camera_facemesh.py
```

**參數調整:**編輯 `configs/default.yaml`:
```yaml
drowsiness:
  ear_threshold: 0.2     # EAR 門檻(實測:張眼~0.30、閉眼<0.02)
  drowsy_seconds: 1.0    # 連續閉眼秒數
alert:
  beep_cooldown_s: 1.5   # 嗶聲最小間隔
```

### 執行測試

```bash
pytest  # 50 綠,無需攝影機、喇叭、GUI
```

## 架構概覽

```
攝影機/影片
    ↓
FaceLandmarker (MediaPipe Tasks API) → 478 個特徵點
    ↓
EAR 計算 → 眼睛張開比例(0.0~1.0)
    ↓
疲勞判定狀態機 → 時間制(秒數,非幀數)
    ↓
NO_MEASUREMENT / EYES_OPEN / EYES_CLOSING / DROWSY
    ↓
視覺警示(紅框) + 聲音警示(嗶聲)
```

詳見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 與 [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)。

## 核心特色

### 1. EAR 純函式

- 從 Soukupová & Čech 2016 論文實現
- 不變性:平移、旋轉、等比縮放下 EAR 不變
- 特徵點 index 對官方 `FaceLandmarksConnections` 驗證
- 單元測試驗證所有數學不變性

### 2. 時間制狀態機

- 連續閉眼**秒數**而非幀數 → 跨裝置一致
- NaN 容錯:短暫掉偵測不清零累積時間
- 實測數據驅動:門檻依現場 live test 設定

### 3. 跨平台聲音

- numpy 合成 + sounddevice 播放(不用 winsound)
- 部署 Pi 時無需改代碼
- 單元測試:可注入假 player,驗證冷卻邏輯無需喇叭

### 4. 參數外部化

- 所有閾值/警示音在 `configs/default.yaml`
- 改參數無需重編譯
- 支援多套配置

## 實測數據

**筆電(2026-06-12):**
| 狀態 | EAR | FPS |
|---|---|---|
| 雙眼張開 | ~0.30 | 30~32 |
| 單眼閉 | ~0.15 | - |
| 雙眼閉 | <0.02 | - |

→ 門檻 0.2 充足;單眼閉也會觸發(開車閉單眼 1 秒值得警示)。

## 法規對應

**歐盟 GSR(General Safety Regulation)** — 2026/7 新車強制
- **ADDW**:駕駛分心警示
- **DDAW**:駕駛疲勞/注意力警示(本專案對應)

本系統:
- 邊緣端本地運算(隱私合規)
- 不依賴臉部辨識(只用幾何特徵點)
- 反應時間 1 秒(符合法規要求)

詳見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 測試

```bash
pytest -v  # 詳細輸出

# 或分別跑
pytest tests/test_ear.py          # 18 tests:EAR 公式驗證
pytest tests/test_drowsiness.py   # 20 tests:狀態機邏輯
pytest tests/test_alert.py        # 12 tests:波形/冷卻
```

**特色:**全部無頭測試,不需攝影機/喇叭/GUI,跑速 0.37s。

## 資料夾結構

```
src/dms/              核心模組(可被 import)
  ├─ face.py          FaceLandmarker Tasks API 封裝
  ├─ ear.py           EAR 計算(純函式)
  ├─ drowsiness.py    疲勞判定狀態機
  └─ alert.py         視覺/聲音警示

scripts/              進入點(可執行腳本)
  ├─ run_dms.py       完整 pipeline
  └─ test_camera_facemesh.py  冒煙測試

tests/                單元測試(50 綠)
configs/              YAML 參數
docs/                 設計文件 + Mermaid 圖
```

## 設計決策

詳見 [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md),包括:
- **EAR 與座標轉換**:為什麼要乘回像素座標
- **時間制 vs 幀數制**:為什麼用秒而非幀
- **MediaPipe 升級**:從舊版 solutions 遷移到 Tasks API
- **聲音跨平台**:為什麼用 numpy + sounddevice
- **無頭測試**:為什麼所有測試都不需設備

## 階段規劃

### ✅ 階段一(完成):開發 + 驗收
- [x] EAR 純函式 + 單元測試(18 tests)
- [x] 疲勞判定狀態機(20 tests)
- [x] 視覺/聲音警示(12 tests)
- [x] 完整 pipeline + live 驗收(FPS 30~32)
- [x] git 初始化 + 首 commit

### 🔄 階段二(規劃中):部署優化
- [ ] Raspberry Pi 5 / Jetson Nano
- [ ] INT8 量化(模型壓縮)
- [ ] FPS/延遲 before-after 量測
- [ ] 邊緣推理優化(CPU vs GPU)
- [ ] 多人 tracking

### 📊 階段三(遠期):監控/數據
- [ ] 疲勞事件上報(時間戳、EAR、持續時長)
- [ ] 多人統計(駕駛技能評分)
- [ ] 數據存儲(SQLite/PostgreSQL)
- [ ] OTA 更新(模型/參數遠端下發)

## 開發紀錄

見 [devlog.md](devlog.md):
- Day 0:環境決策、冒煙測試、MediaPipe 確認
- Day 1:EAR 實現 + 驗收、drowsiness 狀態機、alert 與 pipeline、git init

## 貢獻 & 授權

本專案是個人作品集。歡迎參考設計思路,但複製代碼前請先聯絡。

---

**下一步:**
1. ⭐ Star 此 repo,幫我宣傳
2. 👀 查看 [docs/](docs/) 裡的設計決策
3. 🚀 建議:建 GitHub Issues 討論改進方向(多人 tracking、個人化閾值 etc.)
