# DMS 設計決策紀錄

> 本文記錄為什麼做這些選擇,以及遇到的坑與解決方案。

## 環境決策:Windows 原生 vs WSL2

**決策:Windows 原生**

Day 0 原本打算在 WSL2 開發,但遇到問題:
- WSL2 預設核心未編入 UVC/V4L2 驅動,無法訪問筆電內建鏡頭
- 即使用 `usbipd-win` 也卡在多重依賴與kernel 編譯
- 時間成本過高,與專案無關

**取捨:**
- 損失:少一點「Linux 環境開發體驗」的展示
- 獲得:零環境摩擦,快速進入正題
- 補償:部署階段的 Pi/Jetson 就是真 Linux,那時補回來

**學到:**在邊緣 AI 專案,跨平台開發通常比單一平台更值得:筆電用舒服的環境,終端裝置用適合的 Linux。

---

## EAR 公式:正規化座標陷阱

**問題:**

MediaPipe 給的是「正規化座標」(x/影像寬、y/影像高)。乍看很方便,但眼睛特徵點若用正規化座標直接算 EAR:

```python
# ❌ 錯誤:正規化座標下長寬比被扭曲
norm_p1 = (0.2, 0.3)
norm_p4 = (0.6, 0.3)
h_width = norm_p4[0] - norm_p1[0]  # 0.4(正規化)

# 影像 640x480:橫向 span = 0.4 * 640 = 256px,但縱向 span = 0.4 * 480 = 192px
# 不同軸上的「0.4」代表不同的物理距離 → 長寬比扭曲 → EAR 不可信
```

**解法:**

在 `eye_points_from_landmarks()` 裡乘回像素座標:

```python
def eye_points_from_landmarks(landmarks, indices, image_width, image_height):
    return np.array([
        (landmarks[i].x * image_width,   # 還原 x 軸像素
         landmarks[i].y * image_height)  # 還原 y 軸像素
        for i in indices
    ])
```

這樣 EAR 才能對不同長寬比的影像(16:9、4:3 等)穩定工作。

**學到:**邊緣視覺任務裡,「正規化」是為了方便傳輸與儲存,不是最終計算座標。涉及物理量(距離、面積)的算法要小心轉回像素空間。

---

## 時間制 vs 幀數制

**決策:時間制(連續閉眼秒數)**

Day 1 實現狀態機時,原規劃是「連續閉眼幀數」,改成「秒數」的理由:

### 問題:幀數依賴於裝置

```python
# ❌ 幀數制
drowsy_frames = 30  # 假設 30 FPS

# 筆電 30 FPS 上:30 frames = 1 秒 ✓
# 樹莓派 15 FPS 上:30 frames = 2 秒 ✗(變成 2 秒才警示,不合需求)
```

### 解法:用秒數

```python
drowsy_seconds = 1.0  # 1 秒,與 FPS 無關

# 所有裝置上都是 1 秒(timestamp 驅動,不是幀計數)
state = detector.update(ear=0.05, timestamp=time.monotonic())
```

### 法規驗證

歐盟 DDAW 對「疲勞警示反應時間」的要求本來就是秒級:
- 法規文本:"Within X seconds of detection..."
- 我們選 1.0s 是在合理範圍內

**衍生設計:**NaN(臉掉偵測)時保留累積時間,是因為:
- 閉眼本來就容易掉臉(眼睛幾乎閉上,特徵點信心低)
- 0.1~0.2s 的短暫掉偵測不應清零已累積的「疲勞證據」
- 但 NaN 也不能累積(沒看到不能當閉眼)

```python
# NaN 時:
if math.isnan(ear):
    return DrowsinessState.NO_MEASUREMENT  # 不改變 _closed_since

# 臉回來閉眼:從原起點繼續算
state = detector.update(ear=0.05, timestamp=1.0)  # 仍在同一輪閉眼
```

---

## MediaPipe 破壞性變更:Solutions → Tasks API

**問題:**

Day 0 寫的冒煙測試用 `mp.solutions.face_mesh`,但在安裝 mediapipe 0.10.35 後跑不動:

```python
# ❌ 舊版(Day 0)
from mediapipe.python.solutions import face_mesh
face_mesh_obj = face_mesh.FaceMesh(...)
```

報錯:MediaPipe 0.10.3x 已移除 `mp.solutions`,整個遷移到 Tasks API。

**解法:**

1. 建 `src/dms/face.py` 統一負責 Tasks API 使用
2. 兩支 scripts(冒煙測試、pipeline)共用 `FaceLandmarkerVideo` 類別

```python
# ✅ 新版(封裝)
from dms.face import FaceLandmarkerVideo

with FaceLandmarkerVideo() as face:
    landmarks = face.detect(bgr_frame)
```

**關鍵細節:**

Tasks API 有個坑:
```python
# ❌ Pylance 抱怨
from mediapipe.tasks.python import vision
options = vision.FaceLandmarkerOptions(...)  # 把 Options 當變數,不能放型別註記
```

因為 `vision/__init__.py` 是用賦值轉出類別,不是真正的類別定義。解法:
```python
# ✅ 從定義模組直接 import
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions
```

**教訓:**開源框架的大版本變更常有不向後相容的 API 改動。邊緣部署應該:
1. 版本固定(requirements.txt 鎖版本)
2. 假設一陣子內不會自動升級
3. 主要邏輯用自己的「薄封裝」包一層,換 SDK 時只需改封裝

---

## 聲音警示:為什麼用 sounddevice

**決策:numpy 合成 + sounddevice 播放(而非 winsound)**

### 為什麼不用 winsound?

```python
import winsound  # 只有 Windows 有
winsound.Beep(880, 400)
```

- 平台相依(Windows-only)
- 部署到樹莓派時需要改代碼或換方案
- 不能通過單元測試(沒喇叭時會卡)

### 為什麼 numpy + sounddevice?

```python
# sounddevice 是 mediapipe 依賴的一部分,不增加額外相依
import sounddevice as sd
wave = numpy.sin(...)  # numpy 已有
sd.play(wave, samplerate=44100)
```

優點:
1. 跨平台(Windows/Linux/macOS 都支援)
2. 部署 Pi 時 sounddevice 已經在(mediapipe 附帶),無需安裝
3. **可測**:在單元測試時注入假 player,驗證冷卻邏輯不需真的播放

```python
class _FakePlayer:
    def __call__(self, waveform, sample_rate):
        self.calls.append((waveform, sample_rate))

sound = SoundAlert(player=_FakePlayer())
sound.trigger(now=0.0)  # 無聲執行,驗證邏輯
```

### 波形合成細節

```python
def make_beep(frequency_hz=880, duration_s=0.4, ...):
    wave = volume * np.sin(2π * f * t)
    # 頭尾加 5ms 淡入出(linear ramp)
    # → 避免播放時的爆音(click)
```

880 Hz 是 A5 音高:高於語音頻率、易於察覺、不刺耳。

---

## 單元測試:為什麼全部無頭

**設計原則:**

所有核心模組都設計成「無需外部設備」可測:

| 模組 | 測試方法 | 優點 |
|---|---|---|
| EAR | 手算期望值 + 合成座標 | 快(18 tests in 0.1s),可驗證不變性 |
| drowsiness | 時間序列 + 假時鐘 | 可模擬 30 FPS 完整情境 |
| alert | 假 player + numpy 波形 | 驗證冷卻邏輯,無需喇叭 |

**好處:**

1. **CI 友善**:GitHub Actions 無需特殊硬體,純 CPU 跑 pytest
2. **開發快速**:改一行代碼,測試在 0.4s 內反饋
3. **可靠**:不依賴攝影機、喇叭、GUI,結果可重現

**代價:**

- 不能測「實際的視覺警示長什麼樣」
  → 解法:用冒煙測試(`test_camera_facemesh.py`)live 驗收
- 不能測「播放聲音是否聽得清」
  → 解法:`scripts/run_dms.py` live 驗收

---

## 參數外部化:YAML vs 硬編

**決策:全部放 YAML (`configs/default.yaml`)**

初始衝動是「參數就幾個,直接寫程式裡」,但這樣的後果:
- 改門檻值要改代碼、重啟、重新部署
- 部署多台裝置時無法統一調參(每台配置不同)
- 線上問題診斷時無法 A/B test 不同門檻

**YAML 版本:**

```yaml
drowsiness:
  ear_threshold: 0.2
  drowsy_seconds: 1.0
```

用 pyyaml 在 `run_dms.py` 載入:

```python
cfg = yaml.safe_load(open('configs/default.yaml'))
detector = DrowsinessDetector(
    DrowsinessConfig(**cfg['drowsiness'])
)
```

**優點:**

1. 參數改動無需重編譯
2. 支援多套配置(預設/激進/保守等)
3. 易於版本管理和審計

---

## 設計檢查清單

實現時都驗證過:

- [x] EAR index 對官方 `FaceLandmarksConnections` 驗證過
- [x] 時間制狀態機在邊緣(timestamp 邊界)無 bug
- [x] NaN 邏輯經過「掉偵測→恢復→繼續」情景測試
- [x] 聲音波形不會爆音(淡入出)
- [x] 警示冷卻不會連續嗶(測試驗證)
- [x] 50 個單元測試全綠
- [x] 冒煙測試 live 實測通過(張眼/閉眼/眨眼分別驗證)
- [x] git 初始化,首 commit 鎖定設計決策

---

## 後續改進空間

1. **多人追蹤**
   - 目前只偵測第一張臉(`num_faces=1`)
   - 升級到 `num_faces=5` + per-face 狀態機

2. **姿態檢測**
   - 加入 pose estimation,偵測頭倒(睡著時頭會垂)
   - 結合 EAR + 姿態 → 疲勞信心度提升

3. **個人化閾值**
   - EAR 因人而異(眼睛大小、臉型)
   - 學習個人基線,動態調整 `ear_threshold`

4. **INT8 量化**
   - 階段二:模型壓縮,FPS 提升
   - 部署 Pi 時測延遲

5. **事件上報**
   - 疲勞事件時間戳、EAR 值、閉眼時長送到後端
   - 多人駕駛統計分析
