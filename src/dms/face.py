"""
face.py — MediaPipe FaceLandmarker 封裝(Tasks API)。

把「模型下載、landmarker 建立、BGR→mp.Image 轉換、timestamp 嚴格遞增」
這些樣板集中在這裡,讓 scripts/ 的進入點只需要:

    with FaceLandmarkerVideo() as fl:
        landmarks = fl.detect(bgr_frame)   # 第一張臉的 478 點,沒臉回 None

注意:mediapipe 0.10.3x 已移除舊版 mp.solutions,本模組只走 Tasks API。
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions

# 從定義模組直接 import 類別(vision/__init__ 是用賦值轉出,
# Pylance 會把 vision.FaceLandmarker 當變數、不能放型別註記)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 官方模型下載點(MediaPipe Face Landmarker 文件提供的 float16 版本)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """模型檔不存在就從官方下載點抓一份(~3.8MB,只需一次)。"""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 下載 FaceLandmarker 模型 → {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"[INFO] 下載完成({path.stat().st_size / 1e6:.1f} MB)")
    return path


class FaceLandmarkerVideo:
    """VIDEO 模式 FaceLandmarker 的薄封裝(同步呼叫,攝影機/影片檔都適用)。

    處理兩件煩人的事:
    - BGR(OpenCV)→ SRGB mp.Image 轉換
    - VIDEO 模式要求 timestamp 嚴格遞增(影片檔讀太快會撞毫秒)
    """

    def __init__(
        self,
        model_path: Path | None = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_model(model_path or MODEL_PATH))
            ),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker: FaceLandmarker = FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect_faces(self, bgr_frame) -> list:
        """對一幀 BGR 影像做偵測,回傳**所有**臉的特徵點列表;沒臉回空 list。

        注意:回傳順序由 MediaPipe 決定,幀與幀之間**不保證**同一張臉
        在同一個 index —— 多人場景請配 dms.tracking.CentroidTracker 使用。
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        return list(result.face_landmarks)

    def detect(self, bgr_frame):
        """對一幀 BGR 影像做偵測,回傳第一張臉的特徵點列表;沒臉回 None。

        單臉場景的便利介面(smoke test / benchmark 沿用)。
        """
        faces = self.detect_faces(bgr_frame)
        return faces[0] if faces else None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkerVideo":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
