"""一次性自檢:模型下載 + FaceLandmarkerVideo + 黑畫面偵測(不需鏡頭)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dms.face import MODEL_PATH, FaceLandmarkerVideo

with FaceLandmarkerVideo() as face:
    print(f"model: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = face.detect(black)
    assert landmarks is None, "black frame should have no face"
    print("selfcheck OK - black frame -> no face (expected)")
