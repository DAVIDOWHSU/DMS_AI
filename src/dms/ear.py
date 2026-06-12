"""
ear.py — EAR (Eye Aspect Ratio, 眼睛張開比例) 計算。

純函式模組:輸入特徵點座標、輸出數值。不 import MediaPipe、不碰攝影機,
所以單元測試可以用合成座標直接驗證,對 CI 友善。

EAR 公式(出處:Soukupová & Čech, 2016,
"Real-Time Eye Blink Detection using Facial Landmarks", CVWW 2016):

        EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)

    p1、p4 是眼睛水平方向的兩個端點(眼角),
    (p2, p6) 與 (p3, p5) 是上下眼瞼的兩組對應點。
    眼睛張開時 EAR 大約落在 0.25~0.35,閉眼時趨近 0。
    EAR 對平移/旋轉/等比縮放不變,適合做跨人臉、跨距離的閉眼判斷。

座標系注意事項:
    MediaPipe 給的是「正規化座標」(x 除以影像寬、y 除以影像高)。
    若影像不是正方形,直接拿正規化座標算 EAR 會被長寬比扭曲,
    必須先乘回像素尺寸 —— 用 `eye_points_from_landmarks()` 就會處理好。
"""

import math
from typing import Sequence

import numpy as np

# MediaPipe FaceMesh(468 點)中,左右眼各取 6 點、依 EAR 公式的 p1~p6 排列。
#
# index 出處:MediaPipe Tasks API 的官方眼部連線集
# (mediapipe.tasks.python.vision.FaceLandmarksConnections 的
#  FACE_LANDMARKS_LEFT_EYE / FACE_LANDMARKS_RIGHT_EYE;
#  舊版 mp.solutions 的 FACEMESH_LEFT_EYE/RIGHT_EYE 同一套拓撲)。
# 已用 scripts/_verify_eye_idx.py 對裝機版 mediapipe 0.10.35 驗證為其子集。
# 這 6 點子集合是社群通用的 EAR 對應(等價於 dlib 68 點的 36~41 / 42~47)。
#
# 「右眼/左眼」是以「被拍攝者本人」的方向為準(不是畫面左右)。
#
#   p1=眼角, p2/p6=靠眼角側的上/下眼瞼, p3/p5=靠鼻側的上/下眼瞼, p4=另一眼角
RIGHT_EYE_IDX: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX: tuple[int, ...] = (362, 385, 387, 263, 373, 380)


def compute_ear(eye_points: np.ndarray) -> float:
    """計算單眼 EAR。

    Args:
        eye_points: shape (6, 2) 的座標陣列,依 p1~p6 順序(像素座標,
            或至少是「x、y 同尺度」的座標)。

    Returns:
        EAR 值(float)。若眼睛水平寬度退化為 0(特徵點不可信),
        回傳 NaN,呼叫端應跳過該幀,而不是當成「閉眼」。

    Raises:
        ValueError: eye_points 形狀不是 (6, 2)。
    """
    pts = np.asarray(eye_points, dtype=np.float64)
    if pts.shape != (6, 2):
        raise ValueError(f"eye_points 形狀必須是 (6, 2),收到 {pts.shape}")

    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = float(np.linalg.norm(p2 - p6))
    vertical_2 = float(np.linalg.norm(p3 - p5))
    horizontal = float(np.linalg.norm(p1 - p4))

    # 退化保護:眼寬趨近 0 代表特徵點壞掉(臉太側/偵測失敗),
    # 回傳 NaN 讓上層當「無量測」處理;回 0 會被誤判成閉眼而誤觸警示。
    if horizontal < 1e-9:
        return float("nan")

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def eye_points_from_landmarks(
    landmarks: Sequence,
    indices: Sequence[int],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """從 MediaPipe 特徵點列表取出單眼 6 點,並換算回像素座標。

    Args:
        landmarks: MediaPipe 的 landmark 列表(每個元素有 .x/.y 正規化座標)。
            這裡用 duck typing,單元測試可傳假物件,不需裝 MediaPipe。
        indices: RIGHT_EYE_IDX 或 LEFT_EYE_IDX。
        image_width: 影像寬(像素)。
        image_height: 影像高(像素)。

    Returns:
        shape (6, 2) 的像素座標陣列,可直接丟給 compute_ear()。
    """
    return np.array(
        [(landmarks[i].x * image_width, landmarks[i].y * image_height) for i in indices],
        dtype=np.float64,
    )


def average_ear(left_ear: float, right_ear: float) -> float:
    """雙眼 EAR 取平均;單眼為 NaN 時用另一眼,雙眼都 NaN 才回 NaN。

    這樣偶發的單眼特徵點失效(側臉、反光)不會讓整幀量測作廢。
    """
    values = [v for v in (left_ear, right_ear) if not math.isnan(v)]
    if not values:
        return float("nan")
    return sum(values) / len(values)
