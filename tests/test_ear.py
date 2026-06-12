"""
test_ear.py — EAR 純函式的單元測試。

全部用合成座標,不需要攝影機、不需要 MediaPipe。
跑法(專案根目錄):  python -m pytest
"""

import math

import numpy as np
import pytest

from dms.ear import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    average_ear,
    compute_ear,
    eye_points_from_landmarks,
)

# ---------------------------------------------------------------------------
# 合成眼睛:手算期望值
#
#   p1=(0,0)  p4=(4,0)        → 水平寬 4
#   p2=(1,1)  p6=(1,-1)       → 垂直距 2
#   p3=(3,1)  p5=(3,-1)       → 垂直距 2
#
#   EAR = (2 + 2) / (2 * 4) = 0.5
# ---------------------------------------------------------------------------
OPEN_EYE = np.array(
    [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)],
    dtype=np.float64,
)
OPEN_EYE_EAR = 0.5


class TestComputeEar:
    def test_open_eye_matches_hand_computed_value(self):
        assert compute_ear(OPEN_EYE) == pytest.approx(OPEN_EYE_EAR)

    def test_closed_eye_is_zero(self):
        # 上下眼瞼重合(垂直距離 0)→ EAR 必須是 0
        closed = np.array(
            [(0, 0), (1, 0), (3, 0), (4, 0), (3, 0), (1, 0)],
            dtype=np.float64,
        )
        assert compute_ear(closed) == pytest.approx(0.0)

    def test_scale_invariance(self):
        # 等比放大 7.3 倍(臉離鏡頭遠近)不該改變 EAR
        assert compute_ear(OPEN_EYE * 7.3) == pytest.approx(OPEN_EYE_EAR)

    def test_translation_invariance(self):
        assert compute_ear(OPEN_EYE + np.array([123.4, -56.7])) == pytest.approx(
            OPEN_EYE_EAR
        )

    def test_rotation_invariance(self):
        # 頭歪 30 度不該改變 EAR(距離在旋轉下不變)
        theta = math.radians(30)
        rot = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )
        rotated = OPEN_EYE @ rot.T
        assert compute_ear(rotated) == pytest.approx(OPEN_EYE_EAR)

    def test_degenerate_eye_width_returns_nan(self):
        # p1 == p4(眼寬 0):特徵點壞掉,要回 NaN 而不是 0(0 會被誤判閉眼)
        degenerate = np.zeros((6, 2))
        assert math.isnan(compute_ear(degenerate))

    @pytest.mark.parametrize("bad_shape", [(5, 2), (6, 3), (12,), (2, 6)])
    def test_wrong_shape_raises(self, bad_shape):
        with pytest.raises(ValueError):
            compute_ear(np.zeros(bad_shape))


class TestEyeIndices:
    def test_each_eye_has_six_unique_indices(self):
        for idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            assert len(idx) == 6
            assert len(set(idx)) == 6

    def test_eyes_do_not_share_indices(self):
        assert not set(LEFT_EYE_IDX) & set(RIGHT_EYE_IDX)

    def test_indices_within_facemesh_range(self):
        # MediaPipe FaceMesh 基本特徵點為 0~467
        for i in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            assert 0 <= i < 468


class _FakeLandmark:
    """模擬 MediaPipe landmark:只要有 .x/.y 就行(duck typing)。"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class TestEyePointsFromLandmarks:
    def test_scales_normalized_coords_to_pixels(self):
        # 468 個假 landmark,全部放在 (0.5, 0.25)
        landmarks = [_FakeLandmark(0.5, 0.25)] * 468
        pts = eye_points_from_landmarks(landmarks, RIGHT_EYE_IDX, 640, 480)

        assert pts.shape == (6, 2)
        # x: 0.5 * 640 = 320,y: 0.25 * 480 = 120 → 長寬比已正確還原
        np.testing.assert_allclose(pts, np.tile([320.0, 120.0], (6, 1)))

    def test_picks_the_requested_indices(self):
        # 每個 landmark 的 x 編碼自己的 index,驗證取點順序 = indices 順序
        landmarks = [_FakeLandmark(i / 1000.0, 0.0) for i in range(468)]
        pts = eye_points_from_landmarks(landmarks, LEFT_EYE_IDX, 1000, 1000)
        np.testing.assert_allclose(pts[:, 0], LEFT_EYE_IDX)


class TestAverageEar:
    def test_average_of_two_values(self):
        assert average_ear(0.2, 0.4) == pytest.approx(0.3)

    def test_one_eye_nan_uses_the_other(self):
        assert average_ear(float("nan"), 0.4) == pytest.approx(0.4)
        assert average_ear(0.2, float("nan")) == pytest.approx(0.2)

    def test_both_nan_returns_nan(self):
        assert math.isnan(average_ear(float("nan"), float("nan")))
