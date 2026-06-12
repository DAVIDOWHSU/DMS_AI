"""
test_drowsiness.py — 疲勞判定狀態機的單元測試。

全部用合成 (EAR, timestamp) 序列,不需要攝影機。
時間軸以 t=0.0 起算,模擬 30 FPS 即每幀 +1/30 秒。
"""

import math

import pytest

from dms.drowsiness import DrowsinessConfig, DrowsinessDetector, DrowsinessState

OPEN = 0.30    # 實測:雙眼張開
CLOSED = 0.02  # 實測:雙眼閉上
NAN = float("nan")


def make_detector(threshold: float = 0.2, drowsy_seconds: float = 1.0) -> DrowsinessDetector:
    return DrowsinessDetector(
        DrowsinessConfig(ear_threshold=threshold, drowsy_seconds=drowsy_seconds)
    )


class TestConfig:
    def test_default_values(self):
        cfg = DrowsinessConfig()
        assert cfg.ear_threshold == pytest.approx(0.2)
        assert cfg.drowsy_seconds == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_threshold_raises(self, bad):
        with pytest.raises(ValueError):
            DrowsinessConfig(ear_threshold=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_invalid_drowsy_seconds_raises(self, bad):
        with pytest.raises(ValueError):
            DrowsinessConfig(drowsy_seconds=bad)


class TestStateMachine:
    def test_open_eyes_stay_open(self):
        det = make_detector()
        for i in range(60):  # 2 秒張眼
            assert det.update(OPEN, i / 30) is DrowsinessState.EYES_OPEN

    def test_blink_does_not_trigger_drowsy(self):
        # 眨眼 0.2 秒(6 幀)→ 只該出現 EYES_CLOSING,絕不能 DROWSY
        det = make_detector()
        t = 0.0
        det.update(OPEN, t)
        states = []
        for i in range(6):
            t = (i + 1) / 30
            states.append(det.update(CLOSED, t))
        assert all(s is DrowsinessState.EYES_CLOSING for s in states)
        assert det.update(OPEN, t + 1 / 30) is DrowsinessState.EYES_OPEN

    def test_sustained_closure_triggers_drowsy_at_threshold(self):
        det = make_detector(drowsy_seconds=1.0)
        assert det.update(CLOSED, 0.0) is DrowsinessState.EYES_CLOSING  # 起算點
        assert det.update(CLOSED, 0.99) is DrowsinessState.EYES_CLOSING
        assert det.update(CLOSED, 1.00) is DrowsinessState.DROWSY  # 恰達門檻
        assert det.update(CLOSED, 5.00) is DrowsinessState.DROWSY  # 持續閉眼維持 DROWSY

    def test_reopen_resets_accumulation(self):
        det = make_detector(drowsy_seconds=1.0)
        det.update(CLOSED, 0.0)
        det.update(CLOSED, 0.9)               # 累積 0.9 秒
        det.update(OPEN, 1.0)                 # 張眼 → 歸零
        assert det.update(CLOSED, 1.1) is DrowsinessState.EYES_CLOSING
        # 重新累積:1.1 起算,到 2.0 只有 0.9 秒,還不到門檻
        assert det.update(CLOSED, 2.0) is DrowsinessState.EYES_CLOSING
        assert det.update(CLOSED, 2.1) is DrowsinessState.DROWSY

    def test_ear_exactly_at_threshold_counts_as_open(self):
        det = make_detector(threshold=0.2)
        assert det.update(0.2, 0.0) is DrowsinessState.EYES_OPEN


class TestNanHandling:
    def test_nan_returns_no_measurement(self):
        det = make_detector()
        assert det.update(NAN, 0.0) is DrowsinessState.NO_MEASUREMENT

    def test_nan_gap_preserves_accumulation(self):
        # 閉眼 → 短暫掉偵測(NaN)→ 還是閉眼:累積時間不該清零
        det = make_detector(drowsy_seconds=1.0)
        det.update(CLOSED, 0.0)
        assert det.update(NAN, 0.5) is DrowsinessState.NO_MEASUREMENT
        # 臉回來且仍閉眼:從 0.0 起算已 1.0 秒 → DROWSY
        assert det.update(CLOSED, 1.0) is DrowsinessState.DROWSY

    def test_nan_then_open_resets(self):
        det = make_detector()
        det.update(CLOSED, 0.0)
        det.update(NAN, 0.5)
        assert det.update(OPEN, 1.0) is DrowsinessState.EYES_OPEN
        assert det.closed_duration(1.0) == pytest.approx(0.0)


class TestClosedDurationAndReset:
    def test_closed_duration_while_open_is_zero(self):
        det = make_detector()
        det.update(OPEN, 0.0)
        assert det.closed_duration(1.0) == pytest.approx(0.0)

    def test_closed_duration_accumulates(self):
        det = make_detector()
        det.update(CLOSED, 2.0)
        assert det.closed_duration(2.75) == pytest.approx(0.75)

    def test_reset_clears_accumulation(self):
        det = make_detector(drowsy_seconds=1.0)
        det.update(CLOSED, 0.0)
        det.update(CLOSED, 0.9)
        det.reset()
        assert det.closed_duration(0.9) == pytest.approx(0.0)
        # reset 後重新起算
        assert det.update(CLOSED, 1.0) is DrowsinessState.EYES_CLOSING

    def test_default_config_when_none(self):
        det = DrowsinessDetector()
        assert det.config.ear_threshold == pytest.approx(0.2)


class TestRealisticScenario:
    def test_thirty_fps_drowsiness_episode(self):
        """模擬 30 FPS 完整情境:清醒 → 眨眼 → 清醒 → 打瞌睡 → 驚醒。"""
        det = make_detector(drowsy_seconds=1.0)
        fps = 30
        timeline = (
            [(OPEN, DrowsinessState.EYES_OPEN)] * fps          # 1 秒清醒
            + [(CLOSED, DrowsinessState.EYES_CLOSING)] * 9     # 0.3 秒眨眼
            + [(OPEN, DrowsinessState.EYES_OPEN)] * fps        # 1 秒清醒
            + [(CLOSED, None)] * (fps * 2)                     # 2 秒閉眼(後段應 DROWSY)
            + [(OPEN, DrowsinessState.EYES_OPEN)] * 3          # 驚醒
        )
        drowsy_seen_at = []
        for frame, (ear, expected) in enumerate(timeline):
            t = frame / fps
            state = det.update(ear, t)
            if expected is not None:
                assert state is expected, f"frame {frame} (t={t:.2f}s): {state}"
            elif state is DrowsinessState.DROWSY:
                drowsy_seen_at.append(t)

        assert drowsy_seen_at, "長閉眼段必須出現 DROWSY"
        # 第一次 DROWSY 應該在閉眼開始後 ~1.0 秒(容忍一幀誤差)
        closure_start = (fps + 9 + fps) / fps
        assert drowsy_seen_at[0] - closure_start == pytest.approx(1.0, abs=1.5 / fps)
