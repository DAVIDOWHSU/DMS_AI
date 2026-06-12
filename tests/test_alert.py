"""
test_alert.py — 警示模組的單元測試。

聲音用假 player(不需要喇叭),畫面用黑底 numpy 陣列(不需要 GUI)。
"""

import numpy as np
import pytest

from dms.alert import SoundAlert, draw_alert, make_beep
from dms.drowsiness import DrowsinessState


class TestMakeBeep:
    def test_waveform_shape_and_dtype(self):
        wave = make_beep(frequency_hz=880, duration_s=0.4, sample_rate=44100)
        assert wave.dtype == np.float32
        assert wave.shape == (int(0.4 * 44100),)

    def test_volume_bounds(self):
        wave = make_beep(volume=0.5)
        assert np.max(np.abs(wave)) <= 0.5 + 1e-6
        assert np.max(np.abs(wave)) > 0.4  # 確實有訊號,不是一片靜音

    def test_fade_in_out_kills_clicks(self):
        wave = make_beep()
        assert wave[0] == pytest.approx(0.0, abs=1e-6)
        assert wave[-1] == pytest.approx(0.0, abs=1e-6)


class _FakePlayer:
    """記錄被呼叫幾次的假播放器。"""

    def __init__(self):
        self.calls: list[tuple[np.ndarray, int]] = []

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> None:
        self.calls.append((waveform, sample_rate))


class TestSoundAlertCooldown:
    def test_first_trigger_plays(self):
        player = _FakePlayer()
        alert = SoundAlert(player=player, cooldown_s=1.5)
        assert alert.trigger(now=0.0) is True
        assert len(player.calls) == 1

    def test_trigger_within_cooldown_is_skipped(self):
        player = _FakePlayer()
        alert = SoundAlert(player=player, cooldown_s=1.5)
        alert.trigger(now=0.0)
        assert alert.trigger(now=0.5) is False
        assert alert.trigger(now=1.49) is False
        assert len(player.calls) == 1

    def test_trigger_after_cooldown_plays_again(self):
        player = _FakePlayer()
        alert = SoundAlert(player=player, cooldown_s=1.5)
        alert.trigger(now=0.0)
        assert alert.trigger(now=1.5) is True
        assert len(player.calls) == 2

    def test_player_receives_sample_rate(self):
        player = _FakePlayer()
        alert = SoundAlert(player=player, sample_rate=22050)
        alert.trigger(now=0.0)
        assert player.calls[0][1] == 22050

    def test_invalid_cooldown_raises(self):
        with pytest.raises(ValueError):
            SoundAlert(player=_FakePlayer(), cooldown_s=0.0)


class TestDrawAlert:
    @staticmethod
    def black_frame(h: int = 480, w: int = 640) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_drowsy_draws_red_border(self):
        frame = self.black_frame()
        draw_alert(frame, DrowsinessState.DROWSY, ear=0.02, closed_duration=1.2)
        # 邊框角落應為紅色(BGR = 0,0,255)
        assert tuple(frame[0, 0]) == (0, 0, 255)
        assert tuple(frame[-1, -1]) == (0, 0, 255)

    def test_non_drowsy_has_no_border(self):
        for state in (
            DrowsinessState.EYES_OPEN,
            DrowsinessState.EYES_CLOSING,
            DrowsinessState.NO_MEASUREMENT,
        ):
            frame = self.black_frame()
            draw_alert(frame, state, ear=0.3, closed_duration=0.0)
            assert tuple(frame[0, 0]) == (0, 0, 0), state
            assert tuple(frame[-1, -1]) == (0, 0, 0), state

    def test_status_text_is_drawn(self):
        # 狀態列文字會讓左上區域出現非零像素
        frame = self.black_frame()
        draw_alert(frame, DrowsinessState.EYES_OPEN, ear=0.3, closed_duration=0.0)
        assert frame[:50, :400].sum() > 0

    def test_nan_ear_does_not_crash(self):
        frame = self.black_frame()
        draw_alert(
            frame, DrowsinessState.NO_MEASUREMENT, ear=float("nan"), closed_duration=0.0
        )
        assert frame[:50, :400].sum() > 0
