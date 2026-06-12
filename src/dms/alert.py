"""
alert.py — 疲勞警示(畫面 + 聲音)。

- 視覺:DROWSY 時在畫面上加紅框 + 大字警告(draw_alert,直接改 frame)。
- 聲音:numpy 合成正弦波嗶聲,用 sounddevice 播放(mediapipe 已附帶,
  跨平台、之後上 Pi 也能用)。SoundAlert 內建冷卻時間,DROWSY 連續多幀
  也不會每幀都嗶。

聲音播放器以「可注入的 callable」設計:單元測試傳假 player 進來,
不需要喇叭就能驗證冷卻邏輯。
"""

from typing import Callable

import cv2
import numpy as np

from dms.drowsiness import DrowsinessState

# player 的介面:接 (波形, 取樣率),不等播放完就返回
PlayerFn = Callable[[np.ndarray, int], None]


def make_beep(
    frequency_hz: float = 880.0,
    duration_s: float = 0.4,
    sample_rate: int = 44100,
    volume: float = 0.5,
) -> np.ndarray:
    """合成一段嗶聲波形(float32,單聲道,值域 ±volume)。

    頭尾各加 5ms 線性淡入/淡出,避免播放時的爆音(click)。
    """
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    wave = (volume * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)

    fade_n = min(int(0.005 * sample_rate), n // 2)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        wave[:fade_n] *= ramp
        wave[-fade_n:] *= ramp[::-1]
    return wave


def _sounddevice_player(waveform: np.ndarray, sample_rate: int) -> None:
    """預設播放器:sounddevice 非阻塞播放(lazy import,測試環境不需要喇叭)。"""
    import sounddevice as sd

    sd.play(waveform, samplerate=sample_rate)


class SoundAlert:
    """帶冷卻時間的聲音警示。

    用法:
        sound = SoundAlert()
        if state is DrowsinessState.DROWSY:
            sound.trigger(time.monotonic())   # 冷卻中會自動略過
    """

    def __init__(
        self,
        player: PlayerFn | None = None,
        cooldown_s: float = 1.5,
        frequency_hz: float = 880.0,
        duration_s: float = 0.4,
        sample_rate: int = 44100,
        volume: float = 0.5,
    ) -> None:
        if cooldown_s <= 0.0:
            raise ValueError(f"cooldown_s 必須 > 0,收到 {cooldown_s}")
        self._player = player or _sounddevice_player
        self._cooldown_s = cooldown_s
        self._sample_rate = sample_rate
        self._waveform = make_beep(frequency_hz, duration_s, sample_rate, volume)
        self._last_played: float | None = None

    def trigger(self, now: float) -> bool:
        """要求播放警示音;冷卻中回 False(不播),否則播放並回 True。"""
        if self._last_played is not None and now - self._last_played < self._cooldown_s:
            return False
        self._player(self._waveform, self._sample_rate)
        self._last_played = now
        return True


def draw_alert(
    frame: np.ndarray,
    state: DrowsinessState,
    ear: float,
    closed_duration: float,
) -> None:
    """把狀態與警示畫到 frame 上(就地修改)。

    - 一律顯示左上角狀態列:狀態名稱 + EAR + 連續閉眼秒數。
    - DROWSY 時:整圈紅色粗框 + 置中大字「DROWSY!」。
    """
    h, w = frame.shape[:2]

    is_drowsy = state is DrowsinessState.DROWSY
    color = {
        DrowsinessState.EYES_OPEN: (0, 200, 0),       # 綠
        DrowsinessState.EYES_CLOSING: (0, 200, 200),  # 黃
        DrowsinessState.DROWSY: (0, 0, 255),          # 紅
        DrowsinessState.NO_MEASUREMENT: (180, 180, 180),  # 灰
    }[state]

    ear_text = f"EAR:{ear:.3f}" if ear == ear else "EAR: ---"  # NaN 檢查
    cv2.putText(
        frame,
        f"{state.name}  {ear_text}  closed:{closed_duration:.2f}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    if is_drowsy:
        thickness = max(8, w // 60)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)
        cv2.putText(
            frame,
            "DROWSY!",
            (w // 2 - 120, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 255),
            4,
        )
