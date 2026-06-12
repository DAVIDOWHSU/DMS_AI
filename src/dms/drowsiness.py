"""
drowsiness.py — 連續閉眼的疲勞判定狀態機。

純邏輯模組:輸入 (EAR, 時間戳),輸出狀態。不碰攝影機、不碰 MediaPipe,
可用合成序列做單元測試。

設計決策:
- 用「連續閉眼**秒數**」而非幀數 —— FPS 會隨裝置變(筆電 ~30、Pi 可能 10~15),
  幀數門檻會跟著漂移;DDAW 法規的反應時間本來就以秒為單位。
- EAR 為 NaN(臉沒抓到/特徵點退化)時**不累積也不重置**:
  短暫掉幀不應清掉已累積的閉眼時間(閉眼挑戰偵測本來就比較容易掉),
  但也不能把「沒看到」當成「閉眼」累積。
- 門檻依 2026-06-12 實測:張眼 EAR ≈ 0.30、閉眼 < 0.02,取經典值 0.2,
  餘裕充足。注意單眼閉(平均 ≈ 0.15)也會低於 0.2 —— 對 DMS 而言,
  開車閉單眼超過 1 秒同樣值得警示,視為可接受行為。
"""

import math
from dataclasses import dataclass
from enum import Enum, auto


class DrowsinessState(Enum):
    """狀態機輸出。"""

    NO_MEASUREMENT = auto()  # 本幀沒有可信的 EAR(NaN)
    EYES_OPEN = auto()       # 眼睛張開
    EYES_CLOSING = auto()    # 閉眼中,但還沒超過門檻(可能只是眨眼)
    DROWSY = auto()          # 連續閉眼 ≥ drowsy_seconds → 疲勞,該警示了


@dataclass(frozen=True)
class DrowsinessConfig:
    """疲勞判定參數(之後由 configs/default.yaml 載入)。

    Attributes:
        ear_threshold: EAR 低於此值視為閉眼。實測張眼 ~0.30 / 閉眼 <0.02。
        drowsy_seconds: 連續閉眼達此秒數判定為疲勞。眨眼約 0.1~0.4s,
            預設 1.0s 可區分眨眼與打瞌睡。
    """

    ear_threshold: float = 0.2
    drowsy_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.ear_threshold < 1.0:
            raise ValueError(f"ear_threshold 必須在 (0, 1),收到 {self.ear_threshold}")
        if self.drowsy_seconds <= 0.0:
            raise ValueError(f"drowsy_seconds 必須 > 0,收到 {self.drowsy_seconds}")


class DrowsinessDetector:
    """連續閉眼時間的狀態機。每幀呼叫 update() 一次。

    用法:
        detector = DrowsinessDetector(DrowsinessConfig())
        state = detector.update(ear=0.05, timestamp=time.monotonic())
        if state is DrowsinessState.DROWSY:
            ...警示...
    """

    def __init__(self, config: DrowsinessConfig | None = None) -> None:
        self.config = config or DrowsinessConfig()
        self._closed_since: float | None = None  # 這輪閉眼的起始時間戳

    def update(self, ear: float, timestamp: float) -> DrowsinessState:
        """餵入一幀的 EAR 與時間戳,回傳當前狀態。

        Args:
            ear: 該幀的(平均)EAR;NaN 表示無可信量測。
            timestamp: 單調遞增的時間戳(秒),例如 time.monotonic()。

        Returns:
            DrowsinessState。
        """
        if math.isnan(ear):
            # 沒量測:不累積、不重置,維持 _closed_since 等臉回來再說
            return DrowsinessState.NO_MEASUREMENT

        if ear >= self.config.ear_threshold:
            self._closed_since = None
            return DrowsinessState.EYES_OPEN

        # ear < threshold:閉眼中
        if self._closed_since is None:
            self._closed_since = timestamp

        closed_duration = max(0.0, timestamp - self._closed_since)
        if closed_duration >= self.config.drowsy_seconds:
            return DrowsinessState.DROWSY
        return DrowsinessState.EYES_CLOSING

    def closed_duration(self, now: float) -> float:
        """目前這輪連續閉眼已持續幾秒;眼睛張開時為 0。"""
        if self._closed_since is None:
            return 0.0
        return max(0.0, now - self._closed_since)

    def reset(self) -> None:
        """清掉累積狀態(例如換人、重新開始偵測)。"""
        self._closed_since = None
