"""
tracking.py — 質心追蹤(centroid tracking),給多人臉場景用。

MediaPipe 每幀回傳的臉**沒有穩定順序**:這幀的第 0 張臉,下一幀可能變第 1 張。
若直接拿「偵測順序」對應狀態機,A 的閉眼計時會張冠李戴到 B 身上。
解法:每個 track 代表一個人,帶穩定 id 與上次的臉中心位置;
每幀把新偵測到的臉中心與現有 tracks 做「就近距離」貪婪匹配:

    距離最近且 ≤ max_match_distance → 同一人,更新位置
    沒匹配到的新中心            → 新人,建新 track
    沒匹配到的舊 track           → 記一次消失,連續消失太多幀才刪除
                                   (短暫掉偵測不應立刻忘記這個人)

純幾何模組:不 import MediaPipe、不碰攝影機,可用合成座標做單元測試。
每個 track 可掛一個由 data_factory 產生的專屬物件(DMS 用 DrowsinessDetector;
之後分心偵測等其他「按人累積」的狀態也能複用,所以這裡不寫死型別)。

座標系:臉中心用 MediaPipe 的正規化座標(x/y 都在 0~1),
max_match_distance 也是正規化單位 —— 與解析度無關,換鏡頭不用調。
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

Centroid = tuple[float, float]


def face_centroid(landmarks: Sequence) -> Centroid:
    """臉中心:所有特徵點正規化座標的平均。

    landmarks 用 duck typing(每個元素有 .x/.y),單元測試不需裝 MediaPipe。
    """
    n = len(landmarks)
    if n == 0:
        raise ValueError("landmarks 不可為空")
    return (
        sum(lm.x for lm in landmarks) / n,
        sum(lm.y for lm in landmarks) / n,
    )


def face_bbox(
    landmarks: Sequence, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """臉的外接框 (x1, y1, x2, y2),像素座標,夾在影像範圍內。"""
    if len(landmarks) == 0:
        raise ValueError("landmarks 不可為空")
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = max(0, int(min(xs) * image_width))
    y1 = max(0, int(min(ys) * image_height))
    x2 = min(image_width - 1, int(max(xs) * image_width))
    y2 = min(image_height - 1, int(max(ys) * image_height))
    return (x1, y1, x2, y2)


@dataclass
class Track:
    """一個被追蹤的人。

    Attributes:
        track_id: 穩定的識別編號(從 1 起跳,只增不重用)。
        centroid: 最近一次匹配到的臉中心(正規化座標)。
        data: data_factory 產生的專屬物件(例如 DrowsinessDetector)。
        missed_frames: 連續幾幀沒匹配到(匹配到就歸零)。
    """

    track_id: int
    centroid: Centroid
    data: Any = None
    missed_frames: int = 0


class CentroidTracker:
    """就近距離的貪婪質心匹配。每幀呼叫 update() 一次。

    用法:
        tracker = CentroidTracker(data_factory=DrowsinessDetector)
        tracks = tracker.update([(0.3, 0.5), (0.7, 0.5)])
        # tracks 與輸入同順序:tracks[i] 對應第 i 個中心
        tracks[0].data.update(ear, now)
    """

    def __init__(
        self,
        max_match_distance: float = 0.15,
        max_missed_frames: int = 15,
        data_factory: Callable[[], Any] | None = None,
    ) -> None:
        """
        Args:
            max_match_distance: 兩幀間臉中心移動超過此距離(正規化單位)
                就不視為同一人。0.15 ≈ 畫面寬的 15%,30 FPS 下單幀位移
                遠小於此,而車內兩人臉距通常大於此值。
            max_missed_frames: 連續消失達此幀數才刪除 track
                (容忍短暫掉偵測;30 FPS 下 15 幀 ≈ 0.5 秒)。
            data_factory: 每建一個新 track 呼叫一次,產生該人的專屬物件。
        """
        if max_match_distance <= 0.0:
            raise ValueError(f"max_match_distance 必須 > 0,收到 {max_match_distance}")
        if max_missed_frames < 1:
            raise ValueError(f"max_missed_frames 必須 ≥ 1,收到 {max_missed_frames}")
        self._max_match_distance = max_match_distance
        self._max_missed_frames = max_missed_frames
        self._data_factory = data_factory
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        """目前所有存活的 tracks(含本幀沒匹配到、還在容忍期內的)。"""
        return list(self._tracks)

    def update(self, centroids: Sequence[Centroid]) -> list[Track]:
        """餵入本幀所有臉中心,回傳與輸入**同順序**的 track 列表。

        Args:
            centroids: 本幀偵測到的臉中心列表(正規化座標)。

        Returns:
            list[Track],第 i 個元素就是 centroids[i] 對應的人
            (既有的人就更新位置,新出現的人就建新 track)。
        """
        # 1) 算所有 (舊 track, 新中心) 配對距離,由近到遠貪婪認領
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            for ci, c in enumerate(centroids):
                d = math.dist(track.centroid, c)
                if d <= self._max_match_distance:
                    pairs.append((d, ti, ci))
        pairs.sort()

        matched: dict[int, Track] = {}  # centroid index -> track
        used_tracks: set[int] = set()
        for _, ti, ci in pairs:
            if ti in used_tracks or ci in matched:
                continue
            track = self._tracks[ti]
            track.centroid = tuple(centroids[ci])
            track.missed_frames = 0
            matched[ci] = track
            used_tracks.add(ti)

        # 2) 沒被認領的舊 track:記一次消失,連續太多幀就刪
        survivors: list[Track] = []
        for ti, track in enumerate(self._tracks):
            if ti not in used_tracks:
                track.missed_frames += 1
                if track.missed_frames >= self._max_missed_frames:
                    continue  # 刪除:不放進 survivors
            survivors.append(track)
        self._tracks = survivors

        # 3) 沒匹配到的新中心:新人,建新 track
        result: list[Track] = []
        for ci, c in enumerate(centroids):
            track = matched.get(ci)
            if track is None:
                data = self._data_factory() if self._data_factory else None
                track = Track(track_id=self._next_id, centroid=tuple(c), data=data)
                self._next_id += 1
                self._tracks.append(track)
            result.append(track)
        return result
