"""
test_tracking.py — 質心追蹤器的單元測試。

全部用合成的正規化座標,不需要攝影機與 MediaPipe。
核心要驗的是多人臉的痛點:偵測順序洗牌時,id(與掛在 track 上的
專屬物件)必須跟著「人」走,不能跟著「順序」走。
"""

from types import SimpleNamespace

import pytest

from dms.tracking import CentroidTracker, Track, face_bbox, face_centroid

# 兩個相距夠遠的臉位置(車內駕駛/副駕的典型擺位)
A = (0.30, 0.50)
B = (0.70, 0.50)


def shift(c, dx=0.0, dy=0.0):
    return (c[0] + dx, c[1] + dy)


class TestInit:
    @pytest.mark.parametrize("bad", [0.0, -0.1])
    def test_invalid_max_match_distance_raises(self, bad):
        with pytest.raises(ValueError):
            CentroidTracker(max_match_distance=bad)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_invalid_max_missed_frames_raises(self, bad):
        with pytest.raises(ValueError):
            CentroidTracker(max_missed_frames=bad)


class TestMatching:
    def test_single_face_keeps_id_while_moving(self):
        tracker = CentroidTracker()
        t0 = tracker.update([A])[0]
        # 連續小幅移動(每幀 0.01,遠小於 max_match_distance)
        c = A
        for _ in range(30):
            c = shift(c, dx=0.01)
            t = tracker.update([c])[0]
            assert t.track_id == t0.track_id
            assert t.centroid == pytest.approx(c)

    def test_detection_order_shuffle_keeps_ids(self):
        # MediaPipe 不保證順序:這幀 [A, B]、下幀 [B, A],id 必須跟人走
        tracker = CentroidTracker()
        ta, tb = tracker.update([A, B])
        tb2, ta2 = tracker.update([B, A])
        assert ta2.track_id == ta.track_id
        assert tb2.track_id == tb.track_id

    def test_result_order_matches_input_order(self):
        tracker = CentroidTracker()
        tracker.update([A, B])
        result = tracker.update([B, A])
        assert result[0].centroid == pytest.approx(B)
        assert result[1].centroid == pytest.approx(A)

    def test_new_face_gets_new_id(self):
        tracker = CentroidTracker()
        ta = tracker.update([A])[0]
        ta2, tb = tracker.update([A, B])
        assert ta2.track_id == ta.track_id
        assert tb.track_id != ta.track_id

    def test_far_jump_is_a_new_track(self):
        # 移動超過 max_match_distance → 不是同一人
        tracker = CentroidTracker(max_match_distance=0.15)
        ta = tracker.update([A])[0]
        tb = tracker.update([shift(A, dx=0.4)])[0]
        assert tb.track_id != ta.track_id

    def test_greedy_assigns_nearest(self):
        # 兩張臉都動了一點,各自該配回最近的舊 track
        tracker = CentroidTracker()
        ta, tb = tracker.update([A, B])
        ta2, tb2 = tracker.update([shift(A, dx=0.02), shift(B, dx=-0.02)])
        assert ta2.track_id == ta.track_id
        assert tb2.track_id == tb.track_id


class TestDisappearance:
    def test_brief_dropout_keeps_track(self):
        # 短暫掉偵測(< max_missed_frames)後回來,還是同一人
        tracker = CentroidTracker(max_missed_frames=15)
        ta = tracker.update([A])[0]
        for _ in range(14):
            tracker.update([])
        assert tracker.update([A])[0].track_id == ta.track_id

    def test_long_disappearance_deletes_track(self):
        # 連續消失達 max_missed_frames → 刪除,回來時是新 id
        tracker = CentroidTracker(max_missed_frames=15)
        ta = tracker.update([A])[0]
        for _ in range(15):
            tracker.update([])
        assert tracker.tracks == []
        assert tracker.update([A])[0].track_id != ta.track_id

    def test_match_resets_missed_counter(self):
        tracker = CentroidTracker(max_missed_frames=3)
        ta = tracker.update([A])[0]
        for _ in range(5):  # 消失 2 幀、回來 1 幀,循環多次都不該被刪
            tracker.update([])
            tracker.update([])
            assert tracker.update([A])[0].track_id == ta.track_id

    def test_one_face_leaving_does_not_affect_other(self):
        tracker = CentroidTracker(max_missed_frames=3)
        ta, tb = tracker.update([A, B])
        for _ in range(3):  # B 離開到被刪除為止
            tracker.update([A])
        ta2 = tracker.update([A])[0]
        assert ta2.track_id == ta.track_id
        assert [t.track_id for t in tracker.tracks] == [ta.track_id]


class TestDataFactory:
    def test_each_track_gets_own_data(self):
        tracker = CentroidTracker(data_factory=list)
        ta, tb = tracker.update([A, B])
        assert ta.data is not tb.data

    def test_data_follows_person_not_order(self):
        # 把「狀態跟人走」直接驗在 data 上:順序洗牌後 data 物件不變
        tracker = CentroidTracker(data_factory=list)
        ta, tb = tracker.update([A, B])
        ta.data.append("a-history")
        tb2, ta2 = tracker.update([B, A])
        assert ta2.data is ta.data
        assert ta2.data == ["a-history"]
        assert tb2.data is tb.data

    def test_no_factory_data_is_none(self):
        tracker = CentroidTracker()
        assert tracker.update([A])[0].data is None


class TestTrackDataclass:
    def test_defaults(self):
        t = Track(track_id=1, centroid=A)
        assert t.data is None
        assert t.missed_frames == 0


def fake_landmarks(points):
    """(x, y) 列表 → 有 .x/.y 屬性的假 landmark 列表(不需 MediaPipe)。"""
    return [SimpleNamespace(x=x, y=y) for x, y in points]


class TestFaceGeometry:
    def test_centroid_is_mean(self):
        lms = fake_landmarks([(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)])
        assert face_centroid(lms) == pytest.approx((0.4, 0.6))

    def test_centroid_empty_raises(self):
        with pytest.raises(ValueError):
            face_centroid([])

    def test_bbox_pixel_coords(self):
        lms = fake_landmarks([(0.25, 0.10), (0.75, 0.50)])
        assert face_bbox(lms, 1280, 720) == (320, 72, 960, 360)

    def test_bbox_clamped_to_image(self):
        # 特徵點可能略超出畫面(臉貼邊),框要夾回影像範圍
        lms = fake_landmarks([(-0.1, -0.2), (1.2, 1.1)])
        assert face_bbox(lms, 1280, 720) == (0, 0, 1279, 719)

    def test_bbox_empty_raises(self):
        with pytest.raises(ValueError):
            face_bbox([], 1280, 720)
