"""一次性驗證:ear.py 的眼睛 index 是否屬於 MediaPipe 官方眼部點集。

MediaPipe 0.10.3x 已移除舊版 mp.solutions,
改用 Tasks API 的 FaceLandmarksConnections 作為官方出處。
"""
import sys

sys.path.insert(0, "src")

from mediapipe.tasks.python.vision import FaceLandmarksConnections

from dms.ear import LEFT_EYE_IDX, RIGHT_EYE_IDX

def to_index_set(connections) -> set[int]:
    return {i for c in connections for i in (c.start, c.end)}

official_left = to_index_set(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
official_right = to_index_set(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)

print("official LEFT_EYE  points:", sorted(official_left))
print("official RIGHT_EYE points:", sorted(official_right))
print("LEFT_EYE_IDX  subset of official left :", set(LEFT_EYE_IDX) <= official_left)
print("RIGHT_EYE_IDX subset of official right:", set(RIGHT_EYE_IDX) <= official_right)
