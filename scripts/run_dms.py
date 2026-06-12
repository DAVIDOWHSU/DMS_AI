"""
run_dms.py — DMS 完整即時 pipeline(支援多人臉)。

    攝影機/影片 → FaceLandmarker(num_faces 張臉)→ 質心追蹤(穩定 id)
    → 每人專屬 EAR + 疲勞狀態機 → 按臉畫框/狀態 + 聲音警示

多人規則:
    - 每張臉獨立累積閉眼時間(tracker 保證狀態跟「人」走,不跟偵測順序走)。
    - 「駕駛」= 當幀臉框面積最大者(離鏡頭最近)。只有駕駛 DROWSY
      才觸發全屏紅框 + 嗶聲;其他人疲勞只在他的臉框顯示紅色狀態。

用法:
    python scripts/run_dms.py                          # 攝影機 0 + configs/default.yaml
    python scripts/run_dms.py --source 1               # 攝影機 1
    python scripts/run_dms.py --source data/samples/test.mp4
    python scripts/run_dms.py --config configs/my.yaml

按 q 離開。閉眼超過 drowsy_seconds(預設 1 秒)會觸發警示。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dms.alert import SoundAlert, draw_drowsy_banner, draw_face_status  # noqa: E402
from dms.drowsiness import (  # noqa: E402
    DrowsinessConfig,
    DrowsinessDetector,
    DrowsinessState,
)
from dms.ear import (  # noqa: E402
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    average_ear,
    compute_ear,
    eye_points_from_landmarks,
)
from dms.face import FaceLandmarkerVideo  # noqa: E402
from dms.tracking import CentroidTracker, face_bbox, face_centroid  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path) -> dict:
    """讀 YAML 設定;檔案不存在就回空 dict(全部用程式內預設值)。"""
    if not path.exists():
        print(f"[WARN] 找不到設定檔 {path},使用預設參數。")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_source(value: str):
    """純數字字串 -> 攝影機 index (int);其餘 -> 當成檔案路徑 (str)。"""
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="DMS realtime pipeline")
    parser.add_argument(
        "--source",
        default="0",
        help="攝影機 index(0/1/...)或影片檔路徑。預設 0",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML 設定檔路徑。預設 {DEFAULT_CONFIG}",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    drowsy_cfg = cfg.get("drowsiness", {})
    alert_cfg = cfg.get("alert", {})
    camera_cfg = cfg.get("camera", {})
    face_cfg = cfg.get("face", {})
    tracking_cfg = cfg.get("tracking", {})

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.get("width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.get("height", 480))

    if not cap.isOpened():
        print(f"[ERROR] 無法開啟來源:{source}")
        print("  - 確認沒有其他程式佔用鏡頭,或用 --source 指定影片檔路徑。")
        return

    drowsiness_config = DrowsinessConfig(
        ear_threshold=drowsy_cfg.get("ear_threshold", 0.2),
        drowsy_seconds=drowsy_cfg.get("drowsy_seconds", 1.0),
    )
    # 每個 track(= 每個人)一台專屬狀態機,閉眼計時互不干擾
    tracker = CentroidTracker(
        max_match_distance=tracking_cfg.get("max_match_distance", 0.15),
        max_missed_frames=tracking_cfg.get("max_missed_frames", 15),
        data_factory=lambda: DrowsinessDetector(drowsiness_config),
    )
    sound = SoundAlert(
        cooldown_s=alert_cfg.get("beep_cooldown_s", 1.5),
        frequency_hz=alert_cfg.get("beep_frequency_hz", 880.0),
        duration_s=alert_cfg.get("beep_duration_s", 0.4),
        volume=alert_cfg.get("beep_volume", 0.5),
    )
    eye_idx = set(LEFT_EYE_IDX) | set(RIGHT_EYE_IDX)

    num_faces = face_cfg.get("num_faces", 2)
    print(f"[INFO] DMS pipeline 啟動(最多 {num_faces} 張臉),按 q 離開。")
    with FaceLandmarkerVideo(num_faces=num_faces) as face:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] 沒有更多畫面(影片結束或鏡頭中斷)。")
                break

            h, w = frame.shape[:2]
            now = time.monotonic()

            faces = face.detect_faces(frame)
            tracks = tracker.update([face_centroid(lms) for lms in faces])

            # 駕駛 = 當幀臉框面積最大者(離鏡頭最近)
            bboxes = [face_bbox(lms, w, h) for lms in faces]
            driver_i = -1
            if bboxes:
                driver_i = max(
                    range(len(bboxes)),
                    key=lambda i: (bboxes[i][2] - bboxes[i][0])
                    * (bboxes[i][3] - bboxes[i][1]),
                )

            for i, (landmarks, track) in enumerate(zip(faces, tracks)):
                left = compute_ear(
                    eye_points_from_landmarks(landmarks, LEFT_EYE_IDX, w, h)
                )
                right = compute_ear(
                    eye_points_from_landmarks(landmarks, RIGHT_EYE_IDX, w, h)
                )
                ear = average_ear(left, right)

                # 眼點畫黃點,方便目視確認特徵點品質
                for idx in eye_idx:
                    lm = landmarks[idx]
                    cv2.circle(
                        frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 255), -1
                    )

                detector: DrowsinessDetector = track.data
                state = detector.update(ear, now)
                is_driver = i == driver_i
                draw_face_status(
                    frame,
                    state,
                    ear,
                    detector.closed_duration(now),
                    bboxes[i],
                    track.track_id,
                    is_driver=is_driver,
                )
                # 全屏紅框 + 嗶聲只留給駕駛;乘客疲勞只在他的臉框顯示
                if is_driver and state is DrowsinessState.DROWSY:
                    draw_drowsy_banner(frame)
                    sound.trigger(now)

            if not faces:
                cv2.putText(
                    frame,
                    "NO FACE",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (180, 180, 180),
                    2,
                )

            cv2.imshow("DMS - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
