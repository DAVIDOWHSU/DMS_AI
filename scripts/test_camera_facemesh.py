"""
test_camera_facemesh.py
冒煙測試 (smoke test):確認「攝影機 + MediaPipe FaceLandmarker」能正常運作。

⚠ Tasks API 版:mediapipe 0.10.3x 已移除舊版 mp.solutions,
   改用 dms.face 的 FaceLandmarkerVideo 封裝(Tasks API)。
   模型檔 models/face_landmarker.task 第一次執行會自動下載(~3.8MB)。

它回答三個問題:
    1. 畫面進得來嗎?(攝影機 / 影片檔)
    2. 臉抓得到嗎?(FaceLandmarker)
    3. EAR 算得出來嗎?(用 src/dms/ear.py 對真實特徵點計算,順便驗證整合)

用法:
    python scripts/test_camera_facemesh.py                # 預設用攝影機 0
    python scripts/test_camera_facemesh.py --source 1     # 用攝影機 1
    python scripts/test_camera_facemesh.py --source data/samples/test.mp4  # 影片檔

按 q 離開。

完整的「疲勞判定 + 警示」請跑 scripts/run_dms.py;這支只看偵測鏈路與 EAR。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# 讓 scripts/ 下的腳本能 import src/dms(不需要先 pip install -e)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dms.ear import (  # noqa: E402
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    average_ear,
    compute_ear,
    eye_points_from_landmarks,
)
from dms.face import FaceLandmarkerVideo  # noqa: E402


def parse_source(value: str):
    """純數字字串 -> 攝影機 index (int);其餘 -> 當成檔案路徑 (str)。"""
    return int(value) if value.isdigit() else value


def draw_landmarks(frame, landmarks, width: int, height: int) -> None:
    """把 478 個特徵點畫成小點;EAR 用到的 12 個眼點放大標黃。"""
    eye_idx = set(LEFT_EYE_IDX) | set(RIGHT_EYE_IDX)
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * width), int(lm.y * height)
        if i in eye_idx:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # 黃:EAR 眼點
        else:
            cv2.circle(frame, (x, y), 1, (0, 200, 0), -1)    # 綠:其餘網格點


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera + MediaPipe FaceLandmarker (Tasks API) smoke test"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="攝影機 index(0/1/...)或影片檔路徑。預設 0",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)

    # 只有實體攝影機才需要設定解析度;影片檔由檔案本身決定
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] 無法開啟來源:{source}")
        print("  - 確認沒有其他程式(Teams / Zoom 等)佔用鏡頭,")
        print("    或用 --source 指定影片檔路徑。")
        return

    print("[INFO] 啟動成功,按 q 離開。")
    prev_t = time.time()
    fps = 0.0

    with FaceLandmarkerVideo() as face:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] 沒有更多畫面(影片結束或鏡頭中斷)。")
                break

            h, w = frame.shape[:2]
            landmarks = face.detect(frame)

            face_found = landmarks is not None
            ear = float("nan")
            if face_found:
                draw_landmarks(frame, landmarks, w, h)
                left = compute_ear(
                    eye_points_from_landmarks(landmarks, LEFT_EYE_IDX, w, h)
                )
                right = compute_ear(
                    eye_points_from_landmarks(landmarks, RIGHT_EYE_IDX, w, h)
                )
                ear = average_ear(left, right)

            # FPS:用指數移動平均讓數字穩定一點,不要每幀亂跳
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps

            status = "FACE" if face_found else "NO FACE"
            color = (0, 200, 0) if face_found else (0, 0, 200)  # BGR
            ear_text = f"EAR:{ear:.3f}" if ear == ear else "EAR: ---"  # NaN 檢查
            cv2.putText(
                frame,
                f"{status}  FPS:{fps:5.1f}  {ear_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            cv2.imshow("DMS smoke test (Tasks API) - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
