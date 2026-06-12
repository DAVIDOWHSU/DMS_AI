"""
benchmark.py — DMS pipeline 性能量測(無 GUI,可在任何裝置跑)。

量測整條鏈路的分階段延遲與 FPS:
    讀幀 → FaceLandmarker 推理 → EAR 計算 → 狀態機

這支腳本就是「before/after」對比的量測工具:
    - 筆電(現在):建立 baseline
    - Pi 5 / Jetson(之後):同一支腳本直接跑,數據可比

用法:
    python scripts/benchmark.py                     # 攝影機 0,300 幀
    python scripts/benchmark.py --source 1
    python scripts/benchmark.py --source data/samples/test.mp4
    python scripts/benchmark.py --synthetic         # 合成幀(無鏡頭環境/CI)
    python scripts/benchmark.py --frames 600 --output docs/benchmarks/laptop.json

注意:--synthetic 用隨機雜訊幀(無臉),只能量「偵測路徑」的推理成本;
正式 baseline 請用真鏡頭對著臉跑,landmark 追蹤路徑才會被量到。
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dms.drowsiness import DrowsinessDetector  # noqa: E402
from dms.ear import (  # noqa: E402
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    average_ear,
    compute_ear,
    eye_points_from_landmarks,
)
from dms.face import FaceLandmarkerVideo  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    """簡單百分位(values 會被排序複本處理)。"""
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stage_stats(name: str, samples_ms: list[float]) -> dict:
    """一個階段的統計摘要(ms)。"""
    return {
        "stage": name,
        "mean_ms": round(statistics.mean(samples_ms), 2),
        "p50_ms": round(percentile(samples_ms, 50), 2),
        "p95_ms": round(percentile(samples_ms, 95), 2),
        "max_ms": round(max(samples_ms), 2),
    }


class SyntheticSource:
    """合成幀來源(隨機雜訊,無臉):無鏡頭環境用來驗證腳本與量推理成本。"""

    def __init__(self, width: int, height: int):
        rng = np.random.default_rng(42)
        # 預生成 10 張幀循環使用,避免把「生成雜訊」算進讀幀時間
        self._frames = [
            rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        self._i = 0

    def read(self):
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return True, frame

    def release(self) -> None:
        pass


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="DMS pipeline benchmark")
    parser.add_argument("--source", default="0", help="攝影機 index 或影片路徑")
    parser.add_argument("--synthetic", action="store_true", help="用合成幀(無鏡頭)")
    parser.add_argument("--frames", type=int, default=300, help="量測幀數")
    parser.add_argument("--warmup", type=int, default=30, help="暖機幀數(不計入)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", default=None, help="結果輸出 JSON 路徑")
    args = parser.parse_args()

    if args.synthetic:
        cap = SyntheticSource(args.width, args.height)
        source_desc = f"synthetic {args.width}x{args.height} (no face)"
    else:
        source = parse_source(args.source)
        cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print(f"[ERROR] 無法開啟來源:{source}")
            return
        source_desc = str(source)

    detector = DrowsinessDetector()
    t_read, t_infer, t_ear, t_total = [], [], [], []
    face_frames = 0

    print(f"[INFO] 來源: {source_desc}")
    print(f"[INFO] 暖機 {args.warmup} 幀 + 量測 {args.frames} 幀 ...")

    with FaceLandmarkerVideo() as face:
        # 暖機:讓 XNNPACK/快取/自動曝光穩定,不計入統計
        for _ in range(args.warmup):
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] 暖機階段就沒幀了(影片太短?)")
                return
            face.detect(frame)

        # 計時用 perf_counter:Windows 上 time.monotonic() 解析度僅 ~15.6ms,
        # 量單幀毫秒級延遲會失真;perf_counter 是高解析度單調時鐘
        wall_start = time.perf_counter()
        for _ in range(args.frames):
            f0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 幀來源提前結束,以實際量到的幀數統計。")
                break
            f1 = time.perf_counter()

            landmarks = face.detect(frame)
            f2 = time.perf_counter()

            ear = float("nan")
            if landmarks is not None:
                face_frames += 1
                h, w = frame.shape[:2]
                left = compute_ear(eye_points_from_landmarks(landmarks, LEFT_EYE_IDX, w, h))
                right = compute_ear(eye_points_from_landmarks(landmarks, RIGHT_EYE_IDX, w, h))
                ear = average_ear(left, right)
            detector.update(ear, f2)
            f3 = time.perf_counter()

            t_read.append((f1 - f0) * 1000)
            t_infer.append((f2 - f1) * 1000)
            t_ear.append((f3 - f2) * 1000)
            t_total.append((f3 - f0) * 1000)
        wall = time.perf_counter() - wall_start

    cap.release()

    if not t_total:
        print("[ERROR] 沒有量到任何幀。")
        return

    n = len(t_total)
    result = {
        "date": date.today().isoformat(),
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "source": source_desc,
        "frames_measured": n,
        "face_detected_frames": face_frames,
        "fps_end_to_end": round(n / wall, 1),
        "stages": [
            stage_stats("read_frame", t_read),
            stage_stats("facelandmarker_inference", t_infer),
            stage_stats("ear_plus_state_machine", t_ear),
            stage_stats("total_per_frame", t_total),
        ],
    }

    print()
    print(f"=== DMS benchmark ({result['date']}) ===")
    print(f"平台: {result['platform']['system']} / {result['platform']['machine']}")
    print(f"幀數: {n}(偵測到臉: {face_frames})")
    print(f"端到端 FPS: {result['fps_end_to_end']}")
    print(f"{'階段':<28} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8}  (ms)")
    for s in result["stages"]:
        print(
            f"{s['stage']:<28} {s['mean_ms']:>8} {s['p50_ms']:>8}"
            f" {s['p95_ms']:>8} {s['max_ms']:>8}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[INFO] 結果已存 {out}")


if __name__ == "__main__":
    main()
