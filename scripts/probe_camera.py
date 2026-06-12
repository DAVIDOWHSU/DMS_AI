"""probe_camera.py — 探測攝影機支援的解析度(嘗試設定常見值,讀回實際生效值)。"""
import argparse

import cv2

COMMON = [
    (640, 480), (800, 600), (1280, 720), (1280, 960),
    (1920, 1080), (2560, 1440), (3840, 2160),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟攝影機 {args.source}")
        return

    print(f"攝影機 {args.source} 支援的解析度(要求 -> 實際生效):")
    print("-" * 48)
    seen = set()
    for w, h in COMMON:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mark = "" if (aw, ah) in seen else "  <- 新"
        seen.add((aw, ah))
        print(f"  {w}x{h:<5} -> {aw}x{ah}{mark}")

    print("-" * 48)
    print(f"實際支援的不同解析度: {sorted(seen)}")
    cap.release()


if __name__ == "__main__":
    main()
