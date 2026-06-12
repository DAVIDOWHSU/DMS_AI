"""
inspect_model.py — 拆開 MediaPipe .task 模型檔,看內部組成。

.task 檔本質是 zip 容器,內含多個 TFLite 模型 + metadata。
要評估 INT8 量化的可行性,第一步是知道裡面到底有哪些模型、各佔多大
(量化目標是其中最大的那顆,不是整個 .task)。

用法:
    python scripts/inspect_model.py [path/to/model.task]
"""

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"


def inspect_task_file(path: Path) -> None:
    """列出 .task 容器內容與大小佔比。"""
    total = path.stat().st_size
    print(f"檔案: {path}")
    print(f"總大小: {total / 1e6:.2f} MB")
    print("-" * 64)

    if not zipfile.is_zipfile(path):
        print("不是 zip 容器(無法拆解);可能是單一 TFLite flatbuffer。")
        return

    with zipfile.ZipFile(path) as zf:
        entries = sorted(zf.infolist(), key=lambda i: i.file_size, reverse=True)
        print(f"{'內容':<44} {'大小':>10} {'佔比':>6}")
        for info in entries:
            pct = info.file_size / total * 100
            print(f"{info.filename:<44} {info.file_size / 1e6:>8.2f}MB {pct:>5.1f}%")

    print("-" * 64)
    print("量化評估:佔比最大的 .tflite 才是量化主要目標;")
    print("metadata 與小模型量化收益有限。")


if __name__ == "__main__":
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    if not model.exists():
        print(f"[ERROR] 找不到 {model};先跑過一次 run_dms.py 讓它自動下載。")
        sys.exit(1)
    inspect_task_file(model)
