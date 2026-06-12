"""
quantization_exploration.py

探索 MediaPipe 官方提供的量化模型版本。
MediaPipe Face Landmarker 提供多个版本：
- float32（标准，最大精度）
- float16（中等压缩）
- INT8（最高压缩）

本脚本检查官方下载点，对比不同版本。
"""

import urllib.request
from pathlib import Path

# MediaPipe Face Landmarker 官方下载点
BASE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker"

VARIANTS = {
    "float32": f"{BASE_URL}/float32/latest/face_landmarker.task",
    "float16": f"{BASE_URL}/float16/latest/face_landmarker.task",
    # INT8 版本（如果有的话）
    # "int8": f"{BASE_URL}/int8/latest/face_landmarker.task",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "quantization_comparison"


def check_model_url(url: str) -> tuple[bool, int | None]:
    """检查 URL 是否存在，以及文件大小。"""
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        size = int(response.headers.get("Content-Length", 0))
        return True, size
    except Exception as e:
        return False, None


def download_model(url: str, output_path: Path) -> bool:
    """下载模型。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"  已存在: {output_path}")
        return True
    try:
        print(f"  下載中: {url}")
        urllib.request.urlretrieve(url, output_path)
        size_mb = output_path.stat().st_size / 1e6
        print(f"  完成: {output_path.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  失敗: {e}")
        return False


def main():
    print("=" * 70)
    print("MediaPipe Face Landmarker 量化版本探索")
    print("=" * 70)

    print("\n[1] 檢查官方提供的模型版本:")
    print("-" * 70)

    available_variants = {}
    for variant, url in VARIANTS.items():
        exists, size = check_model_url(url)
        status = "[OK]" if exists else "[NG]"
        print(f"  {variant:10s}: {status}", end="")
        if size:
            print(f" ({size / 1e6:.1f} MB)")
        else:
            print()
        if exists:
            available_variants[variant] = url

    if not available_variants:
        print("\n  [WARN] 官方似乎只提供 float32 和 float16")
        print("  INT8 需要用 TensorFlow Lite 工具手動量化")
        return

    print("\n[2] 下載可用版本進行對比:")
    print("-" * 70)

    models = {}
    for variant, url in available_variants.items():
        output_path = MODELS_DIR / f"face_landmarker_{variant}.task"
        if download_model(url, output_path):
            models[variant] = output_path

    if not models:
        print("  下載失敗")
        return

    print("\n[3] 模型大小對比:")
    print("-" * 70)

    sizes = {}
    for variant, path in models.items():
        size_mb = path.stat().st_size / 1e6
        sizes[variant] = size_mb
        print(f"  {variant:10s}: {size_mb:7.2f} MB")

    if len(sizes) > 1:
        base_size = sizes[list(sizes.keys())[0]]
        print("\n  相對縮減:")
        for variant, size in sizes.items():
            ratio = (1 - size / base_size) * 100
            if ratio > 0:
                print(f"    {variant:10s}: {ratio:5.1f}% 更小")

    print("\n[4] 下一步:")
    print("-" * 70)
    print("  [A] 如果官方有 INT8 版本:")
    print("    - 用不同版本的模型跑 live test")
    print("    - 對比 EAR 精度(應該差不多)")
    print("    - 量測 FPS(INT8 應該更快)")
    print()
    print("  [B] 如果官方只有 float32/float16:")
    print("    - 用 TensorFlow Lite Converter 手動量化")
    print("    - 或估算「理論性能提升」(模型小 4 倍 → FPS 快 ~2-3 倍)")
    print()
    print("  參考: https://www.tensorflow.org/lite/performance/post_training_quantization")


if __name__ == "__main__":
    main()
