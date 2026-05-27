"""
live_onnx_kinect — đo ONNXDetector (chữ nhật 960×544, DML) trên FRAME KINECT THẬT.

Khác smoke_onnx_detector (frame trơn): script này lấy color frame thật từ Kinect
V2, resize về INFER_W×INFER_H như pipeline, chạy ONNX nhiều frame và báo
latency thật + số detection trên cảnh thực. KHÔNG mở UI, KHÔNG đụng user_state.json.

Dùng:
    .\.venv310\Scripts\python.exe tools\live_onnx_kinect.py
    .\.venv310\Scripts\python.exe tools\live_onnx_kinect.py --frames 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from models.onnx_detector import ONNXDetector


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Đo ONNX chữ nhật trên frame Kinect thật")
    ap.add_argument("--model", default=config.ONNX_MODEL_PATH)
    ap.add_argument("--frames", type=int, default=100, help="Số frame đo")
    ap.add_argument("--conf", type=float, default=config.INFERENCE_CONF)
    args = ap.parse_args(argv)

    from pykinect2 import PyKinectV2, PyKinectRuntime

    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    print("Kinect runtime mở — chờ frame màu...")

    det = ONNXDetector(model_path=args.model, conf=args.conf, execution_provider="auto")
    det.load()
    print(f"backend: {det._active_provider} | input shape: {det._in_shape}")

    lat_ms: list[float] = []
    det_counts: list[int] = []
    captured = 0
    t_deadline = time.time() + 30  # tối đa 30s chờ đủ frame

    while captured < args.frames and time.time() < t_deadline:
        if not kinect.has_new_color_frame():
            time.sleep(0.005)
            continue
        raw = kinect.get_last_color_frame()  # BGRA phẳng 1920×1080×4
        color = raw.reshape((config.COLOR_H, config.COLOR_W, 4))[:, :, :3]
        frame = cv2.resize(color, (config.INFER_W, config.INFER_H))

        t0 = time.perf_counter()
        dets = det.predict(frame)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        det_counts.append(len(dets))
        captured += 1

        if captured <= 3 or captured % 25 == 0:
            names = ", ".join(sorted({d.class_name for d in dets})) or "—"
            print(f"  frame {captured:3d}: {lat_ms[-1]:5.1f}ms  dets={len(dets)}  [{names}]")

    if not lat_ms:
        print("KHÔNG lấy được frame Kinect (chưa cắm thiết bị?)")
        return 1

    arr = np.asarray(lat_ms[3:] or lat_ms)  # bỏ 3 frame warmup
    print(
        f"\nĐo {len(arr)} frame (bỏ 3 warmup):"
        f"\n  latency: trung bình {arr.mean():.1f}ms | p50 {np.median(arr):.1f}ms"
        f" | p95 {np.percentile(arr, 95):.1f}ms"
        f"\n  fps   : {1000.0 / arr.mean():.1f}"
        f"\n  dets/frame trung bình: {np.mean(det_counts):.2f}"
    )
    budget = 1000.0 / config.TARGET_FPS
    print(f"  ngân sách {budget:.1f}ms/frame ({config.TARGET_FPS}fps): "
          f"{'ĐẠT ✓' if arr.mean() <= budget else 'VƯỢT ✗'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
