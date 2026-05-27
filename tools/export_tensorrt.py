"""
Export model .pt → TensorRT .engine để giảm latency inference.

    python tools/export_tensorrt.py --model yolo26s.pt --imgsz 960

GTX 1060 (Pascal) KHÔNG có tensor core → FP16 vô dụng (GP106 tỉ lệ FP16
1:64). Luôn export FP32 (--half=False); TensorRT vẫn cắt latency ~20-35%
nhờ layer fusion. Engine KHOÁ imgsz → phải export đúng INFERENCE_IMG_SIZE.

Sau khi build xong: trỏ config.YOLO_MODEL_PATH = "<model>.engine"
(YOLODetector.load đã YOLO(path) — ultralytics nạp .engine trong suốt,
KHÔNG cần sửa detector).

LƯU Ý: engine phụ thuộc GPU/driver/TensorRT version — phải rebuild trên
mỗi máy / khi đổi imgsz/driver. Build chạy ĐƯỢC chỉ trên rig có GPU +
TensorRT (không test/build ở env CI). Exit 0 nếu OK, 1 nếu lỗi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK = "✓"
BAD = "✗"


def export_engine(
    model_path: str,
    imgsz: int,
    half: bool,
    device,
    yolo_cls: Optional[type] = None,
) -> str:
    """
    Export .pt → .engine. Trả đường dẫn .engine.

    Args:
        model_path: file .pt nguồn.
        imgsz: kích thước input — PHẢI khớp config.INFERENCE_IMG_SIZE.
        half: FP16 (False cho Pascal/GTX 1060).
        device: GPU index (0) hoặc 'cpu' (TensorRT cần GPU).
        yolo_cls: inject để test; None = lazy import ultralytics.YOLO.

    Raises:
        FileNotFoundError: model không tồn tại.
        ValueError: file < 1MB (corrupt).
        Exception: lỗi export (TensorRT thiếu, OOM, ...) — propagate.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model không tồn tại: {model_path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < 1.0:
        raise ValueError(f"Model quá nhỏ ({size_mb:.2f} MB) — có thể corrupt")

    if yolo_cls is None:
        from ultralytics import YOLO  # type: ignore

        yolo_cls = YOLO

    model = yolo_cls(str(path))
    out = model.export(
        format="engine", imgsz=imgsz, half=half, device=device
    )
    return str(out)


def main() -> int:
    from config import INFERENCE_IMG_SIZE, YOLO_MODEL_PATH

    ap = argparse.ArgumentParser(description="Export .pt → TensorRT .engine")
    ap.add_argument("--model", default=YOLO_MODEL_PATH,
                    help="File .pt (mặc định: config.YOLO_MODEL_PATH)")
    ap.add_argument("--imgsz", type=int, default=INFERENCE_IMG_SIZE,
                    help="PHẢI khớp config.INFERENCE_IMG_SIZE")
    ap.add_argument("--half", action="store_true",
                    help="FP16 (KHÔNG dùng trên GTX 1060/Pascal)")
    ap.add_argument("--device", default=0,
                    help="GPU index (mặc định 0)")
    args = ap.parse_args()

    if args.imgsz != INFERENCE_IMG_SIZE:
        print(f"{BAD} --imgsz {args.imgsz} ≠ config.INFERENCE_IMG_SIZE "
              f"{INFERENCE_IMG_SIZE} → engine sẽ lệch runtime. Hủy.")
        return 1

    print(f"Export {args.model} → engine (imgsz={args.imgsz}, "
          f"half={args.half})...")
    try:
        out = export_engine(args.model, args.imgsz, args.half, args.device)
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} Export thất bại: {exc}")
        return 1

    print(f"{OK} Engine: {out}")
    print(f"{OK} Đặt config.YOLO_MODEL_PATH = \"{Path(out).name}\" để dùng.")
    print("  (Rebuild khi đổi imgsz / driver / máy.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
