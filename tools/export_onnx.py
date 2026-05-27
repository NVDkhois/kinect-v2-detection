"""
export_onnx — chuyển model YOLO .pt → .onnx để chạy bằng onnxruntime-directml
trên GPU AMD/Intel (vd RX 580). Chạy OFFLINE một lần cho mỗi model/imgsz.

Vì sao cần: torch không chạy nổi YOLO trên DirectML (op không hỗ trợ), còn
torch-CPU quá chậm. ONNX Runtime + DirectML EP chạy tốt (xem
docs/ke_hoach_chuyen_rx580.md). ONNXDetector load file .onnx này.

LƯU Ý: KHÔNG load .onnx qua ultralytics (nó tự pip-install onnxruntime CPU,
đè onnxruntime-directml). File này chỉ dùng ultralytics để EXPORT.

Dùng:
    .\.venv310\Scripts\python.exe tools\export_onnx.py
    .\.venv310\Scripts\python.exe tools\export_onnx.py --model yolo26s.pt --imgsz 960
    .\.venv310\Scripts\python.exe tools\export_onnx.py --model yolo26s.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_onnx(model_path: str, imgsz: int | list[int], simplify: bool = True) -> str:
    """
    Export .pt → .onnx (FP32, shape cố định — DML EP ổn định hơn dynamic).

    Args:
        imgsz: `int` (vuông) HOẶC `[H, W]` chữ nhật (vd [544, 960]). Chữ nhật khớp
            tỉ lệ camera 16:9 → bỏ pad thừa, nhanh hơn ~36% mà giữ độ phân giải
            (xem docs/ke_hoach_chuyen_rx580.md, Pha 5b). H, W phải chia hết 32.

    Returns:
        Đường dẫn file .onnx sinh ra.

    Raises:
        FileNotFoundError: model .pt không tồn tại (trừ model ultralytics tự
            tải, vd 'yolo26s.pt').
    """
    from ultralytics import YOLO

    out = YOLO(model_path).export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,   # shape cố định batch=1 (vuông imgsz² hoặc chữ nhật H×W)
        simplify=simplify,
        opset=12,        # opset phổ biến, DML EP hỗ trợ tốt
    )
    return str(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export YOLO .pt → .onnx (DirectML)")
    ap.add_argument("--model", default="yolo26s.pt", help="Đường dẫn .pt")
    ap.add_argument(
        "--imgsz", type=int, nargs="+", default=[544, 960],
        help="Input size: 1 số (vuông, vd 640) hoặc 2 số H W (chữ nhật, vd 544 960). "
             "Mặc định 544 960 (tối ưu 16:9 cho RX 580).",
    )
    ap.add_argument("--no-simplify", action="store_true", help="Tắt onnxslim")
    ap.add_argument("--out", default=None, help="Tên file .onnx đích (mặc định tự đặt theo size)")
    args = ap.parse_args(argv)

    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz
    print(f"Export {args.model} @ {imgsz} → .onnx ...")
    try:
        out = export_onnx(args.model, imgsz, simplify=not args.no_simplify)
    except Exception as exc:  # noqa: BLE001
        print(f"LỖI export: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # Đổi tên theo size để khớp ONNX_MODEL_PATH (vd yolo26s_960x544.onnx / yolo26s_640.onnx).
    stem = Path(args.model).stem
    if args.out:
        suffix = args.out
    elif len(args.imgsz) == 2:
        suffix = f"{stem}_{args.imgsz[1]}x{args.imgsz[0]}.onnx"  # _{W}x{H}
    else:
        suffix = f"{stem}_{args.imgsz[0]}.onnx"
    dest = Path(out).with_name(suffix)
    if Path(out).resolve() != dest.resolve():
        Path(out).replace(dest)
        out = str(dest)

    size_mb = Path(out).stat().st_size / (1024 * 1024)
    print(f"OK → {out}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
