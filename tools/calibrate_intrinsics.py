"""
Calibrate intrinsics camera RGB Kinect V2 bằng bàn cờ (chessboard).

    # Từ thư mục ảnh đã chụp sẵn (≥10-15 ảnh bàn cờ nhiều góc):
    python tools/calibrate_intrinsics.py --source dir --dir calib_imgs ^
        --rows 6 --cols 9 --square-mm 25
    # Hoặc chụp trực tiếp từ webcam / Kinect:
    python tools/calibrate_intrinsics.py --source webcam --rows 6 --cols 9 ^
        --square-mm 25 --min-frames 15

`--rows`/`--cols` = số GÓC TRONG (inner corners), không phải số ô. Ghi
intrinsics.json (schema khớp core.intrinsics) → config.py tự override
COLOR_FX/FY/CX/CY ở lần chạy sau. Exit 0 nếu OK, 1 nếu lỗi.

Chạy trên rig có Kinect/webcam — không test ở CI (cần bàn cờ thật).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from core.intrinsics import write_intrinsics  # noqa: E402

OK = "✓"
BAD = "✗"
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _object_points(rows: int, cols: int, square_mm: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(square_mm)


def _find(gray: np.ndarray, pattern: tuple[int, int]):
    ok, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not ok:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)


def _iter_source(args) -> "list[np.ndarray]":
    """Trả list frame BGR theo --source (dir | webcam | kinect)."""
    if args.source == "dir":
        d = Path(args.dir)
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in _IMG_EXTS) if d.is_dir() else []
        return [img for p in files
                if (img := cv2.imread(str(p))) is not None]

    if args.source == "webcam":
        cap = cv2.VideoCapture(0)
        frames = []
        try:
            for _ in range(args.min_frames * 4):
                ok, f = cap.read()
                if ok and f is not None:
                    frames.append(f)
                time.sleep(0.3)
                if len(frames) >= args.min_frames * 4:
                    break
        finally:
            cap.release()
        return frames

    # kinect
    try:
        from pykinect2 import PyKinectRuntime, PyKinectV2
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} pykinect2 không khả dụng: {exc}")
        return []
    kin = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    frames, t0 = [], time.time()
    while len(frames) < args.min_frames * 4 and time.time() - t0 < 60:
        if kin.has_new_color_frame():
            buf = kin.get_last_color_frame()
            frames.append(buf.reshape((1080, 1920, 4))[:, :, :3].copy())
        time.sleep(0.25)
    kin.close()
    return frames


def calibrate(args) -> int:
    pattern = (args.cols, args.rows)
    objp = _object_points(args.rows, args.cols, args.square_mm)

    frames = _iter_source(args)
    if not frames:
        print(f"{BAD} Không có frame nào (source={args.source}).")
        return 1

    objpoints, imgpoints, size = [], [], None
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        corners = _find(gray, pattern)
        if corners is None:
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        size = gray.shape[::-1]  # (w, h)

    n = len(objpoints)
    print(f"{OK} Tìm thấy bàn cờ trong {n}/{len(frames)} frame.")
    if n < args.min_frames:
        print(f"{BAD} Cần ≥ {args.min_frames} view hợp lệ (có {n}). "
              f"Chụp thêm nhiều góc/khoảng cách.")
        return 1

    rms, mtx, _dist, _r, _t = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None
    )
    fx, fy = float(mtx[0, 0]), float(mtx[1, 1])
    cx, cy = float(mtx[0, 2]), float(mtx[1, 2])
    print(f"{OK} RMS reprojection error: {rms:.3f} px "
          f"({'tốt' if rms < 1.0 else 'cao — chụp lại kỹ hơn'})")
    print(f"{OK} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    write_intrinsics(args.out, fx=fx, fy=fy, cx=cx, cy=cy,
                     image_w=size[0], image_h=size[1],
                     rms=float(rms), n_views=n)
    print(f"{OK} Đã ghi {args.out} — config.py sẽ tự override lần chạy sau.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate intrinsics RGB.")
    ap.add_argument("--source", choices=("dir", "webcam", "kinect"),
                    default="dir")
    ap.add_argument("--dir", default="calib_imgs",
                    help="Thư mục ảnh (khi --source dir)")
    ap.add_argument("--rows", type=int, required=True,
                    help="Số GÓC TRONG theo chiều dọc")
    ap.add_argument("--cols", type=int, required=True,
                    help="Số GÓC TRONG theo chiều ngang")
    ap.add_argument("--square-mm", type=float, required=True,
                    help="Cạnh ô vuông bàn cờ (mm)")
    ap.add_argument("--min-frames", type=int, default=12,
                    help="Số view hợp lệ tối thiểu")
    ap.add_argument("--out", default="intrinsics.json")
    return calibrate(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
