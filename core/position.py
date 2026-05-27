"""
Tính toán toạ độ 3D (X, Y, Z) từ depth frame + bounding box.

Gốc toạ độ: tâm quang học của **camera RGB** (color) Kinect V2.
Đơn vị output: mm. Trục X dương = phải, Y dương = xuống, Z dương = xa camera.

Lưu ý: chỉ depth sensor (IR) đo được khoảng cách, vì vậy Z vẫn được đọc từ
depth frame; nhưng X, Y được chiếu bằng intrinsics của camera RGB và bbox
trong không gian color (1920×1080). Khoảng baseline ~52mm giữa RGB và IR
được coi là không đáng kể ở khoảng cách 0.3–5 m (sai số <2% cho Z, X).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from config import (
    COLOR_CX,
    COLOR_CY,
    COLOR_FX,
    COLOR_FY,
    COLOR_H,
    COLOR_W,
    DEPTH_H,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    DEPTH_SCALE,
    DEPTH_W,
)


log = logging.getLogger("position")


def _median_depth_mm(depth_frame: np.ndarray, cx: int, cy: int, half: int = 2) -> float:
    """
    Lấy median của vùng (2*half+1)×(2*half+1) quanh (cx, cy) trong depth frame.

    Bỏ qua các pixel = 0 (depth hole) trước khi lấy median.

    Args:
        depth_frame: array uint16, đơn vị mm
        cx, cy: toạ độ tâm trong không gian depth
        half: bán kính cửa sổ. half=2 → cửa sổ 5×5.

    Returns:
        Giá trị mm dạng float, hoặc NaN nếu không có pixel hợp lệ.
    """
    h, w = depth_frame.shape[:2]
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)

    if x1 <= x0 or y1 <= y0:
        return float("nan")

    patch = depth_frame[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def _from_camera_space(
    cs_map: np.ndarray, u_c: float, v_c: float,
) -> tuple[float, float, float]:
    """
    Đọc toạ độ 3D từ camera-space map của Kinect ICoordinateMapper.

    `cs_map` shape (H, W, 3) float, đơn vị MÉT, hệ camera-space Kinect
    (X phải, **Y LÊN**, Z xa). Project quy ước Y XUỐNG → ĐẢO DẤU Y.
    Trả NaN nếu điểm không hữu hạn / Z ≤ 0 / ngoài [DEPTH_MIN_M, MAX_M].
    """
    h, w = cs_map.shape[:2]
    cu = int(min(w - 1, max(0, round(u_c))))
    cv = int(min(h - 1, max(0, round(v_c))))
    x_m, y_m, z_m = (float(t) for t in cs_map[cv, cu][:3])

    if not all(math.isfinite(t) for t in (x_m, y_m, z_m)) or z_m <= 0:
        return (float("nan"), float("nan"), float("nan"))
    if z_m < DEPTH_MIN_M or z_m > DEPTH_MAX_M:
        return (float("nan"), float("nan"), float("nan"))
    return (x_m * 1000.0, -y_m * 1000.0, z_m * 1000.0)  # đảo dấu Y


def compute_3d_position(
    depth_frame: np.ndarray,
    bbox_color: tuple[int, int, int, int],
    color_w: int = COLOR_W,
    color_h: int = COLOR_H,
    cs_map: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Tính toán toạ độ 3D của vật thể trong **hệ camera RGB** Kinect V2.

    Gốc toạ độ: tâm quang học camera RGB (cx ≈ 959.5 ; cy ≈ 539.5).
    Đơn vị: mm. Trục X dương = phải, Y dương = xuống, Z dương = xa camera.

    Args:
        depth_frame: Depth frame (DEPTH_H × DEPTH_W), giá trị uint16 tính bằng mm.
        bbox_color: (x1, y1, x2, y2) trong không gian COLOR (1920×1080).
        color_w, color_h: kích thước color frame thực tế (default = config).
        cs_map: (tuỳ chọn) camera-space map từ Kinect ICoordinateMapper,
            shape (color_h, color_w, 3), mét. Nếu != None → dùng nó (chính
            xác hơn, SDK tự xử lý baseline/distortion), BỎ QUA depth_frame.
            None (mặc định) → đường pinhole tuyến tính cũ, KHÔNG đổi hành vi.

    Returns:
        (x_mm, y_mm, z_mm) — tuple float, NaN nếu không có depth hợp lệ
        hoặc nằm ngoài [DEPTH_MIN_M, DEPTH_MAX_M].
    """
    if cs_map is None and (depth_frame is None or depth_frame.size == 0):
        return (float("nan"), float("nan"), float("nan"))

    x1, y1, x2, y2 = (int(v) for v in bbox_color)

    # Clamp về kích thước color
    x1 = max(0, min(color_w - 1, x1))
    x2 = max(0, min(color_w - 1, x2))
    y1 = max(0, min(color_h - 1, y1))
    y2 = max(0, min(color_h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return (float("nan"), float("nan"), float("nan"))

    # Tâm bbox trong không gian COLOR
    u_c = (x1 + x2) / 2.0
    v_c = (y1 + y2) / 2.0

    # Đường ICoordinateMapper (chính xác hơn) — bỏ qua depth pinhole.
    if cs_map is not None:
        return _from_camera_space(cs_map, u_c, v_c)

    # Map tâm color → toạ độ depth pixel để đọc Z (xấp xỉ tuyến tính)
    sx = DEPTH_W / float(color_w)
    sy = DEPTH_H / float(color_h)
    u_d = int(round(u_c * sx))
    v_d = int(round(v_c * sy))

    z_mm = _median_depth_mm(depth_frame, u_d, v_d, half=2)
    if math.isnan(z_mm) or z_mm <= 0:
        return (float("nan"), float("nan"), float("nan"))

    z_m = z_mm / DEPTH_SCALE
    if z_m < DEPTH_MIN_M or z_m > DEPTH_MAX_M:
        return (float("nan"), float("nan"), float("nan"))

    # Pinhole projection bằng intrinsics CAMERA RGB
    # → kết quả X, Y, Z tham chiếu gốc = tâm quang học camera RGB
    x_m = (u_c - COLOR_CX) * z_m / COLOR_FX
    y_m = (v_c - COLOR_CY) * z_m / COLOR_FY

    return (x_m * 1000.0, y_m * 1000.0, z_m * 1000.0)
