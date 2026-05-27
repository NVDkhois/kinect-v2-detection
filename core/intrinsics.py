"""
Loader/writer intrinsics camera RGB (override danh định bằng calibrate).

CHỈ dùng stdlib (json, pathlib) — KHÔNG import config/numpy/cv2 → tránh
import cycle (config.py gọi resolve_color_intrinsics lúc import).

Schema intrinsics.json (ghi bởi tools/calibrate_intrinsics.py):
    {"fx","fy","cx","cy", "image_w","image_h", "rms","n_views", ...}

resolve_color_intrinsics() TUYỆT ĐỐI không raise: file lỗi/thiếu/sai →
fallback nominal. (raise = chết import config = chết toàn app.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("intrinsics")

_Quad = tuple[float, float, float, float]

# Khoảng hợp lý cho Kinect V2 color (1920×1080) — chặn giá trị vô lý/corrupt.
_F_MIN, _F_MAX = 100.0, 5000.0
_C_MIN, _C_MAX = 1.0, 10000.0


def _valid(v, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and lo < float(v) < hi


def resolve_color_intrinsics(nominal: _Quad, json_path: str) -> _Quad:
    """
    Trả (fx, fy, cx, cy). Đọc `json_path` nếu hợp lệ; mọi lỗi → `nominal`.

    Không bao giờ raise.
    """
    p = Path(json_path)
    if not p.is_file():
        return nominal
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        fx, fy = data["fx"], data["fy"]
        cx, cy = data["cx"], data["cy"]
    except Exception as exc:  # noqa: BLE001 — không được làm chết import
        log.warning("intrinsics.json không đọc được (%s) — dùng danh định.",
                    exc)
        return nominal

    if not (_valid(fx, _F_MIN, _F_MAX) and _valid(fy, _F_MIN, _F_MAX)
            and _valid(cx, _C_MIN, _C_MAX) and _valid(cy, _C_MIN, _C_MAX)):
        log.warning("intrinsics.json giá trị ngoài range — dùng danh định.")
        return nominal

    log.info("intrinsics.json override: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
             fx, fy, cx, cy)
    return (float(fx), float(fy), float(cx), float(cy))


def write_intrinsics(
    json_path: str,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_w: int,
    image_h: int,
    rms: float,
    n_views: int,
) -> None:
    """Ghi intrinsics.json (schema dùng chung với resolve_color_intrinsics)."""
    payload = {
        "fx": float(fx), "fy": float(fy),
        "cx": float(cx), "cy": float(cy),
        "image_w": int(image_w), "image_h": int(image_h),
        "rms": float(rms), "n_views": int(n_views),
        "note": "Ghi bởi tools/calibrate_intrinsics.py — override config.COLOR_*",
    }
    Path(json_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
