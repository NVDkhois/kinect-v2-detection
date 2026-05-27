"""
Overlay helpers — vẽ bbox, label (class + conf), badge track ID, và bbox
nét đứt khi state='lost'. Màu cố định theo tên class (hash).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np


if TYPE_CHECKING:
    from core.detector import Detection


# Cache màu theo tên class để mỗi class luôn 1 màu cố định.
_COLOR_CACHE: dict[str, tuple[int, int, int]] = {}


def color_for_class(class_name: str) -> tuple[int, int, int]:
    """
    Sinh màu BGR ổn định theo hash của tên class.

    Cùng tên class → luôn ra cùng màu trong các frame liên tiếp.
    """
    if class_name in _COLOR_CACHE:
        return _COLOR_CACHE[class_name]
    digest = hashlib.md5(class_name.encode("utf-8")).digest()
    color = (int(digest[0]), int(digest[1]), int(digest[2]))
    # Tránh màu quá tối
    color = tuple(max(c, 60) for c in color)  # type: ignore[assignment]
    _COLOR_CACHE[class_name] = color  # type: ignore[assignment]
    return color  # type: ignore[return-value]


def _draw_dashed_rect(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 8,
) -> None:
    """Vẽ hcn nét đứt cho state='lost'."""
    x1, y1 = p1
    x2, y2 = p2
    # top + bottom
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_track_id(frame, obj, color) -> None:
    """Badge '#NN' góc trên-trái bbox. Dot nhấp nháy khi state='new'."""
    if obj.track_id <= 0:
        return
    x1, y1, x2, y2 = obj.bbox
    text = f"#{obj.track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 3
    bx1 = x1
    bx2 = bx1 + tw + 2 * pad
    by1 = y1
    by2 = y1 + th + 2 * pad
    # Filled rect — pre-darken để mô phỏng alpha 0.8 mà không blend (avoid full-frame copy)
    bg = tuple(int(c * 0.85) for c in color)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg, -1)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad - 1),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    # 'new' state: dot nhỏ góc trên-phải badge
    if getattr(obj, "state", "") == "new":
        cv2.circle(frame, (bx2 - 3, by1 + 3), 3, (0, 255, 255), -1)


def draw_detections(
    frame: np.ndarray,
    detections: Iterable["Detection"],
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """
    Thứ tự vẽ: bbox + label → track ID badge.

    Tracked objects (có .state) được vẽ thêm badge ID + bbox nét đứt nếu lost.
    """
    if frame is None or frame.size == 0:
        return frame

    cmap = color_map or {}

    for det in detections:
        color = cmap.get(det.class_name) or color_for_class(det.class_name)
        x1, y1, x2, y2 = det.bbox

        # Bbox — nét đứt nếu lost (Kalman predicted)
        state = getattr(det, "state", None)
        if state == "lost":
            _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2, dash=8)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label — bỏ #ID khỏi label vì đã có badge riêng góc trên-phải.
        label = f"{det.class_name} {det.conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly2 = max(0, y1)
        ly1 = max(0, y1 - lh - 6)
        cv2.rectangle(frame, (x1, ly1), (x1 + lw + 4, ly2), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Track ID badge (nếu là TrackedObject)
        if hasattr(det, "state"):
            draw_track_id(frame, det, color)

    return frame
