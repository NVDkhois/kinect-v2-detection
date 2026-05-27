"""
processing/face_crop.py — Phát hiện và crop khuôn mặt từ ảnh tĩnh.

Dùng chung cho:
  - ui/training_panel.py  (Template Manager)
  - tools/prepare_face_templates.py  (script chuẩn bị ảnh mẫu)

Cascade được lazy-init một lần ở cấp module (singleton nhẹ ~1MB).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from config import (
    TEMPLATE_CROP_PAD_X,
    TEMPLATE_CROP_PAD_BOT,
    TEMPLATE_CROP_PAD_TOP,
    TEMPLATE_CROP_W,
)

_cascade: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    """Lazy-init Haar Cascade một lần duy nhất."""
    global _cascade
    if _cascade is None:
        xml = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        _cascade = cv2.CascadeClassifier(xml)
    return _cascade


def crop_face_tight(
    img_bgr: np.ndarray,
    target_w: int = TEMPLATE_CROP_W,
    pad_x: float = TEMPLATE_CROP_PAD_X,
    pad_top: float = TEMPLATE_CROP_PAD_TOP,
    pad_bot: float = TEMPLATE_CROP_PAD_BOT,
) -> Optional[np.ndarray]:
    """
    Phát hiện khuôn mặt lớn nhất trong ảnh BGR, crop chặt và resize.

    Padding (tính theo % kích thước bbox Haar Cascade):
      pad_x   — mỗi bên ngang (giữ viền má, bỏ nền)
      pad_top — phía trên (lấy ít tóc/trán)
      pad_bot — phía dưới (lấy cằm, không lấy cổ/vai)

    Returns:
      Ảnh BGR đã resize về ``target_w`` px (height giữ tỉ lệ gốc).
      None nếu không phát hiện được khuôn mặt hoặc ảnh đầu vào lỗi.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    x0 = max(0, fx - int(fw * pad_x))
    y0 = max(0, fy - int(fh * pad_top))
    x1 = min(w, fx + fw + int(fw * pad_x))
    y1 = min(h, fy + fh + int(fh * pad_bot))

    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    th = int(crop.shape[0] * target_w / crop.shape[1])
    return cv2.resize(crop, (target_w, th), interpolation=cv2.INTER_AREA)
