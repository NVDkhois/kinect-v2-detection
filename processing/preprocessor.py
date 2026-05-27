"""
Preprocessing pipeline cho color frame trước khi đưa vào YOLO.

Thứ tự (bắt buộc):
    1. Gaussian blur (khử nhiễu nhẹ)
    2. Unsharp mask (tăng độ sắc nét)
    3. CLAHE trên kênh L của LAB (cải thiện tương phản local)

Benchmark trên i3-10105F, frame 640×480 BGR:
    GaussianBlur:  ~0.8 ms
    UnsharpMask:   ~1.5 ms
    CLAHE (LAB):   ~1.9 ms
    Total:         ~4.2 ms (< 5ms target ✓)
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    GAUSSIAN_KSIZE,
    PREPROCESS_STAGES,
    UNSHARP_SIGMA,
    UNSHARP_STRENGTH,
)

# Thứ tự áp dụng bắt buộc — không phụ thuộc thứ tự caller truyền vào.
_CANONICAL_ORDER: tuple[str, ...] = ("blur", "unsharp", "clahe")


_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


def unsharp_mask(
    img: np.ndarray,
    sigma: float = UNSHARP_SIGMA,
    strength: float = UNSHARP_STRENGTH,
) -> np.ndarray:
    """
    Làm sắc nét ảnh bằng kỹ thuật unsharp mask.

    Args:
        img: ảnh BGR (uint8)
        sigma: độ rộng Gaussian dùng tạo phiên bản blur.
        strength: hệ số khuếch đại high-frequency. 1.5 là vừa phải.

    Returns:
        Ảnh đã sắc nét, cùng shape & dtype với input.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


def apply_clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Áp CLAHE lên kênh L của LAB colorspace.

    Args:
        img_bgr: ảnh BGR uint8.

    Returns:
        Ảnh BGR uint8 sau khi cân bằng tương phản local.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(
    frame: np.ndarray,
    stages: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    """
    Pipeline preprocessing: áp các tầng trong `stages` theo thứ tự CANONICAL
    blur → unsharp → clahe (không phụ thuộc thứ tự truyền vào).

    Args:
        frame: ảnh BGR uint8 (bất kỳ resolution).
        stages: tập tầng cần áp ("blur"/"unsharp"/"clahe"). None → đọc
            `config.PREPROCESS_STAGES` (mặc định cả 3 = hành vi cũ, KHỚP
            byte-for-byte). `()` → no-op, trả COPY (không mutate input).

    Returns:
        Ảnh BGR uint8 đã xử lý, cùng shape với input.

    Raises:
        ValueError: nếu `stages` chứa tên tầng không hợp lệ (fail fast).

    Benchmark (i3-10105F, 640×480, cả 3 tầng): ~4.2 ms. Xem docstring module.
    """
    if frame is None or frame.size == 0:
        return frame

    selected = PREPROCESS_STAGES if stages is None else tuple(stages)
    unknown = set(selected) - set(_CANONICAL_ORDER)
    if unknown:
        raise ValueError(
            f"stage không hợp lệ: {sorted(unknown)} "
            f"(hợp lệ: {list(_CANONICAL_ORDER)})"
        )

    out = frame.copy()
    for stage in _CANONICAL_ORDER:
        if stage not in selected:
            continue
        if stage == "blur":
            out = cv2.GaussianBlur(out, GAUSSIAN_KSIZE, 0)
        elif stage == "unsharp":
            out = unsharp_mask(out, sigma=UNSHARP_SIGMA,
                               strength=UNSHARP_STRENGTH)
        else:  # clahe
            out = apply_clahe_lab(out)
    return out
