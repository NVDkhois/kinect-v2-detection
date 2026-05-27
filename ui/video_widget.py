"""
VideoWidget — QLabel hiển thị RGB frame với bounding box overlay.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy

from config import DISPLAY_H, DISPLAY_W
from core.detector import Detection
from processing.overlay import draw_detections


log = logging.getLogger("video")


class VideoWidget(QLabel):
    """
    Hiển thị color frame (BGR) đã vẽ overlay detection.

    Cách dùng:
        widget.update_frame(color_bgr, detections)
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(DISPLAY_W, DISPLAY_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; color: #aaa;")
        self.setText("Đang chờ camera...")

        # Profiling rolling stats for draw_detections + frame conversion
        self._prof_t0 = time.perf_counter()
        self._prof_n = 0
        self._prof_overlay_ms_sum = 0.0
        self._prof_convert_ms_sum = 0.0

    def update_frame(
        self,
        color_frame: np.ndarray,
        detections: Iterable[Detection] | None = None,
    ) -> None:
        """
        Cập nhật QLabel với 1 frame mới.

        Args:
            color_frame: BGR uint8.
            detections: list[Detection] (toạ độ bbox theo color_frame).
                Nếu None → không vẽ overlay.
        """
        if color_frame is None or color_frame.size == 0:
            return

        frame = color_frame

        t_ov = time.perf_counter()
        if detections:
            frame = frame.copy()
            draw_detections(frame, detections)
        overlay_ms = (time.perf_counter() - t_ov) * 1000.0

        t_cv = time.perf_counter()
        # Resize giữ aspect ratio theo size hiện tại của widget
        target_w = max(1, self.width())
        target_h = max(1, self.height())
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            # INTER_LINEAR thay vì INTER_AREA: AREA cho chất lượng downscale
            # tốt nhất nhưng chậm nhất (~2-3× LINEAR khi thu 1080p→~640).
            # Đây chỉ là preview hiển thị → LINEAR mắt thường không phân biệt
            # được, nhưng cắt mạnh chi phí convert trên GUI thread.
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg.copy()))
        convert_ms = (time.perf_counter() - t_cv) * 1000.0

        # ---- Rolling stats ----
        self._prof_n += 1
        self._prof_overlay_ms_sum += overlay_ms
        self._prof_convert_ms_sum += convert_ms
        elapsed = time.perf_counter() - self._prof_t0
        if elapsed >= 1.0:
            log.info(
                "overlay_avg=%.1fms  convert_avg=%.1fms  ui_fps=%.1f (n=%d)",
                self._prof_overlay_ms_sum / self._prof_n,
                self._prof_convert_ms_sum / self._prof_n,
                self._prof_n / elapsed,
                self._prof_n,
            )
            self._prof_t0 = time.perf_counter()
            self._prof_n = 0
            self._prof_overlay_ms_sum = 0.0
            self._prof_convert_ms_sum = 0.0
