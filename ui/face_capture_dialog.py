"""
FaceCaptureDialog — popup quét khuôn mặt trực tiếp từ camera Kinect.

Luồng sử dụng:
  1. Dialog mở → live preview @ 15fps, Haar Cascade vẽ khung xanh lên mặt
  2. User bấm "▶ Bắt đầu quét" → đếm ngược 3-2-1
  3. Tự động chụp CAPTURE_COUNT ảnh, mỗi ảnh cách nhau CAPTURE_INTERVAL_MS
  4. Mỗi ảnh hợp lệ: crop face tight → thêm thumbnail bên phải
  5. User có thể lặp lại để thêm góc khác
  6. Bấm "✓ Lưu" → trả về list face crops (BGR ndarray) cho caller
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from processing.face_crop import crop_face_tight

# ── Hằng số giao diện / hành vi ──────────────────────────────────────────────
_PREVIEW_W: int = 480        # chiều rộng live preview (px)
_PREVIEW_H: int = 360        # chiều cao live preview (px)
_THUMB_SIZE: int = 80        # thumbnail ảnh đã chụp (px)
_CAPTURE_COUNT: int = 5      # số ảnh mỗi loạt quét
_CAPTURE_INTERVAL_MS: int = 1200  # ms giữa 2 lần chụp trong loạt
_COUNTDOWN_SECS: int = 3     # giây đếm ngược trước khi chụp
_PREVIEW_FPS: int = 15       # fps live preview (nhẹ CPU hơn 30fps)


def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    """Chuyển numpy BGR → QPixmap (an toàn bộ nhớ)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # tobytes() tạo bytes object độc lập — QImage không giữ reference vào array
    qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class FaceCaptureDialog(QDialog):
    """
    Dialog popup quét khuôn mặt từ camera.

    Args:
        frame_source: callable() → Optional[np.ndarray BGR] — trả về
            frame mới nhất từ Kinect/webcam. Gọi từ main thread (QTimer).
        person_name: tên hiển thị trên tiêu đề dialog.
    """

    def __init__(
        self,
        frame_source: Callable[[], Optional[np.ndarray]],
        person_name: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"📷  Quét khuôn mặt — {person_name}")
        self.setMinimumSize(740, 500)
        self.setModal(True)

        self._frame_source = frame_source
        self._person_name = person_name
        self._captures: list[np.ndarray] = []   # face crops BGR đã lưu
        self._cascade: Optional[cv2.CascadeClassifier] = None

        # State machine: 'idle' | 'countdown' | 'capturing'
        self._state = "idle"
        self._countdown_val = 0
        self._capture_idx = 0

        self._build_ui()

        # Timer: cập nhật live preview ~15fps
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(1000 // _PREVIEW_FPS)

        # Timer: tick đếm ngược (1 giây/tick)
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        # Timer: khoảng cách giữa các ảnh trong loạt
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(_CAPTURE_INTERVAL_MS)
        self._capture_timer.timeout.connect(self._do_capture)

    # ================================================================= BUILD UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(14)
        root.addLayout(self._build_left_col())
        root.addLayout(self._build_right_col())

    def _build_left_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(6)

        # Live preview
        self._preview_lbl = QLabel()
        self._preview_lbl.setFixedSize(_PREVIEW_W, _PREVIEW_H)
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setStyleSheet(
            "background:#111;border:2px solid #444;border-radius:4px;"
        )
        self._preview_lbl.setText("⏳  Đang chờ camera…")
        self._preview_lbl.setStyleSheet(
            "background:#111;color:#888;font-size:13px;"
            "border:2px solid #444;border-radius:4px;"
        )
        col.addWidget(self._preview_lbl)

        # Trạng thái khuôn mặt
        self._face_status = QLabel("⏳  Chờ camera…")
        self._face_status.setAlignment(Qt.AlignCenter)
        self._face_status.setFixedHeight(28)
        self._face_status.setStyleSheet(
            "font-size:12px;padding:3px 8px;border-radius:3px;"
            "background:#f5f5f5;color:#666;"
        )
        col.addWidget(self._face_status)

        # Đếm ngược / tiến trình chụp
        self._progress_lbl = QLabel("")
        self._progress_lbl.setAlignment(Qt.AlignCenter)
        self._progress_lbl.setFixedHeight(40)
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self._progress_lbl.setFont(font)
        self._progress_lbl.setStyleSheet("color:#1565C0;")
        col.addWidget(self._progress_lbl)

        return col

    def _build_right_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)

        # ── Thumbnails ─────────────────────────────────────────────────────
        col.addWidget(QLabel("Ảnh đã chụp:"))

        self._thumb_area = QWidget()
        self._thumb_layout = QHBoxLayout(self._thumb_area)
        self._thumb_layout.setContentsMargins(4, 4, 4, 4)
        self._thumb_layout.setSpacing(4)
        self._thumb_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._thumb_area)
        scroll.setFixedHeight(_THUMB_SIZE + 24)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        col.addWidget(scroll)

        self._count_lbl = QLabel("0 ảnh đã chụp")
        self._count_lbl.setStyleSheet("font-size:11px;color:#666;")
        col.addWidget(self._count_lbl)

        # ── Hướng dẫn ──────────────────────────────────────────────────────
        guide = QLabel(
            "💡 Hướng dẫn:\n"
            "• Giữ khuôn mặt trong khung xanh lá\n"
            "• Chụp nhiều góc: thẳng, trái 30°, phải 30°\n"
            "• Cách camera 0.5 – 1.5 m\n"
            "• Thay đổi biểu cảm để tăng độ nhận diện"
        )
        guide.setWordWrap(True)
        guide.setStyleSheet(
            "font-size:11px;color:#333;"
            "background:#e8f0fe;border:1px solid #b0c4f8;"
            "border-radius:4px;padding:8px 10px;"
        )
        col.addWidget(guide)

        col.addStretch(1)

        # ── Nút Bắt đầu quét ───────────────────────────────────────────────
        self._scan_btn = QPushButton(
            f"▶   Bắt đầu quét  ({_CAPTURE_COUNT} ảnh)"
        )
        self._scan_btn.setMinimumHeight(38)
        self._scan_btn.setStyleSheet(
            "QPushButton{background:#1565C0;color:white;font-weight:bold;"
            "font-size:13px;padding:6px 16px;border-radius:4px;}"
            "QPushButton:hover:enabled{background:#1976D2;}"
            "QPushButton:disabled{background:#bbb;color:#fff;}"
        )
        self._scan_btn.clicked.connect(self._start_scan)
        col.addWidget(self._scan_btn)

        # ── Nút Lưu / Hủy ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._save_btn = QPushButton("✓   Lưu 0 ảnh")
        self._save_btn.setEnabled(False)
        self._save_btn.setMinimumHeight(34)
        self._save_btn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;font-weight:bold;"
            "padding:5px 14px;border-radius:4px;}"
            "QPushButton:hover:enabled{background:#388e3c;}"
            "QPushButton:disabled{background:#bbb;}"
        )
        self._save_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._save_btn)

        cancel_btn = QPushButton("Hủy")
        cancel_btn.setMinimumHeight(34)
        cancel_btn.setStyleSheet(
            "QPushButton{padding:5px 14px;border-radius:4px;"
            "border:1px solid #999;background:white;}"
            "QPushButton:hover{background:#f5f5f5;}"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        col.addLayout(btn_row)
        return col

    # =========================================================== PREVIEW TIMER
    def _get_cascade(self) -> cv2.CascadeClassifier:
        if self._cascade is None:
            xml = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            self._cascade = cv2.CascadeClassifier(xml)
        return self._cascade

    def _update_preview(self) -> None:
        """Cập nhật live preview + trạng thái khuôn mặt @ 15fps."""
        frame = self._frame_source()
        if frame is None:
            return

        # Scale frame về kích thước preview giữ tỉ lệ
        fh, fw = frame.shape[:2]
        scale = min(_PREVIEW_W / fw, _PREVIEW_H / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        display = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Detect face → vẽ khung
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        faces = self._get_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        face_found = len(faces) > 0
        box_color = (0, 210, 0) if face_found else (0, 90, 255)
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)

        # Overlay đếm ngược (mờ nền + số lớn)
        if self._state == "countdown":
            overlay = np.zeros_like(display)
            display = cv2.addWeighted(display, 0.55, overlay, 0.45, 0)
            txt = str(self._countdown_val)
            (tw, th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 10
            )
            cx = (nw - tw) // 2
            cy = (nh + th) // 2
            cv2.putText(display, txt, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0,
                        (255, 255, 255), 10, cv2.LINE_AA)

        # Overlay tiến trình chụp
        elif self._state == "capturing":
            txt = f"CHUP  {self._capture_idx}/{_CAPTURE_COUNT}"
            cv2.putText(display, txt, (8, nh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 128), 2, cv2.LINE_AA)

        # Pad về đúng kích thước _PREVIEW_W × _PREVIEW_H
        canvas = np.zeros((_PREVIEW_H, _PREVIEW_W, 3), dtype=np.uint8)
        y0 = (_PREVIEW_H - nh) // 2
        x0 = (_PREVIEW_W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = display

        self._preview_lbl.setPixmap(_bgr_to_pixmap(canvas))

        # Cập nhật label trạng thái mặt
        if face_found:
            n = len(faces)
            msg = f"✅  Phát hiện {n} khuôn mặt" if n > 1 else "✅  Phát hiện khuôn mặt"
            style = (
                "font-size:12px;padding:3px 8px;border-radius:3px;"
                "background:#e8f5e9;color:#2e7d32;"
            )
        else:
            msg = "❌  Không thấy khuôn mặt"
            style = (
                "font-size:12px;padding:3px 8px;border-radius:3px;"
                "background:#ffebee;color:#c62828;"
            )
        self._face_status.setText(msg)
        self._face_status.setStyleSheet(style)

    # ======================================================= SCAN STATE MACHINE
    def _start_scan(self) -> None:
        if self._state != "idle":
            return
        self._state = "countdown"
        self._countdown_val = _COUNTDOWN_SECS
        self._scan_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._progress_lbl.setText(f"⏱  {self._countdown_val}")
        self._progress_lbl.setStyleSheet("color:#e65100;font-size:22pt;font-weight:bold;")
        self._countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        self._countdown_val -= 1
        if self._countdown_val <= 0:
            self._countdown_timer.stop()
            self._state = "capturing"
            self._capture_idx = 0
            self._do_capture()          # chụp ngay lập tức (không chờ timer)
            self._capture_timer.start()
        else:
            self._progress_lbl.setText(f"⏱  {self._countdown_val}")

    def _do_capture(self) -> None:
        """Chụp 1 frame, crop face, thêm thumbnail. Gọi liên tiếp bởi timer."""
        if self._capture_idx >= _CAPTURE_COUNT:
            self._capture_timer.stop()
            self._state = "idle"
            self._scan_btn.setEnabled(True)
            self._scan_btn.setText(
                f"▶   Chụp thêm góc khác  ({_CAPTURE_COUNT} ảnh)"
            )
            n = len(self._captures)
            self._progress_lbl.setText(f"✓  Xong! Tổng {n} ảnh")
            self._progress_lbl.setStyleSheet(
                "color:#2e7d32;font-size:14pt;font-weight:bold;"
            )
            self._save_btn.setEnabled(n > 0)
            return

        self._progress_lbl.setText(
            f"📸  Đang chụp {self._capture_idx + 1} / {_CAPTURE_COUNT}…"
        )
        self._progress_lbl.setStyleSheet(
            "color:#1565C0;font-size:13pt;font-weight:bold;"
        )

        frame = self._frame_source()
        if frame is not None:
            cropped = crop_face_tight(frame)
            if cropped is not None:
                self._captures.append(cropped)
                self._add_thumbnail(cropped)
                self._update_count_label()

        self._capture_idx += 1

    # =========================================================== THUMBNAIL AREA
    def _add_thumbnail(self, face_bgr: np.ndarray) -> None:
        thumb = cv2.resize(
            face_bgr, (_THUMB_SIZE, _THUMB_SIZE), interpolation=cv2.INTER_AREA
        )
        lbl = QLabel()
        lbl.setPixmap(_bgr_to_pixmap(thumb))
        lbl.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        lbl.setStyleSheet(
            "border:2px solid #2e7d32;border-radius:3px;"
        )
        lbl.setToolTip(f"Ảnh {len(self._captures)}")
        # Insert trước stretch (stretch luôn ở cuối)
        insert_idx = self._thumb_layout.count() - 1
        self._thumb_layout.insertWidget(insert_idx, lbl)

    def _update_count_label(self) -> None:
        n = len(self._captures)
        self._count_lbl.setText(f"{n} ảnh đã chụp")
        self._save_btn.setText(f"✓   Lưu {n} ảnh")

    # ============================================================== PUBLIC API
    def get_captures(self) -> list[np.ndarray]:
        """Trả về danh sách face crops BGR đã chụp."""
        return list(self._captures)

    # ============================================================= CLEANUP
    def done(self, result: int) -> None:
        """Dừng mọi timer trước khi đóng để tránh callback sau destroy."""
        self._preview_timer.stop()
        self._countdown_timer.stop()
        self._capture_timer.stop()
        super().done(result)
