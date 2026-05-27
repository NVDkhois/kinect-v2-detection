"""
VideoPanel — tab xử lý video file (không dùng camera).

Kế thừa DetectionPanel (đầy đủ công cụ: model/class/track/conf), bổ sung
điều khiển video: 📂 Mở video · ▶/⏸ Phát-Tạm dừng · ⟲ Phát lại.

Sở hữu pipeline riêng (LAZY — chỉ tạo khi tab Video được kích hoạt lần
đầu, tránh load model thứ 2 lúc khởi động → bảo vệ VRAM 3GB):
    VideoFileCaptureThread → queue(maxsize) → DetectionThread (riêng)

Phối hợp VRAM (host gọi qua activate()/deactivate() khi đổi tab):
  • activate()   : reload detector video (nếu đã unload trước đó).
  • deactivate() : pause playback + unload detector video + empty_cache
                   → trả VRAM cho Kinect (chỉ 1 model YOLO resident/lúc).

Video file không có depth → đẩy (color, None) vào queue;
DetectionThread._scale_and_project bỏ qua 3D → cột X/Y/Z hiện '---'.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QPushButton

from config import QUEUE_MAXSIZE
from core.detector import DetectionThread
from core.video_capture import VideoFileCaptureThread
from ui.detection_panel import DetectionPanel

log = logging.getLogger("ui")

_VIDEO_FILTER = (
    "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;Tất cả file (*)"
)


class VideoPanel(DetectionPanel):
    """Tab Video — DetectionPanel + nguồn frame từ file + điều khiển phát."""

    def __init__(self, parent=None) -> None:
        super().__init__(
            persist_backend=False,            # không hijack active_backend
            video_placeholder="Chưa nạp video — bấm '📂 Mở video'",
            parent=parent,
        )
        self._capture: Optional[VideoFileCaptureThread] = None
        self._det_queue: "queue.Queue" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._threads_started = False
        self._suspended = False               # detector video đã unload chưa
        self._last_dir = self._restore_last_dir()

        # Inference-pause của base panel thừa với video (pause playback đã
        # dừng mọi thứ) → ẩn để tránh 2 khái niệm pause gây rối.
        self.pause_btn.hide()

        self._build_video_controls()

    # --------------------------------------------------------- toolbar bổ sung
    def _build_video_controls(self) -> None:
        """Chèn nút video vào cuối toolbar (sau stretch, trước pause ẩn)."""
        tb = self._extra_toolbar_slot

        self.open_btn = QPushButton("📂 Mở video")
        self.open_btn.setMinimumWidth(120)
        self.open_btn.clicked.connect(self._on_open_clicked)

        self.play_btn = QPushButton("▶ Phát")
        self.play_btn.setCheckable(True)
        self.play_btn.setEnabled(False)
        self.play_btn.setMinimumWidth(120)
        self.play_btn.toggled.connect(self._on_play_toggled)

        self.replay_btn = QPushButton("⟲ Phát lại")
        self.replay_btn.setEnabled(False)
        self.replay_btn.setMinimumWidth(100)
        self.replay_btn.clicked.connect(self._on_replay_clicked)

        # Chèn trước pause_btn (widget cuối cùng trong layout)
        insert_at = tb.count() - 1
        for w in (self.open_btn, self.play_btn, self.replay_btn):
            tb.insertWidget(insert_at, w)
            insert_at += 1

    # --------------------------------------------------------- vòng đời thread
    def _ensure_threads(self) -> None:
        """Lazy: tạo + start capture/detector lần đầu tiên cần đến."""
        if self._threads_started:
            return
        log.info("VideoPanel: khởi tạo pipeline video (lazy).")
        self._capture = VideoFileCaptureThread()
        self._video_detector = DetectionThread(self._det_queue)
        self.attach_detector(self._video_detector)

        # Capture (thread riêng) → slot chạy ở GUI thread (queued):
        # bơm (color, None) vào queue detector + hiển thị frame.
        self._capture.frame_ready.connect(self._on_video_frame)
        self._capture.video_loaded.connect(self._on_video_loaded)
        self._capture.video_finished.connect(self._on_video_finished)
        self._capture.error.connect(self._on_video_error)
        self._video_detector.detections_ready.connect(self.display_detections)

        self._capture.start()
        self._video_detector.start()
        self._threads_started = True

    def activate(self) -> None:
        """Host gọi khi tab Video thành active (Kinect đã bị suspend)."""
        self._ensure_threads()
        if self._suspended and self.current_backend is not None:
            # Reload model video đã unload lúc rời tab.
            self.status_message.emit("Đang nạp lại model video…")
            self.request_backend(self.current_backend, reload=True)
        self._suspended = False

    def deactivate(self) -> None:
        """Host gọi khi rời tab Video — trả VRAM cho Kinect."""
        if not self._threads_started or self._suspended:
            return
        # Dừng phát
        if self._capture is not None:
            self._capture.pause()
        self.play_btn.blockSignals(True)
        self.play_btn.setChecked(False)
        self.play_btn.setText("▶ Phát")
        self.play_btn.blockSignals(False)

        # Pause + unload detector video → empty_cache
        det = self._video_detector
        det.paused = True
        if det.detector is not None:
            try:
                det.detector.unload()
                log.info("VideoPanel: detector unloaded — trả VRAM.")
            except Exception as exc:  # noqa: BLE001
                log.warning("VideoPanel unload lỗi (bỏ qua): %s", exc)
        from core import device
        device.empty_cache()  # no-op nếu không CUDA (ONNX-DML/CPU)
        self._suspended = True

    def shutdown(self) -> None:
        """Dừng an toàn (gọi từ MainWindow.closeEvent)."""
        if not self._threads_started:
            return
        log.info("VideoPanel: shutdown threads.")
        if self._capture is not None:
            self._capture.stop()
        self._video_detector.stop()
        if self._capture is not None and not self._capture.wait(2000):
            log.warning("Video capture thread không dừng kịp (>2s).")
        # Detector cần lâu hơn: nếu stop() rơi vào giữa detector.load()
        # (~4s, không ngắt được) thì thread chỉ thoát NGAY SAU khi load xong.
        if not self._video_detector.wait(5000):
            log.warning("Video detector thread không dừng kịp (>5s).")

    # ---------------------------------------------------------------- slots
    @pyqtSlot(np.ndarray)
    def _on_video_frame(self, color: np.ndarray) -> None:
        # Bơm vào queue detector (drop frame cũ nếu đầy — không buffer)
        try:
            self._det_queue.put_nowait((color, None))
        except queue.Full:
            try:
                self._det_queue.get_nowait()
                self._det_queue.put_nowait((color, None))
            except queue.Empty:
                pass
        self.display_frame(color)

    @pyqtSlot(float, int)
    def _on_video_loaded(self, fps: float, frame_count: int) -> None:
        self.play_btn.setEnabled(True)
        self.replay_btn.setEnabled(True)
        self.play_btn.blockSignals(True)
        self.play_btn.setChecked(False)
        self.play_btn.setText("▶ Phát")
        self.play_btn.blockSignals(False)
        self.clear_log()
        secs = frame_count / fps if fps else 0.0
        self.status_message.emit(
            f"Đã nạp video: {fps:.0f} fps · {frame_count} frame · ~{secs:.0f}s"
            "  — bấm ▶ Phát"
        )

    @pyqtSlot()
    def _on_video_finished(self) -> None:
        # Giữ frame cuối; nút về trạng thái sẵn sàng phát lại.
        self.play_btn.blockSignals(True)
        self.play_btn.setChecked(False)
        self.play_btn.setText("▶ Phát")
        self.play_btn.blockSignals(False)
        self.status_message.emit(
            "Video phát hết — dừng ở frame cuối. Bấm ⟲ Phát lại hoặc 📂 Mở video."
        )

    @pyqtSlot(str)
    def _on_video_error(self, msg: str) -> None:
        self.play_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        QMessageBox.critical(self, "Lỗi mở video", msg)

    # --------------------------------------------------------- nút điều khiển
    def _on_open_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", self._last_dir, _VIDEO_FILTER
        )
        if not path:
            return
        self._ensure_threads()
        self._last_dir = str(Path(path).parent)
        self._persist_last_dir(path)
        if self._capture is not None:
            self._capture.load(path)   # tự pause; load xong → _on_video_loaded

    def _on_play_toggled(self, checked: bool) -> None:
        if self._capture is None:
            return
        if checked:
            self._capture.play()
            self.play_btn.setText("⏸ Tạm dừng")
            self.status_message.emit("Đang phát video…")
        else:
            self._capture.pause()
            self.play_btn.setText("▶ Phát")
            self.status_message.emit("Đã tạm dừng video.")

    def _on_replay_clicked(self) -> None:
        if self._capture is None:
            return
        self._capture.restart()
        self.play_btn.blockSignals(True)
        self.play_btn.setChecked(True)
        self.play_btn.setText("⏸ Tạm dừng")
        self.play_btn.blockSignals(False)
        self.status_message.emit("Phát lại từ đầu…")

    # ------------------------------------------------------------ app_state
    @staticmethod
    def _restore_last_dir() -> str:
        try:
            import app_state
            p = app_state.get_last_video_path()
            if p and Path(p).parent.is_dir():
                return str(Path(p).parent)
        except Exception:
            pass
        return ""

    @staticmethod
    def _persist_last_dir(path: str) -> None:
        try:
            import app_state
            app_state.set_last_video_path(path)
        except Exception:
            pass
