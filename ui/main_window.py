"""
MainWindow — khung 3 tab của KinectVision.

    🔍 Detection : DetectionPanel gắn Kinect live (capture + detector main).
    🎞 Video     : VideoPanel — xử lý video file, pipeline riêng (lazy).
    🏋 Training   : TrainingPanel.

MainWindow KHÔNG còn chứa logic toolbar/log (đã tách sang DetectionPanel —
DRY). Nhiệm vụ còn lại: host status bar, điều phối VRAM (chỉ 1 model YOLO
resident/lúc) giữa Kinect ↔ Video ↔ Training, và giữ các API delegate cũ
(`on_frame_ready` / `on_detections_ready` / `populate_classes` /
`_request_backend`) để main.py không phải đổi.

Phối hợp VRAM (GTX 1060 3GB): khi Video tab active HOẶC training chạy →
suspend detector Kinect (pause + unload + empty_cache). Khi quay lại tab
Detection → reload model Kinect, khôi phục trạng thái pause của user.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import config as _cfg
from config import UI_WINDOW_TITLE
from ui.detection_panel import DetectionPanel
from ui.video_panel import VideoPanel

log = logging.getLogger("ui")

# Thứ tự tab — dùng hằng số, không hardcode magic index trong logic.
TAB_DETECTION = 0
TAB_VIDEO = 1
TAB_TRAINING = 2


class MainWindow(QMainWindow):
    """Cửa sổ chính. Detector Kinect được attach từ main.py."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(UI_WINDOW_TITLE)
        self.resize(1400, 800)

        # Trạng thái điều phối VRAM
        self._active_tab = TAB_DETECTION
        self._kinect_suspended = False        # detector Kinect đã unload chưa
        self._suspend_reason: Optional[str] = None   # 'video' | 'training'
        self._resume_in_progress = False      # chống double-reload race
        self._loading_custom_after_train = False
        # Frame mới nhất từ Kinect/webcam — dùng bởi FaceCaptureDialog
        self._latest_color_frame: Optional[np.ndarray] = None

        self._build_ui()

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        outer.addWidget(self.tab_widget)

        # ---- Tab Detection (Kinect live) ----
        self.kinect_panel = DetectionPanel(
            persist_backend=True,
            video_placeholder="Đang chờ camera...",
        )
        self.tab_widget.addTab(self.kinect_panel, "🔍 Detection")

        # ---- Tab Video (file) ----
        self.video_panel = VideoPanel()
        self.tab_widget.addTab(self.video_panel, "🎞 Video")

        # ---- Tab Training ----
        from ui.training_panel import TrainingPanel
        self.training_panel = TrainingPanel()
        self.training_panel.set_frame_source(self._get_latest_frame)
        self.tab_widget.addTab(self.training_panel, "🏋 Training")

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Đang khởi tạo...")

        # Panel → status bar / dialog
        for p in (self.kinect_panel, self.video_panel):
            p.status_message.connect(self.statusBar().showMessage)
        self.kinect_panel.backend_error_occurred.connect(
            self._on_kinect_backend_load_error
        )
        self.video_panel.backend_error_occurred.connect(
            lambda m: QMessageBox.critical(self, "Lỗi load model video", m)
        )
        self.kinect_panel.backend_ready.connect(self._on_kinect_backend_ready)

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    # -------------------------------------------------------------- wiring
    def attach_threads(self, detector) -> None:
        """Gắn detector Kinect (main.py) + nối Training panel."""
        self.kinect_panel.attach_detector(detector)

        self.training_panel.training_started.connect(self._on_training_started)
        self.training_panel.training_stopped.connect(self._on_training_stopped)
        self.training_panel.model_loaded.connect(self._on_custom_model_loaded)

    @property
    def _detector(self):
        """Detector Kinect (tương thích code cũ tham chiếu window._detector)."""
        return self.kinect_panel.detector

    # ----------------------------------------------- delegate cho main.py
    def _get_latest_frame(self) -> Optional[np.ndarray]:
        """Trả về bản copy frame mới nhất (dùng bởi FaceCaptureDialog)."""
        return self._latest_color_frame

    @pyqtSlot(np.ndarray, np.ndarray)
    def on_frame_ready(self, color: np.ndarray, depth: np.ndarray) -> None:
        self._latest_color_frame = color.copy()   # copy để tránh race condition
        self.kinect_panel.display_frame(color)

    @pyqtSlot(list)
    def on_detections_ready(self, detections: list) -> None:
        self.kinect_panel.display_detections(detections)

    def populate_classes(self, class_names: dict) -> None:
        self.kinect_panel.populate_classes(class_names)

    def _request_backend(self, backend: str, reload: bool = False) -> None:
        self.kinect_panel.request_backend(backend, reload=reload)

    # ------------------------------------------------ điều phối VRAM (chung)
    def _suspend_kinect_detection(self, reason: str) -> None:
        """
        Pause + unload detector Kinect → trả VRAM. Idempotent.

        Dùng chung cho cả Video tab và Training (DRY — trước đây logic này
        nằm rải trong _on_training_started).
        """
        if self._kinect_suspended:
            return
        self._suspend_reason = reason

        if not _cfg.PAUSE_DETECTION_DURING_TRAINING or self._detector is None:
            # Không unload → vẫn pause inference để nhường CPU/GPU.
            if self._detector is not None:
                self._detector.paused = True
            self._kinect_suspended = True
            return

        det = self._detector
        det.paused = True
        if det.detector is not None:
            try:
                det.detector.unload()
                log.info("Kinect detector unloaded (reason=%s) — VRAM freed.",
                         reason)
            except Exception as exc:  # noqa: BLE001
                log.warning("unload() Kinect lỗi (bỏ qua): %s", exc)
        from core import device
        device.empty_cache()  # no-op nếu không CUDA (ONNX-DML/CPU)
        self._kinect_suspended = True

    def _resume_kinect_detection(self) -> None:
        """Reload model Kinect. Khôi phục pause-state user khi reload xong."""
        if not self._kinect_suspended or self._resume_in_progress:
            return
        reason = self._suspend_reason
        self._suspend_reason = None

        if (not _cfg.PAUSE_DETECTION_DURING_TRAINING or self._detector is None
                or self.kinect_panel.current_backend is None):
            # Không unload trước đó → chỉ khôi phục pause-state user.
            if self._detector is not None:
                self._detector.paused = self.kinect_panel.is_paused
            self._kinect_suspended = False
            return

        # Reload backend hiện tại. _on_kinect_backend_ready sẽ khôi phục
        # detector.paused = panel.is_paused khi reload xong.
        self._resume_in_progress = True
        self.statusBar().showMessage("Đang nạp lại model Kinect…")
        self.kinect_panel.request_backend(
            self.kinect_panel.current_backend, reload=True
        )
        log.info("Resume Kinect detection (reason cũ=%s).", reason)

    @pyqtSlot(str)
    def _on_kinect_backend_ready(self, label: str) -> None:
        """Backend Kinect switch/reload xong."""
        # Khôi phục pause-state user sau chu kỳ suspend→resume.
        if self._kinect_suspended:
            self._kinect_suspended = False
            self._resume_in_progress = False
            if self._detector is not None:
                self._detector.paused = self.kinect_panel.is_paused
            log.info("Kinect resumed (paused=%s).", self.kinect_panel.is_paused)

        # Thông báo training panel khi load custom model sau train xong.
        if self._loading_custom_after_train:
            self._loading_custom_after_train = False
            self.training_panel.notify_model_loaded_ok(label)

    # ------------------------------------------------------ đổi tab
    @pyqtSlot(int)
    def _on_tab_changed(self, idx: int) -> None:
        old = self._active_tab
        self._active_tab = idx

        # Rời tab Video → trả VRAM (unload detector video).
        if old == TAB_VIDEO and idx != TAB_VIDEO:
            self.video_panel.deactivate()

        if idx == TAB_VIDEO:
            # Suspend Kinect trước, rồi kích hoạt pipeline video.
            self._suspend_kinect_detection("video")
            self.video_panel.activate()
        elif idx == TAB_DETECTION:
            # Quay lại Kinect — reload nếu đang suspend (không phải do training
            # đang chạy; training tự quản lý resume riêng).
            if self._suspend_reason != "training":
                self._resume_kinect_detection()
        # TAB_TRAINING: giữ nguyên trạng thái; nếu đang suspend do video thì
        # vẫn để suspend (training sắp cần VRAM). Resume xảy ra khi quay lại
        # tab Detection.

    # ================================================ Training wiring slots
    @pyqtSlot()
    def _on_training_started(self) -> None:
        """Training bắt đầu → khoá Detection+Video, suspend Kinect."""
        self.tab_widget.setTabEnabled(TAB_DETECTION, False)
        self.tab_widget.setTabEnabled(TAB_VIDEO, False)
        # User có thể đang đứng ở tab Video khi bấm train → detector video
        # vẫn giữ VRAM, cạnh tranh với training. Trả VRAM trước.
        self.video_panel.deactivate()
        # Nếu đang suspend do video → đổi chủ sở hữu sang training.
        if self._kinect_suspended:
            self._suspend_reason = "training"
        else:
            self._suspend_kinect_detection("training")

        if not _cfg.PAUSE_DETECTION_DURING_TRAINING or self._detector is None:
            self.statusBar().showMessage(
                "Training đang chạy — Detection vẫn tiếp tục."
            )
        else:
            self.statusBar().showMessage(
                "Training đang chạy — Detection tạm dừng, VRAM giải phóng."
            )

    @pyqtSlot()
    def _on_training_stopped(self) -> None:
        """Training kết thúc → mở khoá tab, reload Kinect."""
        self.tab_widget.setTabEnabled(TAB_DETECTION, True)
        self.tab_widget.setTabEnabled(TAB_VIDEO, True)

        if not self._kinect_suspended:
            self.statusBar().showMessage("Training kết thúc.")
            return

        self.statusBar().showMessage("Training xong — đang reload model…")
        # Đảm bảo cờ reason không chặn resume.
        self._suspend_reason = "training"
        self._resume_kinect_detection()

    @pyqtSlot(str)
    def _on_kinect_backend_load_error(self, msg: str) -> None:
        """Load model Kinect thất bại — xóa cờ resume để không bị kẹt."""
        self._resume_in_progress = False
        QMessageBox.critical(self, "Lỗi load model", msg)

    @pyqtSlot(str)
    def _on_custom_model_loaded(self, path: str) -> None:
        """TrainingPanel đã copy best.pt + cập nhật CUSTOM_MODEL_PATH."""
        log.info("Custom model sẵn sàng: %s", path)
        self.kinect_panel.populate_model_combo()
        self._loading_custom_after_train = True
        self.kinect_panel.request_backend("custom")
        self.statusBar().showMessage("Đang load custom model vào backend…")

    # ---------------------------------------------------------------- close
    def closeEvent(self, event) -> None:  # noqa: N802 — Qt API
        log.info("MainWindow closeEvent — yêu cầu shutdown.")
        try:
            self.video_panel.shutdown()
        except Exception as exc:  # noqa: BLE001
            log.warning("video_panel.shutdown lỗi (bỏ qua): %s", exc)
        super().closeEvent(event)
