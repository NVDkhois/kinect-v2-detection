"""
DetectionPanel — widget tái dùng: toolbar + video + log table của 1 nguồn
detection (Kinect live HOẶC video file).

Tách từ MainWindow để tab Detection và tab Video dùng chung (DRY), tránh
copy-paste ~300 dòng. Panel KHÔNG sở hữu QStatusBar — phát
`status_message(str)` để host (MainWindow) hiển thị.

Panel chỉ làm việc với một `detector` (DetectionThread) generic qua các
signal/slot có sẵn (class_names_changed / backend_changed / backend_error
/ on_backend_switched). Nguồn frame được host bơm vào qua
`display_frame()` / `display_detections()` (không tự bind Qt signal →
tái dùng được cho cả Kinect router lẫn video router).

Khác biệt theo nguồn:
  • persist_backend=True  → ghi backend đang dùng vào app_state (chỉ panel
    Kinect cần, để restart khôi phục). Panel Video = False (không hijack).
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import YOLO_CONF_DEFAULT
from core.detector import Detection
from models.factory import ModelFactory
from processing.overlay import color_for_class
from ui.video_widget import VideoWidget

log = logging.getLogger("ui")

# Label "tất cả classes" trong combobox. Em-dash để không trùng tên class
# thật (không class COCO nào chứa "—").
ALL_CLASSES_LABEL = "— Tất cả —"
ALL_TRACKS_LABEL = "— Tất cả —"


class DetectionPanel(QWidget):
    """
    Khối hiển thị + điều khiển 1 luồng detection.

    Signals (host nối để cập nhật status bar / điều phối):
        status_message(str) — thông điệp ngắn cho status bar
        backend_ready(str)  — switch/reload backend hoàn tất (label)
        backend_busy(bool)  — True khi đang switch (controls disabled)
    """

    status_message = pyqtSignal(str)
    backend_ready = pyqtSignal(str)
    backend_busy = pyqtSignal(bool)
    backend_error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        persist_backend: bool = False,
        video_placeholder: str = "Đang chờ camera...",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._persist_backend = persist_backend

        self._detector = None

        self._paused = False
        self._last_color: Optional[np.ndarray] = None
        self._last_detections: list[Detection] = []

        self._current_backend: Optional[str] = None
        self._pending_backend_ui: Optional[str] = None

        self._selected_track_id: Optional[int] = None
        self._row_by_tid: dict[int, int] = {}

        self._last_status_t = 0.0
        self._frame_counter = 0

        self._build_ui(video_placeholder)

    # ----------------------------------------------------------------- UI
    def _build_ui(self, video_placeholder: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(self._build_toolbar())
        layout.addLayout(self._build_panels(video_placeholder), 1)
        layout.addWidget(self._build_log_table())

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # ---- Model selector ----
        toolbar.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(170)
        self._populate_model_combo()
        self.model_combo.activated[int].connect(self._on_model_activated)
        toolbar.addWidget(self.model_combo)

        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Reload backend hiện tại (không switch)")
        self.reload_btn.setMinimumWidth(70)
        self.reload_btn.clicked.connect(self._on_reload_clicked)
        toolbar.addWidget(self.reload_btn)

        toolbar.addSpacing(16)

        # ---- Class selector ----
        toolbar.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)
        self.class_combo.setInsertPolicy(QComboBox.NoInsert)
        self.class_combo.addItem(ALL_CLASSES_LABEL, userData=None)
        self.class_combo.setMinimumWidth(200)

        completer = self.class_combo.completer()
        if completer is not None:
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.class_combo.activated[int].connect(self._on_class_activated)
        toolbar.addWidget(self.class_combo)

        toolbar.addSpacing(12)

        # ---- Track ID selector ----
        toolbar.addWidget(QLabel("Track:"))
        self.track_combo = QComboBox()
        self.track_combo.setMinimumWidth(90)
        self.track_combo.addItem(ALL_TRACKS_LABEL, userData=None)
        self.track_combo.setToolTip(
            "Chọn Track ID để chỉ hiện bbox của track đó.\n"
            "Chọn '— Tất cả —' để hiện lại tất cả."
        )
        self.track_combo.activated[int].connect(self._on_track_activated)
        toolbar.addWidget(self.track_combo)

        toolbar.addSpacing(16)

        # ---- Confidence slider ----
        toolbar.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setSingleStep(5)
        self.conf_slider.setPageStep(5)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setValue(int(YOLO_CONF_DEFAULT * 100))
        self.conf_slider.setMinimumWidth(200)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        toolbar.addWidget(self.conf_slider)

        self.conf_label = QLabel(f"{int(YOLO_CONF_DEFAULT * 100)}%")
        self.conf_label.setMinimumWidth(40)
        self.conf_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.conf_label)

        toolbar.addSpacing(16)

        # ---- Tracks count label ----
        self.tracks_label = QLabel("Tracks: 0 active")
        self.tracks_label.setStyleSheet("color: #888;")
        self.tracks_label.setMinimumWidth(180)
        toolbar.addWidget(self.tracks_label)

        toolbar.addStretch(1)

        # Vùng cho host chèn thêm control (vd nút video) trước nút Pause.
        self._extra_toolbar_slot = toolbar

        # ---- Pause / Resume button ----
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setMinimumWidth(120)
        self.pause_btn.toggled.connect(self._on_pause_toggled)
        toolbar.addWidget(self.pause_btn)

        return toolbar

    def _build_panels(self, placeholder: str) -> QHBoxLayout:
        panels = QHBoxLayout()
        panels.setSpacing(8)
        self.video_widget = VideoWidget()
        self.video_widget.setText(placeholder)
        panels.addWidget(self.video_widget, 1)
        return panels

    def _build_log_table(self) -> QTableWidget:
        self.log_table = QTableWidget(0, 7)
        self.log_table.setHorizontalHeaderLabels(
            ["ID", "Class", "Conf", "X (mm)", "Y (mm)", "Z (mm)", "Cập nhật"]
        )
        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setMaximumHeight(220)
        self.log_table.cellClicked.connect(self._on_log_row_clicked)
        return self.log_table

    # -------------------------------------------------------------- wiring
    def attach_detector(self, detector) -> None:
        """Lưu tham chiếu detector + nối signal backend."""
        self._detector = detector
        if detector is not None:
            detector.class_names_changed.connect(self._on_class_names_changed)
            detector.backend_changed.connect(self._on_backend_changed)
            detector.backend_error.connect(self._on_backend_error)

    @property
    def detector(self):
        return self._detector

    @property
    def current_backend(self) -> Optional[str]:
        return self._current_backend

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------ model selector
    def _populate_model_combo(self) -> None:
        """Nạp danh sách backend; disable + tooltip cho cái không khả dụng."""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for i, b in enumerate(ModelFactory.list_available()):
            label = b["label"]
            if not b["available"]:
                label = f"{label} ✗"
            self.model_combo.addItem(label, userData=b["name"])
            if not b["available"]:
                item = self.model_combo.model().item(i)
                item.setEnabled(False)
                item.setToolTip(b.get("reason", "Không khả dụng"))
        self._current_backend = self.model_combo.itemData(0)
        self.model_combo.blockSignals(False)

    def populate_model_combo(self) -> None:
        """API công khai cho host (vd refresh sau khi train xong)."""
        self._populate_model_combo()

    def _on_model_activated(self, idx: int) -> None:
        backend = self.model_combo.itemData(idx)
        if backend is None or backend == self._current_backend:
            return
        self.request_backend(backend)

    def _on_reload_clicked(self) -> None:
        if self._current_backend is not None:
            self.request_backend(self._current_backend, reload=True)

    def request_backend(self, backend: str, reload: bool = False) -> None:
        if self._detector is None:
            log.warning("request_backend: detector chưa attach — bỏ qua.")
            return
        action = "Reload" if reload else "Switch"
        self._pending_backend_ui = backend
        self.set_controls_enabled(False)
        self.backend_busy.emit(True)
        self.status_message.emit(f"Đang {action} model: {backend}...")

        # Xoá detection cũ để không hiện bbox backend trước trong lúc load.
        self._last_detections = []
        if self._last_color is not None:
            self.video_widget.update_frame(self._last_color, [])

        self._detector.on_backend_switched(backend)

    def set_controls_enabled(self, enabled: bool) -> None:
        for w in (self.model_combo, self.reload_btn, self.class_combo,
                  self.track_combo, self.conf_slider, self.pause_btn):
            w.setEnabled(enabled)

    @pyqtSlot(list)
    def _on_class_names_changed(self, names: list) -> None:
        self.populate_classes({i: n for i, n in enumerate(names)})

    @pyqtSlot(str)
    def _on_backend_changed(self, backend_label: str) -> None:
        self._current_backend = (self._pending_backend_ui
                                 or self._current_backend)
        self._pending_backend_ui = None

        idx = self.model_combo.findData(self._current_backend)
        if idx >= 0:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(idx)
            self.model_combo.blockSignals(False)

        self.set_controls_enabled(True)
        self.backend_busy.emit(False)
        self.log_table.setRowCount(0)
        self._row_by_tid.clear()
        self._reset_track_filter()
        self.status_message.emit(f"Model: {backend_label}")
        log.info("Backend đã đổi: %s", backend_label)

        if self._persist_backend and self._current_backend:
            try:
                import app_state
                app_state.set_active_backend(self._current_backend)
            except Exception:
                pass

        self.backend_ready.emit(backend_label)

    @pyqtSlot(str)
    def _on_backend_error(self, msg: str) -> None:
        self._pending_backend_ui = None
        self.set_controls_enabled(True)
        self.backend_busy.emit(False)
        idx = self.model_combo.findData(self._current_backend)
        if idx >= 0:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(idx)
            self.model_combo.blockSignals(False)
        self.status_message.emit("Switch model thất bại — giữ model cũ.")
        # Host hiển thị dialog (panel không tự popup để tái dùng linh hoạt)
        self.backend_error_occurred.emit(msg)

    def populate_classes(self, class_names: dict[int, str]) -> None:
        if not class_names:
            return
        current = self.class_combo.currentData()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem(ALL_CLASSES_LABEL, userData=None)
        for _id, name in sorted(class_names.items(), key=lambda kv: kv[1]):
            self.class_combo.addItem(name, userData=name)
        if current:
            idx = self.class_combo.findData(current)
            if idx >= 0:
                self.class_combo.setCurrentIndex(idx)
        self.class_combo.blockSignals(False)

    # ----------------------------------------------------------- frame in
    def display_frame(self, color: np.ndarray) -> None:
        """Host bơm 1 color frame vào panel (Kinect router / video router)."""
        self._last_color = color
        self.video_widget.update_frame(
            color, self._visible_detections(self._last_detections)
        )
        self._tick_status()

    def display_detections(self, detections: list) -> None:
        """
        Host bơm list[Detection] mới.

        KHÔNG re-render frame ở đây. Trước đây hàm này vẽ lại _last_color
        (full-res 1080p) MỖI lần có detection, CỘNG THÊM display_frame vẽ
        mỗi frame → ~2 lần convert ×14ms/frame trên GUI thread → bão hoà
        Qt event loop → frame queued dồn ứ → video lag nặng. Giờ chỉ lưu
        detections; display_frame (chạy mỗi frame, tần suất ≥) sẽ vẽ với
        bộ detection mới nhất. Trade-off: bbox trễ ≤1 frame (~37ms @27fps),
        không cảm nhận được, đổi lại cắt 50% tải render GUI.
        """
        if self._paused:
            return
        self._last_detections = detections
        self._append_log(detections)
        active = sum(1 for d in detections if getattr(d, "state", ""))
        self.tracks_label.setText(f"Tracks: {active} active")

    # ------------------------------------------------------------ controls
    def _on_class_activated(self, idx: int) -> None:
        if self._detector is None:
            return
        data = self.class_combo.itemData(idx)
        self._detector.selected_class = data
        self.log_table.setRowCount(0)
        self._row_by_tid.clear()
        self._reset_track_filter()
        log.info("Filter class: %s", data or ALL_CLASSES_LABEL)

    # --------------------------------------------------------- track filter
    def _visible_detections(self, detections: list) -> list:
        if self._selected_track_id is None:
            return detections
        return [d for d in detections
                if getattr(d, "track_id", -1) == self._selected_track_id]

    def _reset_track_filter(self) -> None:
        self._selected_track_id = None
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        self.track_combo.addItem(ALL_TRACKS_LABEL, userData=None)
        self.track_combo.blockSignals(False)

    def _sync_track_combo(self, active_ids: set[int]) -> None:
        self.track_combo.blockSignals(True)

        for i in range(self.track_combo.count() - 1, 0, -1):
            if self.track_combo.itemData(i) not in active_ids:
                self.track_combo.removeItem(i)

        existing = {self.track_combo.itemData(j)
                    for j in range(1, self.track_combo.count())}
        for tid in sorted(active_ids):
            if tid not in existing:
                self.track_combo.addItem(f"#{tid}", userData=tid)

        if (self._selected_track_id is not None and
                self._selected_track_id not in active_ids):
            self._selected_track_id = None
            self.track_combo.setCurrentIndex(0)

        self.track_combo.blockSignals(False)

    @pyqtSlot(int)
    def _on_track_activated(self, idx: int) -> None:
        self._selected_track_id = self.track_combo.itemData(idx)
        if self._last_color is not None:
            self.video_widget.update_frame(
                self._last_color,
                self._visible_detections(self._last_detections),
            )
        tid_str = (f"#{self._selected_track_id}"
                   if self._selected_track_id else "tất cả")
        self.status_message.emit(f"Track filter: {tid_str}")
        log.info("Track filter: %s", self._selected_track_id)

    @pyqtSlot(int, int)
    def _on_log_row_clicked(self, row: int, _col: int) -> None:
        tid = next((t for t, r in self._row_by_tid.items() if r == row), None)
        if tid is None:
            return
        idx = self.track_combo.findData(tid)
        if idx >= 0:
            self.track_combo.setCurrentIndex(idx)
            self._on_track_activated(idx)

    def _on_conf_changed(self, value: int) -> None:
        snapped = int(round(value / 5.0) * 5)
        if snapped != value:
            self.conf_slider.blockSignals(True)
            self.conf_slider.setValue(snapped)
            self.conf_slider.blockSignals(False)
            value = snapped

        conf = value / 100.0
        self.conf_label.setText(f"{value}%")
        if self._detector is not None:
            self._detector.conf_threshold = conf

    def _on_pause_toggled(self, checked: bool) -> None:
        """Pause/Resume inference của detector gắn với panel này."""
        self.set_paused(checked, from_user=True)

    def set_paused(self, checked: bool, *, from_user: bool = False) -> None:
        self._paused = checked
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(checked)
        self.pause_btn.setText("▶ Resume" if checked else "⏸ Pause")
        self.pause_btn.blockSignals(False)

        if self._detector is not None:
            self._detector.paused = checked

        if checked:
            self._last_detections = []
            if self._last_color is not None:
                self.video_widget.update_frame(self._last_color, [])
            self.log_table.setRowCount(0)
            self._row_by_tid.clear()
            self.status_message.emit(
                "Đã Pause — nguồn vẫn capture, YOLO tạm dừng."
            )
        else:
            self.status_message.emit("Đã Resume — YOLO tiếp tục inference.")

        log.info("Pause toggled: %s (user=%s)", checked, from_user)

    def clear_log(self) -> None:
        self.log_table.setRowCount(0)
        self._row_by_tid.clear()

    # ----------------------------------------------------------------- log
    def _append_log(self, detections: list[Detection]) -> None:
        now = datetime.now().strftime("%H:%M:%S")

        active = [d for d in detections if getattr(d, "track_id", -1) > 0]
        active_ids = {d.track_id for d in active}

        self._sync_track_combo(active_ids)

        stale_tids = [tid for tid in self._row_by_tid if tid not in active_ids]
        for tid in stale_tids:
            row = self._row_by_tid.pop(tid)
            self.log_table.removeRow(row)
            for k, v in self._row_by_tid.items():
                if v > row:
                    self._row_by_tid[k] = v - 1

        for det in active:
            tid = det.track_id
            row = self._row_by_tid.get(tid)
            if row is None:
                row = self.log_table.rowCount()
                self.log_table.insertRow(row)
                self._row_by_tid[tid] = row

            id_item = _centered(f"#{tid}")
            font = id_item.font(); font.setBold(True); id_item.setFont(font)
            self.log_table.setItem(row, 0, id_item)

            b, g, r = color_for_class(det.class_name)
            class_item = QTableWidgetItem(det.class_name)
            class_item.setForeground(QColor(r, g, b))
            cf = class_item.font(); cf.setBold(True); class_item.setFont(cf)
            self.log_table.setItem(row, 1, class_item)

            self.log_table.setItem(row, 2, _centered(f"{det.conf:.2f}"))
            self.log_table.setItem(row, 3, _centered(_fmt_mm(det.x_mm)))
            self.log_table.setItem(row, 4, _centered(_fmt_mm(det.y_mm)))
            self.log_table.setItem(row, 5, _centered(_fmt_mm(det.z_mm)))
            self.log_table.setItem(row, 6, _centered(now))

    # --------------------------------------------------------------- status
    def _tick_status(self) -> None:
        if self._paused:
            return
        self._frame_counter += 1
        t = time.perf_counter()
        if t - self._last_status_t >= 1.0:
            fps = self._frame_counter / (t - self._last_status_t)
            self._last_status_t = t
            self._frame_counter = 0
            self.status_message.emit(f"FPS (UI): {fps:.1f}")


def _fmt_mm(v: float) -> str:
    """Format giá trị mm cho table. NaN/None → '---' (video không có depth)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{v:+.0f}"


def _centered(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    return item
