"""
TrainingPanel — widget chính của tab Training.

Layout: 2 cột
  Trái: Dataset + Hyperparameters + Start/Stop
  Phải: Progress (progressbar + LossChart) + Log + Kết quả
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import config as _cfg
from processing.face_crop import crop_face_tight
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QComboBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from training.trainer import TrainParams, TrainingThread
from training.validator import DatasetInfo, validate_dataset
from ui.loss_chart import LossChart

log = logging.getLogger("training_panel")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TrainingPanel(QWidget):
    """
    Widget tab Training. Emit signals lên MainWindow để điều phối pause/resume
    detection và cập nhật model combo.
    """

    # ── Signals phát lên MainWindow ──────────────────────────────────────
    training_started = pyqtSignal()           # training thread vừa start
    training_finished = pyqtSignal(str)       # best_model_path
    training_stopped = pyqtSignal()           # user stop / error
    model_loaded = pyqtSignal(str)            # path custom.pt đã copy xong

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dataset_info: Optional[DatasetInfo] = None
        self._thread: Optional[TrainingThread] = None
        self._best_model_path: Optional[str] = None
        self._epoch_start_t: float = 0.0
        self._epoch_durations: list[float] = []
        # Callable trả về frame BGR mới nhất từ camera (set bởi MainWindow)
        self._frame_source: Optional[Callable] = None

        self._build_ui()

    # ============================================================= BUILD UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        root.addLayout(self._build_left(), 0)   # cột trái, fixed width
        root.addLayout(self._build_right(), 1)  # cột phải, stretch

    # ------------------------------------------------------------ cột trái
    def _build_left(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)
        col.addWidget(self._build_dataset_group())
        col.addWidget(self._build_params_group())
        col.addWidget(self._build_action_buttons())
        col.addWidget(self._build_template_group())
        col.addStretch(1)
        return col

    def _build_dataset_group(self) -> QGroupBox:
        gb = QGroupBox("Dataset")
        layout = QVBoxLayout(gb)

        # Path picker
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Chưa chọn folder…")
        self.path_edit.setMinimumWidth(220)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dataset)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        # Validation status
        self.dataset_status = QLabel()
        self.dataset_status.setWordWrap(True)
        layout.addWidget(self.dataset_status)

        # Stat cards
        stat_row = QHBoxLayout()
        self.lbl_train_count = QLabel("Train: —")
        self.lbl_val_count = QLabel("Val: —")
        self.lbl_class_count = QLabel("Classes: —")
        for lbl in (self.lbl_train_count, self.lbl_val_count, self.lbl_class_count):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "background:#eef;border:1px solid #ccd;border-radius:4px;padding:2px 6px;"
            )
            stat_row.addWidget(lbl)
        layout.addLayout(stat_row)

        # Class names
        self.lbl_classes = QLabel()
        self.lbl_classes.setWordWrap(True)
        self.lbl_classes.setStyleSheet("color:#555;font-size:11px;")
        layout.addWidget(self.lbl_classes)

        return gb

    def _build_params_group(self) -> QGroupBox:
        gb = QGroupBox("Hyperparameters")
        col = QVBoxLayout(gb)
        col.setSpacing(6)

        # ── Chế độ training ───────────────────────────────────────────────
        radio_row = QHBoxLayout()
        self.pretrained_radio = QRadioButton("Fine-tune (pretrained)")
        self.scratch_radio    = QRadioButton("Train from scratch")
        self.pretrained_radio.setChecked(True)
        self.pretrained_radio.setToolTip(
            "Dùng weights YOLOv8 đã train sẵn trên COCO làm điểm xuất phát.\n"
            "Hội tụ nhanh, cần ít ảnh hơn (~100/class)."
        )
        self.scratch_radio.setToolTip(
            "Khởi tạo weights ngẫu nhiên từ file .yaml kiến trúc.\n"
            "Cần nhiều ảnh hơn (~500+/class) và epochs nhiều hơn."
        )
        radio_row.addWidget(self.pretrained_radio)
        radio_row.addWidget(self.scratch_radio)
        radio_row.addStretch(1)
        col.addLayout(radio_row)

        # Cảnh báo dataset nhỏ khi dùng from-scratch (ẩn mặc định)
        self.scratch_warn = QLabel()
        self.scratch_warn.setWordWrap(True)
        self.scratch_warn.setStyleSheet(
            "color:#856404;background:#fff3cd;"
            "border:1px solid #ffc107;border-radius:4px;padding:4px 8px;"
        )
        self.scratch_warn.setVisible(False)
        col.addWidget(self.scratch_warn)

        # ── Form params ───────────────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(6)
        col.addLayout(form)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(_cfg.TRAIN_EPOCHS)
        form.addRow("Epochs:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 32)
        self.spin_batch.setValue(_cfg.TRAIN_BATCH)
        self.spin_batch.setToolTip("Batch 8 = ~2GB VRAM (safe cho GTX 1060 3GB)")
        form.addRow("Batch size:", self.spin_batch)

        # Label "Base model" ↔ "Architecture" thay đổi theo chế độ
        self.lbl_base_model = QLabel("Base model:")
        self.combo_base = QComboBox()
        self._populate_base_model_combo()
        form.addRow(self.lbl_base_model, self.combo_base)

        self.edit_name = QLineEdit("custom_v1")
        self.edit_name.setToolTip("Tên sub-folder output, ví dụ 'custom_v1'")
        form.addRow("Output name:", self.edit_name)

        # Wire radio toggle
        self.pretrained_radio.toggled.connect(self._on_mode_changed)

        return gb

    def _build_action_buttons(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)

        self.start_btn = QPushButton("▶  Start training")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;font-weight:bold;"
            "padding:6px 14px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:enabled{background:#388e3c;}"
        )
        self.start_btn.clicked.connect(self._on_start)
        row.addWidget(self.start_btn, 1)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#c62828;color:white;padding:6px 14px;"
            "border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:enabled{background:#d32f2f;}"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self.stop_btn)

        return w

    # ----------------------------------------------------------- cột phải
    def _build_right(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)
        col.addWidget(self._build_progress_group(), 1)
        col.addWidget(self._build_log_group())
        col.addWidget(self._build_result_group())
        return col

    def _build_progress_group(self) -> QGroupBox:
        gb = QGroupBox("Progress")
        layout = QVBoxLayout(gb)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("Chưa bắt đầu.")
        self.lbl_progress.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_progress)

        self.loss_chart = LossChart()
        self.loss_chart.setMinimumHeight(160)
        layout.addWidget(self.loss_chart, 1)

        return gb

    def _build_log_group(self) -> QGroupBox:
        gb = QGroupBox("Training log")
        layout = QVBoxLayout(gb)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(130)
        self.log_edit.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        layout.addWidget(self.log_edit)
        return gb

    def _build_result_group(self) -> QGroupBox:
        self.result_group = QGroupBox("Kết quả")
        self.result_group.setVisible(False)
        layout = QVBoxLayout(self.result_group)

        self.lbl_best_path = QLabel()
        self.lbl_best_path.setWordWrap(True)
        self.lbl_best_path.setStyleSheet("font-size:11px;color:#333;")
        layout.addWidget(self.lbl_best_path)

        self.lbl_best_metrics = QLabel()
        layout.addWidget(self.lbl_best_metrics)

        self.load_btn = QPushButton("⚡  Load model ngay vào Custom backend")
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet(
            "QPushButton{background:#1565C0;color:white;font-weight:bold;"
            "padding:6px 14px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:enabled{background:#1976D2;}"
        )
        self.load_btn.clicked.connect(self._on_load_model)
        layout.addWidget(self.load_btn)

        return self.result_group

    # -------------------------------------------------- template manager
    def _build_template_group(self) -> QGroupBox:
        gb = QGroupBox("Template Manager")
        layout = QVBoxLayout(gb)
        layout.setSpacing(6)

        # ── Tên người ──────────────────────────────────────────────────────
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Tên:"))
        self.tmpl_name_edit = QLineEdit()
        self.tmpl_name_edit.setPlaceholderText("VD: NguyenVanA")
        self.tmpl_name_edit.textChanged.connect(self._refresh_template_list)
        name_row.addWidget(self.tmpl_name_edit, 1)
        layout.addLayout(name_row)

        # ── Nút chọn ảnh & thêm / quét camera ─────────────────────────────
        btn_row = QHBoxLayout()

        add_btn = QPushButton("📂  Chọn ảnh")
        add_btn.setToolTip(
            "Chọn một hoặc nhiều ảnh từ file.\n"
            "Hệ thống tự phát hiện khuôn mặt, crop chặt và lưu vào templates/."
        )
        add_btn.setStyleSheet(
            "QPushButton{background:#1565C0;color:white;font-weight:bold;"
            "padding:5px 10px;border-radius:4px;}"
            "QPushButton:hover{background:#1976D2;}"
        )
        add_btn.clicked.connect(self._on_add_templates)
        btn_row.addWidget(add_btn)

        self._scan_btn = QPushButton("📷  Quét camera")
        self._scan_btn.setToolTip(
            "Mở cửa sổ quét khuôn mặt trực tiếp từ Kinect/webcam.\n"
            "Đếm ngược 3-2-1 rồi tự động chụp 5 ảnh."
        )
        self._scan_btn.setStyleSheet(
            "QPushButton{background:#6a1b9a;color:white;font-weight:bold;"
            "padding:5px 10px;border-radius:4px;}"
            "QPushButton:hover{background:#7b1fa2;}"
        )
        self._scan_btn.clicked.connect(self._on_scan_from_camera)
        btn_row.addWidget(self._scan_btn)

        layout.addLayout(btn_row)

        # ── Status sau khi xử lý ───────────────────────────────────────────
        self.tmpl_status = QLabel()
        self.tmpl_status.setWordWrap(True)
        self.tmpl_status.setStyleSheet("font-size:11px;color:#555;")
        layout.addWidget(self.tmpl_status)

        # ── Danh sách templates hiện tại ───────────────────────────────────
        lbl_title = QLabel("Templates hiện tại:")
        lbl_title.setStyleSheet("font-size:11px;font-weight:bold;")
        layout.addWidget(lbl_title)

        self.tmpl_list_lbl = QLabel("—")
        self.tmpl_list_lbl.setWordWrap(True)
        self.tmpl_list_lbl.setStyleSheet(
            "font-size:11px;color:#333;"
            "background:#f5f5f5;border:1px solid #ddd;"
            "border-radius:3px;padding:4px 6px;"
        )
        layout.addWidget(self.tmpl_list_lbl)

        # ── Nút xóa templates của tên hiện tại ────────────────────────────
        del_btn = QPushButton("🗑  Xóa templates của tên này")
        del_btn.setStyleSheet(
            "QPushButton{color:#c62828;border:1px solid #c62828;"
            "padding:4px 10px;border-radius:4px;background:white;}"
            "QPushButton:hover{background:#ffebee;}"
        )
        del_btn.clicked.connect(self._on_delete_templates)
        layout.addWidget(del_btn)

        self._refresh_template_list()
        return gb

    # ========================================================= HELPERS
    def _populate_base_model_combo(self) -> None:
        """Scan thư mục gốc tìm file *.pt để populate combo."""
        pts = sorted(_PROJECT_ROOT.glob("*.pt"), key=lambda p: p.name)
        for pt in pts:
            self.combo_base.addItem(pt.name, userData=str(pt))
        if not pts:
            self.combo_base.addItem(_cfg.TRAIN_BASE_MODEL,
                                    userData=_cfg.TRAIN_BASE_MODEL)
        # Set default TRAIN_BASE_MODEL nếu có
        idx = self.combo_base.findText(_cfg.TRAIN_BASE_MODEL)
        if idx >= 0:
            self.combo_base.setCurrentIndex(idx)

    def _on_mode_changed(self) -> None:
        """
        Radio toggle Pretrained ↔ From-scratch.

        Pretrained → epochs về TRAIN_EPOCHS (50), label "Base model:".
        From scratch → epochs về TRAIN_SCRATCH_EPOCHS (200),
                       label "Architecture:" (vì chọn file .yaml, không phải .pt).
        Gọi _check_scratch_warning() để hiện/ẩn cảnh báo dataset nhỏ.
        """
        is_scratch = self.scratch_radio.isChecked()
        if is_scratch:
            self.spin_epochs.setValue(_cfg.TRAIN_SCRATCH_EPOCHS)
            self.lbl_base_model.setText("Architecture:")
        else:
            self.spin_epochs.setValue(_cfg.TRAIN_EPOCHS)
            self.lbl_base_model.setText("Base model:")
        self._check_scratch_warning()

    def _check_scratch_warning(self) -> None:
        """
        Hiện QLabel cảnh báo vàng nếu chế độ from-scratch VÀ dataset nhỏ
        (< TRAIN_MIN_IMAGES_SCRATCH ảnh train).
        Không disable Start — chỉ warn, không block.
        """
        if not self.scratch_radio.isChecked():
            self.scratch_warn.setVisible(False)
            return
        if self._dataset_info is None or not self._dataset_info.is_valid:
            self.scratch_warn.setVisible(False)
            return
        n = self._dataset_info.train_count
        if n < _cfg.TRAIN_MIN_IMAGES_SCRATCH:
            self.scratch_warn.setText(
                f"⚠ Cần ít nhất {_cfg.TRAIN_MIN_IMAGES_SCRATCH} ảnh/class để "
                f"train from scratch.\n"
                f"Hiện tại: {n} ảnh. Kết quả có thể không tốt."
            )
            self.scratch_warn.setVisible(True)
        else:
            self.scratch_warn.setVisible(False)

    def _build_train_params(self) -> "TrainParams":
        """
        Đọc tất cả widget hyperparameter → trả về TrainParams đúng chế độ.

        Pretrained: warmup/lr0/lrf dùng default fine-tune (3 / 0.01 / 0.01).
        From scratch: warmup=TRAIN_SCRATCH_WARMUP, lr0=LR0, lrf=LRF.
        """
        pretrained = self.pretrained_radio.isChecked()
        base = self.combo_base.currentData() or _cfg.TRAIN_BASE_MODEL
        name = self.edit_name.text().strip() or "custom_v1"

        common = dict(
            pretrained=pretrained,
            base_model=base,
            epochs=self.spin_epochs.value(),
            batch=self.spin_batch.value(),
            imgsz=_cfg.TRAIN_IMGSZ,
            output_dir=_cfg.TRAIN_OUTPUT_DIR,
            output_name=name,
            patience=_cfg.TRAIN_PATIENCE,
            workers=_cfg.TRAIN_WORKERS,
            device=_cfg.TRAIN_DEVICE,
        )
        if pretrained:
            return TrainParams(**common)
        else:
            return TrainParams(
                **common,
                warmup_epochs=_cfg.TRAIN_SCRATCH_WARMUP,
                lr0=_cfg.TRAIN_SCRATCH_LR0,
                lrf=_cfg.TRAIN_SCRATCH_LRF,
            )

    def _append_log(self, text: str) -> None:
        """Thêm dòng vào log (giới hạn 500 dòng), auto-scroll."""
        self.log_edit.append(text)
        doc = self.log_edit.document()
        max_blocks = 500
        while doc.blockCount() > max_blocks:
            cursor = self.log_edit.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # xoá newline
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )

    def _set_training_ui(self, is_training: bool) -> None:
        self.start_btn.setEnabled(not is_training and
                                  (self._dataset_info is not None) and
                                  (self._dataset_info.is_valid))
        self.stop_btn.setEnabled(is_training)
        self.combo_base.setEnabled(not is_training)
        self.spin_epochs.setEnabled(not is_training)
        self.spin_batch.setEnabled(not is_training)
        self.edit_name.setEnabled(not is_training)

    # ================================================= TEMPLATE MANAGER
    def set_frame_source(self, fn: Callable) -> None:
        """Đăng ký callable trả về frame BGR mới nhất từ camera (gọi bởi MainWindow)."""
        self._frame_source = fn

    def _tmpl_dir(self) -> Path:
        return (_PROJECT_ROOT / _cfg.TEMPLATE_DIR).resolve()

    def _next_index_for(self, name: str) -> int:
        """Tìm index kế tiếp cho tên name trong thư mục templates/."""
        tdir = self._tmpl_dir()
        exts = set(_cfg.TEMPLATE_EXTS)
        existing = [
            p for p in tdir.iterdir()
            if p.is_file()
            and p.suffix.lower() in exts
            and p.stem.lower().startswith(name.lower() + "_")
        ]
        if not existing:
            return 1
        indices = []
        for p in existing:
            suffix = p.stem[len(name) + 1:]
            if suffix.isdigit():
                indices.append(int(suffix))
        return max(indices, default=0) + 1

    def _refresh_template_list(self) -> None:
        """Quét templates/ và cập nhật label danh sách."""
        tdir = self._tmpl_dir()
        if not tdir.is_dir():
            self.tmpl_list_lbl.setText("(thư mục templates/ chưa tồn tại)")
            return

        exts = set(_cfg.TEMPLATE_EXTS)
        files = [p for p in tdir.iterdir() if p.is_file() and p.suffix.lower() in exts]

        # Gom theo tên gốc (trước dấu _ cuối nếu có số)
        groups: dict[str, int] = {}
        for p in files:
            stem = p.stem
            parts = stem.rsplit("_", 1)
            key = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            groups[key] = groups.get(key, 0) + 1

        if not groups:
            self.tmpl_list_lbl.setText("(chưa có template nào)")
            return

        cur_name = self.tmpl_name_edit.text().strip().lower()
        lines = []
        for k, cnt in sorted(groups.items()):
            marker = " ◀" if k.lower() == cur_name else ""
            lines.append(f"• {k}: {cnt} mẫu{marker}")
        self.tmpl_list_lbl.setText("\n".join(lines))

    def _on_add_templates(self) -> None:
        """Mở file dialog → crop face → lưu vào templates/."""
        name = self.tmpl_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Thiếu tên", "Nhập tên người trước khi chọn ảnh.")
            return

        # Sanitize tên: chỉ giữ chữ/số/gạch dưới
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        if not safe_name:
            QMessageBox.warning(self, "Tên không hợp lệ",
                                "Tên chỉ được chứa chữ cái, số và dấu gạch dưới.")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, "Chọn ảnh khuôn mặt",
            str(Path.home()),
            "Ảnh (*.jpg *.jpeg *.png *.bmp)"
        )
        if not files:
            return

        tdir = self._tmpl_dir()
        tdir.mkdir(parents=True, exist_ok=True)

        idx = self._next_index_for(safe_name)
        saved, skipped = 0, 0
        skip_names: list[str] = []

        for fpath in files:
            img = cv2.imread(fpath)
            if img is None:
                skipped += 1
                skip_names.append(Path(fpath).name)
                continue

            cropped = crop_face_tight(img)
            if cropped is None:
                skipped += 1
                skip_names.append(f"{Path(fpath).name} (không thấy mặt)")
                continue

            out_path = tdir / f"{safe_name}_{idx}.jpg"
            cv2.imwrite(str(out_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
            log.info("Template saved: %s (%dx%dpx)", out_path.name,
                     cropped.shape[1], cropped.shape[0])
            idx += 1
            saved += 1

        # Cập nhật status
        msg_parts = []
        if saved:
            msg_parts.append(f"✓ Đã lưu {saved} ảnh ({safe_name}_{idx-saved}–{idx-1})")
        if skipped:
            detail = ", ".join(skip_names[:3])
            if len(skip_names) > 3:
                detail += f"… (+{len(skip_names)-3})"
            msg_parts.append(f"⚠ Bỏ qua {skipped}: {detail}")
        self.tmpl_status.setText("\n".join(msg_parts))

        if saved:
            self.tmpl_status.setStyleSheet("font-size:11px;color:#2e7d32;")
        else:
            self.tmpl_status.setStyleSheet("font-size:11px;color:#c62828;")

        self._refresh_template_list()

    def _on_scan_from_camera(self) -> None:
        """Mở FaceCaptureDialog → quét mặt từ camera → lưu vào templates/."""
        name = self.tmpl_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Thiếu tên",
                                "Nhập tên người trước khi quét camera.")
            return

        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        if not safe_name:
            QMessageBox.warning(self, "Tên không hợp lệ",
                                "Tên chỉ được chứa chữ cái, số và dấu gạch dưới.")
            return

        if self._frame_source is None or self._frame_source() is None:
            QMessageBox.warning(
                self, "Camera chưa sẵn sàng",
                "Camera chưa khởi động hoặc chưa có frame.\n"
                "Hãy đảm bảo tab Detection đang hiển thị hình ảnh.",
            )
            return

        from ui.face_capture_dialog import FaceCaptureDialog
        dialog = FaceCaptureDialog(self._frame_source, safe_name, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        captures = dialog.get_captures()
        if not captures:
            self.tmpl_status.setText("⚠ Không có ảnh nào được lưu.")
            self.tmpl_status.setStyleSheet("font-size:11px;color:#c62828;")
            return

        tdir = self._tmpl_dir()
        tdir.mkdir(parents=True, exist_ok=True)
        idx = self._next_index_for(safe_name)
        saved = 0

        for cropped in captures:
            out_path = tdir / f"{safe_name}_{idx}.jpg"
            cv2.imwrite(str(out_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
            log.info("Template (camera) saved: %s (%dx%dpx)",
                     out_path.name, cropped.shape[1], cropped.shape[0])
            idx += 1
            saved += 1

        self.tmpl_status.setText(
            f"✓ Đã lưu {saved} ảnh từ camera  ({safe_name}_{idx-saved}–{idx-1})"
        )
        self.tmpl_status.setStyleSheet("font-size:11px;color:#2e7d32;")
        self._refresh_template_list()

    def _on_delete_templates(self) -> None:
        """Xóa tất cả template của tên đang nhập."""
        name = self.tmpl_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Thiếu tên",
                                "Nhập tên muốn xóa vào ô Tên trước.")
            return

        tdir = self._tmpl_dir()
        exts = set(_cfg.TEMPLATE_EXTS)
        targets = [
            p for p in tdir.iterdir()
            if p.is_file()
            and p.suffix.lower() in exts
            and p.stem.lower().startswith(name.lower() + "_")
        ]
        # Khớp cả file không có suffix số (ví dụ: "NguyenA.jpg")
        targets += [
            p for p in tdir.iterdir()
            if p.is_file()
            and p.suffix.lower() in exts
            and p.stem.lower() == name.lower()
            and p not in targets
        ]

        if not targets:
            QMessageBox.information(self, "Không tìm thấy",
                                    f"Không có template nào tên '{name}'.")
            return

        ans = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Xóa {len(targets)} file template của '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return

        for p in targets:
            try:
                p.unlink()
            except Exception as exc:
                log.warning("Không xóa được %s: %s", p.name, exc)

        self.tmpl_status.setText(f"🗑 Đã xóa {len(targets)} template của '{name}'.")
        self.tmpl_status.setStyleSheet("font-size:11px;color:#555;")
        self._refresh_template_list()

    # ====================================================== SLOTS / EVENTS
    def _browse_dataset(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Chọn folder dataset YOLO", str(_PROJECT_ROOT)
        )
        if not folder:
            return
        self.path_edit.setText(folder)
        info = validate_dataset(folder)
        self._dataset_info = info
        self._update_dataset_ui(info)

    def _update_dataset_ui(self, info: DatasetInfo) -> None:
        if info.is_valid:
            self.dataset_status.setText("✓ Dataset hợp lệ")
            self.dataset_status.setStyleSheet("color:green;font-weight:bold;")
            self.lbl_train_count.setText(f"Train: {info.train_count}")
            self.lbl_val_count.setText(f"Val: {info.val_count}")
            self.lbl_class_count.setText(f"Classes: {len(info.class_names)}")
            self.lbl_classes.setText("  ".join(info.class_names))
            self.start_btn.setEnabled(True)
            if info.warnings:
                for w in info.warnings:
                    self._append_log(f"⚠ {w}")
        else:
            self.dataset_status.setText(f"✗ {info.error}")
            self.dataset_status.setStyleSheet("color:red;")
            self.lbl_train_count.setText("Train: —")
            self.lbl_val_count.setText("Val: —")
            self.lbl_class_count.setText("Classes: —")
            self.lbl_classes.clear()
            self.start_btn.setEnabled(False)
        # Kiểm tra cảnh báo from-scratch sau mỗi lần dataset thay đổi
        self._check_scratch_warning()

    def _on_start(self) -> None:
        if self._dataset_info is None or not self._dataset_info.is_valid:
            QMessageBox.warning(self, "Dataset", "Vui lòng chọn dataset hợp lệ trước.")
            return

        # Re-validate trước khi train
        info = validate_dataset(self._dataset_info.root)
        if not info.is_valid:
            QMessageBox.critical(self, "Dataset không hợp lệ", info.error or "Unknown error")
            return
        self._dataset_info = info

        params = self._build_train_params()

        self._thread = TrainingThread(self._dataset_info, params)
        self._thread.progress.connect(self.on_epoch_end, Qt.QueuedConnection)
        self._thread.finished.connect(self.on_training_finished, Qt.QueuedConnection)
        self._thread.error.connect(self.on_training_error, Qt.QueuedConnection)
        self._thread.log_message.connect(self._append_log, Qt.QueuedConnection)

        # Reset UI
        self.progress_bar.setValue(0)
        self.lbl_progress.setText("Epoch 0 / …")
        self.loss_chart.reset()
        self.log_edit.clear()
        self.result_group.setVisible(False)
        self._epoch_start_t = time.perf_counter()
        self._epoch_durations.clear()
        self._best_model_path = None

        self._set_training_ui(True)
        self.start_btn.setText(f"Training… (0/{params.epochs})")
        self._append_log(f"▶ Start training — {params}")

        self._thread.start()
        self.training_started.emit()

    def _on_stop(self) -> None:
        if self._thread is None:
            return
        ans = QMessageBox.question(
            self, "Dừng training?",
            "Training sẽ dừng sau epoch hiện tại. Tiếp tục?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        self.stop_btn.setEnabled(False)
        self.start_btn.setText("Đang dừng…")
        self._thread.stop()
        self._append_log("⏹ Yêu cầu dừng training sau epoch hiện tại…")

    def _on_load_model(self) -> None:
        if not self._best_model_path:
            return
        src = Path(self._best_model_path)
        if not src.is_file():
            QMessageBox.critical(self, "Lỗi", f"best.pt không tồn tại:\n{src}")
            return

        # Tên đích = output_name.pt trong thư mục gốc project.
        # Sanitize: chỉ lấy thành phần tên cuối (Path(...).name loại bỏ mọi
        # phần thư mục, "..", và đường dẫn tuyệt đối) → chặn path traversal
        # ghi file ra ngoài project root.
        raw_name = self.edit_name.text().strip() or "custom"
        safe_stem = Path(raw_name).name.removesuffix(".pt").strip() or "custom"
        dest = (_PROJECT_ROOT / f"{safe_stem}.pt").resolve()

        # Lớp bảo vệ thứ 2: chắc chắn dest vẫn nằm trong project root.
        if _PROJECT_ROOT.resolve() not in dest.parents:
            QMessageBox.critical(
                self, "Tên không hợp lệ",
                f"Tên output không hợp lệ: {raw_name!r}",
            )
            return
        try:
            shutil.copy2(src, dest)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi copy", str(exc))
            return

        # Cập nhật config runtime để factory biết file + class names mới
        _cfg.CUSTOM_MODEL_PATH = str(dest)
        log.info("CUSTOM_MODEL_PATH updated → %s", dest)

        if self._dataset_info is not None and self._dataset_info.class_names:
            _cfg.CUSTOM_CLASS_NAMES = list(self._dataset_info.class_names)
            log.info("CUSTOM_CLASS_NAMES updated → %s", _cfg.CUSTOM_CLASS_NAMES)

        # Persist xuống đĩa để restart app vẫn nhớ (config.py reset khi import lại)
        try:
            import app_state
            app_state.set_custom_model_path(str(dest))
            if self._dataset_info is not None and self._dataset_info.class_names:
                app_state.set_custom_class_names(self._dataset_info.class_names)
            log.info("Đã lưu custom_model_path vào user_state.json")
        except Exception as exc:  # noqa: BLE001
            log.warning("Không lưu được user_state (bỏ qua): %s", exc)

        self._append_log(f"✓ Đã copy → {dest}")
        self._append_log("Đang load vào Custom backend…")

        self.load_btn.setEnabled(False)
        self.model_loaded.emit(str(dest))

    # ============================================================= SLOTS
    @pyqtSlot(int, int, dict)
    def on_epoch_end(self, epoch: int, total: int, metrics: dict) -> None:
        """Cập nhật progress bar, label, loss chart và tính ETA."""
        pct = int(epoch / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        self.start_btn.setText(f"Training… ({epoch}/{total})")

        now = time.perf_counter()
        if self._epoch_start_t > 0 and epoch > 1:
            self._epoch_durations.append(now - self._epoch_start_t)
        self._epoch_start_t = now

        eta_str = ""
        if self._epoch_durations:
            avg = sum(self._epoch_durations) / len(self._epoch_durations)
            remaining_s = avg * (total - epoch)
            eta_min = remaining_s / 60
            eta_str = f"  · ETA {eta_min:.0f} phút"

        map50 = metrics.get("mAP50", 0.0)
        self.lbl_progress.setText(
            f"Epoch {epoch}/{total}  ·  mAP50: {map50:.3f}{eta_str}"
        )
        self.loss_chart.add_epoch(epoch, metrics)

    @pyqtSlot(str)
    def on_training_finished(self, best_model_path: str) -> None:
        """Training hoàn tất — hiện kết quả và enable nút load."""
        self._best_model_path = best_model_path
        self.progress_bar.setValue(100)
        self.start_btn.setText("▶  Start training")
        self._set_training_ui(False)

        self.result_group.setVisible(True)
        self.lbl_best_path.setText(f"Best model: {best_model_path}")
        self.load_btn.setEnabled(True)

        self._append_log(f"✓ Training hoàn tất! Best: {best_model_path}")
        self.training_finished.emit(best_model_path)
        self.training_stopped.emit()

    @pyqtSlot(str)
    def on_training_error(self, message: str) -> None:
        """Xử lý lỗi — hiện dialog và re-enable UI."""
        self.start_btn.setText("▶  Start training")
        self._set_training_ui(False)
        self._append_log(f"✗ Lỗi: {message}")
        QMessageBox.critical(self, "Training lỗi", message)
        self.training_stopped.emit()

    @pyqtSlot(str)
    def notify_model_loaded_ok(self, backend_label: str) -> None:
        """Gọi từ MainWindow sau khi backend_changed xác nhận load xong."""
        self._append_log(f"✓ Custom backend sẵn sàng: {backend_label}")
        self.load_btn.setEnabled(False)
        QMessageBox.information(
            self, "Load model",
            f"Đã load model mới vào Custom backend.\n{backend_label}",
        )
