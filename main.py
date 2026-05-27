"""
KinectVision — entry point.

Khởi tạo app, dây nối thread tab Detection (Kinect live):
    KinectCaptureThread ─► FrameRouter ─┬─► detector_queue ─► DetectionThread
                                        │                          │
                                        └─► UI (color+depth)        ▼
                                                            DetectionRouter ─► UI

Tab Video (VideoFileCaptureThread) và tab Training (TrainingThread) được tạo
+ wiring nội bộ trong panel tương ứng (ui/video_panel, ui/training_panel),
không khởi tạo ở đây.

Graceful shutdown khi đóng cửa sổ chính: stop capture + detector thread + wait.
"""

from __future__ import annotations

# QUAN TRỌNG: torch PHẢI được import TRƯỚC PyQt5 trên Windows.
# Nếu PyQt5 load trước, Qt DLLs set up DLL search path khiến c10.dll
# của torch fail init ("WinError 1114"). Đã verify thực nghiệm trên
# Python 3.10 + torch 2.4.1 + PyQt5 5.15. Đừng đảo thứ tự import này.
import torch  # noqa: F401  (import-order side effect)

import logging
import os
import queue
import sys
import time
from pathlib import Path

# Tắt auto-install của ultralytics TRƯỚC khi nó được import (ở core.detector/
# training). Nếu bật, ultralytics có thể tự pip-install onnxruntime (CPU) đè
# lên onnxruntime-directml → hỏng backend ONNX-DML (xem docs/ke_hoach_chuyen_rx580.md).
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL, QUEUE_MAXSIZE
from core.detector import DetectionThread
from core.kinect_capture import KinectCaptureThread
from ui.main_window import MainWindow


class _NullStream:
    """
    File-like an toàn thay cho sys.stdout/stderr khi = None.

    pythonw.exe (chạy windowed, không console) → sys.stdout/stderr = None.
    ultralytics/tqdm ghi ra stdout/stderr → "'NoneType' object has no
    attribute 'write'". Gán stream này để mọi write thành no-op an toàn.
    """

    encoding = "utf-8"

    def write(self, s: str) -> int:
        return len(s) if s else 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("no fileno")


def _ensure_std_streams() -> None:
    """
    Đảm bảo sys.stdout/stderr KHÔNG None.

    PHẢI gọi trước _setup_logging() và trước khi import ultralytics —
    logging.StreamHandler & ultralytics.LOGGER bind stream lúc tạo; nếu
    lúc đó None thì handler hỏng vĩnh viễn (crash khi training emit log).
    """
    if sys.stdout is None:
        sys.stdout = _NullStream()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullStream()  # type: ignore[assignment]


def _setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    # Ghi log ra file để chẩn đoán khi chạy windowed (không có console).
    # mode='w' → mỗi lần chạy app làm mới log, dễ đọc.
    try:
        log_path = Path(__file__).resolve().parent / "kinectvision.log"
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handlers.append(fh)
    except Exception:
        pass
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
    )


def _warmup_torch() -> None:
    """
    Pre-import torch & init context inference trong main thread, rẽ nhánh
    theo backend khả dụng (CUDA / ONNX-DirectML / CPU — xem core/device).

    Lý do BẮT BUỘC: nếu torch lần đầu được import từ QThread (DetectionThread),
    c10.dll/CUDA context init có thể fail trên Windows
    ("DLL initialization routine failed"). Khởi tạo từ main thread tránh
    được lỗi này và đồng thời pre-load context để inference đầu tiên không stall.
    """
    from core import device

    log = logging.getLogger("main")
    backend = device.detect_backend()
    log.info("Backend inference: %s", device.describe())

    try:
        if backend == "cuda":
            torch.cuda.init()
            torch.zeros(1, device="cuda:0")  # ép tạo CUDA context
            log.info("CUDA warmup OK (device=%s).", torch.cuda.get_device_name(0))
        elif backend == "onnx-dml":
            # Backend ONNX-DirectML (vd RX 580): warmup ở ONNXDetector.load()
            # (tạo InferenceSession). Không cần init torch.cuda.
            log.info("ONNX·DirectML — GPU AMD/Intel; bỏ qua warmup CUDA.")
        else:
            log.warning("Không có GPU tăng tốc — inference sẽ chạy CPU (chậm).")
    except Exception as exc:
        log.warning("warmup lỗi: %s — detector sẽ tự fallback.", exc)


class FrameRouter(QObject):
    """
    Cầu nối từ KinectCaptureThread → detector queue + UI signal.
    """

    forwarded_to_ui = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(
        self,
        detector_queue: queue.Queue,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._detector_queue = detector_queue

    def on_frame(self, color: np.ndarray, depth: np.ndarray) -> None:
        """Nhận từ KinectCaptureThread.frame_ready."""
        # Bơm vào detector queue (drop frame cũ nếu đầy).
        # Tách "drop cũ" và "queue mới" thành 2 try độc lập: nếu consumer
        # vừa làm rỗng queue giữa chừng khiến get_nowait() raise queue.Empty,
        # put_nowait() vẫn được thử — tránh mất frame mới một cách im lặng.
        try:
            self._detector_queue.put_nowait((color, depth))
        except queue.Full:
            try:
                self._detector_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._detector_queue.put_nowait((color, depth))
            except queue.Full:
                pass  # consumer vừa chiếm slot trở lại; frame tiếp theo sẽ vào

        # Forward sang UI
        self.forwarded_to_ui.emit(color, depth)


class DetectionRouter(QObject):
    """
    Cầu nối DetectionThread → UI signal.
    """

    forwarded_to_ui = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def on_detections(self, detections: list) -> None:
        self.forwarded_to_ui.emit(detections)


def main() -> int:
    _ensure_std_streams()  # PHẢI trước _setup_logging & import ultralytics
    _setup_logging()
    log = logging.getLogger("main")
    log.info("Khởi động KinectVision...")

    _warmup_torch()

    # Khôi phục custom model đã train/load từ lần chạy trước.
    # PHẢI trước MainWindow() — _populate_model_combo() gọi
    # ModelFactory.list_available() đọc _cfg.CUSTOM_MODEL_PATH tại thời điểm đó.
    try:
        import app_state
        import config as _cfg

        saved = app_state.get_custom_model_path()
        if saved and Path(saved).is_file():
            _cfg.CUSTOM_MODEL_PATH = saved
            saved_names = app_state.get_custom_class_names()
            if saved_names:
                _cfg.CUSTOM_CLASS_NAMES = saved_names
            log.info("Khôi phục custom model: %s (%d class)",
                     saved, len(saved_names))
        elif saved:
            log.warning("Custom model đã lưu không còn tồn tại: %s", saved)
    except Exception as exc:  # noqa: BLE001
        log.warning("Khôi phục user_state lỗi (bỏ qua): %s", exc)

    app = QApplication(sys.argv)

    # ---- Queues ----
    detector_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # ---- Threads ----
    capture = KinectCaptureThread()
    detector = DetectionThread(detector_queue)

    # ---- Routers ----
    frame_router = FrameRouter(detector_queue)
    det_router = DetectionRouter()

    # ---- UI ----
    window = MainWindow()
    window.attach_threads(detector)

    # ---- Signal wiring ----
    capture.frame_ready.connect(frame_router.on_frame)
    frame_router.forwarded_to_ui.connect(window.on_frame_ready)

    detector.detections_ready.connect(det_router.on_detections)
    det_router.forwarded_to_ui.connect(window.on_detections_ready)

    # ---- Start threads ----
    capture.start()
    detector.start()

    # Nạp danh sách class sau khi detector load xong (poll vài giây).
    # Sau khi nạp xong → auto-switch về backend đã lưu (nếu khác "yolo").
    def _populate_when_ready(attempts: int = 50) -> None:
        names = detector.class_names
        if names:
            window.populate_classes(names)
            log.info("Đã nạp %d classes vào UI.", len(names))
            # Auto-switch backend: khôi phục lựa chọn backend lần trước.
            try:
                saved_backend = app_state.get_active_backend()
                if saved_backend and saved_backend != "yolo":
                    log.info("Auto-switch backend → %s (từ user_state).", saved_backend)
                    # Delay nhỏ để UI settle trước khi switch
                    QTimer.singleShot(300, lambda: window._request_backend(saved_backend))
            except Exception as exc:
                log.warning("Auto-switch backend lỗi (bỏ qua): %s", exc)
            return
        if attempts <= 0:
            # Model chưa load xong sau 10s fast-poll → chuyển sang slow poll.
            # Tình huống: YOLO/custom load lâu (>10s) do disk chậm / CPU warmup.
            # Thay vì bỏ cuộc, retry mỗi 3s để auto-switch vẫn chạy khi model sẵn.
            log.warning("Detector load chậm — chuyển sang slow poll (3s/lần).")
            QTimer.singleShot(3000, lambda: _populate_when_ready(10))
            return
        QTimer.singleShot(200, lambda: _populate_when_ready(attempts - 1))

    QTimer.singleShot(500, _populate_when_ready)

    window.show()

    # ---- Graceful shutdown ----
    def _shutdown() -> None:
        log.info("Đang dừng các thread...")
        capture.stop()
        detector.stop()
        t0 = time.perf_counter()
        # Detector cần timeout lớn hơn: stop() có thể rơi vào giữa
        # detector.load() (~4s, không ngắt được) → thread chỉ thoát ngay
        # sau khi load xong.
        for th, name, timeout_ms in (
            (capture, "capture", 2000),
            (detector, "detector", 5000),
        ):
            if not th.wait(timeout_ms):
                log.warning(
                    "Thread %s không dừng kịp (>%ds).", name, timeout_ms // 1000
                )
        log.info("Đã dừng threads sau %.2fs.", time.perf_counter() - t0)

    app.aboutToQuit.connect(_shutdown)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
