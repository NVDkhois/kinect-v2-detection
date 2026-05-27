"""
TrainingThread — chạy YOLO training trong QThread riêng, không block Qt.

`model.train()` là blocking call → đặt trong `run()` của QThread. Đúng thiết kế.
Callback Ultralytics emit Qt signal cross-thread → Qt tự dùng queued delivery.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PyQt5.QtCore import QThread, Qt, pyqtSignal

from config import (
    ARCH_YAML_MAP,
    TRAIN_BASE_MODEL,
    TRAIN_BATCH,
    TRAIN_DEVICE,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_OUTPUT_DIR,
    TRAIN_PATIENCE,
    TRAIN_WORKERS,
)
from training.callbacks import TrainingSignals, YOLOTrainingCallback
from training.validator import DatasetInfo

log = logging.getLogger("trainer")


class _QtLogStream:
    """
    File-like: gom ký tự, forward TỪNG DÒNG hoàn chỉnh qua callback.

    tqdm dùng '\\r' (không '\\n') cho progress bar → các cập nhật trung
    gian bị gộp; chỉ lấy đoạn sau '\\r' cuối khi gặp '\\n' → tránh flood
    ô Training log mà vẫn giữ dòng tổng kết mỗi epoch của ultralytics.
    """

    encoding = "utf-8"

    def __init__(self, emit_line) -> None:
        self._emit = emit_line
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            line = line.split("\r")[-1].strip()
            if line:
                try:
                    self._emit(line)
                except Exception:
                    pass
        # Chặn buffer phình vô hạn nếu dòng dài bất thường không có '\n'
        if len(self._buf) > 8192:
            self._buf = self._buf[-1024:]
        return len(s)

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


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------
@dataclass
class TrainParams:
    """Hyperparameter cho một lần training. Defaults từ config."""

    epochs: int        = TRAIN_EPOCHS
    batch: int         = TRAIN_BATCH
    imgsz: int         = TRAIN_IMGSZ
    base_model: str    = TRAIN_BASE_MODEL
    output_dir: str    = TRAIN_OUTPUT_DIR
    output_name: str   = "custom_v1"   # sub-folder trong output_dir
    patience: int      = TRAIN_PATIENCE
    workers: int       = TRAIN_WORKERS
    device: str        = TRAIN_DEVICE
    extra: dict        = field(default_factory=dict)

    # ---- Pretrained / from-scratch toggle ----
    pretrained: bool   = True          # True = fine-tune .pt, False = train from .yaml

    # ---- LR / warmup (fine-tune defaults; from-scratch ghi đè qua UI) ----
    warmup_epochs: int = 3
    lr0: float         = 0.01
    lrf: float         = 0.01          # lr_final = lr0 * lrf

    def __str__(self) -> str:
        mode = "fine-tune" if self.pretrained else "from-scratch"
        return (
            f"[{mode}] base={self.base_model} epochs={self.epochs} "
            f"batch={self.batch} imgsz={self.imgsz} device={self.device}"
        )


# ---------------------------------------------------------------------------
# Thread
# ---------------------------------------------------------------------------
class TrainingThread(QThread):
    """
    Chạy ultralytics model.train() trong thread riêng.

    Training i3-10105F + GTX 1060 3GB: ~15-25 phút / 50 epoch với yolov8n.

    Signals:
        progress(current_epoch, total_epochs, metrics)
        finished(best_model_path)
        error(message)
        log_message(text)
    """

    progress = pyqtSignal(int, int, dict)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, dataset_info: DatasetInfo, params: TrainParams,
                 parent=None) -> None:
        super().__init__(parent)
        self._info = dataset_info
        self._params = params
        self._stop_event = threading.Event()
        self._signals = TrainingSignals()

        # Kết nối internal signals → re-emit dưới dạng TrainingThread signals
        # Dùng QueuedConnection vì _signals sống ở main thread, callback ở trainer thread
        self._signals.epoch_end.connect(self._on_epoch_end, Qt.QueuedConnection)
        self._signals.fit_end.connect(self._on_fit_end, Qt.QueuedConnection)
        self._signals.error.connect(self._on_error, Qt.QueuedConnection)
        self._signals.log_line.connect(self.log_message, Qt.QueuedConnection)

        self._total_epochs = params.epochs
        self._epoch_times: list[float] = []
        self._last_epoch_t: float = 0.0

    # -------------------------------------------------------- internal slots
    def _on_epoch_end(self, epoch: int, metrics: dict) -> None:
        now = time.perf_counter()
        if self._last_epoch_t > 0:
            self._epoch_times.append(now - self._last_epoch_t)
        self._last_epoch_t = now
        self.progress.emit(epoch, self._total_epochs, metrics)

    def _on_fit_end(self, path: str) -> None:
        self.finished.emit(path)

    def _on_error(self, msg: str) -> None:
        self.error.emit(msg)

    # ------------------------------------------------------------------ run
    def run(self) -> None:  # noqa: D401
        """Vòng chạy training — blocking trong QThread, không block Qt event loop."""
        p = self._params
        log.info("Bắt đầu training: %s", p)
        self.log_message.emit(f"Training bắt đầu: {p}")

        # Resolve output_dir về absolute (tránh CWD phụ thuộc)
        project_root = Path(__file__).resolve().parent.parent
        output_dir = Path(p.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        cb = YOLOTrainingCallback(
            signals=self._signals,
            stop_event=self._stop_event,
            total_epochs=p.epochs,
        )

        # Redirect stdout/stderr → forward output ultralytics/tqdm vào ô
        # Training log. Cũng fix crash khi chạy windowed (stdout=None).
        #
        # GIẢ ĐỊNH (M4): đây là swap CẤP PROCESS, không thread-safe. An
        # toàn vì:
        #   1. Mọi thread khác dùng `logging` (handler đã bind stream GỐC
        #      lúc _setup_logging, KHÔNG bị ảnh hưởng bởi swap này).
        #   2. PAUSE_DETECTION_DURING_TRAINING=True (mặc định) → thread
        #      capture/detector đã suspend khi training chạy.
        # Nếu user đặt cờ đó = False: print() trực tiếp (nếu có) từ thread
        # khác trong lúc train sẽ lọt vào ô Training log — chấp nhận được,
        # codebase không dùng print() ở hot path.
        _stream = _QtLogStream(self.log_message.emit)
        _old_out, _old_err = sys.stdout, sys.stderr
        sys.stdout = _stream
        sys.stderr = _stream

        try:
            from ultralytics import YOLO  # type: ignore

            # ── Chọn nguồn model theo chế độ ──────────────────────────────
            if p.pretrained:
                model_source = p.base_model          # "yolov8n.pt"
                self.log_message.emit(
                    f"[Fine-tune] Base: {p.base_model} | Epochs: {p.epochs}"
                )
            else:
                # Lấy file .yaml kiến trúc tương ứng (không có weights)
                yaml_name = ARCH_YAML_MAP.get(
                    p.base_model,
                    p.base_model.replace(".pt", ".yaml"),
                )
                model_source = yaml_name             # "yolov8n.yaml"
                self.log_message.emit(
                    f"[From scratch] Arch: {yaml_name} | "
                    f"Epochs: {p.epochs} | Warmup: {p.warmup_epochs}"
                )

            model = YOLO(model_source)
            model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
            model.add_callback("on_train_end", cb.on_train_end)

            # Loại bỏ keys trùng với explicit params để tránh TypeError
            _explicit = {"warmup_epochs", "lr0", "lrf"}
            safe_extra = {k: v for k, v in p.extra.items() if k not in _explicit}

            # Resolve device: TRAIN_DEVICE mặc định "0" (GPU index) nhưng máy
            # không CUDA (vd RX 580) → ép "cpu". torch KHÔNG train được trên
            # DirectML; ONNX EP chỉ phục vụ inference. Train trên AMD = CPU (chậm).
            from core import device as _dev
            train_device = p.device
            if str(p.device) not in ("cpu",) and not _dev._has_cuda():
                train_device = "cpu"
                log.warning(
                    "TRAIN_DEVICE=%r nhưng không có CUDA — train trên CPU (chậm). "
                    "GPU AMD không train được qua torch.", p.device,
                )
                self.log_message.emit(
                    "⚠ Không có GPU CUDA — training chạy CPU (chậm). "
                    "Cân nhắc giảm epochs/batch."
                )

            self._last_epoch_t = time.perf_counter()
            model.train(
                data=str(self._info.yaml_path),
                epochs=p.epochs,
                batch=p.batch,
                imgsz=p.imgsz,
                warmup_epochs=p.warmup_epochs,
                lr0=p.lr0,
                lrf=p.lrf,
                patience=p.patience,
                workers=p.workers,
                device=train_device,
                project=str(output_dir),
                name=p.output_name,
                exist_ok=True,
                verbose=False,
                **safe_extra,
            )

        except torch.cuda.OutOfMemoryError:
            msg = (
                "CUDA Out of Memory — thử giảm batch size (batch=4) "
                "hoặc dùng device='cpu'."
            )
            log.error(msg)
            self.error.emit(msg)
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            log.warning("Training bị interrupt (KeyboardInterrupt).")
            self.log_message.emit("⚠ Training đã bị dừng (interrupt).")
            torch.cuda.empty_cache()

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            msg = f"Training lỗi: {exc}\n{tb}"
            log.error(msg)
            self.error.emit(str(exc))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        finally:
            # Luôn khôi phục stdout/stderr gốc dù train OK hay lỗi
            sys.stdout, sys.stderr = _old_out, _old_err

        log.info("TrainingThread kết thúc.")

    # ------------------------------------------------------------------ stop
    def stop(self) -> None:
        """
        Yêu cầu dừng training sau epoch hiện tại (clean stop).

        Cơ chế: set threading.Event → callback đặt trainer.stop=True sau
        epoch → ultralytics kết thúc cleanly. Không dùng os.kill để tránh
        gửi SIGINT tới toàn bộ process (sẽ crash Qt main thread).
        """
        log.warning("Stop requested — training sẽ dừng sau epoch hiện tại.")
        self._stop_event.set()
