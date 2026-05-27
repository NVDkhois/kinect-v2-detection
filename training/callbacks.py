"""
Ultralytics callback bridge → Qt signals.

Callback chạy trong TrainingThread. Signals Qt tự dùng queued connection
khi emit cross-thread → an toàn, không cần lock thêm.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    pass

log = logging.getLogger("train.cb")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
class TrainingSignals(QObject):
    """
    Container signal — phải là QObject để Qt quản lý queued delivery.

    epoch_end(epoch, metrics): metrics = {box_loss, cls_loss, mAP50,
                               mAP50-95, lr}
    fit_end(best_model_path): đường dẫn tuyệt đối best.pt
    error(message)
    log_line(text): một dòng log dạng text
    """

    epoch_end = pyqtSignal(int, dict)
    fit_end = pyqtSignal(str)
    error = pyqtSignal(str)
    log_line = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
class YOLOTrainingCallback:
    """
    Đăng ký vào `model.add_callback()` để nhận event mỗi epoch.

    Args:
        signals: TrainingSignals object để emit về UI.
        stop_event: threading.Event — khi set, báo trainer dừng sau epoch.
        total_epochs: để format log.
    """

    def __init__(
        self,
        signals: TrainingSignals,
        stop_event: threading.Event,
        total_epochs: int = 0,
    ) -> None:
        self._sig = signals
        self._stop = stop_event
        self._total = total_epochs

    # ------------------------------------------------ ultralytics hooks
    def on_train_epoch_end(self, trainer) -> None:
        """Gọi sau mỗi epoch train xong. Emit metrics về UI + check stop."""
        epoch = int(getattr(trainer, "epoch", 0)) + 1  # ultralytics 0-indexed

        # Dừng clean sau epoch nếu user yêu cầu
        if self._stop.is_set():
            trainer.stop = True
            log.info("Stop flag set — trainer sẽ dừng sau epoch %d.", epoch)
            return

        metrics: dict = {}
        raw = getattr(trainer, "metrics", None) or {}
        loss_items = getattr(trainer, "loss_items", None)

        # Ultralytics >= 8.x: metrics dict có dạng "metrics/mAP50(B)" v.v.
        # Normalize keys để UI không cần biết version
        def _get(*keys: str, default: float = 0.0) -> float:
            for k in keys:
                if k in raw:
                    try:
                        return float(raw[k])
                    except (TypeError, ValueError):
                        pass
            return default

        metrics["box_loss"] = _get(
            "train/box_loss", "box_loss",
            default=float(loss_items[0]) if loss_items is not None and len(loss_items) > 0 else 0.0,
        )
        metrics["cls_loss"] = _get(
            "train/cls_loss", "cls_loss",
            default=float(loss_items[1]) if loss_items is not None and len(loss_items) > 1 else 0.0,
        )
        metrics["mAP50"] = _get("metrics/mAP50(B)", "mAP50", "val/mAP50")
        metrics["mAP50-95"] = _get("metrics/mAP50-95(B)", "mAP50-95", "val/mAP50-95")

        lr_dict = getattr(trainer, "lf", None)
        if lr_dict is None:
            opt = getattr(trainer, "optimizer", None)
            if opt and hasattr(opt, "param_groups") and opt.param_groups:
                metrics["lr"] = float(opt.param_groups[0].get("lr", 0.0))
            else:
                metrics["lr"] = 0.0
        else:
            metrics["lr"] = float(lr_dict) if isinstance(lr_dict, (int, float)) else 0.0

        log.debug("Epoch %d/%d — %s", epoch, self._total, metrics)
        self._sig.epoch_end.emit(epoch, metrics)
        self._sig.log_line.emit(
            f"Epoch {epoch}/{self._total} "
            f"| box={metrics['box_loss']:.4f} "
            f"| cls={metrics['cls_loss']:.4f} "
            f"| mAP50={metrics['mAP50']:.3f}"
        )

    def on_train_end(self, trainer) -> None:
        """Gọi khi training kết thúc (tất cả epochs hoặc early-stop)."""
        best = getattr(trainer, "best", None)
        if best is None:
            self._sig.error.emit("Training xong nhưng không tìm thấy best.pt")
            return
        best_path = Path(str(best))
        if not best_path.is_file():
            self._sig.error.emit(
                f"Training xong nhưng best.pt không tồn tại trên đĩa: {best_path}\n"
                f"(Có thể early-stop trước khi lưu được checkpoint.)"
            )
            return
        best_str = str(best_path)
        log.info("Training hoàn thành. Best model: %s", best_str)
        self._sig.log_line.emit(f"✓ Training xong! Best model: {best_str}")
        self._sig.fit_end.emit(best_str)
