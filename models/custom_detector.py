"""
CustomDetector — backend dùng model tự train (.pt bất kỳ).

Kế thừa YOLODetector (cùng engine ultralytics) nhưng:
  - Bắt buộc kiểm tra file tồn tại trước khi load → ModelLoadError nếu thiếu.
  - Class names đọc từ metadata model, override bằng CUSTOM_CLASS_NAMES nếu có.
  - KHÔNG giả định class id theo COCO (vd 0 != "person").

DetectionThread vẫn chỉ thấy BaseDetector — không biết đây là custom.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import (
    CUSTOM_CLASS_NAMES,
    CUSTOM_MODEL_PATH,
    INFERENCE_CONF,
    INFERENCE_DEVICE,
    INFERENCE_IMG_SIZE,
    INFERENCE_IOU,
)
from models.base_detector import ModelLoadError
from models.yolo_detector import YOLODetector


log = logging.getLogger("custom_detector")


class CustomDetector(YOLODetector):
    """Backend model tự train. Xem docstring module."""

    def __init__(
        self,
        model_path: str = CUSTOM_MODEL_PATH,
        class_names: Optional[list[str]] = None,
        conf: float = INFERENCE_CONF,
        iou: float = INFERENCE_IOU,
        device: str = INFERENCE_DEVICE,
        imgsz: int = INFERENCE_IMG_SIZE,
    ) -> None:
        # class_names None → lấy CUSTOM_CLASS_NAMES từ config ([] = đọc từ .pt)
        override = class_names if class_names is not None else CUSTOM_CLASS_NAMES
        override = list(override) if override else None
        super().__init__(
            model_path=model_path,
            class_names=override,
            conf=conf,
            iou=iou,
            device=device,
            imgsz=imgsz,
        )

    # ------------------------------------------------------------- load
    def load(self) -> None:
        path = Path(self._model_path)
        if not path.is_file():
            raise ModelLoadError(
                f"Custom model không tồn tại: '{self._model_path}'. "
                "Train model hoặc sửa CUSTOM_MODEL_PATH trong config."
            )
        if path.stat().st_size < 1_000_000:
            raise ModelLoadError(
                f"Custom model '{self._model_path}' quá nhỏ "
                f"({path.stat().st_size} bytes) — có thể corrupt."
            )

        # YOLODetector.load() lo phần YOLO()+names+fp16+override.
        super().load()
        log.info(
            "Custom model loaded: %d class %s",
            len(self._names), self.get_class_names(),
        )

    # -------------------------------------------------------------- info
    def _backend_label(self) -> str:
        stem = Path(self._model_path).name
        return f"Custom · {len(self._names)} class · {stem}"
