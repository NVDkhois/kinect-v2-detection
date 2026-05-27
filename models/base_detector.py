"""
Interface chung cho mọi detection backend (Strategy Pattern).

`DetectionThread` chỉ import `BaseDetector` + `Detection` từ module này —
không bao giờ import YOLODetector / CustomDetector trực tiếp. Nhờ vậy có thể
swap backend (YOLO pretrained ↔ custom .pt) mà không đụng tới logic thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Output chuẩn
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """
    Output chuẩn của mọi detector — không đổi dù backend nào.

    Backward compatible với Detection cũ ở core.detector: cùng tên field,
    construct bằng keyword args ở mọi nơi (overlay, tracker, log table).

    Attributes:
        class_name: Tên lớp (theo model của backend đang dùng).
        conf: Confidence score (0..1).
        bbox: (x1, y1, x2, y2) pixel — trong không gian frame truyền vào
            `predict()`. Caller (DetectionThread) tự scale về color space.
        class_id: ID lớp theo indexing của backend (KHÔNG giả định COCO).
        x_mm, y_mm, z_mm: Toạ độ 3D (mm) — điền sau bởi position.py.
            NaN nếu depth không hợp lệ.
        track_id: ID tracking — điền bởi tracker, -1 nếu chưa track.
    """

    class_name: str
    conf: float
    bbox: tuple[int, int, int, int]
    class_id: int
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    track_id: int = -1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ModelLoadError(Exception):
    """Raise khi load model thất bại (file không tồn tại, corrupt, ...)."""


class InferenceError(Exception):
    """Raise khi inference thất bại (CUDA OOM, invalid input, ...)."""


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class BaseDetector(ABC):
    """
    Interface chung cho mọi detection backend.

    Vòng đời: `load()` một lần → nhiều lần `predict()` → `unload()` khi
    switch backend. Mọi method phải thread-safe ở mức "chỉ gọi từ
    DetectionThread"; switch backend được điều phối bằng pending-flag ở
    DetectionThread nên không cần lock nội bộ.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load model vào memory/GPU. Gọi một lần khi khởi tạo.

        Raises:
            ModelLoadError: nếu thất bại (file thiếu, corrupt, ...).
        """

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Chạy inference trên một frame (H×W×3, uint8, BGR như cv2).

        Args:
            frame: ảnh BGR uint8. Không cần resize về input size của model —
                backend tự xử lý qua tham số imgsz.

        Returns:
            list[Detection] đã filter theo conf threshold + class filter.
            Empty list nếu không detect được gì. bbox theo toạ độ `frame`.

        Raises:
            InferenceError: nếu model crash không thể tự phục hồi.
        """

    @abstractmethod
    def get_class_names(self) -> list[str]:
        """Danh sách tên class model này nhận diện (theo thứ tự class_id)."""

    @abstractmethod
    def get_backend_name(self) -> str:
        """Tên hiển thị, ví dụ 'YOLO · yolo26s · COCO 80 classes'."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True nếu model đã load xong và sẵn sàng predict."""

    # ── Shared setters (optional override) ──────────────────────────────
    def set_conf_threshold(self, conf: float) -> None:
        """Cập nhật confidence threshold lúc runtime. Default: no-op."""

    def set_class_filter(self, class_names: list[str] | None) -> None:
        """
        Chỉ detect các class trong list. None = detect tất cả.
        Default: no-op (backend tự override nếu hỗ trợ).
        """

    def unload(self) -> None:
        """Giải phóng VRAM khi switch backend. Default: no-op."""
