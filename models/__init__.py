"""
Package `models` — backend nhận diện theo Strategy Pattern.

DetectionThread chỉ phụ thuộc vào `BaseDetector` (interface) và
`ModelFactory` (tạo backend). Không bao giờ import YOLODetector /
CustomDetector trực tiếp.
"""

from models.base_detector import (
    BaseDetector,
    Detection,
    InferenceError,
    ModelLoadError,
)

__all__ = [
    "BaseDetector",
    "Detection",
    "InferenceError",
    "ModelLoadError",
]
