"""
ObjectTracker — facade thống nhất cho DetectionThread.

Hiện chỉ support backend "bytetrack": bọc ByteTracker và theo dõi số track
active/lost gần nhất (get_stats) để UI hiển thị, không thêm state nào khác.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config import TRACKER_BACKEND
from core.detector import Detection
from tracking.bytetrack import ByteTracker, TrackedObject


log = logging.getLogger("tracker")


class ObjectTracker:
    """Facade: detector chỉ cần biết update()/reset()/get_stats()."""

    def __init__(self) -> None:
        backend = (TRACKER_BACKEND or "bytetrack").lower()
        if backend != "bytetrack":
            log.warning("Backend '%s' không support — fallback ByteTrack.", backend)
        self._impl = ByteTracker()
        self._last_active = 0
        self._last_lost = 0
        log.info("ObjectTracker backend=ByteTrack.")

    # ----------------------------------------------------------------- API
    def set_class_names(self, names: dict[int, str]) -> None:
        self._impl.set_class_names(names)

    def reset(self) -> None:
        self._impl.reset()
        self._last_active = 0
        self._last_lost = 0

    def update(
        self,
        detections: list[Detection],
        depth_frame: Optional[np.ndarray],
    ) -> list[TrackedObject]:
        tracked = self._impl.update(detections, depth_frame)
        active = sum(1 for o in tracked if o.state != "lost")
        lost = sum(1 for o in tracked if o.state == "lost")
        self._last_active = active
        self._last_lost = lost
        return tracked

    def get_stats(self) -> dict:
        return {
            "active": self._last_active,
            "lost": self._last_lost,
            "tracked_count": self._last_active + self._last_lost,
        }
