"""
FrameDiffChecker — quyết định có cần chạy inference cho frame mới hay không.

Ý tưởng: nếu cảnh không thay đổi đáng kể so với frame cuối đã infer, có thể
bỏ qua inference và emit lại detections cache → tiết kiệm GPU/CPU.

Trade-off: thêm ~0.3-0.5ms cho diff check; cứu được ~15-20ms inference khi skip.
Net positive nếu skip rate >5%.
"""
from __future__ import annotations

import cv2
import numpy as np


class FrameDiffChecker:
    """
    So sánh frame mới với frame infer gần nhất bằng mean absolute pixel diff.

    Quy trình check():
        1. Resize frame về (DIFF_W, DIFF_H) — mặc định 160×120 (~16× nhanh hơn 640×480).
        2. Convert grayscale.
        3. cv2.absdiff(prev_small, curr_small).mean() → diff_score.
        4. should_infer = (diff_score >= threshold) OR (skipped >= max_skip).
        5. Nếu should_infer → cập nhật prev_small, reset skip counter.

    Thread-safe: KHÔNG. Gọi từ 1 thread duy nhất (DetectionThread).
    """

    DIFF_W = 160
    DIFF_H = 120

    def __init__(self, threshold: float = 3.5, max_skip: int = 8) -> None:
        self.threshold = float(threshold)
        self.max_skip = int(max_skip)

        self._prev_small: np.ndarray | None = None
        self._skipped_in_a_row = 0

        # Stats (cumulative)
        self.total_checked = 0
        self.total_skipped = 0

    def check(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Return (should_infer, diff_score).

        Lần đầu gọi (chưa có prev) → luôn infer, diff_score=inf.
        """
        self.total_checked += 1

        small = cv2.resize(frame_bgr, (self.DIFF_W, self.DIFF_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_small is None:
            self._prev_small = gray
            self._skipped_in_a_row = 0
            return True, float("inf")

        diff_score = float(cv2.absdiff(gray, self._prev_small).mean())

        # Force infer nếu đã skip quá lâu (tránh bbox cũ quá hạn).
        force = self._skipped_in_a_row >= self.max_skip

        if diff_score >= self.threshold or force:
            self._prev_small = gray
            self._skipped_in_a_row = 0
            return True, diff_score

        self._skipped_in_a_row += 1
        self.total_skipped += 1
        return False, diff_score

    @property
    def skip_rate(self) -> float:
        """% frame được skip trên tổng đã check (0..100)."""
        if self.total_checked == 0:
            return 0.0
        return 100.0 * self.total_skipped / self.total_checked

    def reset_stats(self) -> None:
        self.total_checked = 0
        self.total_skipped = 0

    def reset(self) -> None:
        """
        Xoá frame cache + skip counter.

        Gọi sau khi switch backend: frame đầu tiên sau switch phải LUÔN
        chạy inference (không bị skip do cảnh giống frame của model cũ).
        Không động tới stats tích luỹ (dùng reset_stats() cho mục đó).
        """
        self._prev_small = None
        self._skipped_in_a_row = 0
