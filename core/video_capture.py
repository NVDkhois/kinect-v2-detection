"""
Thread đọc video file từ đĩa (thay cho Kinect) — dùng cho tab Video.

Tách 2 lớp:
  • `VideoSource`  — wrapper cv2.VideoCapture THUẦN (không Qt), test được.
  • `VideoFileCaptureThread` — QThread mỏng: play/pause/restart/stop,
    giữ nhịp theo FPS gốc (cap TARGET_FPS), emit frame_ready.

Video file KHÔNG có depth → pipeline detect nhận depth=None
(`DetectionThread._scale_and_project` đã xử lý sẵn → toạ độ 3D = NaN).

Hết video: KHÔNG loop. Giữ frame cuối, dừng đọc, emit video_finished
(khớp quyết định "dừng ở frame cuối").
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QThread, pyqtSignal

from config import TARGET_FPS

log = logging.getLogger("video_capture")

__all__ = ["VideoSource", "VideoFileCaptureThread"]

# Nhịp poll khi đang pause / chưa nạp video — đủ mượt để UI phản hồi
# nút bấm mà không bận CPU (50 lần/giây).
_IDLE_POLL_MS = 20


class VideoSource:
    """
    Wrapper cv2.VideoCapture thuần — không phụ thuộc Qt, dễ unit test.

    Không thread-safe tự thân; caller (VideoFileCaptureThread) chịu trách
    nhiệm đồng bộ truy cập.
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = float(TARGET_FPS)
        self._frame_count: int = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """FPS gốc của video (fallback TARGET_FPS nếu metadata thiếu)."""
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def open(self, path: str) -> bool:
        """
        Mở video. Đóng source cũ trước (không leak handle).

        Returns:
            True nếu mở + đọc được metadata; False nếu file lỗi/không có.
        """
        self.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            log.warning("Không mở được video: %s", path)
            return False

        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Một số container trả 0 / NaN / vô lý → fallback an toàn
        self._fps = fps if (fps and 1.0 <= fps <= 240.0) else float(TARGET_FPS)
        self._frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        log.info("Đã mở video: %s (fps=%.1f, frames=%d)",
                 path, self._fps, self._frame_count)
        return True

    def read(self) -> Optional[np.ndarray]:
        """Đọc frame BGR kế tiếp. None khi EOF hoặc source đã đóng."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def restart(self) -> None:
        """Seek về frame 0 (phát lại từ đầu)."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        """Giải phóng handle. An toàn khi gọi nhiều lần."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileCaptureThread(QThread):
    """
    QThread đọc video file → emit frame_ready (chỉ color, không depth).

    API gọi từ Qt main thread (GIL đảm bảo gán bool atomic; truy cập
    VideoSource bọc trong _lock vì open/restart từ main thread còn
    read() chạy trong thread này).

    Signals:
        frame_ready(np.ndarray)   — 1 frame BGR
        video_loaded(float, int)  — fps, frame_count (sau load() thành công)
        video_finished()          — phát tới EOF (giữ frame cuối, tự pause)
        error(str)                — lỗi mở file
    """

    frame_ready = pyqtSignal(np.ndarray)
    video_loaded = pyqtSignal(float, int)
    video_finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._src = VideoSource()
        self._lock = QMutex()

        self._running = False
        self._playing = False          # True = đang phát
        self._has_video = False        # đã load video hợp lệ chưa
        self._restart_req = False      # cờ yêu cầu seek-0 (xử lý trong run)

    # ----------------------------------------------------------- điều khiển
    def load(self, path: str) -> bool:
        """
        Nạp video mới (gọi từ main thread). Tự pause sau khi nạp —
        user bấm Play để bắt đầu. Trả về True nếu mở thành công.
        """
        self._lock.lock()
        try:
            ok = self._src.open(path)
            if ok:
                self._has_video = True
                self._playing = False
                fps, fc = self._src.fps, self._src.frame_count
        finally:
            self._lock.unlock()

        if ok:
            self.video_loaded.emit(fps, fc)
        else:
            self.error.emit(f"Không mở được video:\n{path}")
        return ok

    def play(self) -> None:
        self._lock.lock()
        try:
            if self._has_video:
                self._playing = True
        finally:
            self._lock.unlock()

    def pause(self) -> None:
        self._lock.lock()
        try:
            self._playing = False
        finally:
            self._lock.unlock()

    def restart(self) -> None:
        """Phát lại từ đầu (xử lý seek trong run để tránh race với read).

        Cả _restart_req lẫn _playing phải được set dưới cùng một lock để
        tránh race: nếu run() thấy _restart_req=True nhưng _playing=False
        (do hai assignment không atomic), nó sẽ bỏ qua seek vì outer-check
        tại đầu vòng lặp đã sleep trước khi vào phần locked.
        """
        self._lock.lock()
        try:
            if self._has_video:
                self._restart_req = True
                self._playing = True
        finally:
            self._lock.unlock()

    def stop(self) -> None:
        """Yêu cầu dừng thread an toàn."""
        self._running = False

    # --------------------------------------------------------------- loop
    def run(self) -> None:  # noqa: D401
        self._running = True
        log.info("VideoFileCaptureThread bắt đầu.")

        while self._running:
            if not (self._playing and self._has_video):
                self.msleep(_IDLE_POLL_MS)
                continue

            # Mốc đầu chu kỳ: tính cả thời gian decode + emit vào nhịp phát.
            t_loop = time.perf_counter()

            self._lock.lock()
            try:
                if self._restart_req:
                    self._src.restart()
                    self._restart_req = False
                frame = self._src.read()
            finally:
                self._lock.unlock()

            if frame is None:
                # EOF → giữ frame cuối, tự pause, báo UI (1 lần)
                self._playing = False
                log.info("Video phát hết — dừng ở frame cuối.")
                self.video_finished.emit()
                continue

            self.frame_ready.emit(frame)

            # Pacing theo DEADLINE, cap theo TARGET_FPS (queue maxsize=2 tự
            # drop nếu detect chậm). Trước đây sleep CỐ ĐỊNH 1000/fps CỘNG
            # THÊM lên trên ~15ms decode → playback luôn thấp hơn fps gốc
            # (vd 30fps thực tế chỉ ~21fps). Trừ thời gian đã tốn trong chu
            # kỳ; nếu decode đã chậm hơn period thì không sleep (chạy tối đa).
            fps = min(self._src.fps, float(TARGET_FPS))
            remaining_ms = int((1.0 / fps - (time.perf_counter() - t_loop)) * 1000.0)
            if remaining_ms > 0:
                self.msleep(remaining_ms)

        self._lock.lock()
        try:
            self._src.release()
        finally:
            self._lock.unlock()
        log.info("VideoFileCaptureThread đã dừng.")
