"""
Thread 1 — Kinect V2 capture.

Đọc color + depth frame từ Kinect V2 @ 30fps và emit qua Qt signal.
Nếu pykinect2 không import được (môi trường non-Windows, chưa cắm Kinect),
fallback sang webcam + depth giả (numpy zeros) để các module khác vẫn test
được độc lập.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from config import (
    COLOR_H,
    COLOR_W,
    DEPTH_H,
    DEPTH_W,
    TARGET_FPS,
)


log = logging.getLogger("capture")


# ---------------------------------------------------------------------------
# Detect pykinect2 availability
# ---------------------------------------------------------------------------
try:
    from pykinect2 import PyKinectRuntime, PyKinectV2

    _KINECT_AVAILABLE = True
except Exception as exc:  # ImportError, OSError on non-Windows, etc.
    PyKinectRuntime = None  # type: ignore[assignment]
    PyKinectV2 = None  # type: ignore[assignment]
    _KINECT_AVAILABLE = False
    log.warning("pykinect2 không khả dụng (%s) — sẽ fallback sang webcam.", exc)


class KinectCaptureThread(QThread):
    """
    Thread đọc color + depth từ Kinect V2.

    Signals:
        frame_ready(color_bgr: np.ndarray, depth_mm: np.ndarray):
            color_bgr  shape=(COLOR_H, COLOR_W, 3) dtype=uint8 (BGR)
            depth_mm   shape=(DEPTH_H, DEPTH_W)    dtype=uint16 (mm)

    Khi pykinect2 không khả dụng, fallback:
        - color: webcam (cv2.VideoCapture(0)), resize về (COLOR_W, COLOR_H)
        - depth: numpy zeros (DEPTH_H, DEPTH_W) dtype=uint16
        Nếu cả webcam cũng không có → color = test pattern.
    """

    frame_ready = pyqtSignal(np.ndarray, np.ndarray)

    # ----------------------------------------------------------------------
    # TODO (#3 ICoordinateMapper — HOÃN tới khi validate trên Kinect THẬT):
    # `core.position.compute_3d_position(cs_map=...)` ĐÃ sẵn sàng + test.
    # Còn thiếu phần SẢN XUẤT cs_map (chỉ làm khi có phần cứng để kiểm API
    # pykinect2, vì lib này patched/cũ — không verify được ở CI):
    #   1. Khi `config.POSITION_USE_COORDINATE_MAPPER` và `_KINECT_AVAILABLE`:
    #      sau `_read_kinect()`, gọi
    #      `self._kinect._mapper.MapColorFrameToCameraSpace(depth_buf)` →
    #      CameraSpacePoint[1920*1080] → reshape (1080,1920,3) float32 (mét).
    #      BỌC try/except + fallback None (giữ đường tuyến tính nếu lỗi).
    #   2. Cân nhắc perf: mảng ~24MB/frame + chi phí map ~5-10ms — ĐO bằng
    #      profiling log; nếu vỡ 30fps cân nhắc map ở độ phân giải depth.
    #   3. Truyền cs_map xuống: đổi signal → `pyqtSignal(np.ndarray,
    #      np.ndarray, object)` (thêm cs_map=None) và cập nhật main.py
    #      FrameRouter + detector_queue tuple + DetectionThread._scale_and
    #      _project/ByteTracker để forward cs_map vào compute_3d_position.
    #      (Đây là thay đổi signature có ripple UI — làm 1 lần khi validate.)
    # KHÔNG wiring trước khi đo trên Kinect: tránh latent bug ở hot path
    # capture mà không test được. config flag mặc định False = an toàn.
    # ----------------------------------------------------------------------

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = False
        self._kinect = None
        self._webcam = None
        self._frame_interval = 1.0 / max(1, TARGET_FPS)

    # ------------------------------------------------------------------ setup
    def _init_source(self) -> None:
        """Khởi tạo Kinect hoặc fallback webcam."""
        if _KINECT_AVAILABLE:
            try:
                flags = (
                    PyKinectV2.FrameSourceTypes_Color
                    | PyKinectV2.FrameSourceTypes_Depth
                )
                self._kinect = PyKinectRuntime.PyKinectRuntime(flags)
                log.info("Kinect V2 đã được khởi tạo (Color + Depth).")
                return
            except Exception as exc:
                log.error("Khởi tạo Kinect V2 thất bại: %s", exc)
                self._kinect = None

        # Fallback: webcam
        try:
            import cv2

            self._webcam = cv2.VideoCapture(0)
            if not self._webcam.isOpened():
                self._webcam = None
                log.warning("Không mở được webcam — sẽ dùng test pattern.")
            else:
                self._webcam.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_W)
                self._webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_H)
                log.info("Fallback: dùng webcam thay cho Kinect.")
        except Exception as exc:
            log.warning("Webcam fallback lỗi: %s", exc)
            self._webcam = None

    # ----------------------------------------------------------------- read
    def _read_kinect(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Đọc 1 cặp color+depth từ Kinect. Return None nếu chưa có frame."""
        assert self._kinect is not None
        if not self._kinect.has_new_color_frame() or not self._kinect.has_new_depth_frame():
            return None

        color_buf = self._kinect.get_last_color_frame()  # BGRA flatten
        depth_buf = self._kinect.get_last_depth_frame()  # uint16 flatten (mm)

        color_bgra = color_buf.reshape((COLOR_H, COLOR_W, 4))
        color_bgr = color_bgra[:, :, :3].copy()
        depth = depth_buf.reshape((DEPTH_H, DEPTH_W)).astype(np.uint16)
        return color_bgr, depth

    def _read_fallback(self) -> tuple[np.ndarray, np.ndarray]:
        """Đọc frame fallback (webcam hoặc test pattern) + depth giả."""
        import cv2

        if self._webcam is not None:
            ok, frame = self._webcam.read()
            if not ok or frame is None:
                frame = self._test_pattern()
            else:
                if (frame.shape[1], frame.shape[0]) != (COLOR_W, COLOR_H):
                    frame = cv2.resize(frame, (COLOR_W, COLOR_H))
        else:
            frame = self._test_pattern()

        depth = np.zeros((DEPTH_H, DEPTH_W), dtype=np.uint16)
        return frame, depth

    @staticmethod
    def _test_pattern() -> np.ndarray:
        """Test pattern khi không có nguồn nào — gradient đơn giản."""
        img = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        xs = np.linspace(0, 255, COLOR_W, dtype=np.uint8)
        img[:, :, 0] = xs[np.newaxis, :]
        img[:, :, 1] = xs[np.newaxis, :]
        img[:, :, 2] = 128
        return img

    # ----------------------------------------------------------------- loop
    def run(self) -> None:  # noqa: D401 — QThread API
        """Vòng lặp đọc frame chính."""
        self._init_source()
        self._running = True

        # Nâng độ phân giải timer Windows về 1ms. Mặc định ~15.6ms khiến
        # msleep(1) ngủ ~15ms → throttle cố định 33ms cũ cần ~3 lần sleep
        # thô = beat-frequency, chu kỳ ~47ms ≈ 21fps (đo bằng benchmark).
        # Có 1ms thì pacing deadline chính xác. Khôi phục ở _cleanup().
        self._winmm = None
        try:
            import ctypes

            self._winmm = ctypes.windll.winmm
            self._winmm.timeBeginPeriod(1)
        except Exception:
            self._winmm = None

        # Profiling rolling counters
        prof_t0 = time.perf_counter()
        prof_frames = 0
        prof_read_ms_sum = 0.0
        log.info("KinectCaptureThread đã bắt đầu.")

        while self._running:
            t_loop = time.perf_counter()

            # Bọc broad-except: driver Kinect đôi khi trả buffer sai kích
            # thước → reshape() raise ValueError. KHÔNG để 1 frame hỏng
            # giết chết cả thread (mất hình, UI không biết). Log + skip,
            # giống broad-except trong DetectionThread.run().
            try:
                pair: tuple[np.ndarray, np.ndarray] | None = None
                if self._kinect is not None:
                    pair = self._read_kinect()
                if pair is None:
                    if self._kinect is None:
                        pair = self._read_fallback()
                    else:
                        # Kinect chưa có frame mới — chờ ngắn, poll lại
                        # (timer 1ms → msleep(1) thật ~1ms).
                        self.msleep(1)
                        continue
            except Exception:
                import traceback as _tb
                log.error("Đọc frame lỗi — skip frame:\n%s", _tb.format_exc())
                self.msleep(5)
                continue
            read_ms = (time.perf_counter() - t_loop) * 1000.0

            color_bgr, depth = pair
            self.frame_ready.emit(color_bgr, depth)

            # ---- Rolling stats ----
            prof_frames += 1
            prof_read_ms_sum += read_ms
            now = time.perf_counter()
            elapsed = now - prof_t0
            if elapsed >= 1.0:
                fps = prof_frames / elapsed
                avg_read = prof_read_ms_sum / prof_frames
                log.info(
                    "fps=%.1f  read_avg=%.1fms  (%d frames / %.2fs)",
                    fps, avg_read, prof_frames, elapsed,
                )
                prof_t0 = now
                prof_frames = 0
                prof_read_ms_sum = 0.0

            # Pacing DEADLINE-based: cap TARGET_FPS, trừ thời gian đã tốn
            # trong vòng (read+emit+stats). Giống core/video_capture.py.
            # Với timer 1ms msleep chính xác → hết beat-frequency 21fps.
            remaining_ms = int(
                (self._frame_interval - (time.perf_counter() - t_loop)) * 1000.0
            )
            if remaining_ms > 0:
                self.msleep(remaining_ms)

        self._cleanup()
        log.info("KinectCaptureThread đã dừng.")

    def stop(self) -> None:
        """Yêu cầu dừng thread an toàn. Gọi `wait()` sau đó từ caller."""
        self._running = False

    def _cleanup(self) -> None:
        # Khôi phục độ phân giải timer Windows (cặp với timeBeginPeriod).
        if getattr(self, "_winmm", None) is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._winmm = None
        if self._kinect is not None:
            try:
                self._kinect.close()
            except Exception:
                pass
            self._kinect = None
        if self._webcam is not None:
            try:
                self._webcam.release()
            except Exception:
                pass
            self._webcam = None
