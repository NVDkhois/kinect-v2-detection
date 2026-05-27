"""
Thread 2 — Detection (backend-agnostic).

DetectionThread KHÔNG biết backend cụ thể: nó tạo detector qua
`ModelFactory` và chỉ gọi `BaseDetector` (load/predict/...). Đổi backend
(YOLO pretrained ↔ custom .pt) lúc runtime qua slot `on_backend_switched`.

Pipeline 1 frame:
    resize color→INFER → adaptive-skip check → preprocess →
    detector.predict() → scale bbox về color space → tính 3D →
    tracker.update() → emit detections_ready.

`Detection` được re-export từ models.base_detector để mọi consumer cũ
(`from core.detector import Detection`) tiếp tục hoạt động.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QThread, pyqtSignal, pyqtSlot

from config import (
    ADAPTIVE_SKIP_ENABLED,
    ACTIVE_BACKEND,
    DIFF_THRESHOLD,
    INFER_BYPASS_DOWNSCALE,
    INFER_H,
    INFER_W,
    INFERENCE_CONF,
    MAX_SKIP_FRAMES,
)
from core.frame_diff import FrameDiffChecker
from core.position import compute_3d_position
from models.base_detector import (  # re-export cho backward compat
    BaseDetector,
    Detection,
    InferenceError,
    ModelLoadError,
)
from models.factory import ModelFactory
from processing.preprocessor import preprocess


log = logging.getLogger("detector")

__all__ = ["Detection", "DetectionThread"]


class DetectionThread(QThread):
    """
    Thread chạy inference qua một `BaseDetector` bất kỳ.

    Input: queue chứa tuple (color_bgr, depth_mm).
    Output signals:
        detections_ready(list[Detection|TrackedObject])
        class_names_changed(list[str]) — phát khi switch backend xong
        backend_changed(str)           — tên backend mới (cho UI status)
        backend_error(str)             — switch thất bại, giữ backend cũ

    Public attrs (sửa runtime từ Qt main thread, GIL đảm bảo gán atomic):
        selected_class: Optional[str] — filter class. None = tất cả.
        conf_threshold: float
        paused: bool
    """

    detections_ready = pyqtSignal(list)
    class_names_changed = pyqtSignal(list)
    backend_changed = pyqtSignal(str)
    backend_error = pyqtSignal(str)

    def __init__(
        self,
        frame_queue: "queue.Queue[tuple[np.ndarray, np.ndarray]]",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._queue = frame_queue
        self._running = False
        # Cờ dừng BỀN VỮNG: sống sót cả khi stop() được gọi TRƯỚC khi run()
        # kịp chạy (race QThread.start). run() phải kiểm tra cờ này quanh
        # bước detector.load() — bước nặng ~4s, không thể ngắt giữa chừng.
        self._stop_requested = False

        self.detector: Optional[BaseDetector] = None
        self._backend_name = ACTIVE_BACKEND

        self.selected_class: Optional[str] = None
        self.conf_threshold: float = INFERENCE_CONF
        self.paused: bool = False

        # Backend switch điều phối qua pending-flag (UI thread set, detector
        # thread đọc & thực hiện giữa 2 lần predict). QMutex chỉ bảo vệ
        # biến _pending_backend.
        self._switch_lock = QMutex()
        self._pending_backend: Optional[str] = None

        # Adaptive frame skip
        self._diff_checker = (
            FrameDiffChecker(threshold=DIFF_THRESHOLD, max_skip=MAX_SKIP_FRAMES)
            if ADAPTIVE_SKIP_ENABLED else None
        )
        self._last_dets: list = []

        self._tracker = None
        self._prev_selected_class: Optional[str] = None
        self._was_paused = False

    # ----------------------------------------------------------- backend
    @pyqtSlot(str)
    def on_backend_switched(self, new_backend: str) -> None:
        """
        Gọi từ Qt main thread khi user chọn backend mới trên UI.

        Chỉ set pending-flag; switch thật sự thực hiện trong run() giữa 2
        lần predict để không cắt ngang inference đang chạy.
        """
        self._switch_lock.lock()
        self._pending_backend = new_backend
        self._switch_lock.unlock()
        log.info("Yêu cầu switch backend → %s (pending).", new_backend)

    def _take_pending(self) -> Optional[str]:
        self._switch_lock.lock()
        pending = self._pending_backend
        self._pending_backend = None
        self._switch_lock.unlock()
        return pending

    def _do_switch(self, new_backend: str) -> None:
        """Thực thi switch (chạy trong detector thread)."""
        old = self.detector
        try:
            self.detector = ModelFactory.switch(old, new_backend)
        except ModelLoadError as exc:
            log.error("Switch backend thất bại: %s — giữ backend cũ.", exc)
            # Khôi phục backend cũ nếu nó đã bị unload (switch() unload trước
            # khi load mới, nên old có thể đã mất VRAM).
            if self.detector is None or not self.detector.is_loaded:
                self.detector = old
                if self.detector is not None and not self.detector.is_loaded:
                    try:
                        self.detector.load()
                    except Exception as reload_exc:
                        # Cả switch lẫn reload backup đều thất bại → detector
                        # vẫn trỏ tới old (unloaded). predict() sẽ raise
                        # InferenceError ở mỗi frame (bị catch bởi broad-except
                        # trong run()), nhưng user cần được thông báo ngay.
                        log.error(
                            "Không khôi phục được backend '%s': %s — "
                            "detection tạm dừng cho đến khi switch lại.",
                            self._backend_name, reload_exc,
                        )
                        self.backend_error.emit(
                            f"Detection tạm dừng: switch thất bại và không "
                            f"khôi phục được backend cũ ({reload_exc}). "
                            f"Hãy thử chọn lại backend."
                        )
                        return
            self.backend_error.emit(str(exc))
            return

        self._backend_name = new_backend
        # Re-apply runtime settings cho detector mới
        self.detector.set_conf_threshold(self.conf_threshold)
        self.detector.set_class_filter(
            [self.selected_class] if self.selected_class else None
        )
        # Reset tracker — ID cũ không còn ý nghĩa với model mới.
        # QUAN TRỌNG: phải set_class_names() lại theo backend MỚI trước khi
        # reset. Nếu bỏ bước này, ByteTracker giữ bảng class của backend cũ
        # (vd 80 class COCO) → custom model class_id=1 bị map nhầm thành
        # 'bicycle'. Đây là root cause của bug "box sai bicycle".
        if self._tracker is not None:
            self._tracker.set_class_names(
                {i: n for i, n in enumerate(self.detector.get_class_names())}
            )
            self._tracker.reset()
        self._last_dets = []

        # Reset diff_checker: xóa frame cache cũ để frame đầu tiên sau switch
        # LUÔN chạy inference ngay (không bị skip do scene tương tự).
        if self._diff_checker is not None:
            self._diff_checker.reset()

        names = self.detector.get_class_names()
        self.class_names_changed.emit(names)
        self.backend_changed.emit(self.detector.get_backend_name())
        log.info("Switch backend hoàn tất: %s", self.detector.get_backend_name())

    # --------------------------------------------------------------- loop
    def run(self) -> None:  # noqa: D401
        """Vòng lặp inference."""
        # Race QThread.start(): nếu stop() đã được gọi trước khi run() kịp
        # chạy thì BỎ QUA load nặng — đừng ghi đè _running = True.
        if self._stop_requested:
            log.info("DetectionThread: stop trước khi khởi động — bỏ qua load.")
            return
        self._running = True

        # Tạo + load detector theo config
        try:
            self.detector = ModelFactory.create(self._backend_name)
            self.detector.load()
        except ModelLoadError as exc:
            log.error("Không load được backend '%s': %s — thread dừng.",
                      self._backend_name, exc)
            self.backend_error.emit(str(exc))
            return

        # stop() có thể đã đến TRONG lúc load() (không ngắt được). Thoát ngay
        # và trả lại VRAM vừa cấp — tối quan trọng trên card 3GB.
        if self._stop_requested:
            log.info("DetectionThread: stop trong lúc load — thoát ngay.")
            try:
                if self.detector is not None:
                    self.detector.unload()
            except Exception as exc:  # noqa: BLE001
                log.warning("Unload sau stop lỗi (bỏ qua): %s", exc)
            return

        self.detector.set_conf_threshold(self.conf_threshold)

        # Lazy init tracker (sau khi có class names)
        from core.tracker import ObjectTracker

        self._tracker = ObjectTracker()
        self._tracker.set_class_names(
            {i: n for i, n in enumerate(self.detector.get_class_names())}
        )
        self._prev_selected_class = self.selected_class
        self._was_paused = False

        # Profiling
        prof_t0 = time.perf_counter()
        prof_n = 0
        prof_pre = prof_inf = prof_post = 0.0
        prof_dets = 0

        log.info("DetectionThread bắt đầu (backend=%s).",
                 self.detector.get_backend_name())

        while self._running:
            # Backend switch / reload nếu có yêu cầu pending.
            # KHÔNG filter pending == self._backend_name: Reload gửi cùng tên
            # backend (muốn load lại) — phải luôn chạy _do_switch khi pending set.
            pending = self._take_pending()
            if pending is not None:
                # Bọc broad-except: nếu _do_switch lỗi bất ngờ thì thread
                # KHÔNG chết âm thầm — emit backend_error để UI biết.
                try:
                    self._do_switch(pending)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    log.error("_do_switch crash (%s):\n%s",
                              exc, traceback.format_exc())
                    self.backend_error.emit(f"Switch lỗi: {exc}")

            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            color_bgr, depth = item

            if self.paused:
                self._was_paused = True
                continue

            # Resume / đổi class filter → reset tracker
            if self._was_paused or self.selected_class != self._prev_selected_class:
                if self._tracker is not None:
                    self._tracker.reset()
                self._last_dets = []
                self._was_paused = False
                if self.selected_class != self._prev_selected_class:
                    self.detector.set_class_filter(
                        [self.selected_class] if self.selected_class else None
                    )
                self._prev_selected_class = self.selected_class

            # Sync conf threshold (cheap — chỉ gán float)
            self.detector.set_conf_threshold(self.conf_threshold)

            # ---- Bọc toàn bộ inference + post-process trong broad try/except ----
            # Bất kỳ exception nào (tracker crash, shape mismatch, CUDA lỗi...)
            # đều được LOG và skip frame — thread KHÔNG chết âm thầm.
            try:
                h0, w0 = color_bgr.shape[:2]
                t_pre = time.perf_counter()
                if INFER_BYPASS_DOWNSCALE:
                    small = color_bgr
                elif (w0, h0) != (INFER_W, INFER_H):
                    small = cv2.resize(color_bgr, (INFER_W, INFER_H),
                                       interpolation=cv2.INTER_AREA)
                else:
                    small = color_bgr

                should_infer = True
                if self._diff_checker is not None:
                    should_infer, _ = self._diff_checker.check(small)

                if not should_infer:
                    pre_ms = (time.perf_counter() - t_pre) * 1000.0
                    inf_ms = post_ms = 0.0
                    dets = self._last_dets
                    self.detections_ready.emit(dets)
                else:
                    frame_pp = preprocess(small)
                    pre_ms = (time.perf_counter() - t_pre) * 1000.0

                    t_inf = time.perf_counter()
                    try:
                        raw = self.detector.predict(frame_pp)
                    except InferenceError as exc:
                        log.error("predict() lỗi: %s — skip frame.", exc)
                        continue
                    inf_ms = (time.perf_counter() - t_inf) * 1000.0

                    t_post = time.perf_counter()
                    _ph, _pw = frame_pp.shape[:2]
                    sx = w0 / float(_pw)
                    sy = h0 / float(_ph)
                    # Khi tracker bật, ByteTracker.update() tự tính lại 3D
                    # trên bbox ĐÃ track (chính xác hơn) → tính 3D ở đây là
                    # thừa & bị vứt. Chỉ project 3D khi KHÔNG có tracker.
                    scaled = self._scale_and_project(
                        raw, sx, sy, w0, h0, depth,
                        project_3d=(self._tracker is None),
                    )
                    if self._tracker is not None:
                        dets = self._tracker.update(scaled, depth)
                    else:
                        dets = scaled
                    post_ms = (time.perf_counter() - t_post) * 1000.0
                    self._last_dets = dets
                    self.detections_ready.emit(dets)

            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                log.error("Frame processing crash — skip frame:\n%s", _tb.format_exc())
                continue

            # ---- Rolling stats ----
            prof_n += 1
            prof_pre += pre_ms
            prof_inf += inf_ms
            prof_post += post_ms
            prof_dets += len(dets)
            elapsed = time.perf_counter() - prof_t0
            if elapsed >= 1.0:
                skip_info = ""
                if self._diff_checker is not None:
                    skip_info = f"  skip={self._diff_checker.skip_rate:.0f}%"
                log.info(
                    "pre=%.1fms  infer=%.1fms  post=%.1fms  dets/frame=%.1f  "
                    "fps=%.1f%s (n=%d)",
                    prof_pre / prof_n, prof_inf / prof_n, prof_post / prof_n,
                    prof_dets / prof_n, prof_n / elapsed, skip_info, prof_n,
                )
                prof_t0 = time.perf_counter()
                prof_n = 0
                prof_pre = prof_inf = prof_post = 0.0
                prof_dets = 0
                if self._diff_checker is not None:
                    self._diff_checker.reset_stats()

        log.info("DetectionThread đã dừng.")

    # --------------------------------------------------------- helpers
    @staticmethod
    def _scale_and_project(
        raw: list[Detection],
        sx: float,
        sy: float,
        color_w: int,
        color_h: int,
        depth: np.ndarray | None,
        project_3d: bool = True,
    ) -> list[Detection]:
        """
        Scale bbox từ inference space → color space (+ tính 3D nếu cần).

        project_3d=False khi tracker sẽ tự tính 3D trên bbox đã track —
        tránh gọi compute_3d_position 2 lần/detection mỗi frame.
        """
        out: list[Detection] = []
        for d in raw:
            x1, y1, x2, y2 = d.bbox
            scaled_bbox = (
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy),
            )
            if depth is not None and project_3d:
                x_mm, y_mm, z_mm = compute_3d_position(
                    depth, scaled_bbox, color_w, color_h
                )
                out.append(replace(
                    d, bbox=scaled_bbox,
                    x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
                ))
            else:
                out.append(replace(d, bbox=scaled_bbox))
        return out

    def stop(self) -> None:
        """Yêu cầu dừng thread an toàn (an toàn gọi trước khi run() chạy)."""
        self._stop_requested = True
        self._running = False

    # ----------------------------------------------------------------- info
    @property
    def class_names(self) -> dict[int, str]:
        """Map {id: name} cho UI populate (rỗng nếu detector chưa load)."""
        if self.detector is None or not self.detector.is_loaded:
            return {}
        return {i: n for i, n in enumerate(self.detector.get_class_names())}

    def get_backend_name(self) -> str:
        if self.detector is None:
            return self._backend_name
        return self.detector.get_backend_name()
