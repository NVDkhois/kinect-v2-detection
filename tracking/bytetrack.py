"""
ByteTrack wrapper trên ultralytics.trackers.byte_tracker.BYTETracker.

Input:  list[Detection] (bbox color-space gốc)
Output: list[TrackedObject] với track_id ổn định qua occlusion.

Ultralytics' BYTETracker.update() yêu cầu một object duck-typed có
.xywh, .conf, .cls, .xyxy + hỗ trợ __len__ và fancy indexing. Adapter
`_BoxesAdapter` bên dưới đáp ứng yêu cầu mà không cần dependency thêm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from config import (
    BBOX_SMOOTH_ALPHA,
    BT_FUSE_SCORE,
    BT_GRACE_LOST_FRAMES,
    BT_MATCH_THRESH,
    BT_NEW_TRACK_THRESH,
    BT_TRACK_BUFFER,
    BT_TRACK_HIGH_THRESH,
    BT_TRACK_LOW_THRESH,
    COLOR_H,
    COLOR_W,
)
from core.detector import Detection
from core.position import compute_3d_position


log = logging.getLogger("bytetrack")


# ---------------------------------------------------------------------------
_BBoxF = tuple[float, float, float, float]


def _ema_bbox(
    prev: Optional[_BBoxF],
    bbox: tuple[int, int, int, int],
    alpha: float,
) -> tuple[tuple[int, int, int, int], _BBoxF]:
    """
    EMA ĐỘC LẬP cả 4 mép (x1, y1, x2, y2) theo track_id → hết nháy cả
    dọc lẫn ngang.

    alpha >= 1.0 hoặc prev is None → trả box thô (byte-for-byte với bbox
    int đầu vào; lần đầu thấy track không bị trễ).

    Return (bbox_int_đã_mượt, state float 4 mép để cache lần sau).
    """
    x1, y1, x2, y2 = bbox
    if alpha >= 1.0 or prev is None:
        s: _BBoxF = (float(x1), float(y1), float(x2), float(y2))
    else:
        px1, py1, px2, py2 = prev
        a, ia = alpha, 1.0 - alpha
        s = (
            a * x1 + ia * px1,
            a * y1 + ia * py1,
            a * x2 + ia * px2,
            a * y2 + ia * py2,
        )
    new_bbox = (
        int(round(s[0])),
        int(round(s[1])),
        int(round(s[2])),
        int(round(s[3])),
    )
    return new_bbox, s


# ---------------------------------------------------------------------------
@dataclass
class TrackedObject(Detection):
    """
    Detection + tracking metadata. Kế thừa Detection nên backward-compatible
    với mọi consumer cũ (overlay, log table) đọc field của Detection.
    """

    state: str = "confirmed"             # "confirmed" | "lost" | "new"
    age: int = 1                         # số frame track đã sống


# ---------------------------------------------------------------------------
class _BoxesAdapter:
    """
    Duck-type Boxes object cho ultralytics BYTETracker.update().

    Cần expose: xyxy, xywh, conf, cls, __len__, __getitem__(mask).
    """

    __slots__ = ("xyxy", "xywh", "conf", "cls")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.astype(np.float32, copy=False)
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls.astype(np.float32, copy=False)
        # xywh format: cx, cy, w, h
        if len(xyxy):
            cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
            cy = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, mask):
        return _BoxesAdapter(self.xyxy[mask], self.conf[mask], self.cls[mask])


# ---------------------------------------------------------------------------
class ByteTracker:
    """
    Wrapper unified: input List[Detection] → output List[TrackedObject].
    """

    def __init__(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        # ultralytics BYTETracker đọc các field này từ args namespace
        args = SimpleNamespace(
            track_high_thresh=BT_TRACK_HIGH_THRESH,
            track_low_thresh=BT_TRACK_LOW_THRESH,
            new_track_thresh=BT_NEW_TRACK_THRESH,
            track_buffer=BT_TRACK_BUFFER,
            match_thresh=BT_MATCH_THRESH,
            fuse_score=BT_FUSE_SCORE,
        )
        self._tracker = BYTETracker(args)
        self._class_names: dict[int, str] = {}
        # Track meta: tid → (age, last_state)
        self._meta: dict[int, dict] = {}
        # EMA bbox: tid → (x1,y1,x2,y2) float đã mượt (xem _ema_bbox).
        self._bbox_ema: dict[int, _BBoxF] = {}
        log.info("ByteTracker initialized (buffer=%d, match=%.2f).",
                 BT_TRACK_BUFFER, BT_MATCH_THRESH)

    # ------------------------------------------------------------ public
    def set_class_names(self, names: dict[int, str]) -> None:
        self._class_names = dict(names)

    def reset(self) -> None:
        """Xoá toàn bộ tracks (gọi khi pause/class change)."""
        # ultralytics expose reset() trên BaseTrack & tracker
        try:
            self._tracker.reset()
        except Exception:
            # Fallback: re-init
            from ultralytics.trackers.byte_tracker import BYTETracker
            args = self._tracker.args
            self._tracker = BYTETracker(args)
        self._meta.clear()
        self._bbox_ema.clear()
        log.info("ByteTracker reset.")

    def _smooth_bbox(
        self, tid: int, bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Làm mượt 4 mép theo track_id; alpha=1.0 → bypass (không tạo state)."""
        if BBOX_SMOOTH_ALPHA >= 1.0:
            return bbox
        new_bbox, new_state = _ema_bbox(
            self._bbox_ema.get(tid), bbox, BBOX_SMOOTH_ALPHA
        )
        self._bbox_ema[tid] = new_state
        return new_bbox

    def update(
        self,
        detections: list[Detection],
        depth_frame: Optional[np.ndarray],
    ) -> list[TrackedObject]:
        """
        Cập nhật tracker với detections frame mới, return list TrackedObject.
        """
        # ---- Build adapter ----
        if detections:
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            conf = np.array([d.conf for d in detections], dtype=np.float32)
            cls = np.array([d.class_id for d in detections], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)

        adapter = _BoxesAdapter(xyxy, conf, cls)

        # ---- Run tracker ----
        try:
            tracks = self._tracker.update(adapter, img=None)
        except Exception as exc:
            log.warning("BYTETracker.update lỗi: %s — return raw detections.", exc)
            return [self._det_to_tracked(d, tid=-1, state="new", depth=depth_frame)
                    for d in detections]

        # ultralytics tracks columns: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        results: list[TrackedObject] = []
        active_ids: set[int] = set()

        # KHÔNG early-return khi tracks rỗng: frame detector miss SẠCH là ca
        # nháy nặng nhất — vẫn phải chạy lost-bridge bên dưới.
        for row in (tracks if tracks is not None else []):
            if len(row) < 7:
                log.warning("BYTETracker output format không đủ cột (%d) — skip row.", len(row))
                continue
            x1, y1, x2, y2 = (int(v) for v in row[:4])
            tid = int(row[4])
            tconf = float(row[5])
            tcls = int(row[6])
            cname = self._class_names.get(tcls, str(tcls))

            meta = self._meta.get(tid, {"age": 0, "state": "new"})
            meta["age"] = meta["age"] + 1
            state = "new" if meta["age"] <= 2 else "confirmed"
            meta["state"] = state
            self._meta[tid] = meta
            active_ids.add(tid)

            sbbox = self._smooth_bbox(tid, (x1, y1, x2, y2))
            obj = TrackedObject(
                class_id=tcls,
                class_name=cname,
                conf=tconf,
                bbox=sbbox,
                track_id=tid,
                state=state,
                age=meta["age"],
            )
            if depth_frame is not None:
                xm, ym, zm = compute_3d_position(
                    depth_frame, obj.bbox, COLOR_W, COLOR_H
                )
                obj.x_mm, obj.y_mm, obj.z_mm = xm, ym, zm
            results.append(obj)

        # --- Lost-bridge: chống nháy ---
        # Detector miss 1-2 frame (conf dao động quanh ngưỡng / preprocess
        # rung) làm box biến mất rồi hiện lại. ByteTrack vẫn giữ track trong
        # lost_stracks (track_buffer ~1s). Vẽ lại nó dạng NÉT ĐỨT (overlay
        # tự nhận state="lost") tối đa BT_GRACE_LOST_FRAMES frame để cầu
        # khoảng trống. Ân hạn NGẮN → Kalman drift không đáng kể (đây là lý
        # do trước kia không vẽ lost: sợ ghost khi mất lâu — nay có cap nên
        # an toàn). BT_GRACE_LOST_FRAMES=0 → về hành vi cũ.
        if BT_GRACE_LOST_FRAMES > 0:
            lost_list = getattr(self._tracker, "lost_stracks", None) or []
            cur_fid = int(getattr(self._tracker, "frame_id", 0))
            for st in lost_list:
                tid = int(getattr(st, "track_id", -1))
                if tid <= 0 or tid in active_ids:
                    continue
                end_f = int(getattr(st, "end_frame", cur_fid))
                if cur_fid - end_f > BT_GRACE_LOST_FRAMES:
                    continue  # mất quá lâu → thôi vẽ, tránh ghost
                try:
                    x1, y1, x2, y2 = (int(v) for v in st.xyxy)
                except Exception as _e:
                    log.debug("lost-bridge: không đọc được xyxy của tid=%d (%s) — skip.", tid, _e)
                    continue
                tcls = int(getattr(st, "cls", 0))
                meta = self._meta.get(tid, {"age": 1, "state": "lost"})
                sbbox = self._smooth_bbox(tid, (x1, y1, x2, y2))
                obj = TrackedObject(
                    class_id=tcls,
                    class_name=self._class_names.get(tcls, str(tcls)),
                    conf=float(getattr(st, "score", 0.0)),
                    bbox=sbbox,
                    track_id=tid,
                    state="lost",
                    age=int(meta.get("age", 1)),
                )
                if depth_frame is not None:
                    obj.x_mm, obj.y_mm, obj.z_mm = compute_3d_position(
                        depth_frame, obj.bbox, COLOR_W, COLOR_H
                    )
                results.append(obj)

        # Cleanup meta/ema: xóa entries cho IDs không còn sống.
        # Không dùng removed_stracks vì behavior của nó (per-frame vs tích lũy)
        # không được đảm bảo qua các version ultralytics. Thay vào đó tính
        # tập "live" = active + đang lost (đang bridge), rồi prune phần còn lại.
        live_ids = active_ids.copy()
        for st in (getattr(self._tracker, "lost_stracks", None) or []):
            tid_lost = int(getattr(st, "track_id", -1))
            if tid_lost > 0:
                live_ids.add(tid_lost)
        stale = [tid for tid in self._meta if tid not in live_ids]
        for tid in stale:
            self._meta.pop(tid, None)
            self._bbox_ema.pop(tid, None)

        return results

    # ------------------------------------------------------------ helpers
    def _det_to_tracked(
        self,
        det: Detection,
        tid: int,
        state: str,
        depth: Optional[np.ndarray],
    ) -> TrackedObject:
        obj = TrackedObject(
            class_id=det.class_id,
            class_name=det.class_name,
            conf=det.conf,
            bbox=det.bbox,
            track_id=tid,
            state=state,
        )
        if depth is not None:
            xm, ym, zm = compute_3d_position(depth, det.bbox, COLOR_W, COLOR_H)
            obj.x_mm, obj.y_mm, obj.z_mm = xm, ym, zm
        return obj
