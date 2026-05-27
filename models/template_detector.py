"""
TemplateDetector — backend template matching ZERO-TRAINING.

Định vị vật chuẩn/cố định bằng `cv2.matchTemplate` (TM_CCOEFF_NORMED) trên
CPU thuần (0 VRAM). Mỗi ảnh trong TEMPLATE_DIR = 1 class, tên class = stem
filename (KHÔNG hardcode — mirror cách YOLODetector lấy class từ model).

Cùng hợp đồng `BaseDetector` như YOLODetector nên DetectionThread + 3D
projection + ByteTrack + overlay + log + 3 tab UI hoạt động không sửa đổi.

Giới hạn: KHÔNG bất biến xoay; đa tỉ lệ phải opt-in qua TEMPLATE_SCALES
(chi phí ×N mỗi template → cân nhắc realtime). Xem README.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    TEMPLATE_DIR,
    TEMPLATE_EXTS,
    TEMPLATE_HYSTERESIS_CONFIRM,
    TEMPLATE_HYSTERESIS_HOLD,
    TEMPLATE_ID_MARGIN,
    TEMPLATE_MATCH_THRESHOLD,
    TEMPLATE_NMS_IOU,
    TEMPLATE_SCALES,
    TEMPLATE_FACE_PREFILTER,
    TEMPLATE_STRIP_SUFFIX,
)
from models.base_detector import (
    BaseDetector,
    Detection,
    InferenceError,
    ModelLoadError,
)


log = logging.getLogger("template_detector")


# ---------------------------------------------------------------------------
class _ClassState:
    """Trạng thái hysteresis per class_id."""
    __slots__ = ("hit", "miss", "confirmed", "last_dets")

    def __init__(self) -> None:
        self.hit: int = 0
        self.miss: int = 0
        self.confirmed: bool = False
        self.last_dets: list = []   # TẤT CẢ instances của class (multi-object)


class _HysteresisFilter:
    """
    Chỉ emit detection sau CONFIRM frame liên tiếp thấy; giữ thêm HOLD frame
    sau khi mất. Triệt tiêu oscillation score quanh threshold — nguyên nhân
    chính gây bbox bật/tắt liên tục.

    Giữ toàn bộ instances per class (không chỉ best) → hỗ trợ multi-object.
    confirm=1 → emit ngay (bypass); hold=0 → không giữ.
    """

    def __init__(self, confirm_frames: int, hold_frames: int) -> None:
        self._confirm = max(1, confirm_frames)
        self._hold = max(0, hold_frames)
        self._states: dict[int, _ClassState] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        # Gom tất cả detections per class_id (giữ nguyên multi-instance)
        by_class: dict[int, list[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_id, []).append(d)

        results: list[Detection] = []
        all_ids = set(self._states.keys()) | set(by_class.keys())

        for cid in all_ids:
            state = self._states.setdefault(cid, _ClassState())

            if cid in by_class:
                state.hit += 1
                state.miss = 0
                state.last_dets = by_class[cid]
                if state.hit >= self._confirm:
                    state.confirmed = True
            else:
                state.hit = 0
                state.miss += 1
                if state.miss > self._hold:
                    state.confirmed = False

            if state.confirmed and state.last_dets:
                results.extend(state.last_dets)

        # Dọn states đã unconfirm lâu
        stale = [
            cid for cid, s in self._states.items()
            if not s.confirmed and s.miss > self._hold + 30
        ]
        for cid in stale:
            del self._states[cid]

        return results

    def reset(self) -> None:
        self._states.clear()


# ---------------------------------------------------------------------------
class TemplateDetector(BaseDetector):
    """
    Backend template matching. Nạp ảnh mẫu một lần ở `load()`, mỗi
    `predict()` trượt mẫu (đa tỉ lệ tuỳ chọn) + NMS gộp đỉnh chồng nhau.
    Hỗ trợ tùy chọn Face Prefiltering và gộp nhãn trùng lặp.
    """

    def __init__(
        self,
        template_dir: str = TEMPLATE_DIR,
        conf: float = TEMPLATE_MATCH_THRESHOLD,
        scales: tuple[float, ...] = TEMPLATE_SCALES,
        nms_iou: float = TEMPLATE_NMS_IOU,
        exts: tuple[str, ...] = TEMPLATE_EXTS,
        face_prefilter: bool = TEMPLATE_FACE_PREFILTER,
        strip_suffix: bool = TEMPLATE_STRIP_SUFFIX,
        hysteresis_confirm: int = TEMPLATE_HYSTERESIS_CONFIRM,
        hysteresis_hold: int = TEMPLATE_HYSTERESIS_HOLD,
        id_margin: float = TEMPLATE_ID_MARGIN,
    ) -> None:
        self._dir = str(template_dir)
        self._conf = float(conf)
        self._scales = tuple(scales) or (1.0,)
        self._nms_iou = float(nms_iou)
        self._exts = tuple(e.lower() for e in exts)
        self._face_prefilter = bool(face_prefilter)
        self._strip_suffix = bool(strip_suffix)
        self._id_margin = float(id_margin)
        self._hysteresis = _HysteresisFilter(hysteresis_confirm, hysteresis_hold)

        # (class_name, gray_template) theo thứ tự nạp
        self._templates: list[tuple[str, np.ndarray]] = []
        self._names: dict[int, str] = {}
        self._filter_ids: Optional[set[int]] = None
        # (class_name, scale) đã cảnh báo "template > frame"
        self._oversized_warned: set[tuple[str, float]] = set()
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._loaded = False

    # ------------------------------------------------------------- load
    def load(self) -> None:
        tdir = Path(self._dir)
        if not tdir.is_dir():
            raise ModelLoadError(
                f"Thư mục template không tồn tại: '{self._dir}'"
            )

        files = sorted(
            p for p in tdir.iterdir()
            if p.is_file() and p.suffix.lower() in self._exts
        )
        templates: list[tuple[str, np.ndarray]] = []
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                log.warning("Bỏ qua ảnh mẫu lỗi/corrupt: %s", p.name)
                continue
            
            cname = p.stem
            if self._strip_suffix and "_" in cname:
                # Loại bỏ hậu tố sau dấu gạch dưới cuối cùng (ví dụ "nam_1" -> "nam")
                cname = cname.rsplit("_", 1)[0]
                
            templates.append((cname, img))

        if not templates:
            raise ModelLoadError(
                f"Không có ảnh mẫu hợp lệ trong '{self._dir}' "
                f"(ext hỗ trợ: {', '.join(self._exts)})"
            )

        self._templates = templates
        
        # Danh sách tên class duy nhất để gán class_id ổn định
        unique_names = []
        for name, _ in templates:
            if name not in unique_names:
                unique_names.append(name)
                
        self._names = {i: name for i, name in enumerate(unique_names)}
        self._oversized_warned.clear()
        self._hysteresis.reset()
        self._loaded = True
        log.info(
            "TemplateDetector đã load (%d mẫu, %d classes: %s).",
            len(templates), len(self._names), ", ".join(self._names.values()),
        )

    # ---------------------------------------------------------- predict
    def predict(self, frame: np.ndarray) -> list[Detection]:
        if not self._loaded:
            raise InferenceError("predict() gọi trước khi load().")
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]

        name_to_id = {name: i for i, name in self._names.items()}
        
        # Gom hits theo class_id
        hits_by_class: dict[int, list[tuple[tuple[int, int, int, int], float]]] = defaultdict(list)

        if self._face_prefilter and self._face_cascade is None:
            xml_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            _cascade = cv2.CascadeClassifier(xml_path)
            if _cascade.empty():
                log.error(
                    "Haar Cascade không load được từ '%s' — face prefilter bị tắt.",
                    xml_path,
                )
                self._face_prefilter = False
            else:
                self._face_cascade = _cascade

        if self._face_prefilter:
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
            )

            if len(faces) == 0:
                return []

            for (xf, yf, wf, hf) in faces:
                # Bbox hiển thị lấy từ Haar Cascade — ổn định, không lệch.
                # Template matching chỉ dùng để phân loại "ai", KHÔNG định vị.
                face_bbox: tuple[int, int, int, int] = (xf, yf, xf + wf, yf + hf)

                # Mở rộng ROI để bao gồm tóc/cằm/viền cho matching.
                pad_x = int(wf * 0.30)
                pad_y = int(hf * 0.45)
                rx0 = max(0, xf - pad_x)
                ry0 = max(0, yf - pad_y)
                rx1 = min(fw, xf + wf + pad_x)
                ry1 = min(fh, yf + hf + pad_y)
                face_roi = gray[ry0:ry1, rx0:rx1]
                roi_h, roi_w = face_roi.shape[:2]

                # Gom best score PER CLASS (không per-template).
                # Nhiều template cùng class → lấy template nào khớp tốt nhất.
                best_by_class: dict[int, float] = {}

                for cname, tmpl in self._templates:
                    cid = name_to_id.get(cname)
                    if cid is None:
                        continue
                    if self._filter_ids is not None and cid not in self._filter_ids:
                        continue

                    th, tw = tmpl.shape[:2]

                    if len(self._scales) == 1 and self._scales[0] == 1.0:
                        s_ideal = min(wf / tw, hf / th)
                        scales_to_test = (s_ideal,)
                    else:
                        scales_to_test = self._scales

                    for s in scales_to_test:
                        if s != 1.0:
                            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                            t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=interp)
                        else:
                            t = tmpl
                        t_h, t_w = t.shape[:2]
                        if t_h > roi_h or t_w > roi_w or t_h < 2 or t_w < 2:
                            continue

                        res = cv2.matchTemplate(face_roi, t, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        if max_val >= self._conf:
                            if max_val > best_by_class.get(cid, 0.0):
                                best_by_class[cid] = float(max_val)

                if not best_by_class:
                    continue

                # Sắp xếp class theo score giảm dần
                ranked = sorted(best_by_class.items(), key=lambda kv: -kv[1])
                winner_cid, winner_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else 0.0

                # Yêu cầu winner cách runner-up ít nhất _id_margin.
                # Ngăn ghost khi 2 class quá gần nhau (không chắc chắn ai).
                if winner_score - second_score >= self._id_margin:
                    hits_by_class[winner_cid].append((face_bbox, winner_score))
        else:
            # Chạy trượt thông thường trên toàn bộ khung hình
            for cname, tmpl in self._templates:
                cid = name_to_id.get(cname)
                if cid is None:
                    continue
                if self._filter_ids is not None and cid not in self._filter_ids:
                    continue
                hits = self._match_one(cname, gray, tmpl, fw, fh)
                hits_by_class[cid].extend(hits)

        dets: list[Detection] = []
        for cid, hits in hits_by_class.items():
            cname = self._names[cid]
            dets.extend(
                Detection(class_name=cname, conf=score, bbox=box, class_id=cid)
                for box, score in self._nms(hits)
            )
        # Loại box chồng nhau giữa các class khác nhau (ghost class trên cùng khuôn mặt)
        dets = self._cross_class_nms(dets)
        return self._hysteresis.update(dets)

    def _match_one(
        self, cname: str, gray: np.ndarray, tmpl: np.ndarray,
        fw: int, fh: int,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Trượt 1 template đa tỉ lệ → list ((x1,y1,x2,y2), score) >= conf."""
        out: list[tuple[tuple[int, int, int, int], float]] = []
        for s in self._scales:
            if s != 1.0:
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=interp)
            else:
                t = tmpl
            th, tw = t.shape[:2]
            if th > fh or tw > fw or th < 2 or tw < 2:
                # Bẫy thường gặp: bỏ ảnh chụp full (vd 2047x1542) vào
                # templates/ → lớn hơn frame infer (960x540) → skip MỌI
                # frame, 0 detection, không lỗi. Cảnh báo MỘT LẦN/scale.
                key = (cname, s)
                if key not in self._oversized_warned:
                    self._oversized_warned.add(key)
                    log.warning(
                        "Template '%s' (%dx%d) > frame (%dx%d) ở scale "
                        "%.2f → SKIP (sẽ không phát hiện được). Crop ảnh "
                        "mẫu nhỏ lại (chỉ riêng vật) hoặc bật "
                        "TEMPLATE_SCALES<1.0.",
                        cname, tw, th, fw, fh, s,
                    )
                continue  # template > frame ở tỉ lệ này → skip, không crash
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= self._conf)
            for x, y in zip(xs.tolist(), ys.tolist()):
                out.append(
                    ((x, y, x + tw, y + th), float(res[y, x]))
                )
        return out

    def _nms(
        self, hits: list[tuple[tuple[int, int, int, int], float]],
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Gộp các đỉnh chồng nhau (NMS) trong cùng 1 class."""
        if not hits:
            return []
        boxes_xywh = [
            [x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), _ in hits
        ]
        scores = [s for _, s in hits]
        idxs = cv2.dnn.NMSBoxes(
            boxes_xywh, scores, self._conf, self._nms_iou
        )
        if idxs is None or len(idxs) == 0:
            return []
        keep = np.array(idxs).flatten().tolist()
        return [hits[i] for i in keep]

    def _cross_class_nms(self, dets: list[Detection]) -> list[Detection]:
        """NMS toàn bộ — loại box chồng nhau giữa các class khác nhau.

        Vd: 'person_a 0.55' và 'person_b 0.58' đè lên cùng khuôn mặt →
        chỉ giữ 'person_b' (score cao hơn). Per-class NMS không xử lý
        được trường hợp này vì 'person_a' và 'person_b' là 2 class riêng.
        """
        if len(dets) <= 1:
            return dets
        boxes_xywh = [
            [x1, y1, x2 - x1, y2 - y1]
            for x1, y1, x2, y2 in (d.bbox for d in dets)
        ]
        scores = [d.conf for d in dets]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, self._conf, self._nms_iou)
        if idxs is None or len(idxs) == 0:
            return []
        keep = np.array(idxs).flatten().tolist()
        return [dets[i] for i in keep]

    # -------------------------------------------------------------- info
    def get_class_names(self) -> list[str]:
        return [self._names[k] for k in sorted(self._names)]

    def get_backend_name(self) -> str:
        return f"Template · {len(self._templates)} mẫu"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ---------------------------------------------------------- setters
    def set_conf_threshold(self, conf: float) -> None:
        self._conf = float(conf)

    def set_class_filter(self, class_names: list[str] | None) -> None:
        if not class_names:
            self._filter_ids = None
            return
        name_to_id = {v: k for k, v in self._names.items()}
        ids = {name_to_id[n] for n in class_names if n in name_to_id}
        self._filter_ids = ids or None

    def unload(self) -> None:
        self._templates = []
        self._names = {}
        self._filter_ids = None
        self._oversized_warned.clear()
        self._hysteresis.reset()
        self._loaded = False
        log.info("TemplateDetector unloaded.")
