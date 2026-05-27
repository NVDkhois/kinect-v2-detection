"""
ONNXDetector — backend chạy YOLO .onnx bằng onnxruntime + DirectML EP.

Đường tăng tốc trên GPU AMD/Intel (vd RX 580) trên Windows, nơi torch không
có CUDA và torch-directml không chạy nổi graph YOLO. Đo thực trên RX 580
(raw infer): yolo26s 960×544 chữ nhật ≈ 52fps (= accuracy 960² nhưng nhanh
hơn 36% nhờ bỏ pad thừa), 640² ≈ 62fps (xem docs/ke_hoach_chuyen_rx580.md).
Input shape (vuông hoặc chữ nhật) đọc thẳng từ ONNX → letterbox theo (H,W).

Output yolo26 export là NMS-free (`end2end=True`): shape [1, N, 6], mỗi dòng
`[x1, y1, x2, y2, conf, class_id]` trong không gian ảnh letterbox → KHÔNG cần
NMS, chỉ lọc conf + undo letterbox + clip. Class names + imgsz đọc từ ONNX
metadata (không hardcode COCO).

KHÔNG load .onnx qua ultralytics (nó tự pip-install onnxruntime CPU đè lên
onnxruntime-directml). Dùng onnxruntime trực tiếp.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core import device
from models.base_detector import (
    BaseDetector,
    Detection,
    InferenceError,
    ModelLoadError,
)

log = logging.getLogger("onnx_detector")

_PAD_VALUE = 114  # giá trị pad letterbox chuẩn YOLO


# ---------------------------------------------------------------------------
# Hàm thuần (test không cần GPU/onnxruntime)
# ---------------------------------------------------------------------------
def letterbox(
    frame: np.ndarray, new_size: int | tuple[int, int]
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize giữ tỉ lệ + pad về kích thước đích (chuẩn YOLO letterbox).

    Args:
        frame: ảnh BGR uint8 (H×W×3).
        new_size: cạnh vuông `int` (vd 640) HOẶC `(H, W)` chữ nhật (vd (544, 960)).
            Chữ nhật khớp tỉ lệ camera 16:9 → bỏ vùng pad thừa, GPU đỡ ~36% compute
            mà giữ nguyên độ phân giải ảnh (xem docs/ke_hoach_chuyen_rx580.md, Pha 5b).

    Returns:
        (blob, ratio, (dw, dh)):
          blob  — float32 [1,3,H,W], RGB, chia 255 (NCHW).
          ratio — hệ số scale đã dùng (đồng nhất cả 2 chiều — uniform letterbox).
          (dw,dh) — padding (px) mỗi bên trái/trên trong ảnh letterbox.
    """
    new_h, new_w = (new_size, new_size) if isinstance(new_size, int) else new_size
    h, w = frame.shape[:2]
    r = min(new_h / h, new_w / w)  # scaleup=True (cho phép phóng to), uniform
    new_w_r, new_h_r = round(w * r), round(h * r)
    dw = (new_w - new_w_r) / 2.0
    dh = (new_h - new_h_r) / 2.0
    new_w, new_h = new_w_r, new_h_r

    if (w, h) != (new_w, new_h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        frame, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
    )

    # BGR→RGB, HWC→CHW, /255, thêm batch dim
    img = padded[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return blob[None], r, (dw, dh)


def decode_detections(
    raw: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    conf_thresh: float,
    names: dict[int, str],
    orig_w: int,
    orig_h: int,
    filter_ids: Optional[list[int]] = None,
) -> list[Detection]:
    """
    Parse output NMS-free [N,6] → list[Detection] theo toạ độ frame gốc.

    Args:
        raw: mảng [N,6] = [x1,y1,x2,y2,conf,cls] (letterbox space).
        ratio, pad: từ letterbox() để undo.
        conf_thresh: ngưỡng confidence.
        names: {class_id: name} đọc từ ONNX metadata.
        orig_w, orig_h: kích thước frame gốc để clip.
        filter_ids: nếu set, chỉ giữ các class_id này.

    Returns:
        list[Detection] (bbox int, toạ độ frame gốc, đã clip).
    """
    dets: list[Detection] = []
    if raw is None or raw.size == 0:
        return dets

    dw, dh = pad
    inv = 1.0 / ratio if ratio else 1.0

    for row in raw:
        conf = float(row[4])
        if conf < conf_thresh:
            continue
        cid = int(round(float(row[5])))
        if filter_ids is not None and cid not in filter_ids:
            continue

        x1 = (float(row[0]) - dw) * inv
        y1 = (float(row[1]) - dh) * inv
        x2 = (float(row[2]) - dw) * inv
        y2 = (float(row[3]) - dh) * inv

        x1 = min(max(x1, 0.0), orig_w)
        y1 = min(max(y1, 0.0), orig_h)
        x2 = min(max(x2, 0.0), orig_w)
        y2 = min(max(y2, 0.0), orig_h)

        dets.append(
            Detection(
                class_name=names.get(cid, str(cid)),
                conf=conf,
                bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                class_id=cid,
            )
        )
    return dets


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class ONNXDetector(BaseDetector):
    """
    Backend ONNX Runtime (DirectML EP ưu tiên, CPU EP fallback).

    imgsz + class names đọc từ ONNX metadata khi load(). Nếu metadata thiếu,
    dùng imgsz truyền vào + class_names override.
    """

    def __init__(
        self,
        model_path: str,
        class_names: Optional[list[str]] = None,
        conf: float = 0.30,
        imgsz: int = 640,
        execution_provider: str = "auto",
    ) -> None:
        self._model_path = str(model_path)
        self._names_override = list(class_names) if class_names else None
        self._conf = float(conf)
        self._imgsz = int(imgsz)
        self._ep = execution_provider

        self._session = None
        self._input_name: str = "images"
        self._in_shape: tuple[int, int] = (self._imgsz, self._imgsz)  # (H, W) đích letterbox
        self._names: dict[int, str] = {}
        self._filter_ids: Optional[list[int]] = None
        self._active_provider: str = ""

    # ------------------------------------------------------------- load
    def load(self) -> None:
        if not Path(self._model_path).is_file():
            raise ModelLoadError(
                f"File ONNX '{self._model_path}' không tồn tại. "
                f"Chạy tools/export_onnx.py để sinh."
            )
        try:
            import onnxruntime as ort

            providers = device.onnx_providers(self._ep)
            self._session = ort.InferenceSession(self._model_path, providers=providers)
            self._active_provider = self._session.get_providers()[0]
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name

            self._read_metadata()          # names + imgsz (default vuông cho dynamic)
            self._resolve_input_shape(inp.shape)  # override khi shape cố định (rect/vuông)
            if self._names_override:
                self._names = {i: n for i, n in enumerate(self._names_override)}

            log.info(
                "%s đã load (EP=%s, input=%dx%d, %d classes).",
                self.get_backend_name(), self._active_provider,
                self._in_shape[1], self._in_shape[0], len(self._names),
            )
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._session = None
            raise ModelLoadError(
                f"Load ONNX '{self._model_path}' thất bại: {exc}"
            ) from exc

    def _resolve_input_shape(self, shape: list) -> None:
        """
        Xác định (H, W) đích letterbox từ input shape ONNX [N,C,H,W].

        Fixed-shape export (dynamic=False) → H,W là int (authoritative, hỗ trợ cả
        vuông lẫn chữ nhật 960×544). Dynamic export → H/W là str → fallback imgsz vuông.
        """
        try:
            in_h, in_w = shape[2], shape[3]
            if isinstance(in_h, int) and isinstance(in_w, int) and in_h > 0 and in_w > 0:
                self._in_shape = (in_h, in_w)
                self._imgsz = max(in_h, in_w)  # giữ tương thích logging/getter cũ
                return
        except (IndexError, TypeError):
            pass
        self._in_shape = (self._imgsz, self._imgsz)  # fallback vuông

    def _read_metadata(self) -> None:
        """Đọc names + imgsz từ ONNX metadata (ultralytics nhúng khi export)."""
        try:
            meta = self._session.get_modelmeta().custom_metadata_map
        except Exception:
            meta = {}

        raw_names = meta.get("names")
        if raw_names:
            try:
                parsed = ast.literal_eval(raw_names)
                self._names = {int(k): str(v) for k, v in dict(parsed).items()}
            except Exception:
                log.warning("Không parse được 'names' từ ONNX metadata.")

        raw_imgsz = meta.get("imgsz")
        if raw_imgsz:
            try:
                val = ast.literal_eval(raw_imgsz)
                self._imgsz = int(val[0] if isinstance(val, (list, tuple)) else val)
                self._in_shape = (self._imgsz, self._imgsz)  # default vuông; rect override sau
            except Exception:
                pass

    # ---------------------------------------------------------- predict
    def predict(self, frame: np.ndarray) -> list[Detection]:
        if self._session is None:
            raise InferenceError("predict() gọi trước khi load().")
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        blob, ratio, pad = letterbox(frame, self._in_shape)
        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"ONNX inference lỗi: {exc}") from exc

        raw = np.asarray(outputs[0])
        if raw.ndim == 3:  # [1, N, 6] → [N, 6]
            raw = raw[0]
        return decode_detections(
            raw, ratio, pad, self._conf, self._names, w, h, self._filter_ids
        )

    # -------------------------------------------------------------- info
    def get_class_names(self) -> list[str]:
        if not self._names:
            return []
        return [self._names[k] for k in sorted(self._names)]

    def get_backend_name(self) -> str:
        stem = Path(self._model_path).name
        ep = self._active_provider.replace("ExecutionProvider", "") or "?"
        return f"ONNX · {stem} · {ep} · {len(self._names)} classes"

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    # ---------------------------------------------------------- setters
    def set_conf_threshold(self, conf: float) -> None:
        self._conf = float(conf)

    def set_class_filter(self, class_names: list[str] | None) -> None:
        if not class_names:
            self._filter_ids = None
            return
        name_to_id = {v: k for k, v in self._names.items()}
        ids = [name_to_id[n] for n in class_names if n in name_to_id]
        self._filter_ids = ids or None

    def unload(self) -> None:
        self._session = None
        device.empty_cache()
        log.info("ONNXDetector unloaded.")
