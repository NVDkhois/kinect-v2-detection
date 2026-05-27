"""
YOLODetector — backend dùng YOLO pretrained (COCO 80 classes).

Port lại logic load/inference/parse từ core.detector cũ vào đây để
DetectionThread chỉ còn làm việc với BaseDetector. Class names đọc từ
`model.names` sau khi load — KHÔNG hardcode.

CustomDetector kế thừa class này (chỉ override load + backend name), nên
mọi sửa đổi ở predict/parse áp dụng cho cả 2 backend.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config import (
    INFERENCE_CONF,
    INFERENCE_DEVICE,
    INFERENCE_IMG_SIZE,
    INFERENCE_IOU,
    YOLO_CLASS_NAMES,
    YOLO_MODEL_PATH,
)
from models.base_detector import (
    BaseDetector,
    Detection,
    InferenceError,
    ModelLoadError,
)


log = logging.getLogger("yolo_detector")


class YOLODetector(BaseDetector):
    """
    Backend YOLO pretrained COCO (mặc định yolo26s, generic mọi model
    ultralytics theo YOLO_MODEL_PATH). Tự download weights nếu file chưa có.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        class_names: Optional[list[str]] = YOLO_CLASS_NAMES,
        conf: float = INFERENCE_CONF,
        iou: float = INFERENCE_IOU,
        device: str = INFERENCE_DEVICE,
        imgsz: int = INFERENCE_IMG_SIZE,
    ) -> None:
        self._model_path = str(model_path)
        self._names_override = list(class_names) if class_names else None
        self._conf = float(conf)
        self._iou = float(iou)
        self._device = str(device)
        self._imgsz = int(imgsz)

        self._model = None
        self._names: dict[int, str] = {}
        self._filter_ids: Optional[list[int]] = None
        self._fp16: bool = False

    # ----------------------------------------------------------- helpers
    def _backend_label(self) -> str:
        """Override ở CustomDetector."""
        stem = self._model_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return f"YOLO · {stem} · COCO {len(self._names)} classes"

    # ------------------------------------------------------------- load
    def load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self._model_path)
            raw_names = getattr(self._model, "names", None) or {}
            self._names = {int(k): str(v) for k, v in dict(raw_names).items()}

            if self._names_override:
                # Override theo thứ tự index (0..n-1)
                self._names = {i: n for i, n in enumerate(self._names_override)}
                try:
                    self._model.names = dict(self._names)
                except Exception:
                    pass

            # FP16 đã BỎ: GTX 1060 (Pascal, không tensor core) không tăng
            # tốc FP16 — benchmark thực đo FP16≈FP32, FP32 còn nhỉnh hơn,
            # lại ổn định số hơn. Luôn chạy FP32 (half=False).
            self._fp16 = False
            log.info(
                "%s đã load (device=%s, FP32, %d classes).",
                self.get_backend_name(), self._device, len(self._names),
            )
        except Exception as exc:
            self._model = None
            raise ModelLoadError(
                f"Load model '{self._model_path}' thất bại: {exc}"
            ) from exc

    # ---------------------------------------------------------- predict
    def _run(self, frame: np.ndarray, device: str, half: bool):
        return self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=device,
            half=half,
            classes=self._filter_ids,
            verbose=False,
        )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            raise InferenceError("predict() gọi trước khi load().")
        if frame is None or frame.size == 0:
            return []

        try:
            results = self._run(frame, self._device, self._fp16)
        except Exception as exc:
            # Chỉ coi là OOM khi message khớp rõ ràng HOẶC đúng kiểu
            # torch.cuda.OutOfMemoryError. KHÔNG dùng "cuda" in msg — quá
            # rộng, nuốt cả lỗi driver/transient thành fallback CPU VĨNH
            # VIỄN (giết FPS cả session). Đồng bộ với ModelFactory.switch.
            msg = str(exc).lower()
            is_oom = (
                "out of memory" in msg
                or "cuda out of memory" in msg
                or type(exc).__name__ == "OutOfMemoryError"
            )
            if not is_oom:
                # Lỗi KHÔNG phải OOM (driver hiccup, transient) → không
                # degrade GPU. Raise để DetectionThread skip frame này;
                # frame kế tiếp vẫn thử lại trên GPU bình thường.
                raise InferenceError(f"Inference lỗi: {exc}") from exc

            # OOM thật trên card 3GB: VRAM đã cạn → frame sau gần như chắc
            # cũng OOM. Latch CPU cho phần còn lại của session để tránh
            # death-spiral OOM liên tục. Có log cảnh báo rõ.
            log.warning("CUDA OOM — fallback CPU vĩnh viễn cho session. (%s)", exc)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            self._device = "cpu"
            self._fp16 = False
            try:
                results = self._run(frame, "cpu", False)
            except Exception as exc2:
                raise InferenceError(f"Inference CPU fallback lỗi: {exc2}") from exc2

        return self._parse(results)

    def _parse(self, results) -> list[Detection]:
        dets: list[Detection] = []
        if not results:
            return dets
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                cid = int(cls_ids[i])
                dets.append(
                    Detection(
                        class_name=self._names.get(cid, str(cid)),
                        conf=float(confs[i]),
                        bbox=(
                            int(xyxy[i][0]), int(xyxy[i][1]),
                            int(xyxy[i][2]), int(xyxy[i][3]),
                        ),
                        class_id=cid,
                    )
                )
        return dets

    # -------------------------------------------------------------- info
    def get_class_names(self) -> list[str]:
        if not self._names:
            return []
        return [self._names[k] for k in sorted(self._names)]

    def get_backend_name(self) -> str:
        return self._backend_label()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

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
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("YOLODetector unloaded (VRAM freed).")
