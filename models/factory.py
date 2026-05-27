"""
ModelFactory — tạo đúng backend theo tên, liệt kê backend khả dụng, và
switch backend lúc runtime (giải phóng VRAM trước khi load model mới).

DetectionThread chỉ gọi ModelFactory + BaseDetector — không biết class
backend cụ thể.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config as _cfg  # import module để đọc attrs lúc call-time (không snapshot)
from core import device
from models.base_detector import BaseDetector, ModelLoadError


log = logging.getLogger("model_factory")


def _vram_mb() -> float:
    """VRAM allocated (MB). 0 nếu không có CUDA (qua core/device)."""
    return device.memory_allocated_mb()


def _empty_cache() -> None:
    device.empty_cache()


def default_backend() -> str:
    """
    Backend mặc định theo phần cứng khả dụng:
      - máy CUDA  → "yolo"  (YOLODetector torch, giữ như cũ)
      - máy AMD/Intel có DirectML + có file .onnx → "onnx"
      - còn lại   → "yolo"  (sẽ tự fallback CPU)

    Dùng khi config.ACTIVE_BACKEND = "auto".
    """
    if device.detect_backend() == "onnx-dml" and Path(_cfg.ONNX_MODEL_PATH).is_file():
        return "onnx"
    return "yolo"


class ModelFactory:
    """Factory + switch logic cho các detection backend."""

    _BACKENDS = ("yolo", "onnx", "custom", "template")

    # --------------------------------------------------------------- create
    @staticmethod
    def create(backend: str, **kwargs) -> BaseDetector:
        """
        Tạo detector theo backend name.

        Args:
            backend: "yolo" | "onnx" | "custom" | "template" | "auto"
                (alias "both" → "yolo"; "auto" → default_backend() theo phần cứng).
            **kwargs: override config — model_path, class_names, conf,
                device, imgsz.

        Raises:
            ModelLoadError: nếu backend không hợp lệ.
        """
        name = (backend or "yolo").lower()
        if name == "both":
            log.warning("Backend 'both' chưa hỗ trợ — dùng 'yolo'.")
            name = "yolo"
        if name == "auto":
            name = default_backend()
            log.info("Backend 'auto' → '%s' (theo phần cứng).", name)

        if name == "onnx":
            from models.onnx_detector import ONNXDetector

            kw = {
                "model_path": _cfg.ONNX_MODEL_PATH,
                "conf": _cfg.INFERENCE_CONF,
                "imgsz": _cfg.ONNX_IMG_SIZE,
                "execution_provider": _cfg.EXECUTION_PROVIDER,
            }
            if _cfg.YOLO_CLASS_NAMES:
                kw["class_names"] = list(_cfg.YOLO_CLASS_NAMES)
            for k in ("model_path", "class_names", "conf", "imgsz"):
                if k in kwargs:
                    kw[k] = kwargs[k]
            # device='cpu' (từ OOM fallback) → ép CPU EP
            if kwargs.get("device") == "cpu":
                kw["execution_provider"] = "cpu"
            return ONNXDetector(**kw)

        if name == "yolo":
            from models.yolo_detector import YOLODetector

            kw = {}
            if "model_path" in kwargs:
                kw["model_path"] = kwargs["model_path"]
            if "class_names" in kwargs:
                kw["class_names"] = kwargs["class_names"]
            for k in ("conf", "device", "imgsz"):
                if k in kwargs:
                    kw[k] = kwargs[k]
            return YOLODetector(**kw)

        if name == "custom":
            from models.custom_detector import CustomDetector

            # Đọc cả CUSTOM_MODEL_PATH và CUSTOM_CLASS_NAMES từ module lúc call-time
            # (không snapshot lúc import) để lấy giá trị mới nhất do training_panel
            # hoặc main.py restore cập nhật.
            kw: dict = {"model_path": _cfg.CUSTOM_MODEL_PATH}
            if _cfg.CUSTOM_CLASS_NAMES:
                kw["class_names"] = list(_cfg.CUSTOM_CLASS_NAMES)
            if "model_path" in kwargs:
                kw["model_path"] = kwargs["model_path"]
            if "class_names" in kwargs:
                kw["class_names"] = kwargs["class_names"]
            for k in ("conf", "device", "imgsz"):
                if k in kwargs:
                    kw[k] = kwargs[k]
            return CustomDetector(**kw)

        if name == "template":
            from models.template_detector import TemplateDetector

            kw = {}
            if "template_dir" in kwargs:
                kw["template_dir"] = kwargs["template_dir"]
            if "conf" in kwargs:
                kw["conf"] = kwargs["conf"]
            return TemplateDetector(**kw)

        raise ModelLoadError(f"Backend không hợp lệ: '{backend}'")

    # ------------------------------------------------------- list_available
    @staticmethod
    def list_available() -> list[dict]:
        """
        Liệt kê backend + trạng thái khả dụng để UI populate dropdown.

        Returns:
            list[dict] với keys: name, label, available, path, reason?.
        """
        out: list[dict] = []

        # YOLO: luôn available (ultralytics tự download nếu thiếu).
        # Nhãn suy ra từ YOLO_MODEL_PATH — KHÔNG hardcode "YOLOv8n" (lệch
        # khi đổi model qua config; tự bám theo model thực).
        _yolo_stem = Path(_cfg.YOLO_MODEL_PATH).stem
        out.append({
            "name": "yolo",
            "label": f"{_yolo_stem} (COCO)",
            "available": True,
            "path": _cfg.YOLO_MODEL_PATH,
        })

        # ONNX (onnxruntime-directml): khả dụng khi file .onnx tồn tại.
        # Nhãn kèm EP sẽ chạy (DML nếu có, không thì CPU).
        opath = Path(_cfg.ONNX_MODEL_PATH)
        _ep = "DML" if device.detect_backend() == "onnx-dml" else "CPU"
        if opath.is_file():
            out.append({
                "name": "onnx",
                "label": f"{opath.stem} (ONNX·{_ep})",
                "available": True,
                "path": str(opath),
            })
        else:
            out.append({
                "name": "onnx",
                "label": "ONNX",
                "available": False,
                "path": str(opath),
                "reason": "Chưa có file .onnx — chạy tools/export_onnx.py",
            })

        # Custom: phụ thuộc file .pt tồn tại (đọc _cfg lúc call-time).
        # Dùng stat() một lần để tránh TOCTOU giữa is_file() và stat().
        cpath = Path(_cfg.CUSTOM_MODEL_PATH)
        try:
            _cst = cpath.stat()
            _custom_valid = _cst.st_size > 1_000_000
        except OSError:
            _cst = None
            _custom_valid = False
        if _custom_valid:
            ncls = len(_cfg.CUSTOM_CLASS_NAMES) if _cfg.CUSTOM_CLASS_NAMES else "?"
            out.append({
                "name": "custom",
                "label": f"Custom ({ncls} class)",
                "available": True,
                "path": str(cpath),
            })
        else:
            reason = (
                "File quá nhỏ (<1MB) — có thể corrupt"
                if _cst is not None
                else "File không tồn tại"
            )
            out.append({
                "name": "custom",
                "label": "Custom",
                "available": False,
                "path": str(cpath),
                "reason": reason,
            })

        # Template: phụ thuộc thư mục templates/ có ≥1 ảnh hợp lệ
        # (đọc _cfg lúc call-time, mirror custom).
        tdir = Path(_cfg.TEMPLATE_DIR)
        texts = tuple(e.lower() for e in _cfg.TEMPLATE_EXTS)
        timgs = (
            [p for p in tdir.iterdir()
             if p.is_file() and p.suffix.lower() in texts]
            if tdir.is_dir() else []
        )
        if timgs:
            out.append({
                "name": "template",
                "label": f"Template ({len(timgs)} mẫu)",
                "available": True,
                "path": str(tdir),
            })
        else:
            treason = (
                "Thư mục templates/ không tồn tại"
                if not tdir.is_dir()
                else "Chưa có ảnh mẫu (.png/.jpg/...)"
            )
            out.append({
                "name": "template",
                "label": "Template",
                "available": False,
                "path": str(tdir),
                "reason": treason,
            })
        return out

    # --------------------------------------------------------------- switch
    @staticmethod
    def switch(current: BaseDetector, new_backend: str) -> BaseDetector:
        """
        Switch backend: unload model cũ → giải phóng VRAM → load model mới.

        OOM khi load model mới → tự fallback device='cpu' (có log cảnh báo).

        Returns:
            Detector mới đã load xong.

        Raises:
            ModelLoadError: nếu cả CPU fallback cũng thất bại.
        """
        old_name = current.get_backend_name() if current else "None"
        vram_before = _vram_mb()

        if current is not None:
            try:
                current.unload()
            except Exception as exc:
                log.warning("unload() backend cũ lỗi (bỏ qua): %s", exc)
        _empty_cache()
        vram_after_unload = _vram_mb()

        new_detector = ModelFactory.create(new_backend)
        try:
            new_detector.load()
        except ModelLoadError:
            raise
        except Exception as exc:
            # Chỉ coi là OOM khi message khớp rõ ràng HOẶC đúng kiểu
            # OutOfMemoryError (heuristic chung ở core/device.is_oom). KHÔNG
            # dùng "cuda" in msg — quá rộng, nuốt cả lỗi driver thành fallback CPU.
            if not device.is_oom(exc):
                raise ModelLoadError(
                    f"Load backend '{new_backend}' thất bại: {exc}"
                ) from exc

            log.error("CUDA OOM khi load '%s' — thử CPU.", new_backend)
            _empty_cache()
            try:
                new_detector = ModelFactory.create(new_backend, device="cpu")
                new_detector.load()
            except Exception as cpu_exc:
                raise ModelLoadError(
                    f"Load backend '{new_backend}' thất bại cả trên CPU "
                    f"(sau OOM): {cpu_exc}"
                ) from cpu_exc

        vram_after = _vram_mb()
        log.info(
            "Switch backend: %s → %s | VRAM %.0f→%.0f→%.0f MB",
            old_name, new_detector.get_backend_name(),
            vram_before, vram_after_unload, vram_after,
        )
        return new_detector
