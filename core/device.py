"""
core/device — lớp trừu tượng chọn backend inference (một nguồn sự thật).

Project hỗ trợ 3 backend, ưu tiên giảm dần:
  1. "cuda"      — torch + GPU NVIDIA (YOLODetector). Giữ cho máy NVIDIA.
  2. "onnx-dml"  — onnxruntime + DirectML EP (ONNXDetector). GPU AMD/Intel
                   trên Windows (vd RX 580). Đo thực (RX 580, 2026-05-24):
                   yolo26s 960×544 chữ nhật ≈ 52fps (CHỐT), 960² vuông ≈ 33fps.
  3. "cpu"       — onnxruntime CPU EP / torch CPU. Fallback cuối, chậm.

torch-directml KHÔNG dùng: không chạy nổi graph YOLO (op không hỗ trợ).
Xem docs/ke_hoach_chuyen_rx580.md (Pha 1).

Mọi nơi trong code gọi qua module này thay vì `torch.cuda.*` trực tiếp, để
khi chạy trên máy không CUDA các hàm cache/sync thành no-op an toàn.
"""

from __future__ import annotations

import logging

log = logging.getLogger("device")

# Provider names chuẩn của onnxruntime
_DML_EP = "DmlExecutionProvider"
_CPU_EP = "CPUExecutionProvider"


# ---------------------------------------------------------------------------
# Helpers dò khả dụng (tách riêng để test monkeypatch được — không cần GPU)
# ---------------------------------------------------------------------------
def _has_cuda() -> bool:
    """True nếu torch thấy GPU CUDA. Bọc try/except: torch có thể vắng."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _available_onnx_providers() -> list[str]:
    """Danh sách EP onnxruntime khả dụng. [] nếu onnxruntime vắng."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Chọn backend
# ---------------------------------------------------------------------------
def detect_backend() -> str:
    """
    Chọn backend tốt nhất khả dụng: "cuda" → "onnx-dml" → "cpu".

    Returns:
        "cuda" | "onnx-dml" | "cpu".
    """
    if _has_cuda():
        return "cuda"
    if _DML_EP in _available_onnx_providers():
        return "onnx-dml"
    return "cpu"


def onnx_providers(preferred: str = "auto") -> list[str]:
    """
    Danh sách execution providers truyền cho onnxruntime.InferenceSession,
    đã lọc theo provider thực sự khả dụng (không bao giờ trả EP không có).

    Args:
        preferred: "auto" (DML nếu có, không thì CPU) | "dml" | "cpu".

    Returns:
        list[str] EP theo thứ tự ưu tiên; luôn có ít nhất CPUExecutionProvider.
    """
    avail = _available_onnx_providers()
    want_dml = preferred in ("auto", "dml") and _DML_EP in avail

    providers: list[str] = []
    if want_dml:
        providers.append(_DML_EP)
    providers.append(_CPU_EP)
    return providers


def torch_device(preferred: str = "auto") -> str:
    """
    Chuỗi device cho torch path (YOLODetector trên máy CUDA).

    Args:
        preferred: "auto" | "cuda:0" | "cpu".

    Returns:
        "cuda:0" nếu CUDA khả dụng và không bị ép cpu; ngược lại "cpu".
    """
    if preferred == "cpu":
        return "cpu"
    if preferred in ("auto", "cuda", "cuda:0"):
        return "cuda:0" if _has_cuda() else "cpu"
    return preferred


# ---------------------------------------------------------------------------
# OOM heuristic (đồng bộ với yolo_detector.predict + factory.switch)
# ---------------------------------------------------------------------------
def is_oom(exc: BaseException) -> bool:
    """
    True nếu exception là Out-Of-Memory thật.

    KHÔNG dùng `"cuda" in msg` (quá rộng → nuốt cả lỗi driver/transient
    thành fallback CPU vĩnh viễn). Chỉ match message OOM rõ ràng HOẶC đúng
    kiểu OutOfMemoryError.
    """
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or type(exc).__name__ == "OutOfMemoryError"
    )


# ---------------------------------------------------------------------------
# CUDA-only ops → no-op an toàn khi không có CUDA
# ---------------------------------------------------------------------------
def empty_cache() -> None:
    """Giải phóng VRAM cache CUDA. No-op nếu không có CUDA."""
    if not _has_cuda():
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        log.debug("empty_cache bỏ qua: %s", exc)


def memory_allocated_mb() -> float:
    """VRAM đang cấp phát (MB). 0.0 nếu không có CUDA."""
    if not _has_cuda():
        return 0.0
    try:
        import torch

        return torch.cuda.memory_allocated(0) / (1024 * 1024)
    except Exception:
        return 0.0


def synchronize() -> None:
    """Đồng bộ CUDA stream (cho benchmark). No-op nếu không có CUDA."""
    if not _has_cuda():
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        log.debug("synchronize bỏ qua: %s", exc)


def describe() -> str:
    """Mô tả backend hiện tại để log lúc khởi động."""
    backend = detect_backend()
    if backend == "cuda":
        try:
            import torch

            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "CUDA"
    if backend == "onnx-dml":
        return "ONNX Runtime · DirectML EP (GPU AMD/Intel)"
    return "CPU"
