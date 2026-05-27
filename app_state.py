"""
Lưu/đọc trạng thái người dùng (JSON) — sống qua các lần restart app.

Dùng cho những lựa chọn runtime cần nhớ sau khi tắt app, ví dụ đường
dẫn custom model vừa train (config.py luôn reset về default khi import
lại nên KHÔNG dùng để persist).

File: user_state.json ở thư mục gốc project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("app_state")

_STATE_FILE = Path(__file__).resolve().parent / "user_state.json"


def load_state() -> dict:
    """Đọc toàn bộ state. Trả về {} nếu chưa có / lỗi parse."""
    try:
        if _STATE_FILE.is_file():
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Đọc user_state.json lỗi (bỏ qua): %s", exc)
    return {}


def save_state(data: dict) -> None:
    """Ghi đè toàn bộ state xuống đĩa."""
    try:
        _STATE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Ghi user_state.json lỗi: %s", exc)


# --------------------------------------------------------- custom model path
def get_custom_model_path() -> str | None:
    """Đường dẫn custom .pt đã load lần trước (None nếu chưa có)."""
    return load_state().get("custom_model_path")


def set_custom_model_path(path: str) -> None:
    """Lưu đường dẫn custom .pt để lần sau restart vẫn nhớ."""
    st = load_state()
    st["custom_model_path"] = str(path)
    save_state(st)


# --------------------------------------------------------- custom class names
def get_custom_class_names() -> list[str]:
    """Tên class của custom model đã load lần trước ([] nếu chưa có)."""
    names = load_state().get("custom_class_names")
    return list(names) if isinstance(names, list) else []


def set_custom_class_names(names: list[str]) -> None:
    """Lưu tên class để label dropdown hiện đúng ngay khi restart."""
    st = load_state()
    st["custom_class_names"] = list(names)
    save_state(st)


# --------------------------------------------------------- active backend
def get_active_backend() -> str | None:
    """Backend đang dùng lần trước ('yolo' | 'custom'). None nếu chưa lưu."""
    return load_state().get("active_backend")


def set_active_backend(backend: str) -> None:
    """Lưu backend đang dùng để restart app tự khôi phục."""
    st = load_state()
    st["active_backend"] = str(backend)
    save_state(st)


# --------------------------------------------------------- last video path
def get_last_video_path() -> str | None:
    """Đường dẫn video file mở lần trước (None nếu chưa có)."""
    return load_state().get("last_video_path")


def set_last_video_path(path: str) -> None:
    """Lưu đường dẫn video để lần sau mở dialog ở đúng thư mục."""
    st = load_state()
    st["last_video_path"] = str(path)
    save_state(st)
