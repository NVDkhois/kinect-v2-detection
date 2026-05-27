"""
patch_pykinect2 — vá 3 bug tương thích pykinect2 trên Python 3.10 + comtypes 1.4.x.

pykinect2 (PyPI) viết cho Python 2 / comtypes cũ; trên .venv310 hiện tại nó vỡ ở
3 chỗ. Patch nằm trong site-packages nên MẤT mỗi lần `pip install --force` /
tạo lại venv → script này vá lại idempotent (chạy nhiều lần không hỏng).

Ba patch:
  1. PyKinectV2.py  — `assert sizeof(tagSTATSTG)==72` sai trên 64-bit (struct
     thiếu field) → đổi sang `assert alignment(...)==8` (không chặn import).
  2. PyKinectV2.py  — `_check_version('')` ở cuối file: pykinect2 sinh code thời
     comtypes version rỗng, không khớp comtypes hiện tại → comment lại.
  3. PyKinectRuntime.py — `time.clock()` bị xoá từ Python 3.8 (dùng 9 chỗ) →
     chèn shim `time.clock = time.perf_counter` ngay sau `import time`.

Dùng:
    .\.venv310\Scripts\python.exe tools\patch_pykinect2.py          # vá
    .\.venv310\Scripts\python.exe tools\patch_pykinect2.py --check  # chỉ kiểm tra

Xem docs/ke_hoach_chuyen_rx580.md (Pha 0/4).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_MARK = "# PATCH (RX580"  # dấu nhận biết đã vá (idempotent)


def _pykinect2_dir() -> Path:
    """Định vị thư mục package pykinect2 trong site-packages đang chạy."""
    try:
        import pykinect2  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "Không import được pykinect2 — cài trước: pip install pykinect2 comtypes==1.1.11"
        ) from exc
    return Path(pykinect2.__file__).resolve().parent


def _patch_assert_sizeof(text: str) -> tuple[str, bool]:
    """Patch 1: assert sizeof(tagSTATSTG)==<n> → assert alignment==8."""
    if "alignment(tagSTATSTG) == 8" in text:
        return text, False  # đã vá
    pat = re.compile(r"^assert sizeof\(tagSTATSTG\) == \d+, sizeof\(tagSTATSTG\)$", re.M)
    if not pat.search(text):
        return text, False  # không tìm thấy (phiên bản khác) → bỏ qua an toàn
    repl = (
        "# PATCH (RX580/comtypes 1.4.x): assert kích thước struct sai trên 64-bit\n"
        "# (pykinect2 định nghĩa thiếu field pwcsName) → nới thành alignment check.\n"
        "assert alignment(tagSTATSTG) == 8, alignment(tagSTATSTG)"
    )
    return pat.sub(repl, text, count=1), True


def _patch_check_version(text: str) -> tuple[str, bool]:
    """Patch 2: comment dòng gọi _check_version('')."""
    pat = re.compile(r"^from comtypes import _check_version; _check_version\(''\)$", re.M)
    if not pat.search(text):
        return text, False  # đã comment hoặc không có
    repl = (
        "# PATCH (RX580): bỏ guard _check_version('') — code sinh thời comtypes\n"
        "# version rỗng, không khớp comtypes hiện tại nhưng định nghĩa COM vẫn dùng được.\n"
        "# from comtypes import _check_version; _check_version('')"
    )
    return pat.sub(repl, text, count=1), True


def _patch_time_clock(text: str) -> tuple[str, bool]:
    """Patch 3: chèn shim time.clock = time.perf_counter sau `import time`."""
    if "time.clock = time.perf_counter" in text:
        return text, False  # đã vá
    pat = re.compile(r"^import time$", re.M)
    if not pat.search(text):
        return text, False
    shim = (
        "import time\n"
        "# PATCH (RX580/Py3.10): time.clock() bị xoá từ Python 3.8. pykinect2 còn\n"
        "# dùng → shim sang perf_counter để Kinect init được.\n"
        "if not hasattr(time, 'clock'):\n"
        "    time.clock = time.perf_counter"
    )
    return pat.sub(shim, text, count=1), True


def _apply(path: Path, patchers, check_only: bool) -> int:
    """Áp các patcher lên 1 file. Trả về số patch đã/cần áp."""
    if not path.is_file():
        print(f"  ⚠ không thấy {path.name} — bỏ qua")
        return 0
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    applied = 0
    for fn in patchers:
        text, changed = fn(text)
        if changed:
            applied += 1
            print(f"  {'[CẦN VÁ]' if check_only else '[ĐÃ VÁ ]'} {fn.__name__} @ {path.name}")
    if applied and not check_only:
        path.write_text(text, encoding="utf-8", errors="surrogateescape")
    return applied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Vá pykinect2 cho Python 3.10 (RX 580)")
    ap.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không sửa file")
    args = ap.parse_args(argv)

    pkg = _pykinect2_dir()
    print(f"pykinect2: {pkg}")

    total = _apply(
        pkg / "PyKinectV2.py",
        (_patch_assert_sizeof, _patch_check_version),
        args.check,
    )
    total += _apply(
        pkg / "PyKinectRuntime.py",
        (_patch_time_clock,),
        args.check,
    )

    if total == 0:
        print("✓ Tất cả patch đã có sẵn (idempotent — không cần làm gì).")
    elif args.check:
        print(f"→ {total} patch CHƯA áp. Chạy lại không có --check để vá.")
        return 1
    else:
        print(f"✓ Đã áp {total} patch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
