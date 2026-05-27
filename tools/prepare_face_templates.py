"""
Crop khuôn mặt + resize ảnh mẫu rồi lưu vào templates/.

Chạy:  python tools/prepare_face_templates.py <thư_mục_ảnh_nguồn>
Logic crop dùng processing/face_crop.py (chung với Template Manager trong UI).
"""

import sys
from pathlib import Path

import cv2

# Thêm project root vào sys.path để import được config / processing
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from processing.face_crop import crop_face_tight  # noqa: E402

# Thư mục ảnh nguồn: truyền qua tham số dòng lệnh, mặc định ./input_faces
SRC = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "input_faces")
DST = str(_ROOT / "templates")


def main() -> None:
    from pathlib import Path as _Path
    import os
    os.makedirs(DST, exist_ok=True)

    files = sorted(_Path(SRC).glob("*.jpg"))
    if not files:
        print(f"Không có file .jpg trong: {SRC}")
        return

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  SKIP (đọc lỗi): {p.name}")
            continue

        cropped = crop_face_tight(img)
        if cropped is None:
            print(f"  SKIP (không detect mặt): {p.name}")
            continue

        out = str(_Path(DST) / p.name)
        cv2.imwrite(out, cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = cropped.shape[:2]
        print(f"  {p.name}  →  {w}×{h}px  →  {out}")

    print("\nHoàn thành!")


if __name__ == "__main__":
    main()
