"""
Kiểm tra folder dataset YOLO trước khi train.

Hỗ trợ CẢ HAI cấu trúc export phổ biến:

    Split-first (LabelImg / một số tool):     Folder-first (Roboflow YOLOv8 mặc định):
    folder/                                    folder/
    ├── data.yaml                              ├── data.yaml
    ├── images/                                ├── train/
    │   ├── train/                             │   ├── images/
    │   └── valid/  (hoặc val/)                │   └── labels/
    └── labels/                                └── valid/   (hoặc val/)
        ├── train/                                 ├── images/
        └── valid/                                 └── labels/

Sau khi validate OK, sinh thêm `data_kinectvision.yaml` trong root với
train/val đường dẫn TUYỆT ĐỐI — tránh quirk `../train/images` của Roboflow
khiến ultralytics resolve sai khi CWD khác.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("validator")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Tên file yaml chuẩn hoá do app sinh ra (không đụng data.yaml gốc).
_NORM_YAML_NAME = "data_kinectvision.yaml"


# ---------------------------------------------------------------------------
# Kết quả validate
# ---------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    root: Path
    yaml_path: Path                 # yaml để truyền cho model.train()
    train_count: int
    val_count: int
    class_names: list[str]
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in _IMG_EXTS)


def _count_labels(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.rglob("*.txt") if f.is_file())


def _resolve_split_dir(
    root: Path, split_names: list[str], kind: str
) -> Path | None:
    """
    Tìm thư mục cho 1 split, thử cả 2 layout.

    Args:
        split_names: ví dụ ["train"] hoặc ["valid", "val"].
        kind: "images" | "labels".

    Returns:
        Path tới thư mục đầu tiên tồn tại, hoặc None.
    """
    for split in split_names:
        # Folder-first: root/<split>/<kind>   (Roboflow YOLOv8)
        p = root / split / kind
        if p.is_dir():
            return p
        # Split-first: root/<kind>/<split>     (LabelImg)
        p = root / kind / split
        if p.is_dir():
            return p
    return None


def _load_yaml(yaml_path: Path) -> tuple[dict, str | None]:
    """Parse YAML. Trả về (dict, None) nếu OK hoặc ({}, error_msg)."""
    try:
        import yaml  # ultralytics cài sẵn PyYAML
    except ImportError:
        # Fallback: parse thủ công dạng key: value đơn giản
        data: dict = {}
        try:
            for line in yaml_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    data[k.strip()] = v.strip()
        except Exception as exc:
            return {}, f"Không đọc được data.yaml: {exc}"
        return data, None

    try:
        with yaml_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data, None
    except Exception as exc:
        return {}, f"data.yaml không parse được: {exc}"


def _write_normalized_yaml(
    root: Path, img_train: Path, img_val: Path, class_names: list[str]
) -> Path:
    """
    Sinh data_kinectvision.yaml với train/val đường dẫn tuyệt đối (posix).

    Dùng forward slash để ultralytics đọc OK trên cả Windows; tránh lỗi
    relative path `../train/images` của Roboflow phụ thuộc CWD.
    """
    out = root / _NORM_YAML_NAME
    payload = {
        "path": root.as_posix(),
        "train": img_train.as_posix(),
        "val": img_val.as_posix(),
        "nc": len(class_names),
        "names": list(class_names),
    }
    try:
        import yaml

        text = yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True
        )
    except ImportError:
        names_str = "[" + ", ".join(f"'{n}'" for n in class_names) + "]"
        text = (
            f"path: {payload['path']}\n"
            f"train: {payload['train']}\n"
            f"val: {payload['val']}\n"
            f"nc: {payload['nc']}\n"
            f"names: {names_str}\n"
        )
    out.write_text(text, encoding="utf-8")
    log.info("Đã sinh yaml chuẩn hoá: %s", out)
    return out


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------
def validate_dataset(folder: str | Path) -> DatasetInfo:
    """
    Kiểm tra folder dataset có đúng cấu trúc YOLO (cả 2 layout).

    Returns:
        DatasetInfo với is_valid=True + yaml_path trỏ tới yaml chuẩn hoá,
        hoặc is_valid=False và error message nếu fail.
    """
    root = Path(folder).expanduser().resolve()

    def _fail(msg: str) -> DatasetInfo:
        log.warning("Dataset invalid: %s", msg)
        return DatasetInfo(
            root=root, yaml_path=root / "data.yaml",
            train_count=0, val_count=0, class_names=[],
            is_valid=False, error=msg,
        )

    # 1. Folder tồn tại
    if not root.is_dir():
        return _fail(f"Folder không tồn tại: {root}")

    # 2. data.yaml tồn tại
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return _fail(f"Thiếu data.yaml trong {root}")

    # 3. Parse yaml + lấy class names
    yaml_data, yaml_err = _load_yaml(yaml_path)
    if yaml_err:
        return _fail(yaml_err)

    names_raw = yaml_data.get("names", None)
    if not names_raw:
        return _fail("data.yaml thiếu key 'names' (danh sách class).")
    if isinstance(names_raw, dict):
        class_names = [str(names_raw[k]) for k in sorted(names_raw)]
    elif isinstance(names_raw, list):
        class_names = [str(n) for n in names_raw]
    else:
        return _fail(
            f"'names' trong data.yaml không đúng format: {type(names_raw)}"
        )

    # 4. Thư mục ảnh train — thử cả folder-first và split-first
    img_train = _resolve_split_dir(root, ["train"], "images")
    if img_train is None:
        return _fail(
            "Không tìm thấy thư mục ảnh train "
            "(train/images/ hoặc images/train/)"
        )
    train_count = _count_images(img_train)
    if train_count == 0:
        return _fail(f"{img_train} không có file ảnh ({sorted(_IMG_EXTS)})")

    # 5. Thư mục ảnh valid / val
    img_val = _resolve_split_dir(root, ["valid", "val"], "images")
    if img_val is None:
        return _fail(
            "Không tìm thấy thư mục ảnh valid "
            "(valid/images/, val/images/ hoặc images/valid/)"
        )
    val_count = _count_images(img_val)

    warnings: list[str] = []
    if val_count == 0:
        warnings.append("Thư mục valid không có ảnh — val sẽ empty")

    # 6. Kiểm tra labels/ tương ứng (cảnh báo nếu lệch)
    lbl_train = _resolve_split_dir(root, ["train"], "labels")
    lbl_count = _count_labels(lbl_train) if lbl_train is not None else 0
    if lbl_count == 0:
        warnings.append("Thư mục labels/train trống hoặc không tồn tại")
    elif abs(lbl_count - train_count) > max(5, train_count * 0.05):
        warnings.append(
            f"Số ảnh train ({train_count}) lệch nhiều so với "
            f"số file label ({lbl_count})"
        )

    # 7. Sinh yaml chuẩn hoá (absolute paths) để model.train() dùng
    try:
        norm_yaml = _write_normalized_yaml(
            root, img_train, img_val, class_names
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Không sinh được yaml chuẩn hoá (%s) — dùng data.yaml gốc", exc)
        norm_yaml = yaml_path
        warnings.append(
            "Không ghi được yaml chuẩn hoá — nếu train lỗi path, "
            "kiểm tra quyền ghi thư mục dataset."
        )

    log.info(
        "Dataset OK: %d train / %d val / %d classes (%s) [layout=%s]",
        train_count, val_count, len(class_names), root,
        "folder-first" if (root / "train" / "images").is_dir()
        else "split-first",
    )
    if warnings:
        for w in warnings:
            log.warning("Dataset warning: %s", w)

    return DatasetInfo(
        root=root,
        yaml_path=norm_yaml,
        train_count=train_count,
        val_count=val_count,
        class_names=class_names,
        is_valid=True,
        warnings=warnings,
    )
