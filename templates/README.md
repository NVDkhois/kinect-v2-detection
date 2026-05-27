# templates/ — ảnh mẫu cho backend Template Matching

Mỗi file ảnh trong thư mục này là **một class** cho backend `template`
(zero-training, chạy CPU, 0 VRAM).

## Quy ước

- Tên class = **stem filename**. Ví dụ `bottle_cap.png` → class `bottle_cap`.
- Định dạng hỗ trợ: `.png`, `.jpg`, `.jpeg`, `.bmp` (xem `config.TEMPLATE_EXTS`).
- `class_id` ổn định theo thứ tự **sort tên file** (a→z) — không đổi giữa các
  frame trong cùng session (cần cho ByteTrack).
- Ảnh mẫu nên crop sát vật, đủ texture, cùng tỉ lệ với lúc xuất hiện trên
  camera (template matching **không** bất biến xoay; scale chỉ bất biến nếu
  bật `TEMPLATE_SCALES` đa tỉ lệ trong `config.py`).

## Cách dùng

1. Đặt ≥1 ảnh mẫu vào đây (vd `marker.png`).
2. Chạy app → tab Detection/Video → dropdown **Model** chọn
   `Template (N mẫu)`.
3. Slider Confidence = ngưỡng `TM_CCOEFF_NORMED` (mặc định 0.80).

## Tinh chỉnh (`config.py`)

| Hằng | Ý nghĩa |
|---|---|
| `TEMPLATE_MATCH_THRESHOLD` | Ngưỡng match mặc định (UI slider override được) |
| `TEMPLATE_SCALES` | `(1.0,)` = single-scale nhanh; thêm tỉ lệ → bất biến scale nhưng ×N chi phí |
| `TEMPLATE_NMS_IOU` | Gộp đỉnh chồng nhau khi multi-match |

## Khi nào dùng

Vật chuẩn/cố định không có trong 80 class COCO, marker calibration, đếm sản
phẩm giống hệt, hoặc fallback khi không GPU. **Không** hợp cho vật xoay/biến
dạng tự do — khi đó dùng backend `yolo`/`custom`.

> Thư mục rỗng → backend `template` hiển thị "không khả dụng" trong dropdown.
