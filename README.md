# KinectVision

Nhận diện vật thể realtime + tracking + toạ độ 3D bằng **Kinect V2**, với
detection backend đổi runtime (**YOLO26s** / model custom tự train) và giao
diện PyQt5 3 tab.

![python](https://img.shields.io/badge/python-3.10-blue) ![platform](https://img.shields.io/badge/platform-Windows-lightgrey) ![gpu](https://img.shields.io/badge/GPU-GTX%201060%203GB-76b900)

> Trạng thái: 2026-05-18. Point cloud / Open3D **chưa triển khai** (xem
> [Hướng phát triển](#hướng-phát-triển)).

---

## Tính năng

- 📷 Capture color (1920×1080) + depth (512×424) từ Kinect V2 @ 30 FPS
  (tự fallback webcam → test pattern nếu không có Kinect)
- 🧠 Detection backend **đổi runtime** qua Strategy pattern (`BaseDetector`
  + `ModelFactory`): YOLO26s pretrained (COCO) ↔ model custom tự train ↔
  **Template matching** (zero-training, CPU, 0 VRAM)
- 🎯 **ByteTrack**: gán `track_id` ổn định qua occlusion + lost-bridge chống
  nháy bbox (vẽ nét đứt khi detector miss 1–2 frame)
- 📐 Toạ độ 3D `(X, Y, Z) mm` cho mỗi vật, gốc = tâm camera RGB
- 🎨 Preprocessing: Gaussian blur → unsharp mask → CLAHE
- 🖥️ **3 tab**:
  - **Detection** — Kinect live
  - **Video** — phát file video qua cùng pipeline (không depth → 3D = NaN)
  - **Training** — train/fine-tune model custom trong app, loss chart realtime
- 🎛️ Filter realtime theo class & track ID, ngưỡng confidence, pause/resume
- 💾 Persist custom model + backend + class names qua `user_state.json`
- ♻️ Điều phối VRAM: chỉ 1 model resident, suspend/unload khi đổi tab

## Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| OS | Windows 10/11 | Windows 11 |
| CPU | Intel i3 4 cores | i5/i7 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU | NVIDIA GTX 1060 3GB (CUDA 11+) | RTX 30xx |
| Kinect | Kinect V2 (Xbox One) + adapter USB 3.0 | — |

> ⚠️ **VRAM 3GB**: `yolo26s.pt` ~744MB resident. KHÔNG dùng m-class
> (yolo26m/yolo11m) — ~38ms/frame vỡ realtime 30fps. Chỉ 1 model resident
> tại một thời điểm (`ModelFactory.switch` luôn unload + empty_cache).

## Cài đặt

### 1. Tạo virtual env (Python 3.10)

```powershell
cd "C:\path\to\kinect v2"
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
```

### 2. Cài PyTorch với CUDA (chọn đúng phiên bản CUDA driver)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Không có GPU: bỏ flag `--index-url` để cài bản CPU. Chương trình tự fallback
> CPU khi CUDA không khả dụng/OOM.

### 3. Cài còn lại

```powershell
pip install -r requirements.txt
```

### 4. Model weights (KHÔNG kèm trong repo)

Các file `.pt` / `.onnx` không được commit (binary lớn). Tự tải/export sau khi cài:

```powershell
# Pretrained YOLO (ultralytics tự tải về thư mục gốc khi gọi lần đầu,
# hoặc tải thủ công từ https://github.com/ultralytics/assets/releases)
#   yolo26s.pt  (detection mặc định, máy NVIDIA)

# Máy AMD/RX 580 — export ONNX chữ nhật 16:9 cho DirectML:
python tools\export_onnx.py --model yolo26s.pt --imgsz 544 960
# → sinh yolo26s_960x544.onnx (model mặc định backend onnx)
```

> Model custom (`custom.pt`) và ảnh template (`templates/*.jpg`) không kèm theo —
> tự train hoặc thêm ảnh mẫu theo `templates/README.md`.

### 5. Kinect V2 (tuỳ chọn)

- Cắm Kinect qua adapter USB 3.0 chính hãng vào cổng **USB 3.0**
- Cài [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)
- Verify bằng **Kinect Studio** v2.0 → Monitor

> Chưa có Kinect → tự fallback webcam (depth = pattern test, toạ độ 3D = NaN).

## Chạy

```powershell
.\.venv310\Scripts\Activate.ps1
python main.py
# hoặc:  .\run.bat
```

## Sử dụng UI

### Tab Detection / Video

| Thành phần | Chức năng |
|---|---|
| Model dropdown | Chọn backend: YOLO26s (COCO) / Custom / Template; nút **Reload** |
| Class dropdown | Lọc 1 class hoặc "Tất cả"; gõ để tìm kiếm |
| Track dropdown | Lọc theo track ID cụ thể |
| Confidence slider | Ngưỡng YOLO (10–90%, bước 5%) |
| Pause / Resume | Tạm dừng inference, nguồn vẫn capture |
| Log table | track_id \| class \| conf \| X \| Y \| Z \| cập nhật (click → lọc track) |
| (Tab Video) | Nút mở file + Play/Pause/Restart |

### Tab Training

- Chọn dataset (YOLO format), hyperparams (epochs/batch/imgsz/lr…)
- Fine-tune (`.pt`) hoặc from-scratch (`.yaml`)
- Loss chart realtime; model train xong tự nạp được làm backend Custom

### Backend Template Matching (zero-training)

Đặt ảnh mẫu vào `templates/` (mỗi file = 1 class, tên = filename) → chọn
backend **Template** ở dropdown. Không cần train, chạy CPU thuần (0 VRAM).

| Ứng dụng | Vì sao hợp |
|---|---|
| Vật ngoài 80 class COCO (linh kiện, logo, hộp cụ thể) | Chỉ cần 1 ảnh mẫu, không train |
| Fiducial / calibration marker | Định vị marker chuẩn để hiệu chuẩn 3D |
| Pick-and-place / robot arm | Định vị pixel-level vật cố định → stream `(X,Y,Z)` |
| Đếm sản phẩm giống hệt trên băng chuyền | Multi-match + NMS |
| Fallback khi không GPU / CUDA OOM | 0 VRAM, không phụ thuộc CUDA |

> Giới hạn: **không** bất biến xoay; scale chỉ bất biến nếu bật
> `TEMPLATE_SCALES` đa tỉ lệ (chi phí ×N → cân nhắc 30fps). Vật xoay/biến
> dạng tự do → dùng `yolo`/`custom`. Chi tiết: [`templates/README.md`](templates/README.md).

## Kiến trúc thread

```
KinectCaptureThread ─► FrameRouter ─┬─► detector_queue ─► DetectionThread ─► UI
                                    └─► UI (color+depth)         │
VideoFileCaptureThread ─► (tab Video, cùng pipeline)             └─► detection log
TrainingThread ─► (tab Training, ultralytics model.train)
```

- Queue `maxsize=2`, drop frame nếu đầy → không backpressure
- **torch import trước PyQt5** + CUDA warmup từ main thread (tránh `WinError 1114`)
- DetectionThread backend-agnostic; switch backend qua pending-flag giữa 2 predict

## Cấu trúc thư mục

```
kinect v2/
├── main.py                # Entry point, wiring threads/routers
├── config.py              # Constants tập trung (một nguồn sự thật)
├── app_state.py           # Persist user_state.json
├── core/
│   ├── kinect_capture.py  # Capture Kinect/webcam
│   ├── video_capture.py   # Đọc video file (tab Video)
│   ├── detector.py        # DetectionThread (backend-agnostic)
│   ├── tracker.py         # ObjectTracker facade
│   ├── position.py        # Projection 3D (intrinsics RGB)
│   └── frame_diff.py      # Adaptive skip (đang tắt)
├── models/                # base_detector / yolo_detector / custom_detector
│                          # / template_detector / factory
├── templates/             # ảnh mẫu cho backend Template (xem templates/README.md)
├── tracking/bytetrack.py  # ByteTracker wrap ultralytics
├── training/              # trainer / callbacks / validator
├── processing/            # preprocessor / overlay
├── ui/                    # main_window + detection/video/training panel + widgets
├── tools/                 # benchmark / smoke_tracker / validate_model
│                          # / eval_pipeline / export_tensorrt
│                          # / calibrate_intrinsics
└── tests/                 # pytest (48 tests)
```

## Cấu hình

Toàn bộ constant trong [`config.py`](config.py):

```python
INFER_W, INFER_H      = 960, 540        # resize trước inference
YOLO_MODEL_PATH       = "yolo26s.pt"    # KHÔNG đổi sang m-class
INFERENCE_CONF        = 0.30            # UI slider chỉnh được
INFERENCE_IOU         = 0.45
INFERENCE_IMG_SIZE    = 960            # 800→960 (đo A/B 2026-05-19: det/f ×3)
PREPROCESS_STAGES     = ("unsharp",)   # bỏ blur+clahe (đo: +accuracy, -11ms)
INFERENCE_DEVICE      = "cuda:0"        # tự fallback "cpu" khi OOM
DEPTH_MIN_M, DEPTH_MAX_M = 0.3, 5.0     # range Z hợp lệ
QUEUE_MAXSIZE         = 2               # drop frame nếu đầy
COLOR_FX = COLOR_FY   = 1081.37         # tiêu cự camera RGB
COLOR_CX, COLOR_CY    = 959.5, 539.5    # tâm RGB = gốc (0,0,0)
BT_GRACE_LOST_FRAMES  = 6               # lost-bridge chống nháy (~0.2s)
PAUSE_DETECTION_DURING_TRAINING = True  # unload detection trước khi train
```

> `YOLO_MODEL/YOLO_IOU/YOLO_IMG_SIZE` là **alias** của `INFERENCE_*` (một
> nguồn sự thật cho benchmark + UI), không phải namespace riêng.

## Hệ trục toạ độ

- **Gốc (0,0,0)**: tâm quang học camera **RGB** (cx≈959.5, cy≈539.5)
- **X+**: sang phải · **Y+**: xuống dưới · **Z+**: ra xa camera (luôn > 0)
- Đơn vị: **mm**. Z đọc từ depth (median 5×5, bỏ pixel hole), X/Y chiếu pinhole
  bằng intrinsics RGB. Baseline RGB↔IR ~52mm coi như không đáng kể (sai số <2%).

## Tinh chỉnh hiệu suất / độ chính xác (đo A/B)

Trước khi đổi `PREPROCESS_STAGES`, `INFER_BYPASS_DOWNSCALE`,
`ADAPTIVE_SKIP_ENABLED` hay `INFERENCE_IMG_SIZE` — **đo bằng số liệu**,
không đổi theo cảm tính:

```powershell
.\.venv310\Scripts\python.exe tools\eval_pipeline.py --video clip.mp4
.\.venv310\Scripts\python.exe tools\eval_pipeline.py --frames 120   # synthetic
.\.venv310\Scripts\python.exe tools\eval_pipeline.py --webcam --frames 200
```

In bảng so sánh các biến thể (no-CLAHE / unsharp-only / no-preprocess /
bypass-downscale / imgsz-960 / adaptive-skip) với baseline = config hiện
tại: `dets/frame`, `conf` TB, latency `pre/infer`, `skip%`, và `Δtot`.
Default config KHÔNG đổi tự động — chỉ chỉnh tay sau khi cân nhắc số liệu.

## Test

```powershell
.\.venv310\Scripts\python.exe -m pytest -q
```

## Troubleshooting

| Triệu chứng | Nguyên nhân & xử lý |
|---|---|
| `WinError 1114` khi import torch | torch PHẢI import trước PyQt5 — đã fix `main.py` |
| Kinect không nhận | Kinect Studio verify; cắm USB 3.0 chính hãng |
| `CUDA out of memory` (inference) | Tự latch CPU session (có log); cân nhắc model nhẹ hơn |
| OOM khi switch/training | switch tự fallback CPU; training giảm batch hoặc device=cpu |
| FPS thấp <15 | Xem log `pre/infer/post` ms mỗi tick; bottleneck thường là inference |
| Z = NaN mọi vật | Vật ngoài [0.3, 5.0]m, bề mặt phản xạ IR kém, hoặc nguồn không có depth (video) |
| Box custom hiện sai tên class | Đã fix: switch backend set_class_names trước reset tracker |

## Đánh giá hiệu suất

GTX 1060 3GB + i3-10105F, `yolo26s.pt` FP32:

| Khâu | Thời gian | Ghi chú |
|---|---|---|
| YOLO26s inference (CUDA) | ~20 ms | benchmark thực: ~50fps thô, VRAM 744MB |
| Preprocessing | ~5 ms | trên frame 960×540 |
| Post + 3D projection | ~2 ms | (3D chỉ tính 1 lần khi có tracker) |
| Overlay + Qt convert | ~5 ms | preview resize INTER_LINEAR |
| **End-to-end UI** | — | **~25–30 FPS** |

## Hướng phát triển

- [ ] **Point cloud 3D realtime + Open3D** — spec gốc, chưa build
- [~] **ICoordinateMapper 3D** — code + test xong (`compute_3d_position(cs_map=)`,
      flag `POSITION_USE_COORDINATE_MAPPER`); còn wiring cs_map từ Kinect
      (TODO `core/kinect_capture.py`) — validate trên Kinect thật
- [~] **Calibrate intrinsics** — `tools/calibrate_intrinsics.py` xong; chạy
      với bàn cờ thật → `intrinsics.json` tự override `COLOR_*`
- [~] **TensorRT** — `tools/export_tensorrt.py` xong; build `.engine` trên
      rig GPU (imgsz=960) rồi trỏ `YOLO_MODEL_PATH`
- [ ] Export detection log CSV/JSON
- [ ] YOLO-seg lấy median Z trong mask thay vì bbox center
- [ ] Stream `(X, Y, Z)` qua TCP/MQTT cho robot arm

## License

Internal use. Kinect SDK © Microsoft, YOLO/ultralytics © Ultralytics (AGPL-3.0).
