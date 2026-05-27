"""
Cấu hình toàn cục cho KinectVision.

Tất cả constants (intrinsics, kích thước frame, tham số xử lý, paths)
được khai báo tập trung tại đây. KHÔNG hardcode trong logic ở chỗ khác.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Resize color frame trước khi preprocessing + YOLO inference
# ---------------------------------------------------------------------------
# Aspect ratio 16:9 giữ nguyên với color (1920×1080).
# 960×540 → preprocessing nhanh hơn ~4× so với 1920×1080 (giảm 75% pixel).
# YOLO sau đó tự resize về INFERENCE_IMG_SIZE nội bộ — không mất chi tiết
# vì 960×540 ≈ 1.5× yolo input. Bbox được scale ngược về color space gốc.
INFER_W: int = 960
INFER_H: int = 540

# Bỏ qua bước downscale 1920→960 trước inference: feed thẳng frame gốc cho
# detector (YOLO tự letterbox về imgsz). False = hành vi hiện tại (downscale
# rồi mới preprocess+infer, scale bbox ngược lại). Đặt True chỉ sau khi
# tools/eval_pipeline.py xác nhận accuracy/latency có lợi (xem README).
INFER_BYPASS_DOWNSCALE: bool = False


# ---------------------------------------------------------------------------
# Model backend (Strategy Pattern — xem models/)
# ---------------------------------------------------------------------------
# DetectionThread chỉ làm việc với BaseDetector. Chọn backend qua đây hoặc
# qua dropdown trên UI lúc runtime.
ACTIVE_BACKEND: str = "auto"          # "auto"=theo phần cứng (CUDA→yolo,
                                      # AMD/Intel-DirectML→onnx) | "yolo" |
                                      # "onnx" | "custom" | "template" | "both"

# --- YOLO backend (pretrained COCO) ---
# yolo26s: kiến trúc 2025 mới nhất, NMS-free. Benchmark thực đo trên GTX
# 1060 3GB: ~20ms/frame (50fps thô, realtime an toàn vs ngân sách 33ms),
# VRAM 744MB (dư lớn so ngân sách 3GB, còn headroom cho training). mAP COCO
# ~47-48 vs ~39.5 của nano → +7-8 điểm, cải thiện rõ vật nhỏ/khó.
# Quay về nếu cần: "yolo11s.pt" (~same, arch cũ ổn định hơn) hoặc
# "yolo11n.pt"/"yolov8n.pt" (nano, ~13ms, mAP thấp hơn).
# m-class (yolo26m/yolo11m) KHÔNG dùng: ~38ms > 33ms → vỡ realtime 30fps.
YOLO_MODEL_PATH: str = "yolo26s.pt"
YOLO_CLASS_NAMES = None               # None = dùng COCO 80 classes mặc định

# --- Custom backend (model tự train) ---
CUSTOM_MODEL_PATH: str = "custom.pt"  # đường dẫn .pt tự train
CUSTOM_CLASS_NAMES: list[str] = []    # ["class_a", ...] override; [] / None = đọc từ .pt

# --- ONNX backend (onnxruntime-directml — GPU AMD/Intel trên Windows) ---
# Đường tăng tốc cho máy KHÔNG có CUDA (vd RX 580). torch không chạy nổi YOLO
# trên DirectML; ONNX Runtime + DirectML EP thì chạy tốt. Sinh file .onnx bằng
# tools/export_onnx.py. imgsz + class names đọc TỪ ONNX metadata (ONNX_IMG_SIZE
# chỉ là fallback).
#
# CHỐT model — đo THỰC trên RX 580 (raw infer, DmlEP, 2026-05-24):
#   yolo26s 640×640 (vuông)     = 16.1ms → 62.3fps
#   yolo26s 960×960 (vuông)     = 29.8ms → 33.5fps
#   yolo26s 960×544 (CHỮ NHẬT)  = 19.0ms → 52.5fps  ✅ CHỐT
# → Frame vào detector = 960×540 (16:9). Letterbox về VUÔNG 960² phí ~420 hàng
#   pad xám → GPU tính thừa. Bản CHỮ NHẬT 960×544 có ĐỘ PHÂN GIẢI ẢNH Y HỆT 960²
#   (cùng accuracy) nhưng nhanh hơn 36% → đạt realtime ~52fps. Đây là "960 tối ưu".
#   Input shape đọc thẳng từ ONNX (rect/vuông đều được). Re-đo: bench inline / smoke.
ONNX_MODEL_PATH: str = "yolo26s_960x544.onnx"
ONNX_IMG_SIZE: int = 960              # fallback nếu ONNX không có shape cố định (dynamic)
EXECUTION_PROVIDER: str = "auto"      # "auto" (DML nếu có) | "dml" | "cpu"

# --- Shared inference params (áp dụng cho cả 2 backend) ---
INFERENCE_CONF: float = 0.30         # hạ 0.50→0.30: recall cao hơn, bắt
                                     # nhiều vật hơn; UI slider chỉnh tay được
INFERENCE_IOU: float = 0.45
INFERENCE_DEVICE: str = "cuda:0"      # tự fallback "cpu" nếu OOM
INFERENCE_IMG_SIZE: int = 960        # 800→960 (đo A/B 2026-05-19 clip thật):
                                     # det/f ×3 (1.96→3.25), conf 0.48→0.52,
                                     # tot 28ms < ngân sách 33ms. LƯU Ý: infer
                                     # đo trên GPU khác — xác nhận FPS thực
                                     # trên GTX 1060 qua log profiling; nếu
                                     # vỡ 30fps, hạ về 800 (vẫn > baseline).


# ---------------------------------------------------------------------------
# Benchmark + UI defaults — ALIAS của config runtime (một nguồn sự thật)
# ---------------------------------------------------------------------------
# Trước đây nhóm YOLO_* là namespace SONG SONG, tách rời INFERENCE_* và đặt
# YOLO_MODEL="yolov8s.pt" → (1) vi phạm ràng buộc VRAM 3GB trong CLAUDE.md,
# (2) tools/benchmark.py đo một config KHÁC hẳn cái chạy thật → số liệu vô
# nghĩa. Nay alias trực tiếp sang hằng runtime: benchmark đo đúng pipeline
# thật, không thể lệch, và không còn bẫy OOM yolov8s.
YOLO_MODEL: str = YOLO_MODEL_PATH          # benchmark — luôn khớp runtime
YOLO_CONF_DEFAULT: float = INFERENCE_CONF  # default slider Confidence (UI)
YOLO_IOU: float = INFERENCE_IOU
YOLO_IMG_SIZE: int = INFERENCE_IMG_SIZE


# ---------------------------------------------------------------------------
# Template Matching backend (models/template_detector.py)
# ---------------------------------------------------------------------------
# Backend ZERO-TRAINING: định vị vật chuẩn/cố định bằng ảnh mẫu, chạy CPU
# thuần (0 VRAM — hữu ích khi không GPU hoặc detection model OOM). Mỗi file
# ảnh trong TEMPLATE_DIR = 1 class, tên class = stem filename (KHÔNG hardcode,
# mirror cách YOLODetector lấy class từ model). Hợp cho marker/calibration/
# linh kiện cố định; KHÔNG bất biến xoay, scale phải opt-in — xem README.
TEMPLATE_DIR: str = "templates"
TEMPLATE_MATCH_THRESHOLD: float = 0.55   # dùng với minMaxLoc: chỉ lấy 1 đỉnh tốt nhất/template
                                         # nên threshold thấp hơn an toàn; real face ~0.58
TEMPLATE_NMS_IOU: float = 0.50           # gộp box chồng nhau; cũng dùng cho cross-class NMS
TEMPLATE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
# Đa tỉ lệ: (1.0,) = single-scale (nhanh, giữ 30fps — MẶC ĐỊNH). Thêm tỉ lệ
# (vd (0.75, 1.0, 1.25)) tăng bất biến scale nhưng chi phí ×N mỗi template →
# cân nhắc realtime (xem profiling log DetectionThread: pre/infer/post ms).
TEMPLATE_SCALES: tuple[float, ...] = (1.0,)

# Cấu hình tối ưu hóa chuyên sâu cho nhận diện khuôn mặt
# TEMPLATE_FACE_PREFILTER: True = chỉ chạy matchTemplate trên các vùng khuôn mặt phát hiện bởi Haar Cascade
# Crop khuôn mặt cho template (dùng bởi processing/face_crop.py).
# Padding tính theo tỉ lệ kích thước bbox Haar Cascade.
TEMPLATE_CROP_W: int = 160          # chiều rộng output (px); h tự tính giữ tỉ lệ
TEMPLATE_CROP_PAD_X: float = 0.10   # % mỗi bên ngang — giữ viền má, bỏ nền 2 bên
TEMPLATE_CROP_PAD_TOP: float = 0.18 # % phía trên — lấy ít tóc/trán
TEMPLATE_CROP_PAD_BOT: float = 0.08 # % phía dưới — lấy cằm, không lấy cổ/vai

# Hysteresis: chỉ emit detection sau CONFIRM frame liên tiếp thấy; giữ thêm
# HOLD frame sau khi mất. Triệt tiêu oscillation score quanh threshold.
# CONFIRM=0 hoặc 1 = tắt (emit ngay); HOLD=0 = không giữ.
TEMPLATE_HYSTERESIS_CONFIRM: int = 4   # ~0.13s @30fps (tăng từ 3 → loại ghost ngắn)
TEMPLATE_HYSTERESIS_HOLD: int = 5     # ~0.17s @30fps (giảm từ 10 → ghost biến nhanh hơn)

# Margin tối thiểu giữa class thắng và class về nhì trong face prefilter.
# Khi 2 class score quá gần nhau (không chắc chắn) → bỏ qua, không emit.
# 0.0 = tắt (chỉ cần thắng là đủ).
TEMPLATE_ID_MARGIN: float = 0.04

TEMPLATE_FACE_PREFILTER: bool = True
# TEMPLATE_STRIP_SUFFIX: True = tự động loại bỏ hậu tố gạch dưới và số cuối tên file (ví dụ "nam_1", "nam_2" -> "nam")
TEMPLATE_STRIP_SUFFIX: bool = True


# ---------------------------------------------------------------------------
# Kinect V2 — kích thước frame
# ---------------------------------------------------------------------------
COLOR_W: int = 1920
COLOR_H: int = 1080
DEPTH_W: int = 512
DEPTH_H: int = 424
TARGET_FPS: int = 30


# ---------------------------------------------------------------------------
# Kinect V2 — Camera intrinsics
# ---------------------------------------------------------------------------
# Color camera (1920×1080). Mặc định = intrinsics DANH ĐỊNH Kinect V2.
# Nếu có intrinsics.json hợp lệ (sinh bởi tools/calibrate_intrinsics.py từ
# bàn cờ) → override để giảm sai số X/Y hệ thống theo từng cá thể Kinect.
# File lỗi/thiếu/sai → giữ danh định, KHÔNG crash import (xem core/intrinsics).
INTRINSICS_JSON: str = "intrinsics.json"
from core.intrinsics import resolve_color_intrinsics as _resolve_ci  # noqa: E402
COLOR_FX, COLOR_FY, COLOR_CX, COLOR_CY = _resolve_ci(
    nominal=(1081.37, 1081.37, 959.5, 539.5),
    json_path=INTRINSICS_JSON,
)

# Scale: depth frame lưu giá trị mm → chia cho DEPTH_SCALE để ra mét
DEPTH_SCALE: float = 1000.0

# 3D bằng Kinect ICoordinateMapper thay vì pinhole tuyến tính + map color→
# depth xấp xỉ (đường hiện tại bỏ qua baseline RGB↔IR ~52mm → sai số tới cm
# ở gần). False = đường tuyến tính cũ (mặc định, an toàn). Bật True: cần
# wiring cs_map từ KinectCaptureThread → DetectionThread → compute_3d_position
# (xem TODO trong core/kinect_capture.py; phần này validate trên Kinect THẬT).
POSITION_USE_COORDINATE_MAPPER: bool = False


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
DEPTH_MIN_M: float = 0.3           # gần hơn 30cm bỏ qua
DEPTH_MAX_M: float = 5.0           # xa hơn 5m bỏ qua
QUEUE_MAXSIZE: int = 2             # drop frame nếu queue đầy

# Adaptive frame skip: bỏ qua inference khi cảnh không đổi đáng kể.
# Cứu ~15-20ms/frame khi skip, thêm ~0.4ms diff check.
#
# ĐANG TẮT (thử nghiệm): mọi frame chạy inference → bbox luôn tươi, không
# trễ trên vật chuyển động. yolo26s ~20ms/frame (benchmark cột FP16 không
# skip: 50fps) vẫn realtime vs ngân sách 33ms. Đánh đổi: GPU bận hơn, hết
# headroom dự phòng (queue maxsize=2 tự drop nếu tụt → không tích luỹ lag).
# Bật lại: đổi ADAPTIVE_SKIP_ENABLED = True (DIFF_THRESHOLD / MAX_SKIP_FRAMES
# bên dưới inert khi tắt, giữ nguyên để bật lại dùng ngay).
ADAPTIVE_SKIP_ENABLED: bool = False
DIFF_THRESHOLD: float = 2.0        # 3.5→2.0: nhạy hơn với thay đổi nhỏ →
                                   # bám sát vật chuyển động chậm tốt hơn
MAX_SKIP_FRAMES: int = 3           # 8→3: bbox vật chuyển động không trễ tới
                                   # ~267ms nữa (giờ ~100ms). Đổi GPU headroom
                                   # đang thừa (skip 65-80%) lấy độ bám sát


# ---------------------------------------------------------------------------
# Tracking — multi-object ID stability across occlusion
# ---------------------------------------------------------------------------
TRACKER_BACKEND: str = "bytetrack"  # chỉ "bytetrack" được support (DeepSORT bỏ)

# ByteTrack params (mapping sang ultralytics BYTETracker args namespace)
BT_TRACK_HIGH_THRESH: float = 0.5   # det conf để vào round 1 (high-quality)
BT_TRACK_LOW_THRESH: float = 0.1    # det conf để vào round 2 (low-quality)
BT_NEW_TRACK_THRESH: float = 0.25   # conf để tạo track mới
BT_TRACK_BUFFER: int = 30           # số FRAME giữ track khi mất (ultralytics dùng
                                    # trực tiếp làm max_frames_lost; 30 frame ≈ 1s @ 30fps)
BT_MATCH_THRESH: float = 0.8        # IoU threshold matching
BT_FUSE_SCORE: bool = True

# Chống nháy bbox: detector miss 1-2 frame (conf dao động quanh ngưỡng,
# preprocess CLAHE rung) làm box biến mất rồi hiện lại. ByteTrack vẫn giữ
# track nội bộ → vẽ nó dạng NÉT ĐỨT thêm tối đa N frame để cầu khoảng
# trống. Ngắn → Kalman drift không đáng kể (0 = tắt, về hành vi cũ).
BT_GRACE_LOST_FRAMES: int = 6       # ~0.2s @30fps

# Làm mượt bbox theo track_id (chống nháy: x1/y1/x2/y2 dao động từng
# frame — nhiễu regression detector, Kalman ByteTrack tune cho MOT nên
# gần như không lọc). EMA độc lập CẢ 4 MÉP → hết nháy cả dọc lẫn ngang.
# Đánh đổi: toạ độ 3D (tính theo tâm bbox) có độ trễ NHỎ khi vật di
# chuyển nhanh. 1.0 = TẮT (box thô, byte-for-byte hành vi cũ). 0.3–0.5 =
# khuyến nghị; <0.3 mượt hơn nhưng box "đuổi" theo vật chậm hơn.
BBOX_SMOOTH_ALPHA: float = 0.4      # BẬT (mượt 4 mép); 1.0 = tắt


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
GAUSSIAN_KSIZE: tuple[int, int] = (3, 3)
UNSHARP_SIGMA: float = 1.0
UNSHARP_STRENGTH: float = 1.5
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)

# Tầng preprocessing, thứ tự CANONICAL: blur → unsharp → clahe.
# ĐÃ ĐO A/B (tools/eval_pipeline.py, clip thật, 2026-05-19): bỏ blur+clahe,
# chỉ giữ unsharp → det/f +77% (1.11→1.96), conf +0.046, pre 15→4ms. Xác
# nhận giả thuyết: CLAHE/blur lệch phân phối input COCO của yolo26s → giảm
# chất lượng detect. Quay về ("blur","unsharp","clahe") nếu cảnh khác cần.
PREPROCESS_STAGES: tuple[str, ...] = ("unsharp",)


# ---------------------------------------------------------------------------
# UI / Display
# ---------------------------------------------------------------------------
DISPLAY_W: int = 640
DISPLAY_H: int = 480

UI_WINDOW_TITLE: str = "KinectVision — Realtime 3D Object Detection"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_EPOCHS: int = 50
TRAIN_BATCH: int = 8              # safe cho GTX 1060 3GB với imgsz=640
TRAIN_IMGSZ: int = 640
TRAIN_BASE_MODEL: str = "yolov8n.pt"
TRAIN_OUTPUT_DIR: str = "runs/train"
TRAIN_PATIENCE: int = 15          # early stopping nếu không cải thiện
TRAIN_WORKERS: int = 2            # i3-10105F = 4 core → dùng 2
TRAIN_DEVICE: str = "0"           # "0"=GPU, "cpu"=CPU fallback

# Khi training dùng GPU, detection model cũng đang giữ VRAM.
# True → unload detection model trước khi train (tránh OOM trên 3GB).
PAUSE_DETECTION_DURING_TRAINING: bool = True

# From-scratch defaults — khác với fine-tune (ít epoch, warmup ngắn)
TRAIN_SCRATCH_EPOCHS:     int   = 200
TRAIN_SCRATCH_WARMUP:     int   = 5
TRAIN_SCRATCH_LR0:        float = 0.01
TRAIN_SCRATCH_LRF:        float = 0.01   # lr_final = LR0 * LRF
TRAIN_MIN_IMAGES_SCRATCH: int   = 300    # cảnh báo nếu dataset nhỏ hơn

# Map .pt → .yaml kiến trúc tương ứng (ultralytics tự tìm trong package).
# Dùng khi pretrained=False để YOLO("yolov8n.yaml") thay vì YOLO("yolov8n.pt").
ARCH_YAML_MAP: dict[str, str] = {
    "yolov8n.pt": "yolov8n.yaml",
    "yolov8s.pt": "yolov8s.yaml",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s"
LOG_DATEFMT: str = "%H:%M:%S"
